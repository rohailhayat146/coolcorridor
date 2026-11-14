
import os
import joblib
import networkx as nx
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyproj import Transformer

# --------------------------------------------------
# LOAD ROUTING ENGINE
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GRAPH_PATH = os.path.join(
    BASE_DIR,
    "graph",
    "coolcorridor_routing_engine.joblib"
)

package = joblib.load(GRAPH_PATH)

G = package["graph"]
EDGE_GEOMETRIES = package["edge_geometries"]
NODE_COORDINATES = package["node_coordinates"]
METADATA = package["metadata"]

TRANSFORMER = Transformer.from_crs(
    METADATA["graph_crs"],
    METADATA["output_crs"],
    always_xy=True
)

EDGE_KEY_LOOKUP = {
    (u, v, key): key
    for u, v, key in G.edges(keys=True)
}

# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="CoolCorridor API",
    description="Heat-aware walking route planner for Phoenix.",
    version="1.0.0"
)

# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class RouteRequest(BaseModel):

    origin: int = Field(..., description="Origin OSM node ID")
    destination: int = Field(..., description="Destination OSM node ID")

# --------------------------------------------------
# ROUTING FUNCTIONS
# --------------------------------------------------

def calculate_route(origin, destination, scenario, weight):

    try:
        path = nx.shortest_path(
            G,
            source=origin,
            target=destination,
            weight=weight,
            method="dijkstra"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise HTTPException(
            status_code=404,
            detail="No walking route found."
        )

    coordinates = []
    segment_data = []

    for u, v in zip(path[:-1], path[1:]):

        parallel_edges = G.get_edge_data(u, v)

        best_key, best_data = min(
            parallel_edges.items(),
            key=lambda item: item[1][weight]
        )

        geometry = EDGE_GEOMETRIES[(u, v, best_key)]
        coords = list(geometry.coords)

        start_xy = NODE_COORDINATES.loc[u, ["x", "y"]].to_numpy(dtype=float)
        end_xy = NODE_COORDINATES.loc[v, ["x", "y"]].to_numpy(dtype=float)

        forward_error = np.sum((np.array(coords[0][:2]) - start_xy) ** 2)
        reverse_error = np.sum((np.array(coords[-1][:2]) - start_xy) ** 2)

        if reverse_error < forward_error:
            coords.reverse()

        if coordinates:
            coords = coords[1:]

        for x, y in coords:
            lon, lat = TRANSFORMER.transform(x, y)
            coordinates.append([lon, lat])

        segment_data.append(best_data)

    lengths = np.array([d["length"] for d in segment_data])
    scores = np.array([d["heat_exposure_score"] for d in segment_data])
    lst = np.array([d["median_lst_c"] for d in segment_data])
    ndvi = np.array([d["median_ndvi"] for d in segment_data])

    total_distance = lengths.sum()

    return {
        "type": "Feature",
        "properties": {
            "scenario": scenario,
            "distance_km": float(total_distance / 1000),
            "walking_time_min": float(total_distance / 1.2 / 60),
            "segments": len(segment_data),
            "mean_exposure_score": float(np.average(scores, weights=lengths)),
            "mean_lst_c": float(np.average(lst, weights=lengths)),
            "mean_ndvi": float(np.average(ndvi, weights=lengths)),
            "exposure_proxy": float(np.sum(lengths * scores / 100))
        },
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates
        }
    }

# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "project": "CoolCorridor",
        "status": "running",
        "city": METADATA["city"]
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "strongly_connected": nx.is_strongly_connected(G)
    }


@app.post("/route")
def get_routes(request: RouteRequest):

    origin = request.origin
    destination = request.destination

    if origin == destination:
        raise HTTPException(
            status_code=400,
            detail="Origin and destination must be different."
        )

    if origin not in G or destination not in G:
        raise HTTPException(
            status_code=404,
            detail="Origin or destination is outside the supported network."
        )

    scenarios = [
        ("Shortest", "length"),
        ("Lower exposure (lambda=1)", "cost_lambda_1"),
        ("Lower exposure (lambda=2)", "cost_lambda_2")
    ]

    features = [
        calculate_route(origin, destination, name, weight)
        for name, weight in scenarios
    ]

    return {
        "type": "FeatureCollection",
        "metadata": {
            "study_area": "Downtown Phoenix, Arizona",
            "coordinate_system": "EPSG:4326",
            "origin_node": origin,
            "destination_node": destination,
            "walking_speed_mps": 1.2,
            "exposure_score_note": (
                "Relative environmental proxy, not measured pedestrian heat exposure."
            )
        },
        "features": features
    }

# --------------------------------------------------
# GEOGRAPHIC INPUT AND AUTOMATIC NETWORK SNAPPING
# --------------------------------------------------

from scipy.spatial import cKDTree

class CoordinateRouteRequest(BaseModel):

    origin_latitude: float = Field(..., ge=-90, le=90)
    origin_longitude: float = Field(..., ge=-180, le=180)

    destination_latitude: float = Field(..., ge=-90, le=90)
    destination_longitude: float = Field(..., ge=-180, le=180)


# Build spatial index once when the application loads
NODE_IDS = np.asarray(NODE_COORDINATES.index, dtype=np.int64)

NODE_XY = NODE_COORDINATES[
    ["x", "y"]
].to_numpy(dtype=float)

NODE_TREE = cKDTree(NODE_XY)

MAX_SNAP_DISTANCE_METRES = 250


def snap_to_walking_network(latitude, longitude):

    input_transformer = Transformer.from_crs(
        "EPSG:4326",
        METADATA["graph_crs"],
        always_xy=True
    )

    x, y = input_transformer.transform(longitude, latitude)

    distance, index = NODE_TREE.query(
        [x, y],
        k=1
    )

    if distance > MAX_SNAP_DISTANCE_METRES:
        raise HTTPException(
            status_code=422,
            detail=(
                "No walking-network node found within "
                "250 metres of the selected location."
            )
        )

    return int(NODE_IDS[index]), float(distance)


@app.post("/route/coordinates")
def get_routes_from_coordinates(request: CoordinateRouteRequest):

    origin, origin_snap_distance = snap_to_walking_network(
        request.origin_latitude,
        request.origin_longitude
    )

    destination, destination_snap_distance = snap_to_walking_network(
        request.destination_latitude,
        request.destination_longitude
    )

    if origin == destination:
        raise HTTPException(
            status_code=400,
            detail="Origin and destination snap to the same walking-network node."
        )

    scenarios = [
        ("Shortest", "length"),
        ("Lower exposure (lambda=1)", "cost_lambda_1"),
        ("Lower exposure (lambda=2)", "cost_lambda_2")
    ]

    features = [
        calculate_route(origin, destination, name, weight)
        for name, weight in scenarios
    ]

    return {
        "type": "FeatureCollection",

        "metadata": {
            "study_area": "Downtown Phoenix, Arizona",
            "coordinate_system": "EPSG:4326",
            "origin_node": origin,
            "destination_node": destination,
            "origin_snap_distance_m": origin_snap_distance,
            "destination_snap_distance_m": destination_snap_distance,
            "walking_speed_mps": 1.2,
            "exposure_score_note": (
                "Relative environmental proxy, not measured pedestrian heat exposure."
            )
        },

        "features": features
    }


# --------------------------------------------------


# CORS CONFIGURATION

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173"
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)
