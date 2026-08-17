"""
config.py

Centralized configuration for the NovaRoute.AI Ruthvesh module
(Nagpur road-network data pipeline).

This file does NOT download data or implement routing logic.
It only defines constants and paths used by other modules
(graph_builder.py, graph_utils.py, travel_time.py, routing.py).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root / base directories
# ---------------------------------------------------------------------------

# src/ -> ruthvesh/ (module root)
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# OSMnx query configuration (Nagpur)
# ---------------------------------------------------------------------------

# Place name used for OSMnx's graph_from_place() query.
PLACE_NAME = "Nagpur, Maharashtra, India"

# Network type passed to OSMnx (drive = roads usable by motor vehicles).
NETWORK_TYPE = "drive"

# ---------------------------------------------------------------------------
# Raw graph data paths
# ---------------------------------------------------------------------------

# Raw graph pulled directly from OSMnx, stored before any preprocessing.
RAW_GRAPH_PATH = RAW_DATA_DIR / "nagpur_drive.graphml"

# ---------------------------------------------------------------------------
# Processed graph data paths
# ---------------------------------------------------------------------------

# Graph after cleaning/preprocessing (e.g. simplified, projected, travel
# times attached), ready to be used for routing.
PROCESSED_GRAPH_PATH = PROCESSED_DATA_DIR / "nagpur_routing.graphml"

# Optional: processed graph edges/nodes exported as tabular data,
# useful for inspection or downstream analysis.
PROCESSED_EDGES_CSV = PROCESSED_DATA_DIR / "nagpur_edges.csv"
PROCESSED_NODES_CSV = PROCESSED_DATA_DIR / "nagpur_nodes.csv"

# ---------------------------------------------------------------------------
# Default speed assumptions (km/h)
# ---------------------------------------------------------------------------

# Fallback speed used when an edge has no "maxspeed" tag from OSM data.
DEFAULT_SPEED_KMPH = 30

# Speed assumptions per OSM highway type, used when imputing missing
# maxspeed values. Values are illustrative defaults for an Indian city
# context and can be tuned later.
HIGHWAY_SPEED_DEFAULTS_KMPH = {
    "motorway": 80,
    "trunk": 60,
    "primary": 50,
    "secondary": 40,
    "tertiary": 30,
    "residential": 25,
    "living_street": 15,
    "unclassified": 25,
    "service": 15,
}

# ---------------------------------------------------------------------------
# Congestion configuration
# ---------------------------------------------------------------------------

# Default congestion factor (multiplier applied to free-flow travel time).
# 1.0 = no congestion. Values > 1.0 slow down travel time on an edge.
# This is a placeholder for the future dynamic congestion-adjusted
# edge-weight feature; for now it is applied uniformly if needed.
DEFAULT_CONGESTION_FACTOR = 1.0

# ---------------------------------------------------------------------------
# Routing-related constants
# ---------------------------------------------------------------------------

# Edge weight attribute name used by Dijkstra-based routing.
WEIGHT_ATTRIBUTE = "travel_time"

# Coordinate Reference System used for distance-based calculations
# (metric CRS, meters as unit). Used when projecting the graph.
PROJECTED_CRS = "EPSG:32643"  # UTM zone 43N, covers Nagpur

# Geographic CRS used for raw OSM data (lat/lon).
GEOGRAPHIC_CRS = "EPSG:4326"