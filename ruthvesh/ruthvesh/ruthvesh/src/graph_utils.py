"""
graph_utils.py

Reusable utility functions for working with the Nagpur road graph.

This module provides helpers for loading the saved graph, inspecting it,
finding the nearest node to a coordinate, and validating that required
attributes are present. It does NOT implement routing or congestion logic.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import osmnx as ox

try:
    from src.config import RAW_GRAPH_PATH, WEIGHT_ATTRIBUTE
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import RAW_GRAPH_PATH, WEIGHT_ATTRIBUTE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Loading the saved graph
# ---------------------------------------------------------------------------

def load_graph(graph_path: Path = RAW_GRAPH_PATH) -> nx.MultiDiGraph:
    """
    Load a road network graph from a GraphML file.

    Parameters
    ----------
    graph_path : Path
        Path to the .graphml file to load. Defaults to the raw graph
        saved by graph_builder.py.

    Returns
    -------
    networkx.MultiDiGraph
        The loaded graph.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    """
    graph_path = Path(graph_path)

    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph file not found at '{graph_path}'. "
            "Run graph_builder.py first to download and save the graph."
        )

    logger.info("Loading graph from: %s", graph_path)
    graph = ox.load_graphml(graph_path)
    logger.info(
        "Graph loaded: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


# ---------------------------------------------------------------------------
# 2. Basic graph statistics
# ---------------------------------------------------------------------------

def get_basic_stats(graph: nx.MultiDiGraph) -> Dict[str, Any]:
    """
    Compute basic statistics for a road network graph.

    This function does not modify the input graph. A projected copy is
    used internally where distance-based statistics require it.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to summarize.

    Returns
    -------
    dict
        Dictionary containing:
        - num_nodes
        - num_edges
        - is_directed
        - avg_degree (mean out-degree across nodes)
        - basic_stats: OSMnx's ox.basic_stats() output (computed on a
          projected copy of the graph)
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    if num_nodes == 0:
        avg_degree = 0.0
    else:
        avg_degree = sum(dict(graph.out_degree()).values()) / num_nodes

    try:
        # project_graph returns a new graph; does not mutate the original
        graph_proj = ox.project_graph(graph)
        osmnx_stats = ox.basic_stats(graph_proj)
    except Exception as exc:
        logger.warning("Could not compute OSMnx basic_stats: %s", exc)
        osmnx_stats = {}

    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "is_directed": graph.is_directed(),
        "avg_out_degree": avg_degree,
        "basic_stats": osmnx_stats,
    }


# ---------------------------------------------------------------------------
# 3. Finding the nearest graph node to a lat/lon coordinate
# ---------------------------------------------------------------------------

def find_nearest_node(
    graph: nx.MultiDiGraph,
    latitude: float,
    longitude: float,
) -> Optional[int]:
    """
    Find the graph node nearest to a given latitude/longitude coordinate.

    Coordinates are expected and returned in standard lat/lon (EPSG:4326)
    convention. Internally this calls osmnx.distance.nearest_nodes, which
    expects (X=longitude, Y=latitude).

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to search (must have 'x' and 'y' node attributes, as
        produced by OSMnx).
    latitude : float
        Latitude of the query point. Must be in range [-90, 90].
    longitude : float
        Longitude of the query point. Must be in range [-180, 180].

    Returns
    -------
    int or None
        The OSM node ID of the nearest node, or None if the coordinates
        are invalid or the lookup fails.
    """
    if not _is_valid_coordinate(latitude, longitude):
        logger.error(
            "Invalid coordinate provided: latitude=%s, longitude=%s",
            latitude, longitude,
        )
        return None

    if graph.number_of_nodes() == 0:
        logger.error("Graph has no nodes; cannot find nearest node.")
        return None

    try:
        nearest_node = ox.distance.nearest_nodes(graph, X=longitude, Y=latitude)
    except Exception as exc:
        logger.error(
            "Failed to find nearest node for (lat=%s, lon=%s): %s",
            latitude, longitude, exc,
        )
        return None

    logger.info(
        "Nearest node to (lat=%s, lon=%s) is node %s",
        latitude, longitude, nearest_node,
    )
    return nearest_node


def _is_valid_coordinate(latitude: float, longitude: float) -> bool:
    """
    Check whether a latitude/longitude pair is a valid, real coordinate.

    Parameters
    ----------
    latitude : float
    longitude : float

    Returns
    -------
    bool
        True if both values are numeric and within valid geographic ranges.
    """
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False

    if lat != lat or lon != lon:  # NaN check
        return False

    if not (-90.0 <= lat <= 90.0):
        return False

    if not (-180.0 <= lon <= 180.0):
        return False

    return True


# ---------------------------------------------------------------------------
# 4. Inspecting edge attributes
# ---------------------------------------------------------------------------

def inspect_edge_attributes(
    graph: nx.MultiDiGraph,
    sample_size: int = 5,
) -> Dict[str, Any]:
    """
    Inspect edge attributes present in the graph.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to inspect.
    sample_size : int
        Number of sample edges to include in the returned output.

    Returns
    -------
    dict
        Dictionary containing:
        - all_attribute_keys: sorted list of every attribute name found
          across all edges
        - sample_edges: list of up to `sample_size` edges, each as
          (u, v, key, attributes_dict)
    """
    all_keys = set()
    sample_edges: List[Tuple[Any, Any, Any, Dict[str, Any]]] = []

    for i, (u, v, key, data) in enumerate(graph.edges(keys=True, data=True)):
        all_keys.update(data.keys())
        if i < sample_size:
            sample_edges.append((u, v, key, data))

    return {
        "all_attribute_keys": sorted(all_keys),
        "sample_edges": sample_edges,
    }


# ---------------------------------------------------------------------------
# 5. Validating required graph attributes
# ---------------------------------------------------------------------------

def validate_graph_attributes(
    graph: nx.MultiDiGraph,
    required_node_attrs: Iterable[str] = ("x", "y"),
    required_edge_attrs: Iterable[str] = ("length",),
) -> Dict[str, Any]:
    """
    Validate that required node and edge attributes exist in the graph.

    This is a read-only check; it does not modify the graph or raise on
    missing attributes. Callers can inspect the returned report to decide
    how to handle gaps (e.g. imputing missing speed values later).

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to validate.
    required_node_attrs : Iterable[str]
        Node attribute names expected to be present on every node
        (defaults to 'x' and 'y', used for coordinates).
    required_edge_attrs : Iterable[str]
        Edge attribute names expected to be present on every edge
        (defaults to 'length').

    Returns
    -------
    dict
        Dictionary containing:
        - is_valid: True only if every required attribute is present on
          every node/edge
        - missing_node_attrs: dict mapping attribute -> count of nodes
          missing it
        - missing_edge_attrs: dict mapping attribute -> count of edges
          missing it
        - total_nodes, total_edges
    """
    required_node_attrs = list(required_node_attrs)
    required_edge_attrs = list(required_edge_attrs)

    total_nodes = graph.number_of_nodes()
    total_edges = graph.number_of_edges()

    missing_node_attrs = {attr: 0 for attr in required_node_attrs}
    for _, data in graph.nodes(data=True):
        for attr in required_node_attrs:
            if attr not in data:
                missing_node_attrs[attr] += 1

    missing_edge_attrs = {attr: 0 for attr in required_edge_attrs}
    for _, _, data in graph.edges(data=True):
        for attr in required_edge_attrs:
            if attr not in data:
                missing_edge_attrs[attr] += 1

    is_valid = all(count == 0 for count in missing_node_attrs.values()) and all(
        count == 0 for count in missing_edge_attrs.values()
    )

    if is_valid:
        logger.info("Graph validation passed: all required attributes present.")
    else:
        logger.warning(
            "Graph validation found missing attributes. "
            "Missing node attrs: %s | Missing edge attrs: %s",
            missing_node_attrs, missing_edge_attrs,
        )

    return {
        "is_valid": is_valid,
        "missing_node_attrs": missing_node_attrs,
        "missing_edge_attrs": missing_edge_attrs,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    }