"""
routing.py

Dijkstra-based shortest travel-time routing on the processed Nagpur road
graph.

This module is intentionally decoupled from graph downloading/processing
(see graph_builder.py) and from coordinate lookup internals (see
graph_utils.py). It only consumes an already-loaded, already-processed
NetworkX graph and answers routing queries against it.

Weight modes
------------
An edge in the processed graph can carry two different travel-time
attributes:

    base_travel_time     -> stored under config.WEIGHT_ATTRIBUTE
                             (currently "travel_time"), the static,
                             free-flow travel time computed by
                             travel_time.py.
    dynamic_travel_time   -> stored under
                             congestion.DYNAMIC_WEIGHT_ATTRIBUTE
                             (currently "dynamic_travel_time"), computed
                             by congestion.py as:

                                 dynamic_travel_time =
                                     base_travel_time * congestion_factor

Every routing function in this module accepts a `weight_mode` parameter:

    weight_mode="base"    -> route using base_travel_time (default,
                              identical to this module's original,
                              pre-congestion behavior).
    weight_mode="dynamic" -> route using dynamic_travel_time (i.e.
                              congestion-adjusted travel time), if it has
                              been computed on the graph (see
                              congestion.apply_congestion_to_graph()).

This module does NOT compute congestion_factor or dynamic_travel_time
itself (see congestion.py for that) and does NOT implement vehicle
detection (YOLO), officer allocation, or any API layer.

Design note for future integration:
All public functions here return plain, JSON-serializable Python types
(dicts, lists, floats, ints, None) rather than custom classes, so this
module can be called directly from a future FastAPI backend without any
extra translation layer.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

try:
    from src.config import WEIGHT_ATTRIBUTE
    from src.congestion import DYNAMIC_WEIGHT_ATTRIBUTE
    from src.graph_utils import find_nearest_node
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import WEIGHT_ATTRIBUTE
    from src.congestion import DYNAMIC_WEIGHT_ATTRIBUTE
    from src.graph_utils import find_nearest_node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight mode configuration
# ---------------------------------------------------------------------------

# Public constants for the two supported weight modes, so callers don't
# need to hardcode the strings "base" / "dynamic".
WEIGHT_MODE_BASE = "base"
WEIGHT_MODE_DYNAMIC = "dynamic"
VALID_WEIGHT_MODES = (WEIGHT_MODE_BASE, WEIGHT_MODE_DYNAMIC)

# Maps each weight_mode to the underlying graph edge attribute it reads.
_WEIGHT_MODE_TO_ATTRIBUTE = {
    WEIGHT_MODE_BASE: WEIGHT_ATTRIBUTE,          # "travel_time" (base_travel_time)
    WEIGHT_MODE_DYNAMIC: DYNAMIC_WEIGHT_ATTRIBUTE,  # "dynamic_travel_time"
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_weight_attribute(
    weight_mode: str = WEIGHT_MODE_BASE,
    weight_attribute: Optional[str] = None,
) -> str:
    """
    Resolve which graph edge attribute name to use as the Dijkstra weight.

    Resolution order:
    1. If `weight_attribute` is explicitly provided, it always wins (this
       preserves full backward compatibility for any caller that still
       wants to pass a raw attribute name directly, and lets advanced
       callers point at a custom attribute).
    2. Otherwise, `weight_mode` selects the attribute:
       - "base"    -> config.WEIGHT_ATTRIBUTE ("travel_time")
       - "dynamic" -> congestion.DYNAMIC_WEIGHT_ATTRIBUTE
                      ("dynamic_travel_time")

    Parameters
    ----------
    weight_mode : str
        Either "base" or "dynamic". Ignored if `weight_attribute` is set.
    weight_attribute : str or None
        Explicit edge attribute name override.

    Returns
    -------
    str
        The edge attribute name to use as the Dijkstra weight.

    Raises
    ------
    ValueError
        If `weight_attribute` is None and `weight_mode` is not a
        recognized value.
    """
    if weight_attribute is not None:
        return weight_attribute

    if weight_mode not in VALID_WEIGHT_MODES:
        raise ValueError(
            f"Invalid weight_mode {weight_mode!r}; must be one of "
            f"{VALID_WEIGHT_MODES}."
        )

    return _WEIGHT_MODE_TO_ATTRIBUTE[weight_mode]


def _validate_dynamic_weight_available(
    graph: nx.MultiDiGraph,
    weight_mode: str,
    resolved_weight_attribute: str,
) -> Optional[str]:
    """
    Guard against silently-wrong routing when dynamic mode is requested
    but the graph has no congestion-adjusted weights yet.

    NetworkX's Dijkstra falls back to a default weight of 1 for any edge
    missing the requested attribute, rather than raising an error. If a
    caller asks for weight_mode="dynamic" on a graph that never had
    congestion.apply_congestion_to_graph() run on it, every edge would
    silently fall back to weight=1 (i.e. Dijkstra would minimize hop
    count, not travel time) with no indication anything was wrong. This
    check catches that case explicitly instead.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to check.
    weight_mode : str
        The weight mode requested ("base" or "dynamic").
    resolved_weight_attribute : str
        The edge attribute name that resolves to.

    Returns
    -------
    str or None
        A human-readable error message if dynamic routing was requested
        but no edge in the graph has the resolved attribute, else None.
    """
    if weight_mode != WEIGHT_MODE_DYNAMIC:
        return None

    has_any_dynamic_weight = any(
        resolved_weight_attribute in data
        for _, _, data in graph.edges(data=True)
    )

    if not has_any_dynamic_weight:
        return (
            f"weight_mode='dynamic' was requested, but no edge in the "
            f"graph has a '{resolved_weight_attribute}' attribute. "
            "Run congestion.apply_congestion_to_graph() on the graph "
            "first, or use weight_mode='base'."
        )

    return None


def _get_min_weight_edge_data(
    graph: nx.MultiDiGraph,
    u: Any,
    v: Any,
    weight_attribute: str,
) -> Dict[str, Any]:
    """
    Get the attribute dict of the parallel edge between u and v that has
    the minimum value for `weight_attribute`.

    The graph is a MultiDiGraph, so there can be multiple edges between
    the same pair of nodes (u, v). NetworkX's Dijkstra implementation
    resolves this internally by minimizing over parallel edges when the
    weight is given as an attribute name string. This helper mirrors that
    same selection logic, so that distance/time computed after the fact
    is consistent with the edge Dijkstra actually "used".

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph to query.
    u : Any
        Source node of the edge.
    v : Any
        Target node of the edge.
    weight_attribute : str
        The edge attribute name used to select the minimum-weight edge.

    Returns
    -------
    dict
        The attribute dictionary of the selected edge.

    Raises
    ------
    KeyError
        If there is no edge between u and v in the graph.
    """
    edge_dict = graph.get_edge_data(u, v)
    if not edge_dict:
        raise KeyError(f"No edge found between node {u} and node {v}.")

    best_key = min(
        edge_dict,
        key=lambda k: edge_dict[k].get(weight_attribute, float("inf")),
    )
    return edge_dict[best_key]


# ---------------------------------------------------------------------------
# Core Dijkstra routing
# ---------------------------------------------------------------------------

def shortest_path(
    graph: nx.MultiDiGraph,
    origin_node: Any,
    destination_node: Any,
    weight_mode: str = WEIGHT_MODE_BASE,
    weight_attribute: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute the shortest travel-time path between two nodes using
    Dijkstra's algorithm.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The processed road network graph.
    origin_node : Any
        The OSM node ID to start the route from.
    destination_node : Any
        The OSM node ID to route to.
    weight_mode : str
        Which travel-time attribute to route on:
        - "base" (default): static, free-flow travel time
          (config.WEIGHT_ATTRIBUTE, i.e. "travel_time").
        - "dynamic": congestion-adjusted travel time
          (congestion.DYNAMIC_WEIGHT_ATTRIBUTE, i.e.
          "dynamic_travel_time"), which must already be present on the
          graph's edges (see congestion.apply_congestion_to_graph()).
    weight_attribute : str or None
        Optional explicit edge attribute name to use as the Dijkstra
        weight, overriding `weight_mode`. Provided for advanced/backward
        compatible use; most callers should just use `weight_mode`.

    Returns
    -------
    dict
        A JSON-serializable result with the following keys:
        - is_reachable : bool
            Whether a path exists between origin and destination.
        - route_nodes : list or None
            Ordered list of node IDs from origin to destination, or None
            if unreachable / invalid input.
        - total_time_seconds : float or None
            Total travel time along the shortest path, in seconds
            (measured using the same weight attribute that was routed
            on).
        - total_distance_meters : float or None
            Total distance along the shortest path, in meters.
        - weight_mode : str
            Echoes back which weight mode was actually used ("base" or
            "dynamic", or "custom" if an explicit weight_attribute
            override was given).
        - weight_attribute : str
            The actual edge attribute name used as the Dijkstra weight.
        - error : str or None
            Human-readable error message if the route could not be
            computed (e.g. missing node, unreachable destination, or
            dynamic weights unavailable).
    """
    resolved_weight_attribute = _resolve_weight_attribute(weight_mode, weight_attribute)
    reported_weight_mode = weight_mode if weight_attribute is None else "custom"

    result: Dict[str, Any] = {
        "is_reachable": False,
        "route_nodes": None,
        "total_time_seconds": None,
        "total_distance_meters": None,
        "weight_mode": reported_weight_mode,
        "weight_attribute": resolved_weight_attribute,
        "error": None,
    }

    if origin_node not in graph:
        message = f"Origin node {origin_node} does not exist in the graph."
        logger.error(message)
        result["error"] = message
        return result

    if destination_node not in graph:
        message = f"Destination node {destination_node} does not exist in the graph."
        logger.error(message)
        result["error"] = message
        return result

    dynamic_check_error = _validate_dynamic_weight_available(
        graph, reported_weight_mode, resolved_weight_attribute
    )
    if dynamic_check_error is not None:
        logger.error(dynamic_check_error)
        result["error"] = dynamic_check_error
        return result

    if origin_node == destination_node:
        logger.info("Origin and destination are the same node (%s).", origin_node)
        result["is_reachable"] = True
        result["route_nodes"] = [origin_node]
        result["total_time_seconds"] = 0.0
        result["total_distance_meters"] = 0.0
        return result

    try:
        route_nodes = nx.dijkstra_path(
            graph, origin_node, destination_node, weight=resolved_weight_attribute
        )
    except nx.NetworkXNoPath:
        message = (
            f"Destination node {destination_node} is unreachable from "
            f"origin node {origin_node}."
        )
        logger.warning(message)
        result["error"] = message
        return result
    except nx.NodeNotFound as exc:
        message = f"Node not found in graph: {exc}"
        logger.error(message)
        result["error"] = message
        return result

    total_time = calculate_route_time(graph, route_nodes, resolved_weight_attribute)
    total_distance = calculate_route_distance(graph, route_nodes, resolved_weight_attribute)

    logger.info(
        "Route found (weight_mode=%s, attribute=%s): %d nodes, %.1f seconds, %.1f meters.",
        reported_weight_mode, resolved_weight_attribute, len(route_nodes), total_time, total_distance,
    )

    result["is_reachable"] = True
    result["route_nodes"] = route_nodes
    result["total_time_seconds"] = total_time
    result["total_distance_meters"] = total_distance
    return result


def calculate_route_time(
    graph: nx.MultiDiGraph,
    route_nodes: List[Any],
    weight_attribute: str = WEIGHT_ATTRIBUTE,
) -> float:
    """
    Calculate the total travel time (seconds) along an ordered route.

    For each consecutive pair of nodes, the parallel edge with the
    minimum `weight_attribute` is used — consistent with how Dijkstra
    selects edges on a MultiDiGraph.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph the route was computed on.
    route_nodes : list
        Ordered list of node IDs representing the route.
    weight_attribute : str
        Edge attribute holding travel time in seconds. Pass either
        config.WEIGHT_ATTRIBUTE ("travel_time") for base time or
        congestion.DYNAMIC_WEIGHT_ATTRIBUTE ("dynamic_travel_time") for
        congestion-adjusted time (defaults to config.WEIGHT_ATTRIBUTE).

    Returns
    -------
    float
        Total travel time in seconds. Returns 0.0 for a route with fewer
        than 2 nodes.
    """
    if len(route_nodes) < 2:
        return 0.0

    total_time = 0.0
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edge_data = _get_min_weight_edge_data(graph, u, v, weight_attribute)
        edge_time = edge_data.get(weight_attribute)
        if edge_time is None:
            logger.warning(
                "Edge (%s -> %s) is missing '%s'; treating as 0 seconds "
                "for this segment.", u, v, weight_attribute,
            )
            edge_time = 0.0
        total_time += float(edge_time)

    return total_time


def calculate_route_distance(
    graph: nx.MultiDiGraph,
    route_nodes: List[Any],
    weight_attribute: str = WEIGHT_ATTRIBUTE,
) -> float:
    """
    Calculate the total distance (meters) along an ordered route.

    For each consecutive pair of nodes, distance is taken from the same
    edge that would be selected as minimal by `weight_attribute` (so the
    reported distance corresponds to the actual path taken, not an
    independently-minimized distance).

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The graph the route was computed on.
    route_nodes : list
        Ordered list of node IDs representing the route.
    weight_attribute : str
        Edge attribute used to select the relevant parallel edge
        (defaults to config.WEIGHT_ATTRIBUTE, i.e. "travel_time"). Pass
        congestion.DYNAMIC_WEIGHT_ATTRIBUTE to select edges consistent
        with a dynamic-weight route.

    Returns
    -------
    float
        Total distance in meters. Returns 0.0 for a route with fewer
        than 2 nodes.
    """
    if len(route_nodes) < 2:
        return 0.0

    total_distance = 0.0
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edge_data = _get_min_weight_edge_data(graph, u, v, weight_attribute)
        edge_length = edge_data.get("length")
        if edge_length is None:
            logger.warning(
                "Edge (%s -> %s) is missing 'length'; treating as 0 meters "
                "for this segment.", u, v,
            )
            edge_length = 0.0
        total_distance += float(edge_length)

    return total_distance


# ---------------------------------------------------------------------------
# Coordinate-based routing
# ---------------------------------------------------------------------------

def route_between_coordinates(
    graph: nx.MultiDiGraph,
    origin_lat: float,
    origin_lon: float,
    destination_lat: float,
    destination_lon: float,
    weight_mode: str = WEIGHT_MODE_BASE,
    weight_attribute: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute the shortest travel-time route between two lat/lon
    coordinates.

    Each coordinate is first snapped to its nearest graph node (via
    graph_utils.find_nearest_node), then Dijkstra's algorithm is run
    between those two nodes.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The processed road network graph.
    origin_lat : float
        Latitude of the origin point.
    origin_lon : float
        Longitude of the origin point.
    destination_lat : float
        Latitude of the destination point.
    destination_lon : float
        Longitude of the destination point.
    weight_mode : str
        Which travel-time attribute to route on: "base" (default) or
        "dynamic". See `shortest_path` for details.
    weight_attribute : str or None
        Optional explicit edge attribute name overriding `weight_mode`.

    Returns
    -------
    dict
        Same structure as `shortest_path()`, plus:
        - origin_node : int or None
            Graph node nearest to the origin coordinate.
        - destination_node : int or None
            Graph node nearest to the destination coordinate.
        If either coordinate is invalid or cannot be matched to a node,
        `error` is set and `is_reachable` is False.
    """
    resolved_weight_attribute = _resolve_weight_attribute(weight_mode, weight_attribute)
    reported_weight_mode = weight_mode if weight_attribute is None else "custom"

    origin_node = find_nearest_node(graph, origin_lat, origin_lon)
    destination_node = find_nearest_node(graph, destination_lat, destination_lon)

    if origin_node is None:
        message = (
            f"Could not resolve a graph node for origin coordinate "
            f"(lat={origin_lat}, lon={origin_lon})."
        )
        logger.error(message)
        return {
            "is_reachable": False,
            "route_nodes": None,
            "total_time_seconds": None,
            "total_distance_meters": None,
            "weight_mode": reported_weight_mode,
            "weight_attribute": resolved_weight_attribute,
            "origin_node": None,
            "destination_node": None,
            "error": message,
        }

    if destination_node is None:
        message = (
            f"Could not resolve a graph node for destination coordinate "
            f"(lat={destination_lat}, lon={destination_lon})."
        )
        logger.error(message)
        return {
            "is_reachable": False,
            "route_nodes": None,
            "total_time_seconds": None,
            "total_distance_meters": None,
            "weight_mode": reported_weight_mode,
            "weight_attribute": resolved_weight_attribute,
            "origin_node": origin_node,
            "destination_node": None,
            "error": message,
        }

    result = shortest_path(
        graph, origin_node, destination_node,
        weight_mode=weight_mode, weight_attribute=weight_attribute,
    )
    result["origin_node"] = origin_node
    result["destination_node"] = destination_node

    return result