"""
travel_time.py

Calculates realistic base travel time for road-network edges, using edge
length (meters) and speed (km/h) information.

Formula used:
    travel_time_seconds = distance_meters / speed_meters_per_second

This module does NOT implement routing (Dijkstra) or dynamic congestion —
it only computes a static, free-flow travel time per edge.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import networkx as nx

try:
    from src.config import (
        DEFAULT_SPEED_KMPH,
        HIGHWAY_SPEED_DEFAULTS_KMPH,
        WEIGHT_ATTRIBUTE,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import (
        DEFAULT_SPEED_KMPH,
        HIGHWAY_SPEED_DEFAULTS_KMPH,
        WEIGHT_ATTRIBUTE,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Units used throughout this module (explicit, to avoid ambiguity)
# ---------------------------------------------------------------------------
# - Edge "length" attribute (from OSMnx):   meters (m)
# - Speed values (maxspeed, config defaults): kilometers per hour (km/h)
# - Computed travel_time attribute:          seconds (s)

KMPH_TO_MPS = 1000.0 / 3600.0  # conversion factor: km/h -> m/s


# ---------------------------------------------------------------------------
# Speed parsing / resolution helpers
# ---------------------------------------------------------------------------

def _parse_maxspeed_value(raw_value: Union[str, int, float, List[Any]]) -> Optional[float]:
    """
    Parse a raw OSM 'maxspeed' value into a numeric speed in km/h.

    OSM maxspeed data is often messy: it can be a plain number, a string
    like "50" or "50 mph", or a list of such values (e.g. when an edge
    represents multiple merged OSM ways). This function extracts a single
    usable km/h value, or returns None if it cannot be parsed.

    Parameters
    ----------
    raw_value : str | int | float | list
        The raw 'maxspeed' value as stored on the edge.

    Returns
    -------
    float or None
        Speed in km/h, or None if the value is missing/unparseable.
    """
    if raw_value is None:
        return None

    # If it's a list (multiple OSM ways merged into one edge), take the
    # first parseable value.
    if isinstance(raw_value, list):
        for item in raw_value:
            parsed = _parse_maxspeed_value(item)
            if parsed is not None:
                return parsed
        return None

    if isinstance(raw_value, (int, float)):
        speed = float(raw_value)
        return speed if speed > 0 else None

    if isinstance(raw_value, str):
        text = raw_value.strip().lower()
        if not text:
            return None

        is_mph = "mph" in text
        # Strip any non-numeric characters (e.g. "mph", "km/h")
        digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
        if not digits:
            return None

        try:
            value = float(digits)
        except ValueError:
            return None

        if value <= 0:
            return None

        if is_mph:
            value = value * 1.60934  # mph -> km/h

        return value

    return None


def get_edge_speed_kmph(edge_data: Dict[str, Any]) -> float:
    """
    Determine the speed (km/h) to use for a single edge.

    Resolution order:
    1. Use the edge's 'maxspeed' attribute if present and parseable.
    2. Fall back to a highway-type-based default (HIGHWAY_SPEED_DEFAULTS_KMPH)
       using the edge's 'highway' attribute, if present and recognized.
    3. Fall back to the global DEFAULT_SPEED_KMPH.

    This fallback chain is explicit and logged, so speed values are never
    silently invented without a clear, documented source.

    Parameters
    ----------
    edge_data : dict
        The edge's attribute dictionary.

    Returns
    -------
    float
        Resolved speed in km/h. Always > 0.
    """
    # 1. Try maxspeed
    maxspeed_raw = edge_data.get("maxspeed")
    speed_kmph = _parse_maxspeed_value(maxspeed_raw)
    if speed_kmph is not None and speed_kmph > 0:
        return speed_kmph

    # 2. Try highway-type default
    highway = edge_data.get("highway")
    if isinstance(highway, list):
        highway = highway[0] if highway else None

    if highway is not None and highway in HIGHWAY_SPEED_DEFAULTS_KMPH:
        return float(HIGHWAY_SPEED_DEFAULTS_KMPH[highway])

    # 3. Global fallback
    return float(DEFAULT_SPEED_KMPH)


# ---------------------------------------------------------------------------
# Single-edge travel time calculation
# ---------------------------------------------------------------------------

def calculate_edge_travel_time(
    length_m: float,
    speed_kmph: float,
) -> float:
    """
    Calculate travel time (in seconds) for a single edge.

    travel_time_seconds = length_m / speed_mps
    where speed_mps = speed_kmph * (1000 / 3600)

    Parameters
    ----------
    length_m : float
        Edge length in meters. Must be > 0.
    speed_kmph : float
        Speed in kilometers per hour. Must be > 0.

    Returns
    -------
    float
        Travel time in seconds.

    Raises
    ------
    ValueError
        If length_m or speed_kmph is missing, zero, negative, or otherwise
        invalid. Invalid inputs are never silently converted into a
        fabricated travel time.
    """
    if length_m is None:
        raise ValueError("Edge length is missing; cannot calculate travel time.")
    if speed_kmph is None:
        raise ValueError("Edge speed is missing; cannot calculate travel time.")

    try:
        length_m = float(length_m)
        speed_kmph = float(speed_kmph)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Non-numeric length or speed provided (length={length_m!r}, "
            f"speed={speed_kmph!r})."
        ) from exc

    if length_m <= 0:
        raise ValueError(f"Edge length must be positive, got {length_m} m.")

    if speed_kmph <= 0:
        raise ValueError(f"Edge speed must be positive, got {speed_kmph} km/h.")

    speed_mps = speed_kmph * KMPH_TO_MPS
    travel_time_seconds = length_m / speed_mps

    return travel_time_seconds


def calculate_travel_time_for_edge(edge_data: Dict[str, Any]) -> float:
    """
    Calculate travel time (seconds) for a single edge, given its full
    attribute dictionary (as stored in the graph).

    This resolves speed via get_edge_speed_kmph() (using maxspeed ->
    highway-type default -> global default) and requires 'length' to be
    present on the edge.

    Parameters
    ----------
    edge_data : dict
        The edge's attribute dictionary. Must contain a positive 'length'
        value (meters).

    Returns
    -------
    float
        Travel time in seconds.

    Raises
    ------
    ValueError
        If 'length' is missing or not a positive number, or if the
        resolved speed is not positive.
    """
    length_m = edge_data.get("length")
    if length_m is None:
        raise ValueError(
            "Edge is missing required 'length' attribute; "
            "cannot calculate travel time."
        )

    speed_kmph = get_edge_speed_kmph(edge_data)

    return calculate_edge_travel_time(length_m, speed_kmph)


# ---------------------------------------------------------------------------
# Whole-graph travel time calculation
# ---------------------------------------------------------------------------

def add_travel_times_to_graph(
    graph: nx.MultiDiGraph,
    weight_attribute: str = WEIGHT_ATTRIBUTE,
) -> nx.MultiDiGraph:
    """
    Compute and attach a travel_time attribute to every edge in a copy of
    the graph.

    The original graph structure (nodes, edges, all existing attributes)
    is fully preserved. This function operates on and returns a deep copy,
    so the caller's original graph object is never mutated.

    Edges whose travel time cannot be validly computed (e.g. missing or
    non-positive 'length') are skipped and reported, rather than being
    assigned a fabricated value.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The input road network graph. Edges are expected to have a
        'length' attribute (meters), as produced by OSMnx.
    weight_attribute : str
        Name of the edge attribute to store the computed travel time
        under (defaults to config.WEIGHT_ATTRIBUTE, i.e. "travel_time").

    Returns
    -------
    networkx.MultiDiGraph
        A new graph (deep copy of the input) with `weight_attribute` set
        on every successfully-processed edge, in seconds.
    """
    graph_copy = graph.copy()

    num_success = 0
    num_failed = 0
    failed_edges = []

    for u, v, key, data in graph_copy.edges(keys=True, data=True):
        try:
            travel_time_seconds = calculate_travel_time_for_edge(data)
            data[weight_attribute] = travel_time_seconds
            num_success += 1
        except ValueError as exc:
            num_failed += 1
            failed_edges.append((u, v, key, str(exc)))
            logger.warning(
                "Skipping travel_time for edge (%s, %s, %s): %s", u, v, key, exc
            )

    logger.info(
        "Travel time calculation complete: %d edges succeeded, %d edges skipped.",
        num_success, num_failed,
    )

    if num_failed > 0:
        logger.warning(
            "%d edges are missing '%s' and will not have a travel time. "
            "These edges should be handled before routing.",
            num_failed, weight_attribute,
        )

    return graph_copy