"""
cost_matrix.py

Builds the officer -> high-risk-junction response-time cost matrix for
NovaRoute.AI, using real road-network Dijkstra routing (routing.py) —
never straight-line ("as the crow flies") distance.

Pipeline, for every (officer, junction) pair:
    1. Snap the officer's and junction's (lat, lon) to their nearest
       graph node (graph_utils.find_nearest_node).
    2. Run Dijkstra between those two nodes (routing.shortest_path).
    3. Record the actual road-network response time (and distance/path)
       for that pair.
    4. Assemble all pairs into a matrix: rows = officers, columns =
       high-risk junctions, cells = response time.

This module does NOT:
- Implement the allocation algorithm (greedy / ILP / Hungarian, etc.)
  that consumes this matrix — that is a separate module's job.
- Implement a FastAPI layer.
- Use straight-line distance anywhere as a response-time proxy.

Design note for future integration:
The main entry point, build_cost_matrix(), returns a single
JSON/](pandas)-friendly result object (a dict) containing:
- a numpy matrix of response times (officers x junctions), ready to feed
  straight into a scipy/ILP assignment solver,
- the officer/junction ID orderings that index that matrix,
- and the full per-pair route detail (nodes, time, distance) kept
  separately, so the allocation module can pull out "why" a particular
  assignment was chosen (explainability) without recomputing routes.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from src.config import WEIGHT_ATTRIBUTE
    from src.graph_utils import find_nearest_node
    from src.routing import (
        WEIGHT_MODE_BASE,
        VALID_WEIGHT_MODES,
        shortest_path,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import WEIGHT_ATTRIBUTE
    from src.graph_utils import find_nearest_node
    from src.routing import (
        WEIGHT_MODE_BASE,
        VALID_WEIGHT_MODES,
        shortest_path,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Time unit configuration
# ---------------------------------------------------------------------------

# routing.py reports time in seconds; response-time matrices are usually
# discussed/displayed in minutes (as in the concept note's example
# table), so both are supported.
SECONDS_PER_MINUTE = 60.0
VALID_TIME_UNITS = ("seconds", "minutes")

# Sentinel used in the numpy matrix for a pair that could not be routed
# (unreachable, or a location that could not be snapped to a graph
# node). Using +inf (rather than None/NaN) keeps the matrix directly
# usable by numeric assignment solvers (e.g. scipy.optimize.linear_sum_
# assignment), which treat a very high cost as "never choose this pair"
# without needing special-case handling.
UNREACHABLE_COST = float("inf")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_locations(
    locations: Sequence[Dict[str, Any]],
    label: str,
    required_keys: Sequence[str] = ("id", "latitude", "longitude"),
) -> None:
    """
    Validate a list of officer or junction location dicts.

    Parameters
    ----------
    locations : sequence of dict
        The officer or junction records to validate.
    label : str
        A human-readable label ("officer" or "high-risk junction") used
        in error messages.
    required_keys : sequence of str
        Keys that must be present on every record.

    Raises
    ------
    ValueError
        If `locations` is empty, contains duplicate IDs, or any record
        is missing a required key / has a non-numeric coordinate.
    """
    if not locations:
        raise ValueError(f"No {label} records provided; cannot build cost matrix.")

    seen_ids = set()
    for i, record in enumerate(locations):
        for key in required_keys:
            if key not in record:
                raise ValueError(
                    f"{label.capitalize()} record at index {i} is missing "
                    f"required key '{key}': {record!r}"
                )

        record_id = record["id"]
        if record_id in seen_ids:
            raise ValueError(
                f"Duplicate {label} id '{record_id}' found. IDs must be unique."
            )
        seen_ids.add(record_id)

        for coord_key in ("latitude", "longitude"):
            try:
                float(record[coord_key])
            except (TypeError, ValueError):
                raise ValueError(
                    f"{label.capitalize()} '{record_id}' has a non-numeric "
                    f"'{coord_key}': {record[coord_key]!r}"
                )


# ---------------------------------------------------------------------------
# Coordinate -> graph node snapping
# ---------------------------------------------------------------------------

def _snap_locations_to_nodes(
    graph: Any,
    locations: Sequence[Dict[str, Any]],
    label: str,
) -> Tuple[Dict[str, Optional[int]], List[str]]:
    """
    Resolve each location's (latitude, longitude) to its nearest graph
    node.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The processed road network graph.
    locations : sequence of dict
        Officer or junction records, each with 'id', 'latitude',
        'longitude'.
    label : str
        Human-readable label used in log messages ("officer" or
        "high-risk junction").

    Returns
    -------
    (dict, list)
        - Mapping of location id -> nearest graph node id (or None if
          snapping failed for that location).
        - List of location ids that failed to snap (empty if all
          succeeded).
    """
    node_by_id: Dict[str, Optional[int]] = {}
    failed_ids: List[str] = []

    for record in locations:
        record_id = record["id"]
        node = find_nearest_node(
            graph, float(record["latitude"]), float(record["longitude"])
        )
        node_by_id[record_id] = node

        if node is None:
            failed_ids.append(record_id)
            logger.warning(
                "Could not snap %s '%s' (lat=%s, lon=%s) to a graph node; "
                "it will be excluded from the cost matrix.",
                label, record_id, record["latitude"], record["longitude"],
            )

    return node_by_id, failed_ids


# ---------------------------------------------------------------------------
# Core cost matrix construction
# ---------------------------------------------------------------------------

def build_cost_matrix(
    graph: Any,
    officers: Sequence[Dict[str, Any]],
    high_risk_junctions: Sequence[Dict[str, Any]],
    weight_mode: str = WEIGHT_MODE_BASE,
    weight_attribute: Optional[str] = None,
    time_unit: str = "minutes",
) -> Dict[str, Any]:
    """
    Build the officer x high-risk-junction response-time cost matrix
    using real Dijkstra road-network routing (never straight-line
    distance).

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The processed road network graph (as produced by
        graph_builder.py / loaded via graph_utils.load_graph()). Must
        have the edge attribute(s) required by the chosen
        `weight_mode` (see routing.py) — e.g. "travel_time" for
        weight_mode="base", or "dynamic_travel_time" for
        weight_mode="dynamic" (requires congestion.apply_congestion_to_
        graph() to have been run first).
    officers : list of dict
        Officer locations, each:
            {"id": "officer_1", "latitude": ..., "longitude": ...}
    high_risk_junctions : list of dict
        High-risk junction locations, each:
            {"id": "junction_1", "latitude": ..., "longitude": ...,
             "risk_score": ...}
        ("risk_score" is accepted and passed through if present, but is
        not used by this module — it belongs to the risk-scoring
        module.)
    weight_mode : str
        Which travel-time attribute to route on: "base" (default,
        static free-flow travel time) or "dynamic" (congestion-adjusted
        travel time). See routing.py for details.
    weight_attribute : str or None
        Optional explicit edge attribute name overriding `weight_mode`
        (advanced use; see routing.shortest_path).
    time_unit : str
        Unit for response times in the returned matrix: "minutes"
        (default) or "seconds".

    Returns
    -------
    dict
        {
            "officer_ids": [...],       # row order of the matrix
            "junction_ids": [...],      # column order of the matrix
            "matrix": numpy.ndarray,    # shape (n_officers, n_junctions)
                                         # response time in `time_unit`;
                                         # UNREACHABLE_COST (inf) where
                                         # no route/node could be found
            "time_unit": "minutes" | "seconds",
            "weight_mode": str,         # actual mode used ("base",
                                         # "dynamic", or "custom" if an
                                         # explicit weight_attribute was
                                         # given)
            "weight_attribute": str,    # actual edge attribute used
            "routes": {                 # full per-pair detail, keyed by
                (officer_id, junction_id): {   # a plain (id, id) tuple
                    "is_reachable": bool,
                    "route_nodes": list or None,
                    "total_time_seconds": float or None,
                    "total_distance_meters": float or None,
                    "origin_node": int or None,
                    "destination_node": int or None,
                    "error": str or None,
                },
                ...
            },
            "officer_nodes": {officer_id: node_id or None, ...},
            "junction_nodes": {junction_id: node_id or None, ...},
            "unreachable_pairs": [(officer_id, junction_id), ...],
            "unsnapped_officer_ids": [...],   # officers with no nearby node
            "unsnapped_junction_ids": [...],  # junctions with no nearby node
        }

    Raises
    ------
    ValueError
        If `officers` or `high_risk_junctions` is empty, contains
        duplicate/missing IDs or non-numeric coordinates, or if
        `time_unit` is not recognized.
    """
    if time_unit not in VALID_TIME_UNITS:
        raise ValueError(
            f"Invalid time_unit {time_unit!r}; must be one of {VALID_TIME_UNITS}."
        )

    _validate_locations(officers, label="officer")
    _validate_locations(
        high_risk_junctions,
        label="high-risk junction",
        required_keys=("id", "latitude", "longitude"),
    )

    officer_ids = [officer["id"] for officer in officers]
    junction_ids = [junction["id"] for junction in high_risk_junctions]

    logger.info(
        "Building cost matrix: %d officer(s) x %d junction(s), "
        "weight_mode='%s', time_unit='%s'.",
        len(officer_ids), len(junction_ids), weight_mode, time_unit,
    )

    officer_nodes, unsnapped_officer_ids = _snap_locations_to_nodes(
        graph, officers, label="officer"
    )
    junction_nodes, unsnapped_junction_ids = _snap_locations_to_nodes(
        graph, high_risk_junctions, label="high-risk junction"
    )

    matrix = np.full(
        (len(officer_ids), len(junction_ids)), UNREACHABLE_COST, dtype=float
    )
    routes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    unreachable_pairs: List[Tuple[str, str]] = []

    time_divisor = SECONDS_PER_MINUTE if time_unit == "minutes" else 1.0

    for i, officer_id in enumerate(officer_ids):
        origin_node = officer_nodes[officer_id]

        for j, junction_id in enumerate(junction_ids):
            destination_node = junction_nodes[junction_id]

            if origin_node is None or destination_node is None:
                message = (
                    f"Cannot route officer '{officer_id}' -> junction "
                    f"'{junction_id}': "
                    + (
                        f"officer could not be snapped to a graph node. "
                        if origin_node is None else ""
                    )
                    + (
                        f"junction could not be snapped to a graph node."
                        if destination_node is None else ""
                    )
                )
                routes[(officer_id, junction_id)] = {
                    "is_reachable": False,
                    "route_nodes": None,
                    "total_time_seconds": None,
                    "total_distance_meters": None,
                    "origin_node": origin_node,
                    "destination_node": destination_node,
                    "error": message,
                }
                unreachable_pairs.append((officer_id, junction_id))
                continue

            route_result = shortest_path(
                graph, origin_node, destination_node,
                weight_mode=weight_mode, weight_attribute=weight_attribute,
            )
            route_result["origin_node"] = origin_node
            route_result["destination_node"] = destination_node
            routes[(officer_id, junction_id)] = route_result

            if route_result["is_reachable"]:
                matrix[i, j] = route_result["total_time_seconds"] / time_divisor
            else:
                unreachable_pairs.append((officer_id, junction_id))
                logger.warning(
                    "No route found for officer '%s' -> junction '%s': %s",
                    officer_id, junction_id, route_result.get("error"),
                )

    if unreachable_pairs:
        logger.warning(
            "Cost matrix built with %d unreachable pair(s) out of %d total.",
            len(unreachable_pairs), len(officer_ids) * len(junction_ids),
        )
    else:
        logger.info("Cost matrix built successfully; all pairs reachable.")

    # Resolve the actual weight_attribute/weight_mode used, matching the
    # same resolution logic as routing.py, for transparency in the
    # returned result.
    resolved_weight_mode = weight_mode if weight_attribute is None else "custom"
    resolved_weight_attribute = weight_attribute or (
        WEIGHT_ATTRIBUTE if weight_mode == WEIGHT_MODE_BASE else None
    )
    if resolved_weight_attribute is None:
        # weight_mode == "dynamic" and no explicit override; pull the
        # attribute name straight from a routes entry so this stays
        # correct even if routing.py's internal mapping changes.
        any_route = next(iter(routes.values()), None)
        resolved_weight_attribute = (
            any_route.get("weight_attribute") if any_route else weight_mode
        )

    return {
        "officer_ids": officer_ids,
        "junction_ids": junction_ids,
        "matrix": matrix,
        "time_unit": time_unit,
        "weight_mode": resolved_weight_mode,
        "weight_attribute": resolved_weight_attribute,
        "routes": routes,
        "officer_nodes": officer_nodes,
        "junction_nodes": junction_nodes,
        "unreachable_pairs": unreachable_pairs,
        "unsnapped_officer_ids": unsnapped_officer_ids,
        "unsnapped_junction_ids": unsnapped_junction_ids,
    }


# ---------------------------------------------------------------------------
# Convenience accessors (for the future allocation module)
# ---------------------------------------------------------------------------

def get_response_time(
    cost_matrix_result: Dict[str, Any],
    officer_id: str,
    junction_id: str,
) -> Optional[float]:
    """
    Look up the response time for a single (officer, junction) pair from
    a cost matrix result.

    Parameters
    ----------
    cost_matrix_result : dict
        The dict returned by build_cost_matrix().
    officer_id : str
        Officer id.
    junction_id : str
        Junction id.

    Returns
    -------
    float or None
        The response time in `cost_matrix_result["time_unit"]`, or None
        if the pair is unreachable or either id is not in the matrix.
    """
    try:
        i = cost_matrix_result["officer_ids"].index(officer_id)
        j = cost_matrix_result["junction_ids"].index(junction_id)
    except ValueError:
        logger.error(
            "Unknown officer_id '%s' or junction_id '%s' in cost matrix.",
            officer_id, junction_id,
        )
        return None

    value = float(cost_matrix_result["matrix"][i, j])
    return None if value == UNREACHABLE_COST else value


def get_route(
    cost_matrix_result: Dict[str, Any],
    officer_id: str,
    junction_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Look up the full route detail (nodes, time, distance) for a single
    (officer, junction) pair.

    This is what a future explainability feature ("Officer 3 assigned —
    4.2 min via Wardha Road...") or redeployment animation would read
    from, without recomputing Dijkstra.

    Parameters
    ----------
    cost_matrix_result : dict
        The dict returned by build_cost_matrix().
    officer_id : str
        Officer id.
    junction_id : str
        Junction id.

    Returns
    -------
    dict or None
        The route result dict for this pair (see build_cost_matrix()'s
        "routes" entry format), or None if this pair was never computed.
    """
    return cost_matrix_result["routes"].get((officer_id, junction_id))


def to_dataframe(cost_matrix_result: Dict[str, Any]):
    """
    Convert the cost matrix into a pandas DataFrame for easy inspection
    or export — rows = officers, columns = junctions, matching the
    concept note's example table layout.

    Unreachable pairs are shown as NaN rather than `inf`, since that is
    the more conventional "missing value" representation in a
    DataFrame/CSV context.

    Parameters
    ----------
    cost_matrix_result : dict
        The dict returned by build_cost_matrix().

    Returns
    -------
    pandas.DataFrame
        Response-time matrix with officer_ids as the index and
        junction_ids as the columns.

    Raises
    ------
    ImportError
        If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for to_dataframe(); install it or use "
            "cost_matrix_result['matrix'] directly."
        ) from exc

    matrix = cost_matrix_result["matrix"].copy()
    matrix[matrix == UNREACHABLE_COST] = np.nan

    return pd.DataFrame(
        matrix,
        index=cost_matrix_result["officer_ids"],
        columns=cost_matrix_result["junction_ids"],
    )