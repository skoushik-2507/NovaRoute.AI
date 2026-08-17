"""
coverage.py

Identifies high-risk junctions that are insufficiently covered by
currently available officers, using real road-network (Dijkstra)
response times — never a geographic radius.

For every high-risk junction:
    1. Look up the Dijkstra response time from every officer (via
       cost_matrix.py, so routing is never duplicated/recomputed here).
    2. Find the minimum response time across all officers.
    3. Identify which officer achieves that minimum (the
       nearest/fastest-responding officer, by road-network time).
    4. Mark the junction "covered" if that minimum response time is
       <= the configured threshold, "uncovered" (unmanned/under-covered)
       otherwise.

This module does NOT:
- Use geographic/straight-line radius anywhere.
- Implement the officer allocation algorithm.
- Implement any frontend/dashboard.

It only produces structured coverage data for those other pieces (or a
dashboard) to consume.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from src.cost_matrix import (
        UNREACHABLE_COST,
        build_cost_matrix,
        get_route,
    )
    from src.routing import WEIGHT_MODE_BASE
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.cost_matrix import (
        UNREACHABLE_COST,
        build_cost_matrix,
        get_route,
    )
    from src.routing import WEIGHT_MODE_BASE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

# Default response-time threshold, in minutes. 6 minutes is the initial
# demonstration value called out in the concept note; callers can
# override this per-call.
DEFAULT_THRESHOLD_MINUTES = 6.0

SECONDS_PER_MINUTE = 60.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_minutes(value: Optional[float], time_unit: str) -> Optional[float]:
    """
    Convert a response-time value into minutes, regardless of the unit
    it was originally computed in.

    Parameters
    ----------
    value : float or None
        The response time, or None/inf if unreachable.
    time_unit : str
        The unit `value` is currently in: "minutes" or "seconds".

    Returns
    -------
    float or None
        The value converted to minutes, or None if `value` was
        None/inf (unreachable).
    """
    if value is None or value == UNREACHABLE_COST:
        return None

    if time_unit == "seconds":
        return value / SECONDS_PER_MINUTE
    return value


def _risk_score_lookup(
    high_risk_junctions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build a junction_id -> risk_score lookup from the raw junction
    records, so it can be passed through into the coverage report
    (risk_score is optional; junctions without it simply get None).

    Parameters
    ----------
    high_risk_junctions : sequence of dict
        Junction records, each expected to at least have an 'id' key
        and optionally a 'risk_score' key.

    Returns
    -------
    dict
        Mapping of junction id -> risk_score (or None if not provided).
    """
    return {
        record["id"]: record.get("risk_score")
        for record in high_risk_junctions
    }


# ---------------------------------------------------------------------------
# Core coverage analysis (operates on an already-built cost matrix)
# ---------------------------------------------------------------------------

def analyze_coverage_from_cost_matrix(
    cost_matrix_result: Dict[str, Any],
    threshold_minutes: float = DEFAULT_THRESHOLD_MINUTES,
    risk_scores: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Analyze junction coverage from an already-built cost matrix (see
    cost_matrix.build_cost_matrix()).

    This is the core logic and does not touch the graph or re-run any
    routing — it only reads the matrix/routes that were already
    computed, so it's cheap to call repeatedly (e.g. after re-solving
    an allocation) without re-running Dijkstra.

    Parameters
    ----------
    cost_matrix_result : dict
        The dict returned by cost_matrix.build_cost_matrix(). Must
        contain "officer_ids", "junction_ids", "matrix", "time_unit",
        and "routes".
    threshold_minutes : float
        Maximum acceptable response time, in minutes, for a junction to
        be considered "covered". Defaults to
        DEFAULT_THRESHOLD_MINUTES (6.0).
    risk_scores : dict or None
        Optional mapping of junction_id -> risk_score to pass through
        into each junction's report entry (purely informational; not
        used in the covered/uncovered decision).

    Returns
    -------
    dict
        {
            "threshold_minutes": float,
            "junctions": [
                {
                    "junction_id": str,
                    "risk_score": Any or None,
                    "min_response_time_minutes": float or None,
                        # None means no officer could reach this
                        # junction at all.
                    "nearest_officer_id": str or None,
                    "is_covered": bool,
                    "response_times_minutes": {officer_id: float or None, ...},
                        # every officer's response time, for
                        # transparency/explainability.
                    "route": dict or None,
                        # full Dijkstra route detail (nodes, time,
                        # distance) from the nearest officer to this
                        # junction, as returned by routing.shortest_path();
                        # None if unreachable.
                },
                ...
            ],
            "covered_junction_ids": [...],
            "uncovered_junction_ids": [...],
                # sorted by risk_score descending where available (most
                # urgent unmanned junctions first), then by junction_id
                # for a stable order when risk_score is missing/tied.
            "num_covered": int,
            "num_uncovered": int,
            "coverage_rate": float,
                # num_covered / total junctions, in [0.0, 1.0]. 0.0 if
                # there are no junctions.
        }

    Raises
    ------
    ValueError
        If `threshold_minutes` is not a positive number.
    """
    try:
        threshold_minutes = float(threshold_minutes)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"threshold_minutes must be numeric, got {threshold_minutes!r}."
        ) from exc

    if threshold_minutes <= 0:
        raise ValueError(
            f"threshold_minutes must be positive, got {threshold_minutes}."
        )

    officer_ids: List[str] = cost_matrix_result["officer_ids"]
    junction_ids: List[str] = cost_matrix_result["junction_ids"]
    matrix = cost_matrix_result["matrix"]
    time_unit: str = cost_matrix_result["time_unit"]
    risk_scores = risk_scores or {}

    logger.info(
        "Analyzing coverage for %d junction(s) against %d officer(s), "
        "threshold=%.1f minutes.",
        len(junction_ids), len(officer_ids), threshold_minutes,
    )

    junction_reports: List[Dict[str, Any]] = []
    covered_junction_ids: List[str] = []
    uncovered_junction_ids: List[str] = []

    for j, junction_id in enumerate(junction_ids):
        response_times_minutes: Dict[str, Optional[float]] = {}
        best_officer_id: Optional[str] = None
        best_time_minutes: Optional[float] = None

        for i, officer_id in enumerate(officer_ids):
            raw_value = float(matrix[i, j])
            time_minutes = _to_minutes(raw_value, time_unit)
            response_times_minutes[officer_id] = time_minutes

            if time_minutes is None:
                continue

            if best_time_minutes is None or time_minutes < best_time_minutes:
                best_time_minutes = time_minutes
                best_officer_id = officer_id

        is_covered = (
            best_time_minutes is not None and best_time_minutes <= threshold_minutes
        )

        route = None
        if best_officer_id is not None:
            route = get_route(cost_matrix_result, best_officer_id, junction_id)

        report = {
            "junction_id": junction_id,
            "risk_score": risk_scores.get(junction_id),
            "min_response_time_minutes": best_time_minutes,
            "nearest_officer_id": best_officer_id,
            "is_covered": is_covered,
            "response_times_minutes": response_times_minutes,
            "route": route,
        }
        junction_reports.append(report)

        if is_covered:
            covered_junction_ids.append(junction_id)
        else:
            uncovered_junction_ids.append(junction_id)
            if best_officer_id is None:
                logger.warning(
                    "Junction '%s' is UNCOVERED: no officer could reach it.",
                    junction_id,
                )
            else:
                logger.warning(
                    "Junction '%s' is UNCOVERED: fastest response is %.1f min "
                    "(officer '%s'), exceeds threshold of %.1f min.",
                    junction_id, best_time_minutes, best_officer_id, threshold_minutes,
                )

    def _uncovered_sort_key(junction_id: str):
        # Sort by risk_score descending (missing risk_score sorts last),
        # then by junction_id for a stable, deterministic tie-break.
        score = risk_scores.get(junction_id)
        has_score = score is not None
        return (not has_score, -score if has_score else 0, junction_id)

    uncovered_junction_ids.sort(key=_uncovered_sort_key)

    total_junctions = len(junction_ids)
    num_covered = len(covered_junction_ids)
    num_uncovered = len(uncovered_junction_ids)
    coverage_rate = (num_covered / total_junctions) if total_junctions > 0 else 0.0

    logger.info(
        "Coverage analysis complete: %d/%d junctions covered (%.1f%%).",
        num_covered, total_junctions, coverage_rate * 100.0,
    )

    return {
        "threshold_minutes": threshold_minutes,
        "junctions": junction_reports,
        "covered_junction_ids": covered_junction_ids,
        "uncovered_junction_ids": uncovered_junction_ids,
        "num_covered": num_covered,
        "num_uncovered": num_uncovered,
        "coverage_rate": coverage_rate,
    }


# ---------------------------------------------------------------------------
# Convenience entry point (builds the cost matrix internally)
# ---------------------------------------------------------------------------

def analyze_coverage(
    graph: Any,
    officers: Sequence[Dict[str, Any]],
    high_risk_junctions: Sequence[Dict[str, Any]],
    threshold_minutes: float = DEFAULT_THRESHOLD_MINUTES,
    weight_mode: str = WEIGHT_MODE_BASE,
    weight_attribute: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Identify high-risk junctions that are insufficiently covered by the
    given officers, computing road-network (Dijkstra) response times
    from scratch.

    This is a convenience wrapper around cost_matrix.build_cost_matrix()
    + analyze_coverage_from_cost_matrix(). If you already have a cost
    matrix built (e.g. from a previous call, or one shared with the
    allocation module), call analyze_coverage_from_cost_matrix()
    directly instead of recomputing routing here.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The processed road network graph. Must have the edge
        attribute(s) required by `weight_mode` (see routing.py /
        cost_matrix.py) — e.g. "travel_time" for weight_mode="base", or
        "dynamic_travel_time" for weight_mode="dynamic" (requires
        congestion.apply_congestion_to_graph() to have been run first).
    officers : list of dict
        Officer locations, each:
            {"id": "officer_1", "latitude": ..., "longitude": ...}
    high_risk_junctions : list of dict
        High-risk junction locations, each:
            {"id": "junction_1", "latitude": ..., "longitude": ...,
             "risk_score": ...}
        ("risk_score" is optional and passed through into the report if
        present; it is never used to decide coverage.)
    threshold_minutes : float
        Maximum acceptable response time, in minutes, for a junction to
        be considered "covered". Defaults to
        DEFAULT_THRESHOLD_MINUTES (6.0), the concept note's initial
        demonstration value.
    weight_mode : str
        Which travel-time attribute to route on: "base" (default) or
        "dynamic". See routing.py for details.
    weight_attribute : str or None
        Optional explicit edge attribute name overriding `weight_mode`
        (advanced use; see routing.shortest_path).

    Returns
    -------
    dict
        Same structure as analyze_coverage_from_cost_matrix(), plus:
        - "cost_matrix": the full cost_matrix.build_cost_matrix() result
          (matrix, routes, node mappings, etc.), kept alongside the
          coverage report in case the caller needs it (e.g. to hand off
          to an allocation module without recomputing routing).
    """
    cost_matrix_result = build_cost_matrix(
        graph, officers, high_risk_junctions,
        weight_mode=weight_mode, weight_attribute=weight_attribute,
        time_unit="minutes",
    )

    risk_scores = _risk_score_lookup(high_risk_junctions)

    coverage_result = analyze_coverage_from_cost_matrix(
        cost_matrix_result, threshold_minutes=threshold_minutes, risk_scores=risk_scores,
    )
    coverage_result["cost_matrix"] = cost_matrix_result

    return coverage_result


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_junction_report(
    coverage_result: Dict[str, Any],
    junction_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Look up a single junction's coverage report by id.

    Parameters
    ----------
    coverage_result : dict
        The dict returned by analyze_coverage() or
        analyze_coverage_from_cost_matrix().
    junction_id : str
        The junction id to look up.

    Returns
    -------
    dict or None
        That junction's report entry, or None if the id is not present.
    """
    for report in coverage_result["junctions"]:
        if report["junction_id"] == junction_id:
            return report
    return None


def get_uncovered_junctions(
    coverage_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Get the full report entries (not just IDs) for every uncovered
    ("unmanned"/under-covered) junction, already ordered the same way
    as coverage_result["uncovered_junction_ids"] (highest risk_score
    first, where available).

    Parameters
    ----------
    coverage_result : dict
        The dict returned by analyze_coverage() or
        analyze_coverage_from_cost_matrix().

    Returns
    -------
    list of dict
        Report entries for uncovered junctions, in priority order.
    """
    reports_by_id = {
        report["junction_id"]: report for report in coverage_result["junctions"]
    }
    return [
        reports_by_id[junction_id]
        for junction_id in coverage_result["uncovered_junction_ids"]
    ]