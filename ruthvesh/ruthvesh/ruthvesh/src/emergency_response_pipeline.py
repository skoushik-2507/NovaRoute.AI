"""
emergency_response_pipeline.py

Thin end-to-end orchestrator (Prompt 8A) connecting the already-tested
NovaRoute.AI / Ruthvesh components into one emergency-response result:

    Real ML JSON
        -> ml_integration.load_ml_observation() / extract_ml_fields()
           (via ml_dynamic_graph.build_dynamic_graph_from_files() and
           risk_priority.build_risk_priority_map())
        -> congestion_factor (routing side) + risk_score (priority side)
        -> dynamic OSM graph (ml_dynamic_graph.build_dynamic_graph_from_files)
        -> dynamic cost matrix (cost_matrix.build_cost_matrix,
           weight_mode="dynamic")
        -> coverage analysis using real ML risk
           (coverage.analyze_coverage_from_cost_matrix)
        -> emergency-response result

This module is COMPOSITION ONLY. It contains no routing algorithm, no
congestion/risk calculation, and no coverage/allocation logic of its
own - every one of those already exists, elsewhere, unmodified:

    - Dijkstra                 -> routing.py (via cost_matrix.py only;
                                   this module never calls
                                   routing.shortest_path() directly)
    - congestion_factor         -> read verbatim from ML JSON
                                   (ml_integration.py / ml_dynamic_graph.py)
    - risk_score                -> read verbatim from ML JSON
                                   (ml_integration.py / risk_priority.py)
    - cost matrix                -> cost_matrix.build_cost_matrix()
    - coverage / uncovered sort -> coverage.analyze_coverage_from_cost_matrix()

congestion_factor and risk_score remain architecturally independent
through this module exactly as they are in every component it calls:
congestion_factor only ever reaches travel cost (via dynamic_travel_time
on the graph), risk_score only ever reaches junction priority (via
coverage's risk_scores parameter / uncovered-junction sort key). Neither
value is read by the code path that produces the other.

Junction coordinates (needed to build the officer x junction cost
matrix at all - build_cost_matrix requires lat/lon for every junction,
including ones with no ML observation) come from
junction_mapping.PROTOTYPE_JUNCTION_COORDINATES - the same
already-verified prototype coordinates ml_dynamic_graph.py itself uses
for the ML -> OSM segment mapping. This module does NOT import
anything from pipeline_demo.py: pipeline_demo.SAMPLE_HIGH_RISK_
JUNCTIONS bakes in stale hardcoded risk_score values on the old 0-1
demo scale, which must never be mixed with the real 0-100 ML risk
scores this module produces via risk_priority.py.

junction_3 (or any junction_id with no real ML observation file
supplied) is never given a fabricated risk_score: risk_priority.
build_risk_priority_map() simply omits it, so coverage.py's own
existing risk_scores.get(junction_id) -> None behavior applies exactly
as it already does everywhere else. junction_3 still participates
fully in routing/cost-matrix calculations, using only the baseline
dynamic_travel_time == travel_time that ml_dynamic_graph.py already
guarantees for every edge not covered by an ML observation (Prompt 6A's
two-pass initialization) - no special-casing is needed here for that
to work.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import networkx as nx

try:
    from src.coverage import DEFAULT_THRESHOLD_MINUTES, analyze_coverage_from_cost_matrix
    from src.cost_matrix import build_cost_matrix
    from src.junction_mapping import PROTOTYPE_JUNCTION_COORDINATES
    from src.ml_dynamic_graph import DynamicGraphResult, build_dynamic_graph_from_files
    from src.risk_priority import build_risk_priority_map
    from src.routing import VALID_WEIGHT_MODES
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.coverage import DEFAULT_THRESHOLD_MINUTES, analyze_coverage_from_cost_matrix
    from src.cost_matrix import build_cost_matrix
    from src.junction_mapping import PROTOTYPE_JUNCTION_COORDINATES
    from src.ml_dynamic_graph import DynamicGraphResult, build_dynamic_graph_from_files
    from src.risk_priority import build_risk_priority_map
    from src.routing import VALID_WEIGHT_MODES


def _build_high_risk_junction_records(
    junction_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Build {"id", "latitude", "longitude"} records for build_cost_matrix,
    from the existing verified junction_mapping.PROTOTYPE_JUNCTION_
    COORDINATES - never from pipeline_demo.py, and never including a
    "risk_score" key (real risk scores are passed to coverage
    separately, via the `risk_scores` parameter, exactly as Prompt 7B
    already established; baking a value into this record would only
    invite it being confused with pipeline_demo's stale 0-1 numbers).

    Parameters
    ----------
    junction_ids : sequence of str or None
        Defaults to every known prototype junction (junction_1,
        junction_2, junction_3, in that order).
    """
    if junction_ids is None:
        junction_ids = list(PROTOTYPE_JUNCTION_COORDINATES.keys())

    records = []
    for junction_id in junction_ids:
        latitude, longitude = PROTOTYPE_JUNCTION_COORDINATES[junction_id]
        records.append({
            "id": junction_id,
            "latitude": latitude,
            "longitude": longitude,
        })
    return records


def run_emergency_response_pipeline(
    base_graph: nx.MultiDiGraph,
    officers: Sequence[Dict[str, Any]],
    junction_observation_files: Mapping[str, Union[str, Path]],
    weight_mode: str = "dynamic",
    threshold_minutes: float = DEFAULT_THRESHOLD_MINUTES,
    junction_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Run the full ML -> routing -> coverage emergency-response pipeline
    by composing already-existing, already-tested components. This
    function performs no routing, congestion, risk, cost-matrix, or
    coverage computation itself - see module docstring.

    Parameters
    ----------
    base_graph : networkx.MultiDiGraph
        The processed Nagpur road graph (e.g. via
        graph_utils.load_graph(config.PROCESSED_GRAPH_PATH)). Never
        mutated by this function or by anything it calls -
        build_dynamic_graph_from_files() operates on a graph.copy()
        internally (see ml_dynamic_graph.py), and weight_mode="base"
        routes directly on `base_graph` itself, read-only, exactly as
        cost_matrix.build_cost_matrix() already does for every other
        base-mode caller.
    officers : sequence of dict
        Officer locations, in the existing project representation:
        {"id": ..., "latitude": ..., "longitude": ...} (see
        pipeline_demo.SAMPLE_OFFICERS - no new officer model is
        introduced here).
    junction_observation_files : mapping of junction_id -> path
        Real ML observation JSON files, e.g.
        {"junction_1": ".../junction_1_latest.json",
         "junction_2": ".../junction_2_latest.json"}.
        A junction with no real ML observation (e.g. junction_3, as of
        this writing) must simply be omitted here - never included
        with a placeholder/guessed path (see ml_dynamic_graph.py and
        risk_priority.py, both reused unmodified for this).
    weight_mode : str
        "dynamic" (default) or "base". "dynamic" routes the cost matrix
        on the ML-congestion-adjusted graph (dynamic_travel_time);
        "base" routes directly on `base_graph`, untouched by any ML
        observation (travel_time). Any other value raises ValueError
        early - the same VALID_WEIGHT_MODES check routing.py already
        performs, surfaced here before any routing work begins rather
        than only after the first cost-matrix pair is computed.
    threshold_minutes : float
        Passed straight through to
        coverage.analyze_coverage_from_cost_matrix().
    junction_ids : sequence of str or None
        Which junctions to include as high-risk-junction input records
        (coordinates only - independent of which junctions have a real
        ML observation). Defaults to every junction in
        junction_mapping.PROTOTYPE_JUNCTION_COORDINATES.

    Returns
    -------
    dict
        {
            "dynamic_graph": ml_dynamic_graph.DynamicGraphResult,
                # Always built (ML observations are always loaded,
                # regardless of `weight_mode`), so callers can inspect
                # which junctions/edges were ML-covered even when
                # weight_mode="base" was used for the actual routing.
            "risk_scores": {junction_id: float, ...},
                # Real ML risk_score, 0-100 scale, verbatim. Omits any
                # junction_id with no real observation file supplied.
            "cost_matrix": dict,
                # cost_matrix.build_cost_matrix()'s full result,
                # computed with weight_mode="base" or "dynamic" as
                # requested.
            "coverage": dict,
                # coverage.analyze_coverage_from_cost_matrix()'s full
                # result, built from "cost_matrix" above with
                # risk_scores="risk_scores" above passed through.
        }

    Raises
    ------
    ValueError
        If `weight_mode` is not "base" or "dynamic".
    ml_integration.MLObservationError
        If any file in `junction_observation_files` is missing, not
        valid JSON, or fails ML observation schema validation
        (propagated unmodified from ml_dynamic_graph.py /
        risk_priority.py - both of which load every file independently,
        so a missing/malformed file fails clearly and early, before any
        routing work happens).
    ml_dynamic_graph.JunctionObservationMismatchError,
    risk_priority.RiskJunctionMismatchError
        If a junction_id key does not match the road_segment_id found
        inside its observation JSON (an "unknown junction" case, on the
        congestion side and risk side respectively).
    """
    if weight_mode not in VALID_WEIGHT_MODES:
        raise ValueError(
            f"Invalid weight_mode {weight_mode!r}; must be one of "
            f"{VALID_WEIGHT_MODES}."
        )

    # Always build the dynamic graph and the risk map, regardless of
    # weight_mode: ML observations are the input to both the congestion
    # side (dynamic_travel_time) and the risk side (risk_scores), and
    # callers of this function reasonably expect "dynamic_graph" /
    # "risk_scores" to be populated in the returned result either way
    # (e.g. to inspect ML coverage even while routing in "base" mode).
    dynamic_graph_result: DynamicGraphResult = build_dynamic_graph_from_files(
        base_graph, junction_observation_files
    )
    risk_scores: Dict[str, float] = build_risk_priority_map(
        junction_observation_files
    )

    high_risk_junctions = _build_high_risk_junction_records(junction_ids)

    # weight_mode="base" routes on the original, untouched base_graph
    # (dynamic_graph_result.graph is never used for base-mode routing);
    # weight_mode="dynamic" routes on the ML-congestion-adjusted copy.
    # Either way, build_cost_matrix() remains the single routing entry
    # point - this module never calls routing.shortest_path() itself.
    graph_for_cost_matrix = (
        base_graph if weight_mode == "base" else dynamic_graph_result.graph
    )

    cost_matrix_result = build_cost_matrix(
        graph_for_cost_matrix,
        officers,
        high_risk_junctions,
        weight_mode=weight_mode,
    )

    coverage_result = analyze_coverage_from_cost_matrix(
        cost_matrix_result,
        threshold_minutes=threshold_minutes,
        risk_scores=risk_scores,
    )

    return {
        "dynamic_graph": dynamic_graph_result,
        "risk_scores": risk_scores,
        "cost_matrix": cost_matrix_result,
        "coverage": coverage_result,
    }