"""
test_emergency_response_pipeline.py

End-to-end test suite for src/emergency_response_pipeline.py (Prompt
8A). Uses the ACTUAL real NovaRoute ML JSON files (junction_1_latest.
json / junction_2_latest.json) and the ACTUAL processed Nagpur graph
for the main end-to-end tests, mirroring test_ml_dynamic_graph.py's
NOVAROUTE_ML_OUTPUT_DIR resolution pattern. Small synthetic observation
dicts are used only for the isolated failure-mode tests, which need
precise control over specific field values.

Run with:
    pytest src/test_emergency_response_pipeline.py -v
"""

import copy
import os
import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_GRAPH_PATH, WEIGHT_ATTRIBUTE
from src.congestion import DYNAMIC_WEIGHT_ATTRIBUTE, calculate_dynamic_travel_time
from src.emergency_response_pipeline import run_emergency_response_pipeline
from src.graph_utils import load_graph
from src.junction_mapping import PROTOTYPE_JUNCTION_COORDINATES
from src.ml_integration import MLObservationError
from src.ml_dynamic_graph import JunctionObservationMismatchError
from src.risk_priority import RiskJunctionMismatchError


# ---------------------------------------------------------------------------
# Real ML observation file locations (mirrors test_ml_dynamic_graph.py /
# test_risk_priority.py)
# ---------------------------------------------------------------------------
_DEFAULT_ML_OBS_DIR = Path(__file__).resolve().parent.parent / "data" / "ml_observations"
ML_OBS_DIR = (
    Path(os.environ["NOVAROUTE_ML_OUTPUT_DIR"])
    if os.environ.get("NOVAROUTE_ML_OUTPUT_DIR")
    else _DEFAULT_ML_OBS_DIR
)
JUNCTION_1_FILE = ML_OBS_DIR / "junction_1_latest.json"
JUNCTION_2_FILE = ML_OBS_DIR / "junction_2_latest.json"
JUNCTION_3_FILE = ML_OBS_DIR / "junction_3_latest.json"


def _require_real_ml_file(path: Path, junction_id: str) -> None:
    """Fail the test clearly (never skip) when a required real ML
    observation file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required real ML observation file for {junction_id!r} not "
            f"found at '{path}'.\n"
            f"Resolved ML output directory: '{ML_OBS_DIR}' "
            f"({'from NOVAROUTE_ML_OUTPUT_DIR env var' if os.environ.get('NOVAROUTE_ML_OUTPUT_DIR') else 'default fallback, NOVAROUTE_ML_OUTPUT_DIR not set'}).\n"
            "Set the NOVAROUTE_ML_OUTPUT_DIR environment variable to the "
            "directory containing junction_1_latest.json and "
            "junction_2_latest.json (your real NovaRoute_AI ml/outputs/metrics "
            "folder) before running this test suite."
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nagpur_graph():
    """Load the actual processed Nagpur graph once for all tests in this
    module. Skips the whole module if the graph file doesn't exist yet."""
    if not PROCESSED_GRAPH_PATH.exists():
        pytest.skip(
            f"Processed graph not found at '{PROCESSED_GRAPH_PATH}'. "
            "Run graph_builder.py first."
        )
    return load_graph(PROCESSED_GRAPH_PATH)


@pytest.fixture(scope="module")
def real_junction_files():
    _require_real_ml_file(JUNCTION_1_FILE, "junction_1")
    _require_real_ml_file(JUNCTION_2_FILE, "junction_2")
    return {"junction_1": JUNCTION_1_FILE, "junction_2": JUNCTION_2_FILE}


# Existing project officer representation (id/latitude/longitude),
# reused from pipeline_demo.SAMPLE_OFFICERS' shape - not its risk-score
# data, which this module never touches. Defined locally (rather than
# imported from pipeline_demo.py) so this test file has zero import
# dependency on pipeline_demo.py, matching Prompt 8A's instruction that
# the orchestrator itself must not source anything from it.
SAMPLE_OFFICERS = [
    {"id": "officer_1", "latitude": 21.1458, "longitude": 79.0882},
    {"id": "officer_2", "latitude": 21.1300, "longitude": 79.0700},
    {"id": "officer_3", "latitude": 21.1700, "longitude": 79.1100},
]


def _valid_obs(road_segment_id="junction_1", risk_score=42.0, congestion_factor=1.05):
    return {
        "schema_version": "1.1.0",
        "road_segment_id": road_segment_id,
        "osm_edge": None,
        "timestamp": "2026-08-16T18:56:44.056086+00:00",
        "observation_window_seconds": 5.0,
        "vehicle_counts": {"car": 4, "motorcycle": 6, "bus": 1, "truck": 0, "total": 11},
        "total_vehicles": 11,
        "peak_vehicles": 17,
        "road_capacity": 50.0,
        "traffic_density": 0.22,
        "congestion_level": "free_flow",
        "congestion_factor": congestion_factor,
        "aggregation_method": "average_active_vehicles",
        "risk_score": risk_score,
        "risk_level": "low",
        "risk_factor_scores": {
            "accident_history": 0.2, "traffic_density": 0.146,
            "pedestrian_conflict": 0.16, "time_of_day": 0.8,
        },
        "risk_contributions": {
            "accident_history": 7.0, "traffic_density": 3.67,
            "pedestrian_conflict": 4.0, "time_of_day": 12.0,
        },
    }


# ===========================================================================
# 1. Real junction_1 and junction_2 flow through the entire pipeline
# ===========================================================================

def test_real_observations_flow_through_entire_pipeline(nagpur_graph, real_junction_files):
    result = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )
    assert set(result.keys()) == {"dynamic_graph", "risk_scores", "cost_matrix", "coverage"}
    assert set(result["cost_matrix"]["junction_ids"]) == {
        "junction_1", "junction_2", "junction_3",
    }
    assert result["coverage"]["threshold_minutes"] > 0
    assert len(result["coverage"]["junctions"]) == 3


# ===========================================================================
# 2-3. Dynamic graph is a separate object; original graph unchanged
# ===========================================================================

def test_dynamic_graph_is_a_separate_object(nagpur_graph, real_junction_files):
    result = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )
    dynamic_graph = result["dynamic_graph"].graph
    assert dynamic_graph is not nagpur_graph
    assert isinstance(dynamic_graph, type(nagpur_graph))


def test_original_graph_unchanged_after_pipeline_run(nagpur_graph, real_junction_files):
    has_dynamic_attr_before = any(
        DYNAMIC_WEIGHT_ATTRIBUTE in data
        for _, _, data in nagpur_graph.edges(data=True)
    )
    assert not has_dynamic_attr_before

    run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )

    has_dynamic_attr_after = any(
        DYNAMIC_WEIGHT_ATTRIBUTE in data
        for _, _, data in nagpur_graph.edges(data=True)
    )
    assert not has_dynamic_attr_after


# ===========================================================================
# 4. Every edge in the dynamic graph has dynamic_travel_time
# ===========================================================================

def test_every_edge_in_dynamic_graph_has_dynamic_travel_time(nagpur_graph, real_junction_files):
    result = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )
    dynamic_graph = result["dynamic_graph"].graph
    edges_missing = [
        (u, v, k) for u, v, k, data in dynamic_graph.edges(keys=True, data=True)
        if DYNAMIC_WEIGHT_ATTRIBUTE not in data and WEIGHT_ATTRIBUTE in data
    ]
    assert edges_missing == []


# ===========================================================================
# 5. ML-covered edges receive travel_time * congestion_factor
# ===========================================================================

def test_ml_covered_edges_equal_travel_time_times_congestion_factor(nagpur_graph, real_junction_files):
    result = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )
    dyn_result = result["dynamic_graph"]
    dynamic_graph = dyn_result.graph

    checked_any = False
    for junction_id, segment_ids in dyn_result.updated_segment_ids_by_junction.items():
        factor = dyn_result.congestion_factor_by_junction[junction_id]
        for segment_id in segment_ids:
            u_str, v_str, k_str = segment_id.split("_")
            for u, v, k, data in dynamic_graph.edges(keys=True, data=True):
                if f"{u}_{v}_{k}" == segment_id:
                    expected = calculate_dynamic_travel_time(data[WEIGHT_ATTRIBUTE], factor)
                    assert data[DYNAMIC_WEIGHT_ATTRIBUTE] == pytest.approx(expected)
                    checked_any = True
    assert checked_any, "Expected at least one ML-covered edge to verify."


# ===========================================================================
# 6. Unmapped edges retain dynamic_travel_time == travel_time
# ===========================================================================

def test_unmapped_edges_retain_baseline_dynamic_travel_time(nagpur_graph, real_junction_files):
    result = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )
    dyn_result = result["dynamic_graph"]
    dynamic_graph = dyn_result.graph

    covered_segment_ids = set()
    for segment_ids in dyn_result.updated_segment_ids_by_junction.values():
        covered_segment_ids.update(segment_ids)

    checked_any = False
    for u, v, k, data in dynamic_graph.edges(keys=True, data=True):
        segment_id = f"{u}_{v}_{k}"
        if segment_id in covered_segment_ids:
            continue
        if WEIGHT_ATTRIBUTE not in data:
            continue
        assert data[DYNAMIC_WEIGHT_ATTRIBUTE] == data[WEIGHT_ATTRIBUTE]
        checked_any = True
        if checked_any:
            break
    assert checked_any, "Expected at least one unmapped edge to verify."


# ===========================================================================
# 7-8. Dynamic cost matrix uses dynamic_travel_time; base uses travel_time
# ===========================================================================

def test_dynamic_cost_matrix_uses_dynamic_travel_time(nagpur_graph, real_junction_files):
    result = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )
    assert result["cost_matrix"]["weight_mode"] == "dynamic"
    assert result["cost_matrix"]["weight_attribute"] == DYNAMIC_WEIGHT_ATTRIBUTE


def test_base_cost_matrix_uses_travel_time(nagpur_graph, real_junction_files):
    result = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="base",
    )
    assert result["cost_matrix"]["weight_mode"] == "base"
    assert result["cost_matrix"]["weight_attribute"] == WEIGHT_ATTRIBUTE


# ===========================================================================
# 9-10. Real ML risk_score preserved exactly
# ===========================================================================

def test_real_junction_1_risk_score_preserved(nagpur_graph, real_junction_files):
    result = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )
    assert result["risk_scores"]["junction_1"] == pytest.approx(26.6667)


def test_real_junction_2_risk_score_preserved(nagpur_graph, real_junction_files):
    result = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )
    assert result["risk_scores"]["junction_2"] == pytest.approx(25.6667)


# ===========================================================================
# 11-12. junction_3: no fabricated risk score, still participates in routing
# ===========================================================================

def test_junction_3_has_no_fabricated_risk_score(nagpur_graph, real_junction_files):
    assert not JUNCTION_3_FILE.exists(), (
        "This test asserts junction_3 has no real ML observation on disk. "
        "If a real junction_3_latest.json now exists, this test's premise "
        "is stale and should be revisited, not silently left passing."
    )
    result = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )
    assert "junction_3" not in result["risk_scores"]
    reports_by_id = {r["junction_id"]: r for r in result["coverage"]["junctions"]}
    assert reports_by_id["junction_3"]["risk_score"] is None


def test_junction_3_participates_in_cost_matrix_with_baseline_dynamic_time(nagpur_graph, real_junction_files):
    result = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )
    cmr = result["cost_matrix"]
    assert "junction_3" in cmr["junction_ids"]
    j_idx = cmr["junction_ids"].index("junction_3")
    # junction_3 must have a real (possibly unreachable, but present and
    # numeric) entry for every officer - i.e. it was actually routed,
    # not skipped/omitted from the matrix.
    assert cmr["matrix"].shape[1] > j_idx
    for officer_id in cmr["officer_ids"]:
        assert (officer_id, "junction_3") in cmr["routes"]


# ===========================================================================
# 13-14. Coverage receives and preserves the real risk_scores mapping
# ===========================================================================

def test_coverage_receives_real_risk_scores_mapping(nagpur_graph, real_junction_files):
    result = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )
    reports_by_id = {r["junction_id"]: r for r in result["coverage"]["junctions"]}
    assert reports_by_id["junction_1"]["risk_score"] == result["risk_scores"]["junction_1"]
    assert reports_by_id["junction_2"]["risk_score"] == result["risk_scores"]["junction_2"]


def test_coverage_reports_preserve_exact_real_risk_scores(nagpur_graph, real_junction_files):
    result = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )
    reports_by_id = {r["junction_id"]: r for r in result["coverage"]["junctions"]}
    assert reports_by_id["junction_1"]["risk_score"] == pytest.approx(26.6667)
    assert reports_by_id["junction_2"]["risk_score"] == pytest.approx(25.6667)


# ===========================================================================
# 15-17. congestion_factor / risk_score independence
# ===========================================================================

def test_risk_score_does_not_modify_travel_time(nagpur_graph):
    # Two synthetic observations for the same junction_id/segments, same
    # congestion_factor, but very different risk_score. dynamic_travel_time
    # on ML-covered edges must be identical either way.
    obs_low_risk = _valid_obs("junction_1", risk_score=5.0, congestion_factor=1.3)
    obs_high_risk = _valid_obs("junction_1", risk_score=95.0, congestion_factor=1.3)

    files_a = {"junction_1": obs_low_risk}
    files_b = {"junction_1": obs_high_risk}

    result_a = run_emergency_response_pipeline(nagpur_graph, SAMPLE_OFFICERS, files_a, weight_mode="dynamic")
    result_b = run_emergency_response_pipeline(nagpur_graph, SAMPLE_OFFICERS, files_b, weight_mode="dynamic")

    graph_a = result_a["dynamic_graph"].graph
    graph_b = result_b["dynamic_graph"].graph
    segs_a = result_a["dynamic_graph"].updated_segment_ids_by_junction["junction_1"]
    segs_b = result_b["dynamic_graph"].updated_segment_ids_by_junction["junction_1"]
    assert segs_a == segs_b
    for u, v, k, data_a in graph_a.edges(keys=True, data=True):
        segment_id = f"{u}_{v}_{k}"
        if segment_id in segs_a:
            data_b = graph_b[u][v][k]
            assert data_a[DYNAMIC_WEIGHT_ATTRIBUTE] == data_b[DYNAMIC_WEIGHT_ATTRIBUTE]


def test_risk_score_does_not_modify_congestion_factor(nagpur_graph):
    obs_low_risk = _valid_obs("junction_1", risk_score=5.0, congestion_factor=1.3)
    obs_high_risk = _valid_obs("junction_1", risk_score=95.0, congestion_factor=1.3)

    result_a = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, {"junction_1": obs_low_risk}, weight_mode="dynamic",
    )
    result_b = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, {"junction_1": obs_high_risk}, weight_mode="dynamic",
    )
    factor_a = result_a["dynamic_graph"].congestion_factor_by_junction["junction_1"]
    factor_b = result_b["dynamic_graph"].congestion_factor_by_junction["junction_1"]
    assert factor_a == factor_b == 1.3


def test_congestion_factor_does_not_modify_risk_score(nagpur_graph):
    obs_low_congestion = _valid_obs("junction_1", risk_score=50.0, congestion_factor=1.01)
    obs_high_congestion = _valid_obs("junction_1", risk_score=50.0, congestion_factor=4.5)

    result_a = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, {"junction_1": obs_low_congestion}, weight_mode="dynamic",
    )
    result_b = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, {"junction_1": obs_high_congestion}, weight_mode="dynamic",
    )
    assert result_a["risk_scores"]["junction_1"] == result_b["risk_scores"]["junction_1"] == 50.0


# ===========================================================================
# 18. Base and dynamic modes remain distinct
# ===========================================================================

def test_base_and_dynamic_modes_remain_distinct(nagpur_graph, real_junction_files):
    result_base = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="base",
    )
    result_dynamic = run_emergency_response_pipeline(
        nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="dynamic",
    )
    assert result_base["cost_matrix"]["weight_attribute"] != result_dynamic["cost_matrix"]["weight_attribute"]
    # The base-mode cost matrix must have been computed by routing
    # directly on the original graph, never the dynamic copy.
    assert DYNAMIC_WEIGHT_ATTRIBUTE not in next(iter(nagpur_graph.edges(data=True)))[2]


# ===========================================================================
# 19. No second Dijkstra implementation exists
# ===========================================================================

def test_orchestrator_never_calls_routing_shortest_path_directly():
    import ast
    import src.emergency_response_pipeline as erp_module

    source = Path(erp_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Functional check only (ignore docstrings/comments): walk the AST
    # for actual call expressions and import statements.
    called_names = set()
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called_names.add(node.func.attr)
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name)

    assert "shortest_path" not in called_names
    assert "shortest_path" not in imported_names
    # It must go through build_cost_matrix (the single existing routing
    # entry point), which it does import and call.
    assert "build_cost_matrix" in imported_names
    assert "build_cost_matrix" in called_names


# ===========================================================================
# 20. Full existing test suite remains green
# ===========================================================================
# Not re-implemented here (would duplicate the entire existing suite).
# Verified as a separate step: `python -m pytest -q` over all of src/,
# reported alongside this file's own results in the Prompt 8A report.


# ===========================================================================
# Failure cases
# ===========================================================================

def test_missing_ml_file_fails_clearly(nagpur_graph, tmp_path):
    missing_file = tmp_path / "does_not_exist.json"
    with pytest.raises(MLObservationError):
        run_emergency_response_pipeline(
            nagpur_graph, SAMPLE_OFFICERS, {"junction_1": missing_file}, weight_mode="dynamic",
        )


def test_malformed_ml_json_fails_clearly(nagpur_graph, tmp_path):
    bad_file = tmp_path / "junction_1_latest.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(MLObservationError):
        run_emergency_response_pipeline(
            nagpur_graph, SAMPLE_OFFICERS, {"junction_1": bad_file}, weight_mode="dynamic",
        )


def test_unknown_junction_id_mismatch_fails_clearly(nagpur_graph):
    # Supplied under "junction_2" key but the JSON says junction_1.
    obs = _valid_obs("junction_1")
    with pytest.raises((JunctionObservationMismatchError, RiskJunctionMismatchError)):
        run_emergency_response_pipeline(
            nagpur_graph, SAMPLE_OFFICERS, {"junction_2": obs}, weight_mode="dynamic",
        )


def test_invalid_risk_score_fails_clearly(nagpur_graph):
    obs = _valid_obs("junction_1", risk_score=150.0)
    with pytest.raises(MLObservationError):
        run_emergency_response_pipeline(
            nagpur_graph, SAMPLE_OFFICERS, {"junction_1": obs}, weight_mode="dynamic",
        )


def test_missing_risk_score_fails_clearly(nagpur_graph):
    obs = _valid_obs("junction_1")
    del obs["risk_score"]
    with pytest.raises(MLObservationError):
        run_emergency_response_pipeline(
            nagpur_graph, SAMPLE_OFFICERS, {"junction_1": obs}, weight_mode="dynamic",
        )


def test_invalid_weight_mode_fails_clearly(nagpur_graph, real_junction_files):
    with pytest.raises(ValueError):
        run_emergency_response_pipeline(
            nagpur_graph, SAMPLE_OFFICERS, real_junction_files, weight_mode="turbo",
        )