"""
test_risk_priority.py

Test suite for src/risk_priority.py (Prompt 7B: real ML risk_score ->
coverage.py junction priority integration).

Uses the ACTUAL real NovaRoute ML JSON files (junction_1_latest.json /
junction_2_latest.json) for the integration-style tests, mirroring
test_ml_dynamic_graph.py's NOVAROUTE_ML_OUTPUT_DIR resolution pattern,
plus small synthetic observation dicts (via ml_integration's dict-input
support) for the failure-mode tests that need to control specific field
values precisely.

Run with:
    pytest src/test_risk_priority.py -v
"""

import copy
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import WEIGHT_ATTRIBUTE
from src.coverage import analyze_coverage_from_cost_matrix
from src.ml_integration import MLObservationError
from src.risk_priority import RiskJunctionMismatchError, build_risk_priority_map


# ---------------------------------------------------------------------------
# Real ML observation file locations (mirrors test_ml_dynamic_graph.py)
# ---------------------------------------------------------------------------
# Resolved from the NOVAROUTE_ML_OUTPUT_DIR environment variable, which
# must point at the directory containing the real NovaRoute ML pipeline
# output (junction_1_latest.json, junction_2_latest.json). Falls back to
# a repo-relative data/ml_observations/ directory if unset. This is
# TEST/FIXTURE configuration only - risk_priority.py itself has no
# knowledge of this environment variable or of any absolute path; it
# only ever accepts file paths/dicts as explicit function arguments.
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
    observation file is missing, explaining exactly which file and how
    to configure the directory it should be found in."""
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
# Synthetic observation builders (for failure-mode / controlled-value tests)
# ---------------------------------------------------------------------------

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
            "accident_history": 0.2,
            "traffic_density": 0.146,
            "pedestrian_conflict": 0.16,
            "time_of_day": 0.8,
        },
        "risk_contributions": {
            "accident_history": 7.0,
            "traffic_density": 3.67,
            "pedestrian_conflict": 4.0,
            "time_of_day": 12.0,
        },
    }


# ---------------------------------------------------------------------------
# 1-2. Real junction_1 / junction_2 risk_score loads correctly
# ---------------------------------------------------------------------------

def test_real_junction_1_risk_score_loaded_correctly():
    _require_real_ml_file(JUNCTION_1_FILE, "junction_1")
    result = build_risk_priority_map({"junction_1": JUNCTION_1_FILE})
    assert result == {"junction_1": pytest.approx(26.6667)}


def test_real_junction_2_risk_score_loaded_correctly():
    _require_real_ml_file(JUNCTION_2_FILE, "junction_2")
    result = build_risk_priority_map({"junction_2": JUNCTION_2_FILE})
    assert result == {"junction_2": pytest.approx(25.6667)}


def test_real_both_junctions_loaded_together():
    _require_real_ml_file(JUNCTION_1_FILE, "junction_1")
    _require_real_ml_file(JUNCTION_2_FILE, "junction_2")
    result = build_risk_priority_map(
        {"junction_1": JUNCTION_1_FILE, "junction_2": JUNCTION_2_FILE}
    )
    assert result == {
        "junction_1": pytest.approx(26.6667),
        "junction_2": pytest.approx(25.6667),
    }


# ---------------------------------------------------------------------------
# 3. Missing junction_3 observation does not fabricate a score
# ---------------------------------------------------------------------------

def test_junction_3_omitted_when_no_real_observation_exists():
    # junction_3 simply never appears in `sources` (mirrors how
    # ml_dynamic_graph.load_junction_observations is called in
    # production - only junctions with a real file are included).
    assert not JUNCTION_3_FILE.exists(), (
        "This test asserts junction_3 has no real ML observation on disk. "
        "If a real junction_3_latest.json now exists, this test's premise "
        "is stale and should be revisited, not silently left passing."
    )
    result = build_risk_priority_map({"junction_1": _valid_obs("junction_1")})
    assert "junction_3" not in result


def test_empty_sources_returns_empty_map_no_fabrication():
    assert build_risk_priority_map({}) == {}


# ---------------------------------------------------------------------------
# 4. Invalid ML JSON fails clearly
# ---------------------------------------------------------------------------

def test_invalid_json_file_fails_clearly(tmp_path):
    bad_file = tmp_path / "junction_1_latest.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(MLObservationError):
        build_risk_priority_map({"junction_1": bad_file})


def test_nonexistent_file_fails_clearly(tmp_path):
    missing_file = tmp_path / "does_not_exist.json"
    with pytest.raises(MLObservationError):
        build_risk_priority_map({"junction_1": missing_file})


def test_non_dict_json_body_fails_clearly(tmp_path):
    # A file that IS valid JSON but is not a JSON object (a top-level
    # array instead) - must fail via MLObservationError's "must be a
    # JSON object" check, not fabricate/pass through partial data.
    bad_file = tmp_path / "junction_1_latest.json"
    bad_file.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(MLObservationError):
        build_risk_priority_map({"junction_1": bad_file})


def test_risk_score_out_of_range_fails_clearly():
    obs = _valid_obs("junction_1", risk_score=150.0)
    with pytest.raises(MLObservationError):
        build_risk_priority_map({"junction_1": obs})


# ---------------------------------------------------------------------------
# 5. Missing risk_score fails clearly
# ---------------------------------------------------------------------------

def test_missing_risk_score_field_fails_clearly():
    obs = _valid_obs("junction_1")
    del obs["risk_score"]
    with pytest.raises(MLObservationError):
        build_risk_priority_map({"junction_1": obs})


# ---------------------------------------------------------------------------
# Unknown junction (mismatch) case
# ---------------------------------------------------------------------------

def test_junction_id_mismatch_fails_clearly():
    # Supplied under "junction_2" but the JSON itself says junction_1.
    obs = _valid_obs("junction_1")
    with pytest.raises(RiskJunctionMismatchError):
        build_risk_priority_map({"junction_2": obs})


# ---------------------------------------------------------------------------
# 6. Risk scores remain on the 0-100 scale (never rescaled to 0-1)
# ---------------------------------------------------------------------------

def test_risk_score_stays_on_0_100_scale_not_rescaled_to_0_1():
    obs = _valid_obs("junction_1", risk_score=42.0)
    result = build_risk_priority_map({"junction_1": obs})
    assert result["junction_1"] == 42.0  # exact, verbatim - no /100 division


def test_real_risk_scores_are_in_0_100_range_not_0_1():
    _require_real_ml_file(JUNCTION_1_FILE, "junction_1")
    result = build_risk_priority_map({"junction_1": JUNCTION_1_FILE})
    # 26.6667 would be nonsensical on a 0-1 scale; confirms no rescale.
    assert result["junction_1"] > 1.0
    assert result["junction_1"] <= 100.0


# ---------------------------------------------------------------------------
# 7-8. Risk score does not modify congestion_factor / travel_time
# ---------------------------------------------------------------------------

def test_building_risk_map_does_not_touch_congestion_factor():
    obs = _valid_obs("junction_1", risk_score=42.0, congestion_factor=1.35)
    obs_before = copy.deepcopy(obs)
    build_risk_priority_map({"junction_1": obs})
    # The source observation dict passed in must be completely unchanged -
    # risk_priority.py never mutates congestion_factor (or anything else).
    assert obs == obs_before
    assert obs["congestion_factor"] == 1.35


def test_risk_priority_module_has_no_functional_coupling_to_congestion_or_travel_time():
    import src.risk_priority as rp_module
    # Functional check (not a docstring-wording check): the module must
    # never import anything from congestion.py or ml_dynamic_graph.py -
    # the only way it could touch congestion_factor/travel_time/
    # dynamic_travel_time at all.
    source = Path(rp_module.__file__).read_text(encoding="utf-8")
    assert "from src.congestion" not in source
    assert "import src.congestion" not in source
    assert "from src.ml_dynamic_graph" not in source
    assert "import src.ml_dynamic_graph" not in source
    # And confirm no such names exist as attributes on the loaded module.
    assert not hasattr(rp_module, "calculate_congestion_factor")
    assert not hasattr(rp_module, "calculate_dynamic_travel_time")


# ---------------------------------------------------------------------------
# 9. Risk score is correctly passed through to coverage.py
# ---------------------------------------------------------------------------

def _make_route(is_reachable, total_time_seconds=None):
    return {
        "is_reachable": is_reachable,
        "route_nodes": ["origin", "dest"] if is_reachable else None,
        "total_time_seconds": total_time_seconds,
        "total_distance_meters": (total_time_seconds or 0) * 10 if is_reachable else None,
        "origin_node": None,
        "destination_node": None,
        "error": None if is_reachable else "unreachable",
    }


def _cost_matrix_result(junction_ids, matrix, congestion_factors=None):
    officer_ids = ["officer_1"]
    routes = {}
    for j_idx, jid in enumerate(junction_ids):
        routes[("officer_1", jid)] = _make_route(True, matrix[0][j_idx] * 60.0)
    return {
        "officer_ids": officer_ids,
        "junction_ids": junction_ids,
        "matrix": np.array(matrix),
        "time_unit": "minutes",
        "weight_mode": "base",
        "weight_attribute": WEIGHT_ATTRIBUTE,
        "routes": routes,
        "unreachable_pairs": [],
    }


def test_risk_score_correctly_passed_through_into_coverage_reports():
    cmr = _cost_matrix_result(["junction_1", "junction_2"], [[3.0, 10.0]])
    risk_scores = build_risk_priority_map(
        {
            "junction_1": _valid_obs("junction_1", risk_score=26.6667),
            "junction_2": _valid_obs("junction_2", risk_score=25.6667),
        }
    )
    result = analyze_coverage_from_cost_matrix(
        cmr, threshold_minutes=5.0, risk_scores=risk_scores
    )
    reports_by_id = {r["junction_id"]: r for r in result["junctions"]}
    assert reports_by_id["junction_1"]["risk_score"] == pytest.approx(26.6667)
    assert reports_by_id["junction_2"]["risk_score"] == pytest.approx(25.6667)


def test_junction_without_real_observation_gets_none_risk_score_in_coverage():
    cmr = _cost_matrix_result(["junction_1", "junction_3"], [[3.0, 10.0]])
    # Only junction_1 has a real observation; junction_3 is omitted.
    risk_scores = build_risk_priority_map(
        {"junction_1": _valid_obs("junction_1", risk_score=26.6667)}
    )
    result = analyze_coverage_from_cost_matrix(
        cmr, threshold_minutes=5.0, risk_scores=risk_scores
    )
    reports_by_id = {r["junction_id"]: r for r in result["junctions"]}
    assert reports_by_id["junction_1"]["risk_score"] == pytest.approx(26.6667)
    assert reports_by_id["junction_3"]["risk_score"] is None


# ---------------------------------------------------------------------------
# 10. Higher risk junctions appear earlier in uncovered-junction ordering
# ---------------------------------------------------------------------------

def test_higher_risk_junction_ranks_earlier_in_uncovered_priority():
    # Both junctions unreachable within threshold -> both uncovered.
    cmr = _cost_matrix_result(["junction_1", "junction_2"], [[50.0, 50.0]])
    risk_scores = build_risk_priority_map(
        {
            "junction_1": _valid_obs("junction_1", risk_score=26.6667),
            "junction_2": _valid_obs("junction_2", risk_score=25.6667),
        }
    )
    result = analyze_coverage_from_cost_matrix(
        cmr, threshold_minutes=5.0, risk_scores=risk_scores
    )
    uncovered = result["uncovered_junction_ids"]
    assert uncovered.index("junction_1") < uncovered.index("junction_2")


# ---------------------------------------------------------------------------
# 11. Risk ordering is independent of congestion_factor
# ---------------------------------------------------------------------------

def test_risk_ordering_independent_of_congestion_factor():
    # junction_1 has a much higher congestion_factor than junction_2, but
    # a LOWER risk_score. Uncovered ordering must follow risk_score only.
    cmr = _cost_matrix_result(["junction_1", "junction_2"], [[50.0, 50.0]])
    risk_scores = build_risk_priority_map(
        {
            "junction_1": _valid_obs(
                "junction_1", risk_score=10.0, congestion_factor=5.0
            ),
            "junction_2": _valid_obs(
                "junction_2", risk_score=90.0, congestion_factor=1.01
            ),
        }
    )
    result = analyze_coverage_from_cost_matrix(
        cmr, threshold_minutes=5.0, risk_scores=risk_scores
    )
    uncovered = result["uncovered_junction_ids"]
    # junction_2 has the higher risk_score (90 > 10), despite the far
    # lower congestion_factor, so it must rank first.
    assert uncovered.index("junction_2") < uncovered.index("junction_1")


# ---------------------------------------------------------------------------
# 12. Existing full suite continues to pass
# ---------------------------------------------------------------------------
# Not re-implemented here (that would be redundant with test_cost_matrix.py,
# test_coverage.py, test_ml_dynamic_graph.py, etc. themselves). Verified
# instead by running `python -m pytest -q` for the whole src/ directory as
# a separate step after this file - see Prompt 7B report, item I.
