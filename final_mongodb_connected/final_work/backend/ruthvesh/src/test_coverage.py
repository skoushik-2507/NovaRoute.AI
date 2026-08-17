"""
test_coverage.py

Regression test suite for coverage.py (Prompt 7A).

This is a REGRESSION baseline for CURRENT behavior, written by inspecting
coverage.py's actual implementation -- it does not invent or assume any
behavior that isn't already there. In particular:
- risk_scores is verified to be pass-through/sort-key data only, never a
  multiplier or input to the covered/uncovered decision.
- No exclusive one-to-one officer assignment is assumed anywhere: multiple
  junctions are explicitly tested sharing the same nearest_officer_id,
  since analyze_coverage_from_cost_matrix() does independent per-junction
  argmin lookups, not an assignment algorithm.

No production code is modified by this file.

Fixture strategy
-----------------
analyze_coverage_from_cost_matrix() only reads a cost_matrix_result dict
(officer_ids, junction_ids, matrix, time_unit, routes) -- it never touches
the graph or re-runs routing. So these tests hand-build small, fully
deterministic cost_matrix_result dicts matching that documented schema
directly, rather than going through build_cost_matrix() (which would pull
in graph_utils/osmnx for no benefit here). analyze_coverage()'s thin
graph-based wrapper is exercised separately, with the same monkeypatched
find_nearest_node approach used in test_cost_matrix.py.

Run with:
    pytest src/test_coverage.py -v
"""

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.cost_matrix as cost_matrix_module
from src.coverage import (
    DEFAULT_THRESHOLD_MINUTES,
    analyze_coverage,
    analyze_coverage_from_cost_matrix,
    get_junction_report,
    get_uncovered_junctions,
)
from src.cost_matrix import UNREACHABLE_COST
from src.congestion import DYNAMIC_WEIGHT_ATTRIBUTE
from src.config import WEIGHT_ATTRIBUTE


# ---------------------------------------------------------------------------
# Hand-built cost_matrix_result fixtures
# ---------------------------------------------------------------------------

def _make_route(is_reachable, total_time_seconds=None, origin_node=None, destination_node=None):
    return {
        "is_reachable": is_reachable,
        "route_nodes": ["origin", "dest"] if is_reachable else None,
        "total_time_seconds": total_time_seconds,
        "total_distance_meters": (total_time_seconds or 0) * 10 if is_reachable else None,
        "origin_node": origin_node,
        "destination_node": destination_node,
        "error": None if is_reachable else "unreachable",
    }


@pytest.fixture
def basic_cost_matrix_result():
    """
    2 officers x 2 junctions, all reachable, in minutes:

              junction_1   junction_2
    officer_1     3.0          10.0
    officer_2     5.0           1.0

    junction_1 -> nearest officer_1 (3.0)
    junction_2 -> nearest officer_2 (1.0)
    """
    officer_ids = ["officer_1", "officer_2"]
    junction_ids = ["junction_1", "junction_2"]
    matrix = np.array([
        [3.0, 10.0],
        [5.0, 1.0],
    ])
    routes = {
        ("officer_1", "junction_1"): _make_route(True, 180.0),
        ("officer_1", "junction_2"): _make_route(True, 600.0),
        ("officer_2", "junction_1"): _make_route(True, 300.0),
        ("officer_2", "junction_2"): _make_route(True, 60.0),
    }
    return {
        "officer_ids": officer_ids,
        "junction_ids": junction_ids,
        "matrix": matrix,
        "time_unit": "minutes",
        "weight_mode": "base",
        "weight_attribute": WEIGHT_ATTRIBUTE,
        "routes": routes,
        "unreachable_pairs": [],
    }


@pytest.fixture
def cost_matrix_result_with_unreachable(basic_cost_matrix_result):
    """
    Adds junction_3, unreachable from both officers (inf in the matrix,
    no routes entry marked reachable).
    """
    result = dict(basic_cost_matrix_result)
    result["junction_ids"] = result["junction_ids"] + ["junction_3"]
    extra_col = np.array([[UNREACHABLE_COST], [UNREACHABLE_COST]])
    result["matrix"] = np.hstack([result["matrix"], extra_col])
    result["routes"] = dict(result["routes"])
    result["routes"][("officer_1", "junction_3")] = _make_route(False)
    result["routes"][("officer_2", "junction_3")] = _make_route(False)
    return result


@pytest.fixture
def cost_matrix_result_seconds():
    """Same as basic_cost_matrix_result but time_unit='seconds'."""
    return {
        "officer_ids": ["officer_1", "officer_2"],
        "junction_ids": ["junction_1", "junction_2"],
        "matrix": np.array([
            [180.0, 600.0],
            [300.0, 60.0],
        ]),
        "time_unit": "seconds",
        "weight_mode": "base",
        "weight_attribute": WEIGHT_ATTRIBUTE,
        "routes": {
            ("officer_1", "junction_1"): _make_route(True, 180.0),
            ("officer_1", "junction_2"): _make_route(True, 600.0),
            ("officer_2", "junction_1"): _make_route(True, 300.0),
            ("officer_2", "junction_2"): _make_route(True, 60.0),
        },
        "unreachable_pairs": [],
    }


@pytest.fixture
def cost_matrix_result_shared_officer():
    """
    officer_2 is the fastest responder for BOTH junctions, to verify
    coverage.py does not enforce exclusive one-to-one assignment.

              junction_1   junction_2
    officer_1     9.0          9.0
    officer_2     2.0          1.0
    """
    return {
        "officer_ids": ["officer_1", "officer_2"],
        "junction_ids": ["junction_1", "junction_2"],
        "matrix": np.array([
            [9.0, 9.0],
            [2.0, 1.0],
        ]),
        "time_unit": "minutes",
        "weight_mode": "base",
        "weight_attribute": WEIGHT_ATTRIBUTE,
        "routes": {
            ("officer_1", "junction_1"): _make_route(True, 540.0),
            ("officer_1", "junction_2"): _make_route(True, 540.0),
            ("officer_2", "junction_1"): _make_route(True, 120.0),
            ("officer_2", "junction_2"): _make_route(True, 60.0),
        },
        "unreachable_pairs": [],
    }


@pytest.fixture
def cost_matrix_result_four_junctions():
    """
    4 junctions with a mix of covered/uncovered and risk scores, used for
    threshold + sorting tests.

              junction_1  junction_2  junction_3  junction_4
    officer_1     3.0         10.0        9.0         4.0
    officer_2     5.0          1.0        9.0         8.0

    At threshold=2.0 minutes:
      junction_1: min=3.0  -> uncovered (risk 0.90)
      junction_2: min=1.0  -> covered
      junction_3: min=9.0, all-inf-equivalent by design -> uncovered (risk 0.60)
      junction_4: min=4.0  -> uncovered (no risk_score provided)
    """
    return {
        "officer_ids": ["officer_1", "officer_2"],
        "junction_ids": ["junction_1", "junction_2", "junction_3", "junction_4"],
        "matrix": np.array([
            [3.0, 10.0, 9.0, 4.0],
            [5.0, 1.0, 9.0, 8.0],
        ]),
        "time_unit": "minutes",
        "weight_mode": "base",
        "weight_attribute": WEIGHT_ATTRIBUTE,
        "routes": {
            ("officer_1", "junction_1"): _make_route(True, 180.0),
            ("officer_1", "junction_2"): _make_route(True, 600.0),
            ("officer_1", "junction_3"): _make_route(True, 540.0),
            ("officer_1", "junction_4"): _make_route(True, 240.0),
            ("officer_2", "junction_1"): _make_route(True, 300.0),
            ("officer_2", "junction_2"): _make_route(True, 60.0),
            ("officer_2", "junction_3"): _make_route(True, 540.0),
            ("officer_2", "junction_4"): _make_route(True, 480.0),
        },
        "unreachable_pairs": [],
    }


# ---------------------------------------------------------------------------
# 1. analyze_coverage_from_cost_matrix() basic behavior
# ---------------------------------------------------------------------------

def test_analyze_coverage_returns_expected_keys(basic_cost_matrix_result):
    result = analyze_coverage_from_cost_matrix(basic_cost_matrix_result)
    for key in (
        "threshold_minutes", "junctions", "covered_junction_ids",
        "uncovered_junction_ids", "num_covered", "num_uncovered", "coverage_rate",
    ):
        assert key in result


def test_default_threshold_used_when_not_specified(basic_cost_matrix_result):
    result = analyze_coverage_from_cost_matrix(basic_cost_matrix_result)
    assert result["threshold_minutes"] == DEFAULT_THRESHOLD_MINUTES


# ---------------------------------------------------------------------------
# 2 & 3. Threshold behavior; correct covered/uncovered classification
# ---------------------------------------------------------------------------

def test_junctions_within_threshold_are_covered(basic_cost_matrix_result):
    # Default threshold is 6.0 minutes; junction_1 (3.0) and junction_2
    # (1.0) are both within it.
    result = analyze_coverage_from_cost_matrix(basic_cost_matrix_result)
    assert set(result["covered_junction_ids"]) == {"junction_1", "junction_2"}
    assert result["uncovered_junction_ids"] == []
    assert result["num_covered"] == 2
    assert result["num_uncovered"] == 0
    assert result["coverage_rate"] == pytest.approx(1.0)


def test_lower_threshold_produces_uncovered_junction(basic_cost_matrix_result):
    result = analyze_coverage_from_cost_matrix(basic_cost_matrix_result, threshold_minutes=2.0)
    # junction_1's best is 3.0 > 2.0 -> uncovered; junction_2's best is 1.0 <= 2.0 -> covered
    assert result["covered_junction_ids"] == ["junction_2"]
    assert result["uncovered_junction_ids"] == ["junction_1"]
    assert result["coverage_rate"] == pytest.approx(0.5)


def test_response_time_exactly_at_threshold_is_covered(basic_cost_matrix_result):
    # Boundary: threshold == best_time_minutes uses <=, so it should count
    # as covered (junction_1's best response time is exactly 3.0).
    result = analyze_coverage_from_cost_matrix(basic_cost_matrix_result, threshold_minutes=3.0)
    assert "junction_1" in result["covered_junction_ids"]


def test_threshold_must_be_positive(basic_cost_matrix_result):
    with pytest.raises(ValueError):
        analyze_coverage_from_cost_matrix(basic_cost_matrix_result, threshold_minutes=0)
    with pytest.raises(ValueError):
        analyze_coverage_from_cost_matrix(basic_cost_matrix_result, threshold_minutes=-1.0)


def test_non_numeric_threshold_raises(basic_cost_matrix_result):
    with pytest.raises(ValueError):
        analyze_coverage_from_cost_matrix(basic_cost_matrix_result, threshold_minutes="fast")


def test_unreachable_junction_is_always_uncovered(cost_matrix_result_with_unreachable):
    result = analyze_coverage_from_cost_matrix(cost_matrix_result_with_unreachable)
    assert "junction_3" in result["uncovered_junction_ids"]
    report = get_junction_report(result, "junction_3")
    assert report["min_response_time_minutes"] is None
    assert report["nearest_officer_id"] is None
    assert report["is_covered"] is False


# ---------------------------------------------------------------------------
# 4 & 5. Correct minimum response time; correct nearest officer selection
# ---------------------------------------------------------------------------

def test_min_response_time_and_nearest_officer(basic_cost_matrix_result):
    result = analyze_coverage_from_cost_matrix(basic_cost_matrix_result)

    report_j1 = get_junction_report(result, "junction_1")
    assert report_j1["min_response_time_minutes"] == pytest.approx(3.0)
    assert report_j1["nearest_officer_id"] == "officer_1"

    report_j2 = get_junction_report(result, "junction_2")
    assert report_j2["min_response_time_minutes"] == pytest.approx(1.0)
    assert report_j2["nearest_officer_id"] == "officer_2"


def test_response_times_minutes_includes_every_officer(basic_cost_matrix_result):
    result = analyze_coverage_from_cost_matrix(basic_cost_matrix_result)
    report_j1 = get_junction_report(result, "junction_1")
    assert report_j1["response_times_minutes"] == {"officer_1": pytest.approx(3.0), "officer_2": pytest.approx(5.0)}


def test_seconds_time_unit_converted_to_minutes(cost_matrix_result_seconds):
    result = analyze_coverage_from_cost_matrix(cost_matrix_result_seconds)
    report_j2 = get_junction_report(result, "junction_2")
    # 60 seconds -> 1.0 minute
    assert report_j2["min_response_time_minutes"] == pytest.approx(1.0)
    assert report_j2["nearest_officer_id"] == "officer_2"


def test_route_field_matches_nearest_officer(basic_cost_matrix_result):
    result = analyze_coverage_from_cost_matrix(basic_cost_matrix_result)
    report_j1 = get_junction_report(result, "junction_1")
    assert report_j1["route"] is not None
    assert report_j1["route"]["is_reachable"] is True
    assert report_j1["route"]["total_time_seconds"] == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# 6 & 7 & 8. risk_scores=None; risk_scores as a dict; passed through
# ---------------------------------------------------------------------------

def test_risk_scores_none_yields_none_in_reports(basic_cost_matrix_result):
    result = analyze_coverage_from_cost_matrix(basic_cost_matrix_result, risk_scores=None)
    for report in result["junctions"]:
        assert report["risk_score"] is None


def test_risk_scores_dict_passed_through_into_reports(basic_cost_matrix_result):
    risk_scores = {"junction_1": 0.90, "junction_2": 0.75}
    result = analyze_coverage_from_cost_matrix(basic_cost_matrix_result, risk_scores=risk_scores)

    report_j1 = get_junction_report(result, "junction_1")
    report_j2 = get_junction_report(result, "junction_2")
    assert report_j1["risk_score"] == 0.90
    assert report_j2["risk_score"] == 0.75


def test_risk_score_does_not_affect_covered_uncovered_decision(basic_cost_matrix_result):
    """
    Explicitly confirms the audit finding: risk_scores is pass-through/
    sort-key data only, never a multiplier or input to the covered/
    uncovered decision.
    """
    result_no_risk = analyze_coverage_from_cost_matrix(basic_cost_matrix_result, threshold_minutes=2.0)
    result_with_risk = analyze_coverage_from_cost_matrix(
        basic_cost_matrix_result, threshold_minutes=2.0,
        risk_scores={"junction_1": 99999, "junction_2": 99999},
    )
    assert result_no_risk["covered_junction_ids"] == result_with_risk["covered_junction_ids"]
    assert result_no_risk["uncovered_junction_ids"] == result_with_risk["uncovered_junction_ids"]
    assert result_no_risk["num_covered"] == result_with_risk["num_covered"]


# ---------------------------------------------------------------------------
# 9 & 10. Uncovered junctions sorted by existing risk-score behavior;
#          missing risk score behavior
# ---------------------------------------------------------------------------

def test_uncovered_junctions_sorted_by_risk_score_descending(cost_matrix_result_four_junctions):
    risk_scores = {"junction_1": 0.90, "junction_3": 0.60}  # junction_4 intentionally has no score
    result = analyze_coverage_from_cost_matrix(
        cost_matrix_result_four_junctions, threshold_minutes=2.0, risk_scores=risk_scores,
    )

    # junction_2 is the only covered one (min response 1.0 <= 2.0).
    assert result["covered_junction_ids"] == ["junction_2"]
    # Uncovered, sorted: highest risk_score first, missing risk_score last.
    assert result["uncovered_junction_ids"] == ["junction_1", "junction_3", "junction_4"]


def test_missing_risk_score_sorts_after_scored_junctions(cost_matrix_result_four_junctions):
    # Give junction_3 a LOWER risk_score than junction_1 but still a real
    # number; junction_4 has no entry at all and must still sort last
    # regardless of the numeric values used for the others.
    risk_scores = {"junction_1": 0.10, "junction_3": 0.05}
    result = analyze_coverage_from_cost_matrix(
        cost_matrix_result_four_junctions, threshold_minutes=2.0, risk_scores=risk_scores,
    )
    assert result["uncovered_junction_ids"][-1] == "junction_4"
    assert get_junction_report(result, "junction_4")["risk_score"] is None


def test_uncovered_junctions_tie_break_by_junction_id(cost_matrix_result_four_junctions):
    # No risk_scores at all: every uncovered junction has has_score=False,
    # so the tie-break is purely alphabetical junction_id order.
    result = analyze_coverage_from_cost_matrix(cost_matrix_result_four_junctions, threshold_minutes=2.0)
    uncovered = result["uncovered_junction_ids"]
    assert uncovered == sorted(uncovered)


# ---------------------------------------------------------------------------
# 11 & 12. Multiple junctions can select the same officer; no exclusivity
#           is assumed
# ---------------------------------------------------------------------------

def test_multiple_junctions_can_share_the_same_nearest_officer(cost_matrix_result_shared_officer):
    result = analyze_coverage_from_cost_matrix(cost_matrix_result_shared_officer)

    report_j1 = get_junction_report(result, "junction_1")
    report_j2 = get_junction_report(result, "junction_2")

    # Both junctions' fastest responder is officer_2 -- coverage.py does
    # not prevent or flag this, since no exclusive one-to-one assignment
    # algorithm exists yet.
    assert report_j1["nearest_officer_id"] == "officer_2"
    assert report_j2["nearest_officer_id"] == "officer_2"
    assert report_j1["is_covered"] is True
    assert report_j2["is_covered"] is True


# ---------------------------------------------------------------------------
# Convenience accessors: get_junction_report(), get_uncovered_junctions()
# ---------------------------------------------------------------------------

def test_get_junction_report_unknown_id_returns_none(basic_cost_matrix_result):
    result = analyze_coverage_from_cost_matrix(basic_cost_matrix_result)
    assert get_junction_report(result, "junction_unknown") is None


def test_get_uncovered_junctions_matches_priority_order(cost_matrix_result_four_junctions):
    risk_scores = {"junction_1": 0.90, "junction_3": 0.60}
    result = analyze_coverage_from_cost_matrix(
        cost_matrix_result_four_junctions, threshold_minutes=2.0, risk_scores=risk_scores,
    )
    uncovered_reports = get_uncovered_junctions(result)
    assert [r["junction_id"] for r in uncovered_reports] == result["uncovered_junction_ids"]
    assert [r["junction_id"] for r in uncovered_reports] == ["junction_1", "junction_3", "junction_4"]


# ---------------------------------------------------------------------------
# analyze_coverage() -- the graph-based convenience wrapper. Uses the same
# monkeypatched find_nearest_node strategy as test_cost_matrix.py, since
# this wrapper internally calls cost_matrix.build_cost_matrix().
# ---------------------------------------------------------------------------

@pytest.fixture
def small_graph():
    g = nx.MultiDiGraph()
    for node in ("O1", "O2", "J1", "J2"):
        g.add_node(node, x=0.0, y=0.0)
    g.add_edge("O1", "J1", key=0, **{WEIGHT_ATTRIBUTE: 120.0, "length": 1000.0})
    g.add_edge("O2", "J2", key=0, **{WEIGHT_ATTRIBUTE: 60.0, "length": 500.0})
    return g


_WRAPPER_COORD_TO_NODE = {
    (21.00, 79.00): "O1",
    (21.01, 79.01): "O2",
    (21.02, 79.02): "J1",
    (21.03, 79.03): "J2",
}


def _wrapper_stub_find_nearest_node(graph, latitude, longitude):
    return _WRAPPER_COORD_TO_NODE.get((latitude, longitude))


def test_analyze_coverage_wrapper_builds_matrix_and_coverage(monkeypatch, small_graph):
    monkeypatch.setattr(cost_matrix_module, "find_nearest_node", _wrapper_stub_find_nearest_node)

    officers = [
        {"id": "officer_1", "latitude": 21.00, "longitude": 79.00},
        {"id": "officer_2", "latitude": 21.01, "longitude": 79.01},
    ]
    junctions = [
        {"id": "junction_1", "latitude": 21.02, "longitude": 79.02, "risk_score": 0.90},
        {"id": "junction_2", "latitude": 21.03, "longitude": 79.03, "risk_score": 0.75},
    ]

    result = analyze_coverage(small_graph, officers, junctions)

    assert "cost_matrix" in result
    assert result["cost_matrix"]["matrix"].shape == (2, 2)
    report_j1 = get_junction_report(result, "junction_1")
    assert report_j1["risk_score"] == 0.90
    assert report_j1["min_response_time_minutes"] == pytest.approx(2.0)  # 120s / 60