"""
test_officer_allocation.py

Test suite for src/officer_allocation.py (Prompt 10: officer -> high-risk
junction assignment via Hungarian optimization, applied per risk-priority
tier).

No production code is modified by this file.

Determinism strategy
---------------------
Most tests build small, hand-crafted `cost_matrix_result` dicts directly
(the exact shape cost_matrix.build_cost_matrix() returns), so the
Hungarian result can be verified by hand-computed arithmetic rather than
depending on real road-network geometry. This isolates
officer_allocation.py's own logic (tiering, Hungarian solve, unreachable
handling, non-mutation) from cost_matrix.py / routing.py / osmnx, exactly
as test_cost_matrix.py already isolates cost_matrix.py from osmnx via a
monkeypatched find_nearest_node.

Two tests go further and build a REAL cost_matrix_result via
cost_matrix.build_cost_matrix() (using the same small deterministic graph
+ monkeypatched find_nearest_node pattern as test_cost_matrix.py) and/or
real NovaRoute ML risk scores via risk_priority.build_risk_priority_map()
(using the same NOVAROUTE_ML_OUTPUT_DIR environment variable convention
as test_risk_priority.py / test_ml_dynamic_graph.py), to prove real
end-to-end integration, not just synthetic-input behavior.

Run with:
    pytest src/test_officer_allocation.py -v
"""

import copy
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.cost_matrix as cost_matrix_module
from src.cost_matrix import UNREACHABLE_COST, build_cost_matrix
from src.congestion import DYNAMIC_WEIGHT_ATTRIBUTE
from src.config import WEIGHT_ATTRIBUTE
from src.risk_priority import build_risk_priority_map

from src.officer_allocation import (
    ASSIGNMENT_METHOD,
    InvalidCostMatrixError,
    InvalidRiskScoreError,
    assign_officers,
)


# ---------------------------------------------------------------------------
# Synthetic cost_matrix_result builder
# ---------------------------------------------------------------------------
# Builds a dict in exactly the shape cost_matrix.build_cost_matrix()
# returns (the subset officer_allocation.py actually reads: matrix,
# officer_ids, junction_ids, routes, time_unit), without needing a graph
# or any routing at all.

INF = float("inf")


def make_cost_matrix_result(
    officer_ids: Sequence[str],
    junction_ids: Sequence[str],
    matrix: Sequence[Sequence[float]],
    time_unit: str = "minutes",
) -> Dict[str, Any]:
    matrix_arr = np.array(matrix, dtype=float)
    routes: Dict[Any, Any] = {}
    for i, officer_id in enumerate(officer_ids):
        for j, junction_id in enumerate(junction_ids):
            value = matrix_arr[i, j]
            is_reachable = math.isfinite(value)
            routes[(officer_id, junction_id)] = {
                "is_reachable": is_reachable,
                "route_nodes": ["n_o", "n_j"] if is_reachable else None,
                "total_time_seconds": value * 60.0 if is_reachable else None,
                "total_distance_meters": 1234.0 if is_reachable else None,
                "origin_node": f"node_{officer_id}",
                "destination_node": f"node_{junction_id}",
                "error": None if is_reachable else "no route",
            }
    return {
        "officer_ids": list(officer_ids),
        "junction_ids": list(junction_ids),
        "matrix": matrix_arr,
        "time_unit": time_unit,
        "weight_mode": "dynamic",
        "weight_attribute": "dynamic_travel_time",
        "routes": routes,
        "officer_nodes": {oid: f"node_{oid}" for oid in officer_ids},
        "junction_nodes": {jid: f"node_{jid}" for jid in junction_ids},
        "unreachable_pairs": [],
        "unsnapped_officer_ids": [],
        "unsnapped_junction_ids": [],
    }


def _assignment_map(result: Dict[str, Any]) -> Dict[str, str]:
    """junction_id -> officer_id, for easy lookup in assertions."""
    return {a["junction_id"]: a["officer_id"] for a in result["assignments"]}


# ---------------------------------------------------------------------------
# 1. Equal officers and junctions
# ---------------------------------------------------------------------------

def test_equal_officers_and_junctions_all_assigned():
    cm = make_cost_matrix_result(
        ["officer_1", "officer_2"],
        ["junction_1", "junction_2"],
        [[5.0, 10.0], [10.0, 5.0]],
    )
    result = assign_officers(cm, risk_scores={"junction_1": 80.0, "junction_2": 60.0})

    assert len(result["assignments"]) == 2
    assert result["unassigned_officer_ids"] == []
    assert result["unassigned_junction_ids"] == []
    assert result["assignment_method"] == ASSIGNMENT_METHOD

    assigned = _assignment_map(result)
    assert assigned["junction_1"] == "officer_1"
    assert assigned["junction_2"] == "officer_2"


# ---------------------------------------------------------------------------
# 2. More officers than junctions
# ---------------------------------------------------------------------------

def test_more_officers_than_junctions_extra_officer_unassigned():
    cm = make_cost_matrix_result(
        ["officer_1", "officer_2", "officer_3"],
        ["junction_1"],
        [[3.0], [1.0], [7.0]],
    )
    result = assign_officers(cm, risk_scores={"junction_1": 50.0})

    assert len(result["assignments"]) == 1
    assert result["assignments"][0]["officer_id"] == "officer_2"  # cheapest
    assert result["unassigned_junction_ids"] == []
    assert set(result["unassigned_officer_ids"]) == {"officer_1", "officer_3"}


# ---------------------------------------------------------------------------
# 3. Fewer officers than junctions
# ---------------------------------------------------------------------------

def test_fewer_officers_than_junctions_extra_junction_unassigned():
    cm = make_cost_matrix_result(
        ["officer_1"],
        ["junction_1", "junction_2"],
        [[4.0, 4.0]],
    )
    # Equal cost either way; risk priority alone must decide who gets the
    # one available officer.
    result = assign_officers(
        cm, risk_scores={"junction_1": 90.0, "junction_2": 20.0}
    )

    assert len(result["assignments"]) == 1
    assert result["assignments"][0]["junction_id"] == "junction_1"
    assert result["unassigned_junction_ids"] == ["junction_2"]
    assert result["unassigned_officer_ids"] == []


# ---------------------------------------------------------------------------
# 4. Hungarian chooses the globally minimum response-time assignment
#    within a tier (a case where nearest-first/greedy would be wrong)
# ---------------------------------------------------------------------------

def test_hungarian_chooses_global_minimum_within_tier():
    # officer_1 is fastest to junction_1 (1) but also fine for junction_2 (2).
    # officer_2 is much slower to junction_1 (100) and slower to junction_2 (5).
    # A greedy "give each junction its single nearest officer, in order"
    # policy would give officer_1 to junction_1 (cost 1) and be forced to
    # give officer_2 to junction_2 (cost 5) -> total 6, which happens to
    # match Hungarian here, so use a case where greedy actually loses:
    # process junction_2 first (nearest officer_1, cost 2) -> officer_1 taken
    # -> junction_1 forced onto officer_2 (cost 100) -> total 102, clearly
    # worse than the true optimum of 1 + 5 = 6 (officer_1->junction_1,
    # officer_2->junction_2). Hungarian must find the true optimum
    # regardless of processing order, since both junctions share one tier.
    cm = make_cost_matrix_result(
        ["officer_1", "officer_2"],
        ["junction_1", "junction_2"],
        [[1.0, 2.0], [100.0, 5.0]],
    )
    # Same risk_score -> both junctions in ONE tier, solved together.
    result = assign_officers(
        cm, risk_scores={"junction_1": 50.0, "junction_2": 50.0}
    )

    assigned = _assignment_map(result)
    assert assigned == {"junction_1": "officer_1", "junction_2": "officer_2"}
    total_cost = sum(a["response_time_minutes"] for a in result["assignments"])
    assert total_cost == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# 5. High-risk junction receives priority over lower-risk junction
# ---------------------------------------------------------------------------

def test_high_risk_junction_receives_priority_over_lower_risk():
    # Only one officer. It is actually a WORSE (slower) fit for the
    # high-risk junction, proving priority -- not cost -- decides which
    # junction wins the scarce officer.
    cm = make_cost_matrix_result(
        ["officer_1"],
        ["junction_high", "junction_low"],
        [[9.0, 1.0]],
    )
    result = assign_officers(
        cm, risk_scores={"junction_high": 95.0, "junction_low": 5.0}
    )

    assert len(result["assignments"]) == 1
    assert result["assignments"][0]["junction_id"] == "junction_high"
    assert result["assignments"][0]["response_time_minutes"] == pytest.approx(9.0)
    assert result["unassigned_junction_ids"] == ["junction_low"]


# ---------------------------------------------------------------------------
# 6. Missing risk_score is lowest priority
# ---------------------------------------------------------------------------

def test_missing_risk_score_is_lowest_priority():
    cm = make_cost_matrix_result(
        ["officer_1"],
        ["junction_scored", "junction_unscored"],
        [[5.0, 1.0]],  # officer is actually closer to the unscored junction
    )
    # junction_unscored simply absent from risk_scores.
    result = assign_officers(cm, risk_scores={"junction_scored": 10.0})

    assert len(result["assignments"]) == 1
    assert result["assignments"][0]["junction_id"] == "junction_scored"
    assert result["assignments"][0]["risk_score"] == 10.0
    assert result["unassigned_junction_ids"] == ["junction_unscored"]

    # But an unscored junction remains fully eligible once officers remain
    # after every scored junction has had priority.
    cm2 = make_cost_matrix_result(
        ["officer_1", "officer_2"],
        ["junction_scored", "junction_unscored"],
        [[5.0, 1.0], [2.0, 3.0]],
    )
    result2 = assign_officers(cm2, risk_scores={"junction_scored": 10.0})
    assigned2 = _assignment_map(result2)
    assert "junction_unscored" in assigned2
    assert result2["unassigned_junction_ids"] == []


# ---------------------------------------------------------------------------
# 7. Unreachable pairs (infinity) are never assigned
# ---------------------------------------------------------------------------

def test_unreachable_pairs_never_assigned():
    cm = make_cost_matrix_result(
        ["officer_1", "officer_2"],
        ["junction_1", "junction_2"],
        [[INF, 5.0], [3.0, INF]],
    )
    result = assign_officers(cm, risk_scores=None)

    for assignment in result["assignments"]:
        assert math.isfinite(assignment["response_time_minutes"])

    assigned = _assignment_map(result)
    # officer_1 can only reach junction_2; officer_2 can only reach junction_1
    assert assigned == {"junction_1": "officer_2", "junction_2": "officer_1"}


# ---------------------------------------------------------------------------
# 8. Completely unreachable junction becomes unassigned
# ---------------------------------------------------------------------------

def test_completely_unreachable_junction_becomes_unassigned():
    cm = make_cost_matrix_result(
        ["officer_1", "officer_2"],
        ["junction_1", "junction_2"],
        [[4.0, INF], [6.0, INF]],  # junction_2 unreachable by anyone
    )
    result = assign_officers(cm, risk_scores={"junction_1": 50.0, "junction_2": 99.0})

    assert "junction_2" in result["unassigned_junction_ids"]
    assigned = _assignment_map(result)
    assert "junction_2" not in assigned
    assert assigned["junction_1"] == "officer_1"  # cheaper of the two


# ---------------------------------------------------------------------------
# 9. All officers unreachable -> no assignments, no crash
# ---------------------------------------------------------------------------

def test_all_officers_unreachable_no_assignments():
    cm = make_cost_matrix_result(
        ["officer_1", "officer_2"],
        ["junction_1", "junction_2"],
        [[INF, INF], [INF, INF]],
    )
    result = assign_officers(cm, risk_scores={"junction_1": 50.0})

    assert result["assignments"] == []
    assert set(result["unassigned_junction_ids"]) == {"junction_1", "junction_2"}
    assert set(result["unassigned_officer_ids"]) == {"officer_1", "officer_2"}


# ---------------------------------------------------------------------------
# 10. Assignment response time matches the original cost matrix
# ---------------------------------------------------------------------------

def test_response_time_matches_original_cost_matrix_minutes():
    cm = make_cost_matrix_result(
        ["officer_1"], ["junction_1"], [[7.25]], time_unit="minutes",
    )
    result = assign_officers(cm, risk_scores=None)
    assert result["assignments"][0]["response_time_minutes"] == pytest.approx(7.25)


def test_response_time_matches_original_cost_matrix_seconds_converted():
    cm = make_cost_matrix_result(
        ["officer_1"], ["junction_1"], [[120.0]], time_unit="seconds",
    )
    result = assign_officers(cm, risk_scores=None)
    # seconds -> minutes conversion, same convention as coverage.py's _to_minutes
    assert result["assignments"][0]["response_time_minutes"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 11. Returned route comes from cost_matrix_result and is not recomputed
# ---------------------------------------------------------------------------

def test_route_from_cost_matrix_not_recomputed():
    cm = make_cost_matrix_result(["officer_1"], ["junction_1"], [[5.0]])
    sentinel_route = {"is_reachable": True, "total_time_seconds": 300.0, "marker": "ORIGINAL_ROUTE_OBJECT"}
    cm["routes"][("officer_1", "junction_1")] = sentinel_route

    result = assign_officers(cm, risk_scores=None)

    assert result["assignments"][0]["route"] is sentinel_route


def test_route_is_none_when_missing_from_routes():
    cm = make_cost_matrix_result(["officer_1"], ["junction_1"], [[5.0]])
    del cm["routes"][("officer_1", "junction_1")]

    result = assign_officers(cm, risk_scores=None)
    assert result["assignments"][0]["route"] is None


# ---------------------------------------------------------------------------
# 12. Original cost matrix is not modified
# ---------------------------------------------------------------------------

def test_original_cost_matrix_not_modified():
    cm = make_cost_matrix_result(
        ["officer_1", "officer_2"],
        ["junction_1", "junction_2"],
        [[5.0, INF], [INF, 5.0]],
    )
    matrix_before = cm["matrix"].copy()

    assign_officers(cm, risk_scores={"junction_1": 10.0})

    assert np.array_equal(cm["matrix"], matrix_before, equal_nan=True)
    # inf entries specifically must remain inf (not overwritten by the
    # internal finite sentinel substitution).
    assert cm["matrix"][0, 1] == INF
    assert cm["matrix"][1, 0] == INF


# ---------------------------------------------------------------------------
# 13. Input risk_scores is not modified
# ---------------------------------------------------------------------------

def test_input_risk_scores_not_modified():
    cm = make_cost_matrix_result(["officer_1"], ["junction_1", "junction_2"], [[1.0, 2.0]])
    risk_scores = {"junction_1": 80.0, "junction_2": 20.0}
    risk_scores_before = copy.deepcopy(risk_scores)

    assign_officers(cm, risk_scores=risk_scores)

    assert risk_scores == risk_scores_before


# ---------------------------------------------------------------------------
# 14 & 15. Duplicate officer / junction ids are handled appropriately
# ---------------------------------------------------------------------------

def test_duplicate_officer_ids_raise():
    cm = make_cost_matrix_result(["officer_1", "officer_1"], ["junction_1"], [[1.0], [2.0]])
    with pytest.raises(InvalidCostMatrixError):
        assign_officers(cm, risk_scores=None)


def test_duplicate_junction_ids_raise():
    cm = make_cost_matrix_result(["officer_1"], ["junction_1", "junction_1"], [[1.0, 2.0]])
    with pytest.raises(InvalidCostMatrixError):
        assign_officers(cm, risk_scores=None)


# ---------------------------------------------------------------------------
# 16. Invalid / malformed cost matrix raises a clear error
# ---------------------------------------------------------------------------

def test_non_dict_cost_matrix_raises():
    with pytest.raises(InvalidCostMatrixError):
        assign_officers(["not", "a", "dict"], risk_scores=None)


def test_missing_required_key_raises():
    cm = make_cost_matrix_result(["officer_1"], ["junction_1"], [[1.0]])
    del cm["routes"]
    with pytest.raises(InvalidCostMatrixError):
        assign_officers(cm, risk_scores=None)


def test_shape_mismatch_raises():
    cm = make_cost_matrix_result(["officer_1", "officer_2"], ["junction_1"], [[1.0], [2.0]])
    cm["matrix"] = np.array([[1.0, 2.0, 3.0]])  # wrong shape entirely
    with pytest.raises(InvalidCostMatrixError):
        assign_officers(cm, risk_scores=None)


def test_invalid_time_unit_raises():
    cm = make_cost_matrix_result(["officer_1"], ["junction_1"], [[1.0]])
    cm["time_unit"] = "hours"
    with pytest.raises(InvalidCostMatrixError):
        assign_officers(cm, risk_scores=None)


# ---------------------------------------------------------------------------
# 17. Invalid risk score values handled consistently with the project's
#     validation philosophy (finite, numeric, within [0, 100]; None is
#     valid and simply means "no score")
# ---------------------------------------------------------------------------

def test_risk_score_out_of_range_raises():
    cm = make_cost_matrix_result(["officer_1"], ["junction_1"], [[1.0]])
    with pytest.raises(InvalidRiskScoreError):
        assign_officers(cm, risk_scores={"junction_1": 150.0})


def test_risk_score_negative_raises():
    cm = make_cost_matrix_result(["officer_1"], ["junction_1"], [[1.0]])
    with pytest.raises(InvalidRiskScoreError):
        assign_officers(cm, risk_scores={"junction_1": -1.0})


def test_risk_score_non_numeric_raises():
    cm = make_cost_matrix_result(["officer_1"], ["junction_1"], [[1.0]])
    with pytest.raises(InvalidRiskScoreError):
        assign_officers(cm, risk_scores={"junction_1": "high"})


def test_risk_score_bool_raises():
    cm = make_cost_matrix_result(["officer_1"], ["junction_1"], [[1.0]])
    with pytest.raises(InvalidRiskScoreError):
        assign_officers(cm, risk_scores={"junction_1": True})


def test_risk_score_nan_raises():
    cm = make_cost_matrix_result(["officer_1"], ["junction_1"], [[1.0]])
    with pytest.raises(InvalidRiskScoreError):
        assign_officers(cm, risk_scores={"junction_1": float("nan")})


def test_risk_score_none_is_valid_and_means_no_score():
    cm = make_cost_matrix_result(["officer_1"], ["junction_1"], [[1.0]])
    result = assign_officers(cm, risk_scores={"junction_1": None})
    assert result["assignments"][0]["risk_score"] is None


def test_risk_scores_not_a_mapping_raises():
    cm = make_cost_matrix_result(["officer_1"], ["junction_1"], [[1.0]])
    with pytest.raises(InvalidRiskScoreError):
        assign_officers(cm, risk_scores=["junction_1", 80.0])


# ---------------------------------------------------------------------------
# 18. Real NovaRoute ML data: junction_1 = 26.6667, junction_2 = 25.6667,
#     junction_3 has no ML score.
# ---------------------------------------------------------------------------

_DEFAULT_ML_OBS_DIR = Path(__file__).resolve().parent.parent / "data" / "ml_observations"
ML_OBS_DIR = (
    Path(os.environ["NOVAROUTE_ML_OUTPUT_DIR"])
    if os.environ.get("NOVAROUTE_ML_OUTPUT_DIR")
    else _DEFAULT_ML_OBS_DIR
)
JUNCTION_1_FILE = ML_OBS_DIR / "junction_1_latest.json"
JUNCTION_2_FILE = ML_OBS_DIR / "junction_2_latest.json"


def _require_real_ml_file(path: Path, junction_id: str) -> None:
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


def test_real_ml_risk_scores_priority_and_missing_junction_3():
    _require_real_ml_file(JUNCTION_1_FILE, "junction_1")
    _require_real_ml_file(JUNCTION_2_FILE, "junction_2")

    risk_scores = build_risk_priority_map(
        {"junction_1": JUNCTION_1_FILE, "junction_2": JUNCTION_2_FILE}
    )
    assert risk_scores["junction_1"] == pytest.approx(26.6667)
    assert risk_scores["junction_2"] == pytest.approx(25.6667)
    assert "junction_3" not in risk_scores

    # One scarce officer, equally fast to all three junctions, so ONLY
    # risk priority decides who wins it: junction_1 (highest real score).
    cm = make_cost_matrix_result(
        ["officer_1"],
        ["junction_1", "junction_2", "junction_3"],
        [[5.0, 5.0, 5.0]],
    )
    result = assign_officers(cm, risk_scores=risk_scores)

    assert len(result["assignments"]) == 1
    assert result["assignments"][0]["junction_id"] == "junction_1"
    assert result["assignments"][0]["risk_score"] == pytest.approx(26.6667)
    # junction_3 (no ML score) must never receive a fabricated score.
    assert set(result["unassigned_junction_ids"]) == {"junction_2", "junction_3"}


def test_real_ml_risk_scores_junction_3_still_eligible_with_enough_officers():
    _require_real_ml_file(JUNCTION_1_FILE, "junction_1")
    _require_real_ml_file(JUNCTION_2_FILE, "junction_2")

    risk_scores = build_risk_priority_map(
        {"junction_1": JUNCTION_1_FILE, "junction_2": JUNCTION_2_FILE}
    )
    cm = make_cost_matrix_result(
        ["officer_1", "officer_2", "officer_3"],
        ["junction_1", "junction_2", "junction_3"],
        [[5.0, 6.0, 7.0], [6.0, 5.0, 7.0], [7.0, 7.0, 5.0]],
    )
    result = assign_officers(cm, risk_scores=risk_scores)

    assigned = _assignment_map(result)
    assert set(assigned.keys()) == {"junction_1", "junction_2", "junction_3"}
    assert result["unassigned_junction_ids"] == []
    assert result["assignments"][-1]["junction_id"] == "junction_3"
    assert result["assignments"][-1]["risk_score"] is None


# ---------------------------------------------------------------------------
# 19. Real cost matrix integration (via cost_matrix.build_cost_matrix(),
#     same small deterministic graph + monkeypatch pattern as
#     test_cost_matrix.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_graph():
    g = nx.MultiDiGraph()
    for node in ("O1", "O2", "J1", "J2"):
        g.add_node(node, x=0.0, y=0.0)

    g.add_edge(
        "O1", "J1", key=0,
        **{WEIGHT_ATTRIBUTE: 120.0, DYNAMIC_WEIGHT_ATTRIBUTE: 120.0, "length": 1000.0},
    )
    g.add_edge(
        "O1", "J2", key=0,
        **{WEIGHT_ATTRIBUTE: 600.0, DYNAMIC_WEIGHT_ATTRIBUTE: 600.0, "length": 5000.0},
    )
    g.add_edge(
        "O2", "J1", key=0,
        **{WEIGHT_ATTRIBUTE: 300.0, DYNAMIC_WEIGHT_ATTRIBUTE: 300.0, "length": 2500.0},
    )
    g.add_edge(
        "O2", "J2", key=0,
        **{WEIGHT_ATTRIBUTE: 60.0, DYNAMIC_WEIGHT_ATTRIBUTE: 60.0, "length": 500.0},
    )
    return g


_COORD_TO_NODE = {
    (21.00, 79.00): "O1",
    (21.01, 79.01): "O2",
    (21.02, 79.02): "J1",
    (21.03, 79.03): "J2",
}


def _stub_find_nearest_node(graph, latitude, longitude):
    return _COORD_TO_NODE.get((latitude, longitude))


@pytest.fixture(autouse=True)
def patch_find_nearest_node(monkeypatch):
    monkeypatch.setattr(cost_matrix_module, "find_nearest_node", _stub_find_nearest_node)


def test_real_cost_matrix_integration(small_graph):
    officers = [
        {"id": "officer_1", "latitude": 21.00, "longitude": 79.00},
        {"id": "officer_2", "latitude": 21.01, "longitude": 79.01},
    ]
    junctions = [
        {"id": "junction_1", "latitude": 21.02, "longitude": 79.02},
        {"id": "junction_2", "latitude": 21.03, "longitude": 79.03},
    ]
    cost_matrix_result = build_cost_matrix(
        small_graph, officers, junctions, time_unit="minutes",
    )

    result = assign_officers(
        cost_matrix_result, risk_scores={"junction_1": 70.0, "junction_2": 30.0}
    )

    assigned = _assignment_map(result)
    # O1->J1 = 120s = 2.0 min; O2->J2 = 60s = 1.0 min; this is the global
    # optimum (2 + 1 = 3) versus the alternative pairing (5 + 10 = 15).
    assert assigned == {"junction_1": "officer_1", "junction_2": "officer_2"}
    assert result["unassigned_officer_ids"] == []
    assert result["unassigned_junction_ids"] == []

    for assignment in result["assignments"]:
        route = assignment["route"]
        assert route is not None
        assert route["is_reachable"] is True
        # route must come straight from cost_matrix_result["routes"], i.e.
        # be the exact same dict object, not a recomputation.
        key = (assignment["officer_id"], assignment["junction_id"])
        assert route is cost_matrix_result["routes"][key]


# ---------------------------------------------------------------------------
# 20 & 21. One officer never assigned to two junctions; one junction
#          never assigned two officers.
# ---------------------------------------------------------------------------

def test_no_officer_assigned_twice_no_junction_assigned_twice():
    cm = make_cost_matrix_result(
        ["officer_1", "officer_2", "officer_3"],
        ["junction_1", "junction_2", "junction_3"],
        [
            [1.0, 9.0, 9.0],
            [9.0, 1.0, 9.0],
            [9.0, 9.0, 1.0],
        ],
    )
    result = assign_officers(
        cm,
        risk_scores={"junction_1": 90.0, "junction_2": 50.0, "junction_3": 10.0},
    )

    officer_ids_used = [a["officer_id"] for a in result["assignments"]]
    junction_ids_used = [a["junction_id"] for a in result["assignments"]]

    assert len(officer_ids_used) == len(set(officer_ids_used))
    assert len(junction_ids_used) == len(set(junction_ids_used))
    assert len(result["assignments"]) == 3


def test_no_double_assignment_across_tiers_with_scarce_officers():
    # Two officers, three junctions across three separate risk tiers:
    # once both officers are used up in the first two tiers, the third
    # tier's junction must be unassigned, and no officer can appear twice.
    cm = make_cost_matrix_result(
        ["officer_1", "officer_2"],
        ["junction_1", "junction_2", "junction_3"],
        [[1.0, 5.0, 2.0], [5.0, 1.0, 2.0]],
    )
    result = assign_officers(
        cm,
        risk_scores={"junction_1": 90.0, "junction_2": 60.0, "junction_3": 30.0},
    )

    officer_ids_used = [a["officer_id"] for a in result["assignments"]]
    assert len(officer_ids_used) == len(set(officer_ids_used))
    assert len(result["assignments"]) == 2
    assert result["unassigned_junction_ids"] == ["junction_3"]