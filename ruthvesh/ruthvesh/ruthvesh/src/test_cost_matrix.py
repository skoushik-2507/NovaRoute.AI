"""
test_cost_matrix.py

Regression test suite for cost_matrix.py (Prompt 7A).

This is a REGRESSION baseline for CURRENT behavior, written by inspecting
cost_matrix.py's actual implementation -- it does not invent or assume any
behavior that isn't already there.

No production code is modified by this file.

Determinism strategy
---------------------
cost_matrix.build_cost_matrix() snaps every officer/junction (lat, lon) to
a graph node via graph_utils.find_nearest_node(), which internally calls
osmnx.distance.nearest_nodes() -- a nearest-neighbour search that is
unnecessary complexity for a small hand-built test graph and would make
these tests depend on floating-point nearest-neighbour geometry rather
than on cost_matrix.py's own logic.

Instead, exactly like test_dynamic_routing.py already does for
ml_dynamic_graph.py's file-loading step, these tests monkeypatch
`find_nearest_node` (as imported into src.cost_matrix) with a small,
deterministic coordinate -> node-id lookup table. This tests
cost_matrix.py's own behavior (validation, matrix assembly, unreachable
handling, accessors) without depending on osmnx's nearest-neighbour
implementation, which belongs to graph_utils.py, not this module.

Run with:
    pytest src/test_cost_matrix.py -v
"""

import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

import src.cost_matrix as cost_matrix_module
from src.cost_matrix import (
    UNREACHABLE_COST,
    VALID_TIME_UNITS,
    build_cost_matrix,
    get_response_time,
    get_route,
    to_dataframe,
)
from src.congestion import DYNAMIC_WEIGHT_ATTRIBUTE
from src.config import WEIGHT_ATTRIBUTE
from src.routing import WEIGHT_MODE_BASE, WEIGHT_MODE_DYNAMIC


# ---------------------------------------------------------------------------
# Small deterministic graph
# ---------------------------------------------------------------------------
#
#   O1 --2min/3min(dyn)--> J1
#   O1 --10min/10min(dyn)-> J2
#   O2 --5min/5min(dyn)---> J1
#   O2 --1min/1.5min(dyn)-> J2
#
#   J3 has no incoming edges from O1/O2 -> genuinely unreachable.
#
# Travel times are stored in SECONDS on the edges (matching the real
# graph_builder.py convention that cost_matrix.py itself documents:
# "routing.py reports time in seconds").

@pytest.fixture(scope="module")
def small_graph():
    g = nx.MultiDiGraph()
    for node in ("O1", "O2", "J1", "J2", "J3"):
        g.add_node(node, x=0.0, y=0.0)

    g.add_edge(
        "O1", "J1", key=0,
        **{WEIGHT_ATTRIBUTE: 120.0, DYNAMIC_WEIGHT_ATTRIBUTE: 180.0, "length": 1000.0},
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
        **{WEIGHT_ATTRIBUTE: 60.0, DYNAMIC_WEIGHT_ATTRIBUTE: 90.0, "length": 500.0},
    )
    # J3 is intentionally left with no incoming edges: genuinely unreachable.
    return g


# Deterministic (latitude, longitude) -> node id lookup, standing in for
# graph_utils.find_nearest_node's real nearest-neighbour search.
COORD_TO_NODE = {
    (21.00, 79.00): "O1",
    (21.01, 79.01): "O2",
    (21.02, 79.02): "J1",
    (21.03, 79.03): "J2",
    (21.04, 79.04): "J3",
}

# A coordinate deliberately absent from COORD_TO_NODE, to simulate a
# location that could not be snapped to any graph node.
GHOST_LATITUDE, GHOST_LONGITUDE = 99.99, 99.99


def _stub_find_nearest_node(graph, latitude, longitude):
    return COORD_TO_NODE.get((latitude, longitude))


@pytest.fixture(autouse=True)
def patch_find_nearest_node(monkeypatch):
    """
    Applied to every test in this module: replaces cost_matrix.py's
    imported find_nearest_node with the deterministic stub above, so
    every test in this file exercises cost_matrix.py's own logic, not
    osmnx's nearest-neighbour search.
    """
    monkeypatch.setattr(cost_matrix_module, "find_nearest_node", _stub_find_nearest_node)


@pytest.fixture
def officers():
    return [
        {"id": "officer_1", "latitude": 21.00, "longitude": 79.00},
        {"id": "officer_2", "latitude": 21.01, "longitude": 79.01},
    ]


@pytest.fixture
def junctions_reachable():
    return [
        {"id": "junction_1", "latitude": 21.02, "longitude": 79.02, "risk_score": 0.90},
        {"id": "junction_2", "latitude": 21.03, "longitude": 79.03, "risk_score": 0.75},
    ]


@pytest.fixture
def junctions_with_unreachable(junctions_reachable):
    return junctions_reachable + [
        {"id": "junction_3", "latitude": 21.04, "longitude": 79.04, "risk_score": 0.60},
    ]


# ---------------------------------------------------------------------------
# 1 & 2. build_cost_matrix() with a small deterministic graph; correct
#        officer x junction matrix dimensions
# ---------------------------------------------------------------------------

def test_build_cost_matrix_basic_shape(small_graph, officers, junctions_reachable):
    result = build_cost_matrix(small_graph, officers, junctions_reachable)

    assert isinstance(result, dict)
    assert result["matrix"].shape == (len(officers), len(junctions_reachable))
    assert result["matrix"].shape == (2, 2)


def test_matrix_dimensions_scale_with_input_size(small_graph, officers, junctions_with_unreachable):
    result = build_cost_matrix(small_graph, officers, junctions_with_unreachable)
    assert result["matrix"].shape == (2, 3)


# ---------------------------------------------------------------------------
# 6. Correct officer/junction IDs are preserved (order + identity)
# ---------------------------------------------------------------------------

def test_officer_and_junction_ids_preserved_in_order(small_graph, officers, junctions_reachable):
    result = build_cost_matrix(small_graph, officers, junctions_reachable)
    assert result["officer_ids"] == ["officer_1", "officer_2"]
    assert result["junction_ids"] == ["junction_1", "junction_2"]


# ---------------------------------------------------------------------------
# 3. Base routing uses travel_time
# ---------------------------------------------------------------------------

def test_base_mode_uses_travel_time(small_graph, officers, junctions_reachable):
    result = build_cost_matrix(
        small_graph, officers, junctions_reachable, weight_mode=WEIGHT_MODE_BASE,
    )

    assert result["weight_mode"] == "base"
    assert result["weight_attribute"] == WEIGHT_ATTRIBUTE

    # travel_time seconds -> minutes (default time_unit)
    assert get_response_time(result, "officer_1", "junction_1") == pytest.approx(120.0 / 60.0)
    assert get_response_time(result, "officer_1", "junction_2") == pytest.approx(600.0 / 60.0)
    assert get_response_time(result, "officer_2", "junction_1") == pytest.approx(300.0 / 60.0)
    assert get_response_time(result, "officer_2", "junction_2") == pytest.approx(60.0 / 60.0)


# ---------------------------------------------------------------------------
# 4. Dynamic routing uses dynamic_travel_time when weight_mode="dynamic"
# ---------------------------------------------------------------------------

def test_dynamic_mode_uses_dynamic_travel_time(small_graph, officers, junctions_reachable):
    result = build_cost_matrix(
        small_graph, officers, junctions_reachable, weight_mode=WEIGHT_MODE_DYNAMIC,
    )

    assert result["weight_mode"] == "dynamic"
    assert result["weight_attribute"] == DYNAMIC_WEIGHT_ATTRIBUTE

    # dynamic_travel_time seconds -> minutes; differs from base for the
    # two edges that have congestion applied (O1->J1, O2->J2).
    assert get_response_time(result, "officer_1", "junction_1") == pytest.approx(180.0 / 60.0)
    assert get_response_time(result, "officer_2", "junction_2") == pytest.approx(90.0 / 60.0)
    # Unaffected edges: dynamic == base for these two.
    assert get_response_time(result, "officer_1", "junction_2") == pytest.approx(600.0 / 60.0)
    assert get_response_time(result, "officer_2", "junction_1") == pytest.approx(300.0 / 60.0)


def test_base_and_dynamic_differ_only_on_congested_edges(small_graph, officers, junctions_reachable):
    base_result = build_cost_matrix(small_graph, officers, junctions_reachable, weight_mode=WEIGHT_MODE_BASE)
    dyn_result = build_cost_matrix(small_graph, officers, junctions_reachable, weight_mode=WEIGHT_MODE_DYNAMIC)

    assert base_result["matrix"][0, 0] != dyn_result["matrix"][0, 0]  # officer_1 -> junction_1: congested
    assert base_result["matrix"][0, 1] == dyn_result["matrix"][0, 1]  # officer_1 -> junction_2: unaffected
    assert base_result["matrix"][1, 0] == dyn_result["matrix"][1, 0]  # officer_2 -> junction_1: unaffected
    assert base_result["matrix"][1, 1] != dyn_result["matrix"][1, 1]  # officer_2 -> junction_2: congested


# ---------------------------------------------------------------------------
# Time unit handling (seconds vs minutes) -- part of build_cost_matrix()'s
# existing public contract, exercised alongside base/dynamic mode.
# ---------------------------------------------------------------------------

def test_time_unit_seconds_not_divided(small_graph, officers, junctions_reachable):
    result = build_cost_matrix(small_graph, officers, junctions_reachable, time_unit="seconds")
    assert result["time_unit"] == "seconds"
    assert get_response_time(result, "officer_1", "junction_1") == pytest.approx(120.0)


def test_invalid_time_unit_raises(small_graph, officers, junctions_reachable):
    with pytest.raises(ValueError):
        build_cost_matrix(small_graph, officers, junctions_reachable, time_unit="hours")


# ---------------------------------------------------------------------------
# 5. Unreachable officer/junction pairs receive the existing
#    UNREACHABLE_COST behavior
# ---------------------------------------------------------------------------

def test_unreachable_pair_marked_with_unreachable_cost(small_graph, officers, junctions_with_unreachable):
    result = build_cost_matrix(small_graph, officers, junctions_with_unreachable)

    j3_index = result["junction_ids"].index("junction_3")
    for i in range(len(result["officer_ids"])):
        assert result["matrix"][i, j3_index] == UNREACHABLE_COST

    assert ("officer_1", "junction_3") in result["unreachable_pairs"]
    assert ("officer_2", "junction_3") in result["unreachable_pairs"]

    # Reachable pairs must NOT be in unreachable_pairs.
    assert ("officer_1", "junction_1") not in result["unreachable_pairs"]


def test_get_response_time_returns_none_for_unreachable_pair(small_graph, officers, junctions_with_unreachable):
    result = build_cost_matrix(small_graph, officers, junctions_with_unreachable)
    assert get_response_time(result, "officer_1", "junction_3") is None
    assert get_response_time(result, "officer_2", "junction_3") is None


def test_unreachable_route_has_is_reachable_false(small_graph, officers, junctions_with_unreachable):
    result = build_cost_matrix(small_graph, officers, junctions_with_unreachable)
    route = get_route(result, "officer_1", "junction_3")
    assert route is not None
    assert route["is_reachable"] is False


# ---------------------------------------------------------------------------
# Unsnapped locations (find_nearest_node returns None) -- part of
# build_cost_matrix()'s existing documented behavior (unsnapped_officer_ids
# / unsnapped_junction_ids, excluded via UNREACHABLE_COST).
# ---------------------------------------------------------------------------

def test_unsnapped_officer_excluded_from_matrix(small_graph, junctions_reachable):
    officers_with_ghost = [
        {"id": "officer_ghost", "latitude": GHOST_LATITUDE, "longitude": GHOST_LONGITUDE},
        {"id": "officer_2", "latitude": 21.01, "longitude": 79.01},
    ]
    result = build_cost_matrix(small_graph, officers_with_ghost, junctions_reachable)

    assert "officer_ghost" in result["unsnapped_officer_ids"]
    assert result["officer_nodes"]["officer_ghost"] is None

    ghost_index = result["officer_ids"].index("officer_ghost")
    assert all(result["matrix"][ghost_index, :] == UNREACHABLE_COST)
    assert ("officer_ghost", "junction_1") in result["unreachable_pairs"]

    route = get_route(result, "officer_ghost", "junction_1")
    assert route["is_reachable"] is False
    assert route["origin_node"] is None
    assert route["error"] is not None


# ---------------------------------------------------------------------------
# Input validation -- existing public API behavior (11)
# ---------------------------------------------------------------------------

def test_empty_officers_raises(small_graph, junctions_reachable):
    with pytest.raises(ValueError):
        build_cost_matrix(small_graph, [], junctions_reachable)


def test_empty_junctions_raises(small_graph, officers):
    with pytest.raises(ValueError):
        build_cost_matrix(small_graph, officers, [])


def test_duplicate_officer_id_raises(small_graph, junctions_reachable):
    dup_officers = [
        {"id": "officer_1", "latitude": 21.00, "longitude": 79.00},
        {"id": "officer_1", "latitude": 21.01, "longitude": 79.01},
    ]
    with pytest.raises(ValueError):
        build_cost_matrix(small_graph, dup_officers, junctions_reachable)


def test_missing_required_key_raises(small_graph, junctions_reachable):
    bad_officers = [{"id": "officer_1", "latitude": 21.00}]  # missing longitude
    with pytest.raises(ValueError):
        build_cost_matrix(small_graph, bad_officers, junctions_reachable)


def test_non_numeric_coordinate_raises(small_graph, junctions_reachable):
    bad_officers = [{"id": "officer_1", "latitude": "not-a-number", "longitude": 79.00}]
    with pytest.raises(ValueError):
        build_cost_matrix(small_graph, bad_officers, junctions_reachable)


# ---------------------------------------------------------------------------
# 7. get_response_time() returns the correct matrix value
# ---------------------------------------------------------------------------

def test_get_response_time_correct_value(small_graph, officers, junctions_reachable):
    result = build_cost_matrix(small_graph, officers, junctions_reachable)
    assert get_response_time(result, "officer_2", "junction_2") == pytest.approx(1.0)


def test_get_response_time_unknown_ids_returns_none(small_graph, officers, junctions_reachable):
    result = build_cost_matrix(small_graph, officers, junctions_reachable)
    assert get_response_time(result, "officer_unknown", "junction_1") is None
    assert get_response_time(result, "officer_1", "junction_unknown") is None


# ---------------------------------------------------------------------------
# 8. get_route() returns the expected route
# ---------------------------------------------------------------------------

def test_get_route_returns_expected_route(small_graph, officers, junctions_reachable):
    result = build_cost_matrix(small_graph, officers, junctions_reachable)
    route = get_route(result, "officer_1", "junction_1")

    assert route["is_reachable"] is True
    assert route["route_nodes"] == ["O1", "J1"]
    assert route["total_time_seconds"] == pytest.approx(120.0)
    assert route["total_distance_meters"] == pytest.approx(1000.0)
    assert route["origin_node"] == "O1"
    assert route["destination_node"] == "J1"


def test_get_route_unknown_pair_returns_none(small_graph, officers, junctions_reachable):
    result = build_cost_matrix(small_graph, officers, junctions_reachable)
    assert get_route(result, "officer_unknown", "junction_unknown") is None


# ---------------------------------------------------------------------------
# 9. to_dataframe() returns the expected structure
# ---------------------------------------------------------------------------

def test_to_dataframe_structure(small_graph, officers, junctions_reachable):
    result = build_cost_matrix(small_graph, officers, junctions_reachable)
    df = to_dataframe(result)

    assert list(df.index) == ["officer_1", "officer_2"]
    assert list(df.columns) == ["junction_1", "junction_2"]
    assert df.loc["officer_2", "junction_2"] == pytest.approx(1.0)


def test_to_dataframe_unreachable_pairs_are_nan(small_graph, officers, junctions_with_unreachable):
    result = build_cost_matrix(small_graph, officers, junctions_with_unreachable)
    df = to_dataframe(result)
    assert math.isnan(df.loc["officer_1", "junction_3"])
    assert math.isnan(df.loc["officer_2", "junction_3"])


# ---------------------------------------------------------------------------
# 10. Multiple officers and multiple junctions work correctly
# ---------------------------------------------------------------------------

def test_multiple_officers_and_junctions(small_graph):
    officers_3 = [
        {"id": "officer_1", "latitude": 21.00, "longitude": 79.00},
        {"id": "officer_2", "latitude": 21.01, "longitude": 79.01},
    ]
    junctions_3 = [
        {"id": "junction_1", "latitude": 21.02, "longitude": 79.02},
        {"id": "junction_2", "latitude": 21.03, "longitude": 79.03},
        {"id": "junction_3", "latitude": 21.04, "longitude": 79.04},
    ]
    result = build_cost_matrix(small_graph, officers_3, junctions_3)

    assert result["matrix"].shape == (2, 3)
    assert result["officer_ids"] == ["officer_1", "officer_2"]
    assert result["junction_ids"] == ["junction_1", "junction_2", "junction_3"]
    # Spot-check a couple of specific cells alongside the shape check.
    assert get_response_time(result, "officer_1", "junction_2") == pytest.approx(10.0)
    assert get_response_time(result, "officer_2", "junction_3") is None


# ---------------------------------------------------------------------------
# 11. Existing public API behavior is preserved: weight_attribute override
# and resolved weight_mode="custom" (documented in build_cost_matrix()).
# ---------------------------------------------------------------------------

def test_explicit_weight_attribute_override_resolves_to_custom(small_graph, officers, junctions_reachable):
    result = build_cost_matrix(
        small_graph, officers, junctions_reachable, weight_attribute=WEIGHT_ATTRIBUTE,
    )
    assert result["weight_mode"] == "custom"
    assert result["weight_attribute"] == WEIGHT_ATTRIBUTE