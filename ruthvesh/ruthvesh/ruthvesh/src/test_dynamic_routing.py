"""
test_dynamic_routing.py

Pytest test suite proving the Dijkstra <-> dynamic-graph connection
(dynamic_routing.py) works correctly, without modifying routing.py or
ml_dynamic_graph.py.

Two kinds of fixtures are used, per Prompt 6 instructions:

1. A small, deterministic, hand-built MultiDiGraph (`small_graph`) for
   routing-mode-switching behavior (tests B-H) — fast, exact, and not
   dependent on the large real Nagpur graph's specific topology. This
   graph carries 'travel_time' and 'dynamic_travel_time' directly (as
   ml_dynamic_graph.py would produce them), so these tests isolate
   routing.py's weight_mode behavior specifically.

2. The real processed Nagpur graph + the real NovaRoute ML observation
   files (via the same NOVAROUTE_ML_OUTPUT_DIR environment variable
   convention established for test_ml_dynamic_graph.py) for one true
   end-to-end integration test (test I / the "real integration test").

Run with:
    pytest src/test_dynamic_routing.py -v
"""

import os
import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_GRAPH_PATH, WEIGHT_ATTRIBUTE
from src.congestion import DYNAMIC_WEIGHT_ATTRIBUTE
from src.graph_utils import load_graph
from src import dynamic_routing as dr
from src import ml_dynamic_graph as mdg
from src import routing


# ---------------------------------------------------------------------------
# Deterministic small graph fixture
# ---------------------------------------------------------------------------
#
#        (fast when uncongested, becomes slow in dynamic mode)
#   A -------------------- D
#   |                       |
#   +------ B ------ C -----+
#        (always modest travel_time; unaffected by congestion)
#
# Two parallel routes from A to D:
#   - Direct: A -> D            (short base travel_time, e.g. 10s)
#   - Detour: A -> B -> C -> D  (longer base travel_time, e.g. 24s = 3x8)
#
# In base mode, the direct A->D edge is faster (10s < 24s), so Dijkstra
# picks the direct route.
#
# A dynamic_travel_time is set ONLY on the direct A->D edge; the detour
# edges get NO dynamic_travel_time at all. NOTE: this no longer matches
# real ml_dynamic_graph.py output as of Prompt 6A (which now sets a
# baseline dynamic_travel_time == travel_time on EVERY edge). This
# fixture intentionally models the OLD/incomplete-coverage case, to keep
# exercising routing.py's raw, unmodified fallback behavior in isolation
# (NetworkX defaulting a missing weight to 1) as a documented edge case.
# For the corrected, Prompt-6A-accurate scenario (every edge has a real
# dynamic_travel_time), see `small_graph_full_coverage` below and
# test_dijkstra_switches_to_alternative_route_with_full_coverage().

@pytest.fixture
def small_graph():
    g = nx.MultiDiGraph()
    g.add_node("A", x=0.0, y=0.0)
    g.add_node("B", x=0.0, y=0.1)
    g.add_node("C", x=0.1, y=0.1)
    g.add_node("D", x=0.1, y=0.0)

    # Direct route: congested in dynamic mode.
    g.add_edge("A", "D", key=0, **{WEIGHT_ATTRIBUTE: 10.0, DYNAMIC_WEIGHT_ATTRIBUTE: 50.0, "length": 100.0})

    # Detour route: deliberately has NO dynamic_travel_time (old/
    # incomplete-coverage scenario — see fixture docstring above).
    g.add_edge("A", "B", key=0, **{WEIGHT_ATTRIBUTE: 8.0, "length": 80.0})
    g.add_edge("B", "C", key=0, **{WEIGHT_ATTRIBUTE: 8.0, "length": 80.0})
    g.add_edge("C", "D", key=0, **{WEIGHT_ATTRIBUTE: 8.0, "length": 80.0})

    return g


@pytest.fixture
def small_graph_full_coverage():
    """Same 4-node topology as `small_graph`, but with dynamic_travel_time
    set on EVERY edge — modeling the corrected ml_dynamic_graph.py output
    as of Prompt 6A: baseline dynamic_travel_time == travel_time on
    uncovered edges (the detour), congestion-adjusted on the ML-covered
    edge (the direct route)."""
    g = nx.MultiDiGraph()
    g.add_node("A", x=0.0, y=0.0)
    g.add_node("B", x=0.0, y=0.1)
    g.add_node("C", x=0.1, y=0.1)
    g.add_node("D", x=0.1, y=0.0)

    # Direct route: ML-covered, congestion factor 5.0 applied (10 -> 50).
    g.add_edge("A", "D", key=0, **{WEIGHT_ATTRIBUTE: 10.0, DYNAMIC_WEIGHT_ATTRIBUTE: 50.0, "length": 100.0})

    # Detour route: NOT ML-covered, so dynamic_travel_time == travel_time
    # (baseline only, per the Prompt 6A fix) rather than absent.
    g.add_edge("A", "B", key=0, **{WEIGHT_ATTRIBUTE: 8.0, DYNAMIC_WEIGHT_ATTRIBUTE: 8.0, "length": 80.0})
    g.add_edge("B", "C", key=0, **{WEIGHT_ATTRIBUTE: 8.0, DYNAMIC_WEIGHT_ATTRIBUTE: 8.0, "length": 80.0})
    g.add_edge("C", "D", key=0, **{WEIGHT_ATTRIBUTE: 8.0, DYNAMIC_WEIGHT_ATTRIBUTE: 8.0, "length": 80.0})

    return g


# ---------------------------------------------------------------------------
# A. Base routing still works
# ---------------------------------------------------------------------------

def test_base_routing_still_works_default_mode(small_graph):
    """weight_mode defaults to 'base' — calling shortest_path with no
    weight_mode argument at all must behave exactly as before this
    integration existed."""
    result = routing.shortest_path(small_graph, "A", "D")
    assert result["is_reachable"] is True
    assert result["weight_mode"] == "base"
    assert result["weight_attribute"] == WEIGHT_ATTRIBUTE
    assert result["route_nodes"] == ["A", "D"]
    assert result["total_time_seconds"] == pytest.approx(10.0)


def test_base_routing_works_explicit_mode(small_graph):
    result = routing.shortest_path(small_graph, "A", "D", weight_mode="base")
    assert result["is_reachable"] is True
    assert result["route_nodes"] == ["A", "D"]
    assert result["total_time_seconds"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# B. Dynamic routing works
# ---------------------------------------------------------------------------

def test_dynamic_routing_works(small_graph):
    result = routing.shortest_path(small_graph, "A", "B", weight_mode="dynamic")
    # A->B has no dynamic_travel_time; NetworkX dijkstra defaults missing
    # weight to 1 for that edge, so the call must still succeed (not
    # crash) and correctly report which weight attribute it used.
    assert result["is_reachable"] is True
    assert result["weight_mode"] == "dynamic"
    assert result["weight_attribute"] == DYNAMIC_WEIGHT_ATTRIBUTE


# ---------------------------------------------------------------------------
# C. Base mode uses travel_time
# ---------------------------------------------------------------------------

def test_base_mode_uses_travel_time_attribute(small_graph):
    result = routing.shortest_path(small_graph, "A", "D", weight_mode="base")
    assert result["weight_attribute"] == WEIGHT_ATTRIBUTE
    # The direct A->D edge's travel_time (10.0) must be exactly what's
    # reported, proving dynamic_travel_time (50.0) was never consulted.
    assert result["total_time_seconds"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# D. Dynamic mode uses dynamic_travel_time
# ---------------------------------------------------------------------------

def test_dynamic_mode_uses_dynamic_travel_time_attribute():
    """Isolated two-node graph so the ONLY route is the congested edge —
    proves dynamic mode reads dynamic_travel_time (50.0), not
    travel_time (10.0), with nothing else to fall back to."""
    g = nx.MultiDiGraph()
    g.add_node("A", x=0.0, y=0.0)
    g.add_node("D", x=0.1, y=0.0)
    g.add_edge("A", "D", key=0, **{WEIGHT_ATTRIBUTE: 10.0, DYNAMIC_WEIGHT_ATTRIBUTE: 50.0, "length": 100.0})

    result = routing.shortest_path(g, "A", "D", weight_mode="dynamic")
    assert result["weight_attribute"] == DYNAMIC_WEIGHT_ATTRIBUTE
    assert result["total_time_seconds"] == pytest.approx(50.0)
    assert result["total_time_seconds"] != pytest.approx(10.0)


# ---------------------------------------------------------------------------
# E. A deliberately congested edge becomes more expensive in dynamic mode
# ---------------------------------------------------------------------------

def test_congested_edge_more_expensive_in_dynamic_mode(small_graph):
    base_direct_time = routing.calculate_route_time(small_graph, ["A", "D"], WEIGHT_ATTRIBUTE)
    dynamic_direct_time = routing.calculate_route_time(small_graph, ["A", "D"], DYNAMIC_WEIGHT_ATTRIBUTE)

    assert base_direct_time == pytest.approx(10.0)
    assert dynamic_direct_time == pytest.approx(50.0)
    assert dynamic_direct_time > base_direct_time


# ---------------------------------------------------------------------------
# F. Alternative route selected when it becomes faster in dynamic mode
# ---------------------------------------------------------------------------

def test_dijkstra_switches_to_alternative_route_when_dynamic(small_graph):
    base_result = routing.shortest_path(small_graph, "A", "D", weight_mode="base")
    dynamic_result = routing.shortest_path(small_graph, "A", "D", weight_mode="dynamic")

    # Base mode: direct A->D (10s) beats the detour (24s).
    assert base_result["route_nodes"] == ["A", "D"]
    assert base_result["total_time_seconds"] == pytest.approx(10.0)

    # Dynamic mode: direct A->D is now 50s, detour A->B->C->D has no
    # dynamic_travel_time on any of its edges. NetworkX's dijkstra
    # defaults a missing weight attribute to 1 per edge, so the detour
    # totals 3 (not 24) under weight_mode="dynamic" — still far less
    # than the congested direct edge's 50, so Dijkstra must switch.
    assert dynamic_result["route_nodes"] == ["A", "B", "C", "D"]
    assert dynamic_result["route_nodes"] != base_result["route_nodes"]


# ---------------------------------------------------------------------------
# G. Base and dynamic routing can be compared for the same origin/destination
# ---------------------------------------------------------------------------

def test_compare_base_vs_dynamic_route(small_graph, monkeypatch):
    """Uses dynamic_routing.compare_base_vs_dynamic_route(), but with
    ml_dynamic_graph's file-loading step bypassed (monkeypatched) since
    this fixture is a synthetic graph with no real ML files or verified
    junction mapping behind it — this test targets the comparison
    plumbing itself, not ml_dynamic_graph's file I/O (already covered
    exhaustively in test_ml_dynamic_graph.py, unmodified here)."""

    def fake_build_dynamic_graph_from_files(base_graph, junction_observation_files):
        # small_graph already carries dynamic_travel_time where relevant
        # (as if ml_dynamic_graph.py had already run) — just wrap it in
        # the same result type build_dynamic_graph_from_files returns.
        return mdg.DynamicGraphResult(
            graph=base_graph,
            junction_ids_applied=["synthetic"],
            updated_segment_ids_by_junction={"synthetic": ["A_D_0"]},
            congestion_factor_by_junction={"synthetic": 5.0},
        )

    monkeypatch.setattr(
        dr.ml_dynamic_graph,
        "build_dynamic_graph_from_files",
        fake_build_dynamic_graph_from_files,
    )

    comparison = dr.compare_base_vs_dynamic_route(
        small_graph, {"synthetic": "unused.json"}, "A", "D"
    )

    assert comparison["base"]["route_nodes"] == ["A", "D"]
    assert comparison["base"]["total_time_seconds"] == pytest.approx(10.0)

    assert comparison["dynamic"]["route_nodes"] == ["A", "B", "C", "D"]

    assert comparison["path_changed"] is True


# ---------------------------------------------------------------------------
# H. The original graph remains unchanged
# ---------------------------------------------------------------------------

def test_original_graph_unchanged_after_routing(small_graph):
    before_edges = {
        (u, v, k): dict(data)
        for u, v, k, data in small_graph.edges(keys=True, data=True)
    }

    routing.shortest_path(small_graph, "A", "D", weight_mode="base")
    routing.shortest_path(small_graph, "A", "D", weight_mode="dynamic")
    routing.shortest_path(small_graph, "A", "B", weight_mode="dynamic")

    after_edges = {
        (u, v, k): dict(data)
        for u, v, k, data in small_graph.edges(keys=True, data=True)
    }

    assert before_edges == after_edges
    assert small_graph.number_of_nodes() == 4
    assert small_graph.number_of_edges() == 4


def test_original_base_graph_unchanged_by_compare(small_graph, monkeypatch):
    """compare_base_vs_dynamic_route must not mutate the base_graph it's
    given, even though internally it routes on both the base graph and a
    (mocked, here identical-object-returning) 'dynamic' graph."""

    def fake_build_dynamic_graph_from_files(base_graph, junction_observation_files):
        graph_copy = base_graph.copy()  # genuine copy, as the real function does
        return mdg.DynamicGraphResult(
            graph=graph_copy,
            junction_ids_applied=["synthetic"],
            updated_segment_ids_by_junction={"synthetic": []},
            congestion_factor_by_junction={"synthetic": 5.0},
        )

    monkeypatch.setattr(
        dr.ml_dynamic_graph,
        "build_dynamic_graph_from_files",
        fake_build_dynamic_graph_from_files,
    )

    before = {
        (u, v, k): dict(data)
        for u, v, k, data in small_graph.edges(keys=True, data=True)
    }
    dr.compare_base_vs_dynamic_route(small_graph, {"synthetic": "unused.json"}, "A", "D")
    after = {
        (u, v, k): dict(data)
        for u, v, k, data in small_graph.edges(keys=True, data=True)
    }
    assert before == after


# ---------------------------------------------------------------------------
# I(a). Base mode errors clearly if requested attribute truly absent, and
#       dynamic mode's "no dynamic weights at all" guard still works
#       (routing.py's pre-existing behavior — verified here, unmodified)
# ---------------------------------------------------------------------------

def test_dynamic_mode_guard_when_graph_has_no_dynamic_weights_at_all():
    g = nx.MultiDiGraph()
    g.add_node("A", x=0.0, y=0.0)
    g.add_node("D", x=0.1, y=0.0)
    g.add_edge("A", "D", key=0, **{WEIGHT_ATTRIBUTE: 10.0, "length": 100.0})  # no dynamic_travel_time anywhere

    result = routing.shortest_path(g, "A", "D", weight_mode="dynamic")
    assert result["is_reachable"] is False
    assert "dynamic_travel_time" in result["error"] or "dynamic" in result["error"].lower()


# ---------------------------------------------------------------------------
# Real integration test (Nagpur graph + real ML observation files)
# ---------------------------------------------------------------------------

_DEFAULT_ML_OBS_DIR = Path(__file__).resolve().parent.parent / "data" / "ml_observations"
ML_OBS_DIR = (
    Path(os.environ["NOVAROUTE_ML_OUTPUT_DIR"])
    if os.environ.get("NOVAROUTE_ML_OUTPUT_DIR")
    else _DEFAULT_ML_OBS_DIR
)
JUNCTION_1_FILE = ML_OBS_DIR / "junction_1_latest.json"
JUNCTION_2_FILE = ML_OBS_DIR / "junction_2_latest.json"


@pytest.fixture(scope="module")
def nagpur_graph():
    if not PROCESSED_GRAPH_PATH.exists():
        pytest.skip(f"Processed graph not found at '{PROCESSED_GRAPH_PATH}'.")
    return load_graph(PROCESSED_GRAPH_PATH)


def _require_real_ml_file(path, junction_id):
    if not path.exists():
        raise FileNotFoundError(
            f"Required real ML observation file for {junction_id!r} not "
            f"found at '{path}'. Set NOVAROUTE_ML_OUTPUT_DIR to your real "
            "NovaRoute_AI ml/outputs/metrics folder before running this test."
        )


def test_real_integration_base_vs_dynamic_on_nagpur_graph(nagpur_graph):
    """The real end-to-end integration test requested in Prompt 6: real
    Nagpur graph, real ML observation files, real junction_mapping
    lookup (inside ml_dynamic_graph.py, unmodified), real Dijkstra
    (inside routing.py, unmodified)."""
    _require_real_ml_file(JUNCTION_1_FILE, "junction_1")
    _require_real_ml_file(JUNCTION_2_FILE, "junction_2")

    # junction_1's snapped node (3750261536, per the verified mapping)
    # is the origin; route to junction_2's snapped node (9345140062).
    origin_node = 3750261536
    destination_node = 9345140062

    comparison = dr.compare_base_vs_dynamic_route(
        nagpur_graph,
        {"junction_1": JUNCTION_1_FILE, "junction_2": JUNCTION_2_FILE},
        origin_node,
        destination_node,
    )

    assert comparison["base"]["is_reachable"] is True
    assert comparison["base"]["weight_attribute"] == WEIGHT_ATTRIBUTE

    assert comparison["dynamic"]["is_reachable"] is True
    assert comparison["dynamic"]["weight_attribute"] == DYNAMIC_WEIGHT_ATTRIBUTE

    # Original graph must remain untouched by the whole comparison.
    assert not any(
        DYNAMIC_WEIGHT_ATTRIBUTE in data for _, _, data in nagpur_graph.edges(data=True)
    )

    dgr = comparison["dynamic_graph_result"]
    assert set(dgr.junction_ids_applied) == {"junction_1", "junction_2"}
    assert "junction_3" not in dgr.junction_ids_applied


# ---------------------------------------------------------------------------
# Prompt 6A: corrected full-coverage scenario, small deterministic graph
# ---------------------------------------------------------------------------

def test_dijkstra_uses_real_times_with_full_coverage_not_hop_count(small_graph_full_coverage):
    """With every edge carrying a real dynamic_travel_time (baseline ==
    travel_time on uncovered edges, ML-adjusted on covered edges),
    dynamic-mode Dijkstra must compare 50s (direct) against 24s (detour
    real travel time), NOT 50s against 3 (NetworkX's old implicit
    weight=1 x 3 hops). The detour still wins either way here, but for
    the RIGHT reason — this test checks the reported total_time_seconds
    reflects real time (24.0), not hop count (3)."""
    result = routing.shortest_path(small_graph_full_coverage, "A", "D", weight_mode="dynamic")

    assert result["is_reachable"] is True
    assert result["route_nodes"] == ["A", "B", "C", "D"]
    assert result["total_time_seconds"] == pytest.approx(24.0)  # real time, not 3 (hop count)
    assert result["total_time_seconds"] != pytest.approx(3.0)


def test_congested_direct_edge_vs_real_detour_time_full_coverage(small_graph_full_coverage):
    """Sanity check that both edges being compared have physically
    meaningful values under full coverage: congested direct (50s) is
    still more expensive than the real (not hop-count) detour time (24s)."""
    direct_dynamic_time = routing.calculate_route_time(
        small_graph_full_coverage, ["A", "D"], DYNAMIC_WEIGHT_ATTRIBUTE
    )
    detour_dynamic_time = routing.calculate_route_time(
        small_graph_full_coverage, ["A", "B", "C", "D"], DYNAMIC_WEIGHT_ATTRIBUTE
    )
    assert direct_dynamic_time == pytest.approx(50.0)
    assert detour_dynamic_time == pytest.approx(24.0)
    assert direct_dynamic_time > detour_dynamic_time