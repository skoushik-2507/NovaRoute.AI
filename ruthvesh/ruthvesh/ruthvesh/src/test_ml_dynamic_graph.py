"""
test_ml_dynamic_graph.py

Pytest test suite for src/ml_dynamic_graph.py (ML congestion_factor ->
dynamic_travel_time graph copy stage).

Mirrors test_junction_mapping.py's approach: uses the ACTUAL processed
Nagpur graph, no mocking of the graph itself. Uses the ACTUAL real
NovaRoute ML JSON files (junction_1_latest.json / junction_2_latest.json)
for the integration-style tests, per instructions, plus small synthetic
observations (via ml_integration's dict-input support) for the
failure-mode tests that need to control congestion_factor precisely.

Run with:
    pytest src/test_ml_dynamic_graph.py -v
"""

import copy
import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_GRAPH_PATH, WEIGHT_ATTRIBUTE
from src.congestion import DYNAMIC_WEIGHT_ATTRIBUTE, get_segment_id
from src.graph_utils import load_graph
from src.junction_mapping import build_junction_report
from src import ml_dynamic_graph as mdg
from src import ml_integration as mi


# ---------------------------------------------------------------------------
# Real ML observation file locations
# ---------------------------------------------------------------------------
# Resolved from the NOVAROUTE_ML_OUTPUT_DIR environment variable, which
# must point at the directory containing the real NovaRoute ML pipeline
# output (junction_1_latest.json, junction_2_latest.json), e.g.:
#
#   Windows (PowerShell):
#       $env:NOVAROUTE_ML_OUTPUT_DIR = "C:\Users\koush\OneDrive\Desktop\NovaRoute_AI\ml\outputs\metrics"
#       python -m pytest src/test_ml_dynamic_graph.py -v
#
#   Windows (cmd.exe):
#       set NOVAROUTE_ML_OUTPUT_DIR=C:\Users\koush\OneDrive\Desktop\NovaRoute_AI\ml\outputs\metrics
#
#   macOS/Linux (bash):
#       export NOVAROUTE_ML_OUTPUT_DIR=/path/to/NovaRoute_AI/ml/outputs/metrics
#
# If the variable is unset, this falls back to a repo-relative
# data/ml_observations/ directory (useful for CI or for anyone who
# copies sample files there instead of pointing at a live ML output
# folder). Either way, this is TEST/FIXTURE configuration only — the
# production module (ml_dynamic_graph.py) has no knowledge of this
# environment variable or of any absolute path; it only ever accepts
# file paths as explicit function arguments (see
# load_junction_observations / build_dynamic_graph_from_files).
_DEFAULT_ML_OBS_DIR = Path(__file__).resolve().parent.parent / "data" / "ml_observations"
ML_OBS_DIR = Path(os.environ["NOVAROUTE_ML_OUTPUT_DIR"]) if os.environ.get("NOVAROUTE_ML_OUTPUT_DIR") else _DEFAULT_ML_OBS_DIR
JUNCTION_1_FILE = ML_OBS_DIR / "junction_1_latest.json"
JUNCTION_2_FILE = ML_OBS_DIR / "junction_2_latest.json"


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
def real_junction_observations():
    """Load the two real ML observation files, once per module. Fails
    clearly (does not skip) if either required file is missing — a
    missing real ML file means the integration is unverified, which
    must be visible as a failure, not silently absent from the report."""
    _require_real_ml_file(JUNCTION_1_FILE, "junction_1")
    _require_real_ml_file(JUNCTION_2_FILE, "junction_2")
    return mdg.load_junction_observations({
        "junction_1": JUNCTION_1_FILE,
        "junction_2": JUNCTION_2_FILE,
    })


@pytest.fixture
def base_observation_dict():
    """A minimal, schema-valid ML observation dict (mirrors the shape of
    the real files), for synthetic/failure-mode tests that need precise
    control over congestion_factor or malformed fields."""
    return {
        "schema_version": "1.1.0",
        "road_segment_id": "junction_1",
        "osm_edge": None,
        "timestamp": "2026-08-16T12:00:00+00:00",
        "observation_window_seconds": 5.0,
        "vehicle_counts": {"car": 18, "motorcycle": 7, "bus": 1, "truck": 2, "total": 28},
        "total_vehicles": 28,
        "peak_vehicles": 33,
        "road_capacity": 50.0,
        "traffic_density": 0.56,
        "congestion_level": "moderate",
        "congestion_factor": 1.5,
        "aggregation_method": "average_active_vehicles",
        "risk_score": 49.25,
        "risk_level": "moderate",
        "risk_factor_scores": {
            "accident_history": 0.3, "traffic_density": 0.373333,
            "pedestrian_conflict": 0.2, "time_of_day": 0.75,
        },
        "risk_contributions": {
            "accident_history": 10.5, "traffic_density": 9.333333,
            "pedestrian_conflict": 5.0, "time_of_day": 11.25,
        },
    }


# ---------------------------------------------------------------------------
# 1. Dynamic graph is a separate graph object
# ---------------------------------------------------------------------------

def test_dynamic_graph_is_a_separate_object(nagpur_graph, real_junction_observations):
    result = mdg.apply_ml_congestion_to_graph(nagpur_graph, real_junction_observations)
    assert result.graph is not nagpur_graph
    assert isinstance(result.graph, type(nagpur_graph))


# ---------------------------------------------------------------------------
# 2. Original graph is unchanged
# ---------------------------------------------------------------------------

def test_original_graph_is_unchanged(nagpur_graph, real_junction_observations):
    # Snapshot every edge's attribute keys before, so we can prove
    # nothing was added/mutated on the original graph object.
    before_keys = {
        (u, v, k): set(data.keys())
        for u, v, k, data in nagpur_graph.edges(keys=True, data=True)
    }
    before_node_count = nagpur_graph.number_of_nodes()
    before_edge_count = nagpur_graph.number_of_edges()

    mdg.apply_ml_congestion_to_graph(nagpur_graph, real_junction_observations)

    after_keys = {
        (u, v, k): set(data.keys())
        for u, v, k, data in nagpur_graph.edges(keys=True, data=True)
    }

    assert nagpur_graph.number_of_nodes() == before_node_count
    assert nagpur_graph.number_of_edges() == before_edge_count
    assert before_keys == after_keys
    assert not any(
        DYNAMIC_WEIGHT_ATTRIBUTE in data
        for _, _, data in nagpur_graph.edges(data=True)
    )


# ---------------------------------------------------------------------------
# 3 & 4. junction_1 / junction_2 mapped edges receive their ML factor
# ---------------------------------------------------------------------------

def test_junction_1_mapped_edges_receive_its_ml_factor(nagpur_graph, real_junction_observations):
    result = mdg.apply_ml_congestion_to_graph(nagpur_graph, real_junction_observations)
    factor = result.congestion_factor_by_junction["junction_1"]

    report = build_junction_report(nagpur_graph, "junction_1")
    assert len(report.selected_segments) > 0  # sanity: junction_1 has real incident edges

    for u, v, key in report.selected_segments:
        edge_data = result.graph[u][v][key]
        assert DYNAMIC_WEIGHT_ATTRIBUTE in edge_data
        expected = edge_data[WEIGHT_ATTRIBUTE] * factor
        assert edge_data[DYNAMIC_WEIGHT_ATTRIBUTE] == pytest.approx(expected)


def test_junction_2_mapped_edges_receive_its_ml_factor(nagpur_graph, real_junction_observations):
    result = mdg.apply_ml_congestion_to_graph(nagpur_graph, real_junction_observations)
    factor = result.congestion_factor_by_junction["junction_2"]

    report = build_junction_report(nagpur_graph, "junction_2")
    assert len(report.selected_segments) > 0

    for u, v, key in report.selected_segments:
        edge_data = result.graph[u][v][key]
        assert DYNAMIC_WEIGHT_ATTRIBUTE in edge_data
        expected = edge_data[WEIGHT_ATTRIBUTE] * factor
        assert edge_data[DYNAMIC_WEIGHT_ATTRIBUTE] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5 & 6. travel_time unchanged; dynamic_travel_time = travel_time * factor
# ---------------------------------------------------------------------------

def test_travel_time_remains_unchanged_on_mapped_edges(nagpur_graph, real_junction_observations):
    report = build_junction_report(nagpur_graph, "junction_1")
    before_travel_times = {
        (u, v, k): nagpur_graph[u][v][k][WEIGHT_ATTRIBUTE]
        for u, v, k in report.selected_segments
    }

    result = mdg.apply_ml_congestion_to_graph(nagpur_graph, real_junction_observations)

    for (u, v, k), before_tt in before_travel_times.items():
        assert result.graph[u][v][k][WEIGHT_ATTRIBUTE] == before_tt


def test_dynamic_travel_time_equals_travel_time_times_factor(nagpur_graph, real_junction_observations):
    result = mdg.apply_ml_congestion_to_graph(nagpur_graph, real_junction_observations)

    for junction_id in ("junction_1", "junction_2"):
        factor = result.congestion_factor_by_junction[junction_id]
        for segment_id in result.updated_segment_ids_by_junction[junction_id]:
            u, v, k = next(
                (uu, vv, kk) for uu, vv, kk in result.graph.edges(keys=True)
                if get_segment_id(uu, vv, kk) == segment_id
            )
            edge_data = result.graph[u][v][k]
            assert edge_data[DYNAMIC_WEIGHT_ATTRIBUTE] == pytest.approx(
                edge_data[WEIGHT_ATTRIBUTE] * factor
            )


# ---------------------------------------------------------------------------
# 7. Factor is applied exactly once
# ---------------------------------------------------------------------------

def test_factor_applied_exactly_once_not_squared(nagpur_graph, real_junction_observations):
    """If the factor were (incorrectly) applied twice, dynamic_travel_time
    would equal travel_time * factor**2, not travel_time * factor. This
    test would only pass by coincidence for factor == 1.0 or 0.0, so it
    uses a synthetic factor far from both (2.0) to make double-application
    unambiguously detectable."""
    synthetic_obs = {
        "junction_1": mi.extract_ml_fields({
            "schema_version": "1.1.0",
            "road_segment_id": "junction_1",
            "osm_edge": None,
            "timestamp": "2026-08-16T12:00:00+00:00",
            "observation_window_seconds": 5.0,
            "vehicle_counts": {"car": 1, "motorcycle": 0, "bus": 0, "truck": 0, "total": 1},
            "total_vehicles": 1,
            "peak_vehicles": 1,
            "road_capacity": 50.0,
            "traffic_density": 0.02,
            "congestion_level": "free_flow",
            "congestion_factor": 2.0,
            "aggregation_method": "average_active_vehicles",
            "risk_score": 10.0,
            "risk_level": "low",
            "risk_factor_scores": {
                "accident_history": 0.0, "traffic_density": 0.0,
                "pedestrian_conflict": 0.0, "time_of_day": 0.4,
            },
            "risk_contributions": {
                "accident_history": 0.0, "traffic_density": 0.0,
                "pedestrian_conflict": 0.0, "time_of_day": 6.0,
            },
        })
    }

    result = mdg.apply_ml_congestion_to_graph(nagpur_graph, synthetic_obs)
    report = build_junction_report(nagpur_graph, "junction_1")

    for u, v, k in report.selected_segments:
        edge_data = result.graph[u][v][k]
        travel_time = edge_data[WEIGHT_ATTRIBUTE]
        dynamic = edge_data[DYNAMIC_WEIGHT_ATTRIBUTE]
        assert dynamic == pytest.approx(travel_time * 2.0)
        assert dynamic != pytest.approx(travel_time * 4.0)  # would indicate factor^2

    # Calling apply again on the *original* graph (not the result) with
    # the same observation must produce the identical dynamic_travel_time
    # — proves the function is idempotent per-call, not accumulating
    # state across calls.
    result_again = mdg.apply_ml_congestion_to_graph(nagpur_graph, synthetic_obs)
    for u, v, k in report.selected_segments:
        assert result.graph[u][v][k][DYNAMIC_WEIGHT_ATTRIBUTE] == pytest.approx(
            result_again.graph[u][v][k][DYNAMIC_WEIGHT_ATTRIBUTE]
        )


# ---------------------------------------------------------------------------
# 8. Unmapped edges get baseline dynamic_travel_time (== travel_time),
#    NOT congestion-adjusted, and NOT missing (Prompt 6A)
# ---------------------------------------------------------------------------

def test_unmapped_edges_get_baseline_dynamic_travel_time_not_ml_adjustment(nagpur_graph, real_junction_observations):
    """Prompt 6A: unmapped edges are NOT left without dynamic_travel_time
    anymore (that caused NetworkX's implicit weight=1 fallback during
    Dijkstra). They now get a baseline dynamic_travel_time == travel_time
    -- i.e. dynamic_travel_time IS present, but is NOT congestion-adjusted
    (no ML factor applied to something with no ML observation)."""
    result = mdg.apply_ml_congestion_to_graph(nagpur_graph, real_junction_observations)

    mapped_segment_ids = set()
    for seg_ids in result.updated_segment_ids_by_junction.values():
        mapped_segment_ids.update(seg_ids)

    # Sample: check a broad set of edges (every 500th edge) that are
    # NOT in the ML-mapped set.
    checked = 0
    for i, (u, v, k) in enumerate(result.graph.edges(keys=True)):
        if i % 500 != 0:
            continue
        segment_id = get_segment_id(u, v, k)
        if segment_id in mapped_segment_ids:
            continue
        edge_data = result.graph[u][v][k]
        assert DYNAMIC_WEIGHT_ATTRIBUTE in edge_data  # present...
        assert edge_data[DYNAMIC_WEIGHT_ATTRIBUTE] == pytest.approx(
            edge_data[WEIGHT_ATTRIBUTE]
        )  # ...but exactly equal to travel_time, i.e. unadjusted
        checked += 1

    assert checked > 0  # sanity: we actually checked something


# ---------------------------------------------------------------------------
# 9. junction_3 gets baseline only, no ML factor applied (no observation)
# ---------------------------------------------------------------------------

def test_junction_3_gets_baseline_only_no_ml_factor_applied(nagpur_graph, real_junction_observations):
    """real_junction_observations only contains junction_1 and junction_2
    -- junction_3's incident edges must get the baseline
    dynamic_travel_time == travel_time (present, per Prompt 6A), but must
    NOT have any congestion_factor applied to them, since there is no
    real ML observation for junction_3."""
    result = mdg.apply_ml_congestion_to_graph(nagpur_graph, real_junction_observations)

    assert "junction_3" not in result.junction_ids_applied

    report = build_junction_report(nagpur_graph, "junction_3")
    assert len(report.selected_segments) > 0
    for u, v, k in report.selected_segments:
        edge_data = result.graph[u][v][k]
        assert DYNAMIC_WEIGHT_ATTRIBUTE in edge_data
        assert edge_data[DYNAMIC_WEIGHT_ATTRIBUTE] == pytest.approx(
            edge_data[WEIGHT_ATTRIBUTE]
        )



# ---------------------------------------------------------------------------
# 10. Unknown junction fails clearly
# ---------------------------------------------------------------------------

def test_unknown_junction_fails_clearly(nagpur_graph, base_observation_dict):
    obs = dict(base_observation_dict)
    obs["road_segment_id"] = "junction_999"
    fields = mi.extract_ml_fields(obs)

    with pytest.raises(mdg.UnmappedJunctionError, match="junction_999"):
        mdg.apply_ml_congestion_to_graph(nagpur_graph, {"junction_999": fields})


def test_junction_observation_mismatch_fails_clearly(base_observation_dict):
    """load_junction_observations must reject a file loaded under the
    wrong junction_id key (road_segment_id inside the JSON disagrees)."""
    obs = dict(base_observation_dict)
    obs["road_segment_id"] = "junction_2"  # file content says junction_2...

    with pytest.raises(mdg.JunctionObservationMismatchError):
        # ...but caller claims it's junction_1's observation.
        mdg.load_junction_observations({"junction_1": obs})


# ---------------------------------------------------------------------------
# 11. Missing congestion_factor fails clearly
# ---------------------------------------------------------------------------

def test_missing_congestion_factor_fails_clearly(base_observation_dict):
    obs = dict(base_observation_dict)
    del obs["congestion_factor"]

    with pytest.raises(mi.MLObservationError, match="congestion_factor"):
        mdg.load_junction_observations({"junction_1": obs})


# ---------------------------------------------------------------------------
# 12. Invalid ML JSON fails clearly
# ---------------------------------------------------------------------------

def test_invalid_json_file_fails_clearly(tmp_path):
    bad_file = tmp_path / "not_valid.json"
    bad_file.write_text("{ this is not valid json ]", encoding="utf-8")

    with pytest.raises(mi.MLObservationError, match="not valid JSON"):
        mdg.load_junction_observations({"junction_1": bad_file})


def test_missing_file_fails_clearly(tmp_path):
    missing_file = tmp_path / "does_not_exist.json"

    with pytest.raises(mi.MLObservationError, match="not found"):
        mdg.load_junction_observations({"junction_1": missing_file})


# ---------------------------------------------------------------------------
# Real-file integration test: exact values read from the actual JSON
# ---------------------------------------------------------------------------

def test_real_files_load_expected_congestion_factors(real_junction_observations):
    """Confirms this module reads congestion_factor verbatim from the
    actual JSON files, not a hardcoded/guessed value."""
    assert real_junction_observations["junction_1"].congestion_factor == pytest.approx(1.000351384)
    assert real_junction_observations["junction_2"].congestion_factor == pytest.approx(1.000098304)


def test_conflicting_segment_congestion_raises(nagpur_graph, base_observation_dict, monkeypatch):
    """If two junctions' verified segment sets overlapped with different
    factors, this must raise rather than silently pick one. Simulated by
    monkeypatching junction_mapping.PROTOTYPE_JUNCTION_COORDINATES so
    junction_1 and a fake 'junction_1_dup' resolve to the SAME
    coordinate (and therefore the same incident segments) with different
    congestion factors."""
    import src.junction_mapping as jm

    fake_coords = dict(jm.PROTOTYPE_JUNCTION_COORDINATES)
    fake_coords["junction_1_dup"] = fake_coords["junction_1"]
    # build_junction_report() (called inside ml_dynamic_graph) reads
    # PROTOTYPE_JUNCTION_COORDINATES from the junction_mapping module's
    # own namespace, so that is the reference that must be patched — not
    # ml_dynamic_graph's separately-imported name binding.
    monkeypatch.setattr(jm, "PROTOTYPE_JUNCTION_COORDINATES", fake_coords)
    monkeypatch.setattr(mdg, "PROTOTYPE_JUNCTION_COORDINATES", fake_coords)

    obs_1 = dict(base_observation_dict)
    obs_1["road_segment_id"] = "junction_1"
    obs_1["congestion_factor"] = 1.5

    obs_2 = dict(base_observation_dict)
    obs_2["road_segment_id"] = "junction_1_dup"
    obs_2["congestion_factor"] = 3.0

    observations = {
        "junction_1": mi.extract_ml_fields(obs_1),
        "junction_1_dup": mi.extract_ml_fields(obs_2),
    }

    with pytest.raises(mdg.SegmentCongestionConflictError):
        mdg.apply_ml_congestion_to_graph(nagpur_graph, observations)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def test_build_dynamic_graph_from_files_matches_manual_pipeline(nagpur_graph):
    _require_real_ml_file(JUNCTION_1_FILE, "junction_1")
    _require_real_ml_file(JUNCTION_2_FILE, "junction_2")

    result_via_wrapper = mdg.build_dynamic_graph_from_files(
        nagpur_graph, {"junction_1": JUNCTION_1_FILE, "junction_2": JUNCTION_2_FILE}
    )
    manual_obs = mdg.load_junction_observations(
        {"junction_1": JUNCTION_1_FILE, "junction_2": JUNCTION_2_FILE}
    )
    result_manual = mdg.apply_ml_congestion_to_graph(nagpur_graph, manual_obs)

    assert (
        result_via_wrapper.total_edges_updated == result_manual.total_edges_updated
    )
    assert (
        result_via_wrapper.congestion_factor_by_junction
        == result_manual.congestion_factor_by_junction
    )


# ---------------------------------------------------------------------------
# Prompt 6A: every edge has a valid dynamic_travel_time; no implicit
# NetworkX weight=1 fallback anywhere on the real graph.
# ---------------------------------------------------------------------------

def test_every_edge_in_dynamic_graph_has_dynamic_travel_time(nagpur_graph, real_junction_observations):
    """Requirement 1 (Prompt 6A): EVERY edge in the returned dynamic
    graph copy must carry a dynamic_travel_time attribute — not just the
    14 ML-covered edges out of 110,566."""
    result = mdg.apply_ml_congestion_to_graph(nagpur_graph, real_junction_observations)

    edges_missing_dynamic = [
        (u, v, k) for u, v, k, data in result.graph.edges(keys=True, data=True)
        if DYNAMIC_WEIGHT_ATTRIBUTE not in data
    ]
    # The only edges allowed to be missing dynamic_travel_time are ones
    # that never had a base travel_time to begin with (none expected on
    # this processed graph, but this keeps the test honest rather than
    # silently masking a real gap).
    edges_missing_base = [
        (u, v, k) for u, v, k, data in result.graph.edges(keys=True, data=True)
        if WEIGHT_ATTRIBUTE not in data
    ]
    assert edges_missing_base == []  # sanity: processed graph always has travel_time
    assert edges_missing_dynamic == [], (
        f"{len(edges_missing_dynamic)} edge(s) have travel_time but no "
        f"dynamic_travel_time, e.g. {edges_missing_dynamic[:5]}"
    )


def test_real_nagpur_graph_has_valid_dynamic_weights_on_all_edges(nagpur_graph, real_junction_observations):
    """Requirement 9 (Prompt 6A): on the real Nagpur graph specifically,
    confirm total edge count == count of edges with dynamic_travel_time,
    and every value is a positive, finite number (never a placeholder)."""
    result = mdg.apply_ml_congestion_to_graph(nagpur_graph, real_junction_observations)

    total_edges = result.graph.number_of_edges()
    edges_with_dynamic = sum(
        1 for _, _, data in result.graph.edges(data=True)
        if DYNAMIC_WEIGHT_ATTRIBUTE in data
    )
    assert total_edges == 110566  # matches the verified real graph size
    assert edges_with_dynamic == total_edges

    for _, _, data in result.graph.edges(data=True):
        dtt = data[DYNAMIC_WEIGHT_ATTRIBUTE]
        assert dtt > 0
        assert dtt == dtt  # not NaN
        assert dtt != float("inf")


def test_dynamic_routing_does_not_fall_back_to_networkx_implicit_weight_one(nagpur_graph, real_junction_observations):
    """Requirement 8 (Prompt 6A): with every edge now carrying a real
    dynamic_travel_time, NetworkX's dijkstra_path (weight="dynamic_travel_time")
    must never hit its own implicit default-to-1 fallback anywhere along
    the route it picks between two real, moderately distant graph nodes
    — proven by asserting the reported dynamic total_time is NOT an
    integer-ish "hop count" and is instead consistent with real edge
    dynamic_travel_time sums for the exact nodes on the route."""
    from src import routing

    result = mdg.apply_ml_congestion_to_graph(nagpur_graph, real_junction_observations)

    route_result = routing.shortest_path(
        result.graph, 3750261536, 9345140062, weight_mode="dynamic"
    )
    assert route_result["is_reachable"] is True

    # Recompute the route's total dynamic time by directly summing each
    # edge's real dynamic_travel_time along the reported path, and
    # confirm it matches what Dijkstra/routing.py reported — if any edge
    # had silently fallen back to weight=1, this independent recomputation
    # (which reads the real attribute, never a default) would disagree.
    independent_total = 0.0
    route_nodes = route_result["route_nodes"]
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        edge_dict = result.graph.get_edge_data(u, v)
        best = min(edge_dict.values(), key=lambda d: d[DYNAMIC_WEIGHT_ATTRIBUTE])
        independent_total += best[DYNAMIC_WEIGHT_ATTRIBUTE]

    assert route_result["total_time_seconds"] == pytest.approx(independent_total)
    # A hop-count-minimized route (the old bug) would report a total in
    # the single/low-double digits for an 80+-node route; a real
    # travel-time sum for a multi-km real route should be well above 1
    # second per hop on average.
    assert route_result["total_time_seconds"] > len(route_nodes) * 1.0