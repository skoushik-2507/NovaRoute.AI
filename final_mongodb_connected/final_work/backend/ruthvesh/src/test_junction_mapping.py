"""
test_junction_mapping.py

Pytest test suite for src/junction_mapping.py (PROTOTYPE junction -> OSM
segment mapping stage).

Mirrors test_routing.py's approach: uses the ACTUAL processed Nagpur
graph, no mocking. If the processed graph file does not exist, the
graph-dependent tests are skipped with a clear message rather than
failing.

Run with:
    pytest src/test_junction_mapping.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_GRAPH_PATH
from src.graph_utils import load_graph
from src import junction_mapping as jm


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


# ---------------------------------------------------------------------------
# 1. All three prototype junction coordinates can be processed
# ---------------------------------------------------------------------------

def test_all_prototype_junctions_have_coordinates():
    """All three junction ids must have a simulated coordinate defined."""
    assert set(jm.PROTOTYPE_JUNCTION_COORDINATES.keys()) == {
        "junction_1", "junction_2", "junction_3",
    }


def test_all_prototype_junctions_can_be_reported(nagpur_graph):
    """build_all_junction_reports should process all three junctions
    without error against the real graph."""
    reports = jm.build_all_junction_reports(nagpur_graph)
    assert len(reports) == 3
    assert {r.junction_id for r in reports} == {
        "junction_1", "junction_2", "junction_3",
    }


# ---------------------------------------------------------------------------
# 2. A graph node is found for each junction
# ---------------------------------------------------------------------------

def test_each_junction_resolves_to_a_real_graph_node(nagpur_graph):
    """Every junction's simulated coordinate must snap to a node that
    actually exists in the graph."""
    for junction_id in jm.PROTOTYPE_JUNCTION_COORDINATES:
        report = jm.build_junction_report(nagpur_graph, junction_id)
        assert report.nearest_graph_node is not None
        assert report.nearest_graph_node in nagpur_graph.nodes


def test_distance_to_node_is_non_negative(nagpur_graph):
    """The reported snap distance must be a sane non-negative number."""
    for junction_id in jm.PROTOTYPE_JUNCTION_COORDINATES:
        report = jm.build_junction_report(nagpur_graph, junction_id)
        assert report.distance_to_node_meters >= 0.0


# ---------------------------------------------------------------------------
# 3. Selected edges actually exist in the graph (no fake edge IDs)
# ---------------------------------------------------------------------------

def test_selected_segments_exist_in_graph(nagpur_graph):
    """Every (u, v, key) triple returned for a junction must be a real,
    existing edge in the graph."""
    for junction_id in jm.PROTOTYPE_JUNCTION_COORDINATES:
        report = jm.build_junction_report(nagpur_graph, junction_id)
        for u, v, k in report.selected_segments:
            assert nagpur_graph.has_edge(u, v, k), (
                f"Segment ({u}, {v}, {k}) reported for {junction_id} "
                "does not actually exist in the graph."
            )


def test_selected_segments_are_incident_to_the_snapped_node(nagpur_graph):
    """Every returned segment must actually touch the junction's
    nearest_graph_node (as an endpoint), proving they weren't invented
    or pulled from an unrelated part of the graph."""
    for junction_id in jm.PROTOTYPE_JUNCTION_COORDINATES:
        report = jm.build_junction_report(nagpur_graph, junction_id)
        node = report.nearest_graph_node
        for u, v, _k in report.selected_segments:
            assert node in (u, v), (
                f"Segment ({u}, {v}) for {junction_id} does not touch "
                f"nearest_graph_node {node}."
            )


# ---------------------------------------------------------------------------
# 4. No fake edge IDs are introduced
# ---------------------------------------------------------------------------

def test_segment_ids_match_congestion_module_format(nagpur_graph):
    """Segment ID strings must be generated via congestion.get_segment_id
    (the single source of truth for the 'u_v_key' format), not
    hand-built, and must round-trip back to a real edge."""
    from src.congestion import get_segment_id_mapping

    id_to_edge = get_segment_id_mapping(nagpur_graph)

    for junction_id in jm.PROTOTYPE_JUNCTION_COORDINATES:
        report = jm.build_junction_report(nagpur_graph, junction_id)
        for seg_id in report.selected_segment_ids():
            assert seg_id in id_to_edge, (
                f"Segment id {seg_id!r} for {junction_id} does not "
                "correspond to any real edge via "
                "congestion.get_segment_id_mapping()."
            )


def test_get_incident_segments_reads_live_from_graph(nagpur_graph):
    """get_incident_segments must return exactly graph.out_edges +
    graph.in_edges for the node — proving it derives edges live rather
    than returning a hardcoded/static list."""
    node = next(iter(nagpur_graph.nodes))
    expected = list(nagpur_graph.out_edges(node, keys=True)) + list(
        nagpur_graph.in_edges(node, keys=True)
    )
    actual = jm.get_incident_segments(nagpur_graph, node)
    assert actual == expected


def test_get_incident_segments_unknown_node_raises(nagpur_graph):
    """A node not present in the graph must fail clearly, not silently
    return an empty or fabricated edge list."""
    with pytest.raises(jm.JunctionNodeNotFoundError):
        jm.get_incident_segments(nagpur_graph, "definitely_not_a_real_node_id")


# ---------------------------------------------------------------------------
# 5. Unknown junction ids fail clearly
# ---------------------------------------------------------------------------

def test_unknown_junction_id_raises_on_coordinate_lookup():
    with pytest.raises(jm.UnknownJunctionError, match="junction_999"):
        jm.get_junction_coordinate("junction_999")


def test_unknown_junction_id_raises_on_report_build(nagpur_graph):
    with pytest.raises(jm.UnknownJunctionError, match="junction_999"):
        jm.build_junction_report(nagpur_graph, "junction_999")


# ---------------------------------------------------------------------------
# Reports are clearly labeled as prototype
# ---------------------------------------------------------------------------

def test_report_is_labeled_prototype(nagpur_graph):
    """Every report must self-identify as prototype/simulated — this is
    a hard requirement, not just a docstring note."""
    for junction_id in jm.PROTOTYPE_JUNCTION_COORDINATES:
        report = jm.build_junction_report(nagpur_graph, junction_id)
        assert report.is_prototype is True


def test_print_all_junction_reports_returns_all_three(nagpur_graph, capsys):
    """The CLI-style print function should print and also return all
    three reports for programmatic use."""
    reports = jm.print_all_junction_reports(nagpur_graph)
    captured = capsys.readouterr()

    assert len(reports) == 3
    assert "PROTOTYPE/SIMULATED" in captured.out
    for junction_id in jm.PROTOTYPE_JUNCTION_COORDINATES:
        assert junction_id in captured.out