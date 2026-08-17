"""
test_routing.py

Pytest test suite for the NovaRoute.AI routing module (routing.py).

These tests use the ACTUAL processed Nagpur graph and the ACTUAL
routing.py implementation — no mock routing algorithm is used. Reachable
node pairs are found by walking the real graph structure (BFS), and
"unreachable" is tested by adding a genuinely isolated node to a copy of
the real graph (no edges), so Dijkstra genuinely cannot reach it.

If the processed graph file does not exist yet (i.e. graph_builder.py
has not been run), these tests are skipped with a clear message rather
than failing, since there is nothing to test against.

Run with:
    pytest src/test_routing.py -v
"""

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_GRAPH_PATH
from src.graph_utils import load_graph
from src.routing import shortest_path, route_between_coordinates


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nagpur_graph():
    """
    Load the actual processed Nagpur graph once for all tests in this
    module. Skips the whole module if the graph file doesn't exist yet.
    """
    if not PROCESSED_GRAPH_PATH.exists():
        pytest.skip(
            f"Processed graph not found at '{PROCESSED_GRAPH_PATH}'. "
            "Run graph_builder.py first."
        )
    return load_graph(PROCESSED_GRAPH_PATH)


@pytest.fixture(scope="module")
def reachable_node_pair(nagpur_graph):
    """
    Find a real origin/destination pair that is guaranteed to be
    reachable, by walking outward from an arbitrary starting node using
    a breadth-first search on the actual graph.
    """
    origin = next(iter(nagpur_graph.nodes))

    # BFS outward from origin to find a node a few hops away.
    visited_in_order = [origin]
    for _, target in nx.bfs_edges(nagpur_graph, origin):
        visited_in_order.append(target)
        if len(visited_in_order) > 5:
            break

    if len(visited_in_order) < 2:
        pytest.skip("Could not find a reachable node pair in the graph.")

    destination = visited_in_order[-1]
    return origin, destination


@pytest.fixture(scope="module")
def graph_with_isolated_node(nagpur_graph):
    """
    A copy of the real graph with one extra node that has no edges at
    all, so it is genuinely unreachable from every other node. This is
    test setup only — the routing algorithm itself is untouched.
    """
    graph_copy = nagpur_graph.copy()
    isolated_node_id = "isolated_test_node_for_pytest"
    graph_copy.add_node(isolated_node_id, x=0.0, y=0.0)
    return graph_copy, isolated_node_id


# ---------------------------------------------------------------------------
# 1. Graph loads successfully
# ---------------------------------------------------------------------------

def test_graph_loads_successfully(nagpur_graph):
    """The processed graph should load without errors."""
    assert nagpur_graph is not None


# ---------------------------------------------------------------------------
# 2. Graph contains nodes and edges
# ---------------------------------------------------------------------------

def test_graph_contains_nodes_and_edges(nagpur_graph):
    """The graph should not be empty."""
    assert nagpur_graph.number_of_nodes() > 0
    assert nagpur_graph.number_of_edges() > 0


# ---------------------------------------------------------------------------
# 3. Valid nodes can be routed between
# ---------------------------------------------------------------------------

def test_valid_nodes_can_be_routed(nagpur_graph, reachable_node_pair):
    """Routing between two connected real nodes should succeed."""
    origin, destination = reachable_node_pair

    result = shortest_path(nagpur_graph, origin, destination)

    assert result["is_reachable"] is True
    assert result["error"] is None


# ---------------------------------------------------------------------------
# 4. Dijkstra returns a path
# ---------------------------------------------------------------------------

def test_dijkstra_returns_a_path(nagpur_graph, reachable_node_pair):
    """The returned route should be a non-empty list of nodes."""
    origin, destination = reachable_node_pair

    result = shortest_path(nagpur_graph, origin, destination)

    assert isinstance(result["route_nodes"], list)
    assert len(result["route_nodes"]) >= 2
    assert result["route_nodes"][0] == origin
    assert result["route_nodes"][-1] == destination


# ---------------------------------------------------------------------------
# 5. Travel time is positive
# ---------------------------------------------------------------------------

def test_travel_time_is_positive(nagpur_graph, reachable_node_pair):
    """A real multi-node route should have a positive travel time."""
    origin, destination = reachable_node_pair

    result = shortest_path(nagpur_graph, origin, destination)

    assert result["total_time_seconds"] is not None
    assert result["total_time_seconds"] > 0


# ---------------------------------------------------------------------------
# 6. Distance is positive
# ---------------------------------------------------------------------------

def test_distance_is_positive(nagpur_graph, reachable_node_pair):
    """A real multi-node route should have a positive total distance."""
    origin, destination = reachable_node_pair

    result = shortest_path(nagpur_graph, origin, destination)

    assert result["total_distance_meters"] is not None
    assert result["total_distance_meters"] > 0


# ---------------------------------------------------------------------------
# 7. Origin and destination are handled correctly
# ---------------------------------------------------------------------------

def test_same_origin_and_destination(nagpur_graph):
    """Routing a node to itself should return a trivial zero-cost route."""
    origin = next(iter(nagpur_graph.nodes))

    result = shortest_path(nagpur_graph, origin, origin)

    assert result["is_reachable"] is True
    assert result["route_nodes"] == [origin]
    assert result["total_time_seconds"] == 0.0
    assert result["total_distance_meters"] == 0.0


def test_route_starts_and_ends_at_correct_nodes(nagpur_graph, reachable_node_pair):
    """The first and last node in the route must match the requested
    origin and destination."""
    origin, destination = reachable_node_pair

    result = shortest_path(nagpur_graph, origin, destination)

    assert result["route_nodes"][0] == origin
    assert result["route_nodes"][-1] == destination


# ---------------------------------------------------------------------------
# 8. Invalid nodes are handled cleanly
# ---------------------------------------------------------------------------

def test_invalid_origin_node(nagpur_graph):
    """A non-existent origin node should not crash the routing function."""
    fake_node = "this_node_does_not_exist_in_the_graph"
    real_destination = next(iter(nagpur_graph.nodes))

    result = shortest_path(nagpur_graph, fake_node, real_destination)

    assert result["is_reachable"] is False
    assert result["route_nodes"] is None
    assert result["error"] is not None


def test_invalid_destination_node(nagpur_graph):
    """A non-existent destination node should not crash the routing
    function."""
    real_origin = next(iter(nagpur_graph.nodes))
    fake_node = "this_node_does_not_exist_in_the_graph"

    result = shortest_path(nagpur_graph, real_origin, fake_node)

    assert result["is_reachable"] is False
    assert result["route_nodes"] is None
    assert result["error"] is not None


def test_invalid_coordinates_are_handled_cleanly(nagpur_graph):
    """Out-of-range coordinates should not crash route_between_coordinates."""
    result = route_between_coordinates(
        nagpur_graph,
        origin_lat=999.0,   # invalid latitude
        origin_lon=79.08,
        destination_lat=21.16,
        destination_lon=79.10,
    )

    assert result["is_reachable"] is False
    assert result["error"] is not None


# ---------------------------------------------------------------------------
# 9. Unreachable nodes are handled correctly
# ---------------------------------------------------------------------------

def test_unreachable_node_is_handled_correctly(graph_with_isolated_node):
    """A genuinely isolated node (no edges) should be reported as
    unreachable, not crash or return a fake route."""
    graph_copy, isolated_node_id = graph_with_isolated_node
    real_node = next(
        n for n in graph_copy.nodes if n != isolated_node_id
    )

    result = shortest_path(graph_copy, real_node, isolated_node_id)

    assert result["is_reachable"] is False
    assert result["route_nodes"] is None
    assert result["total_time_seconds"] is None
    assert result["total_distance_meters"] is None
    assert result["error"] is not None