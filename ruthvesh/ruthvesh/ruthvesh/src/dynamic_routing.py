"""
dynamic_routing.py

Connects the already-verified ML -> dynamic-graph pipeline
(ml_dynamic_graph.py) to Ruthvesh's already-existing Dijkstra routing
(routing.py) — WITHOUT modifying either module.

This module exists only because nothing previously actually called both
in sequence for a single query. routing.py already fully implements a
weight_mode="base"/"dynamic" mechanism (see routing.py's module
docstring and shortest_path()); this module does not reimplement any of
that. It simply:

    1. Builds a dynamic graph copy via
       ml_dynamic_graph.build_dynamic_graph_from_files() (unmodified),
    2. Calls routing.shortest_path() / routing.route_between_coordinates()
       (unmodified) on that copy with weight_mode="dynamic",
    3. And, for convenience, offers a side-by-side base-vs-dynamic
       comparison for the same origin/destination.

Required architecture (from Prompt 6), as implemented by composing the
existing modules:

    Base Graph -> routing.shortest_path(weight_mode="base")  -> travel_time
    ML observations -> ml_dynamic_graph.build_dynamic_graph_from_files()
        -> dynamic Graph -> routing.shortest_path(weight_mode="dynamic")
        -> dynamic_travel_time

No second Dijkstra implementation is created — every path computation in
this module goes through routing.shortest_path(), which internally calls
networkx.dijkstra_path() exactly once, the same as it always has.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import networkx as nx

try:
    from src import ml_dynamic_graph
    from src import routing
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src import ml_dynamic_graph
    from src import routing


def route_on_dynamic_graph(
    base_graph: nx.MultiDiGraph,
    junction_observation_files: Mapping[str, Union[str, Path]],
    origin_node: Any,
    destination_node: Any,
) -> Dict[str, Any]:
    """
    Build the ML-congestion dynamic graph from real observation files
    (via ml_dynamic_graph.build_dynamic_graph_from_files(), unmodified)
    and route on it using the existing Dijkstra implementation
    (routing.shortest_path(), unmodified, weight_mode="dynamic").

    `base_graph` itself is never mutated — build_dynamic_graph_from_files
    already guarantees this (it works on graph.copy()); this function
    adds no further mutation.

    Parameters
    ----------
    base_graph : networkx.MultiDiGraph
        The processed Nagpur road graph (or any graph junction_mapping
        can resolve junctions against).
    junction_observation_files : mapping of junction_id -> file path
        Real ML observation JSON files, e.g.
        {"junction_1": ".../junction_1_latest.json",
         "junction_2": ".../junction_2_latest.json"}.
    origin_node, destination_node : Any
        Graph node IDs to route between.

    Returns
    -------
    dict with keys:
        - route_result : the dict returned by routing.shortest_path()
          (weight_mode="dynamic")
        - dynamic_graph_result : the ml_dynamic_graph.DynamicGraphResult
          used to build the graph that was routed on (exposes which
          junctions/edges were actually updated, for inspection).
    """
    dynamic_graph_result = ml_dynamic_graph.build_dynamic_graph_from_files(
        base_graph, junction_observation_files
    )
    route_result = routing.shortest_path(
        dynamic_graph_result.graph,
        origin_node,
        destination_node,
        weight_mode=routing.WEIGHT_MODE_DYNAMIC,
    )
    return {
        "route_result": route_result,
        "dynamic_graph_result": dynamic_graph_result,
    }


def compare_base_vs_dynamic_route(
    base_graph: nx.MultiDiGraph,
    junction_observation_files: Mapping[str, Union[str, Path]],
    origin_node: Any,
    destination_node: Any,
) -> Dict[str, Any]:
    """
    Route the same origin/destination on both the unmodified base graph
    (weight_mode="base") and the ML-congestion dynamic graph
    (weight_mode="dynamic"), for direct side-by-side comparison.

    `base_graph` is used, read-only, for both calls: the base-mode call
    routes on it directly (routing.shortest_path never mutates its
    input), and the dynamic-mode call only ever touches a COPY of it
    (produced inside build_dynamic_graph_from_files).

    Returns
    -------
    dict with keys:
        - base : dict returned by routing.shortest_path(weight_mode="base")
        - dynamic : dict returned by routing.shortest_path(weight_mode="dynamic")
          on the dynamic graph copy
        - dynamic_graph_result : the ml_dynamic_graph.DynamicGraphResult
        - path_changed : bool, True if the ordered route_nodes differ
          between base and dynamic (None if either route is unreachable)
    """
    base_result = routing.shortest_path(
        base_graph, origin_node, destination_node,
        weight_mode=routing.WEIGHT_MODE_BASE,
    )

    dynamic = route_on_dynamic_graph(
        base_graph, junction_observation_files, origin_node, destination_node
    )
    dynamic_result = dynamic["route_result"]

    path_changed = None
    if base_result.get("is_reachable") and dynamic_result.get("is_reachable"):
        path_changed = base_result["route_nodes"] != dynamic_result["route_nodes"]

    return {
        "base": base_result,
        "dynamic": dynamic_result,
        "dynamic_graph_result": dynamic["dynamic_graph_result"],
        "path_changed": path_changed,
    }