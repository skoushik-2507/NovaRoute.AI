"""
junction_mapping.py

PROTOTYPE / SIMULATED junction-to-OSM-segment mapping for the hackathon
demo.

============================================================================
IMPORTANT — READ BEFORE USING
============================================================================
The coordinates in PROTOTYPE_JUNCTION_COORDINATES below are the same
illustrative/demo coordinates already present in src/pipeline_demo.py
(SAMPLE_HIGH_RISK_JUNCTIONS). They are NOT verified real-world camera
locations. They are used here, by explicit hackathon-scope decision, as
SIMULATED camera/junction positions so the ML -> routing integration can
be demonstrated end-to-end.

Do not present the output of this module as confirmed real-world
mapping. Every report this module produces is labeled "prototype":
True for exactly this reason.
============================================================================

What this module does:
- Snaps each simulated junction coordinate to its nearest real graph node
  using graph_utils.find_nearest_node() (existing, unmodified utility —
  this module does not reimplement nearest-node search).
- Lists the real OSM edges incident to that node, read live from the
  graph (graph.out_edges / graph.in_edges), so no edge (u, v, key)
  triple is ever hand-typed or invented. Every edge this module returns
  is guaranteed to exist in the graph, because it was read directly out
  of the graph.
- Produces a per-junction report (junction_id, latitude, longitude,
  nearest_graph_node, distance_from_coordinate_to_node_meters,
  selected_segments) suitable for manual inspection before it is ever
  trusted for congestion integration.

What this module deliberately does NOT do (out of scope for this stage):
- It does NOT build a {segment_id: congestion_factor} map.
- It does NOT create a dynamic_travel_time graph copy.
- It does NOT call congestion.calculate_dynamic_travel_time() or
  congestion.apply_congestion_to_graph().
- It does NOT modify routing.py, Dijkstra, or the graph in any way —
  every function here is read-only with respect to the graph.
That is the next stage (dynamic congestion integration), not this one.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

try:
    from src.graph_utils import find_nearest_node
    from src.congestion import get_segment_id
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.graph_utils import find_nearest_node
    from src.congestion import get_segment_id


class UnknownJunctionError(ValueError):
    """Raised when a junction id has no entry in
    PROTOTYPE_JUNCTION_COORDINATES. This module never guesses a
    coordinate for an unrecognized junction id."""


class JunctionNodeNotFoundError(ValueError):
    """Raised when graph_utils.find_nearest_node() cannot resolve a
    graph node for a junction's simulated coordinate (e.g. invalid
    coordinate, or an empty graph)."""


# ---------------------------------------------------------------------------
# PROTOTYPE / SIMULATED junction coordinates
# ---------------------------------------------------------------------------
# Source: these are the exact same demo coordinates already present in
# src/pipeline_demo.py's SAMPLE_HIGH_RISK_JUNCTIONS. They are reused here
# by explicit instruction, for the hackathon prototype only, as
# simulated camera/junction locations — not as confirmed real-world
# positions. See module docstring above.
PROTOTYPE_JUNCTION_COORDINATES: Dict[str, Tuple[float, float]] = {
    "junction_1": (21.1600, 79.1000),
    "junction_2": (21.1200, 79.0600),
    "junction_3": (21.2200, 79.2200),
}


# ---------------------------------------------------------------------------
# Report type
# ---------------------------------------------------------------------------

@dataclass
class JunctionMappingReport:
    """Read-only report of how a simulated junction coordinate maps onto
    the real graph. Every field is derived live from the graph at call
    time — nothing here is hardcoded."""

    junction_id: str
    latitude: float
    longitude: float
    nearest_graph_node: Any
    distance_to_node_meters: float
    selected_segments: List[Tuple[Any, Any, Any]] = field(default_factory=list)
    is_prototype: bool = True

    def selected_segment_ids(self) -> List[str]:
        """Segment IDs in the same 'u_v_key' string format used
        elsewhere in the codebase (see congestion.get_segment_id),
        derived from selected_segments — never hand-typed."""
        return [get_segment_id(u, v, k) for u, v, k in self.selected_segments]


# ---------------------------------------------------------------------------
# Distance helper (reporting only — NOT used for routing/Dijkstra)
# ---------------------------------------------------------------------------

def _haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two lat/lon points, in meters.

    Used here only to report how far a simulated junction coordinate is
    from the graph node it snapped to (a data-quality signal for manual
    inspection). This is NOT used for routing or Dijkstra anywhere —
    routing.py exclusively uses real road-network travel time, unchanged
    by this module.
    """
    earth_radius_m = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_m * c


# ---------------------------------------------------------------------------
# Core mapping logic
# ---------------------------------------------------------------------------

def get_junction_coordinate(junction_id: str) -> Tuple[float, float]:
    """
    Look up the PROTOTYPE/SIMULATED (latitude, longitude) for a junction
    id.

    Raises
    ------
    UnknownJunctionError
        If junction_id is not in PROTOTYPE_JUNCTION_COORDINATES.
    """
    if junction_id not in PROTOTYPE_JUNCTION_COORDINATES:
        raise UnknownJunctionError(
            f"Unknown junction id {junction_id!r}: no entry in "
            f"PROTOTYPE_JUNCTION_COORDINATES. Known junction ids: "
            f"{sorted(PROTOTYPE_JUNCTION_COORDINATES)}."
        )
    return PROTOTYPE_JUNCTION_COORDINATES[junction_id]


def get_incident_segments(
    graph: nx.MultiDiGraph,
    node: Any,
) -> List[Tuple[Any, Any, Any]]:
    """
    List every real OSM edge (segment) incident to `node`, in both
    directions, read live from the graph.

    Because the graph is a directed MultiDiGraph, an intersection's
    "incident segments" include both edges leaving the node
    (graph.out_edges) and edges arriving at it (graph.in_edges) — these
    are different edges (see congestion.get_segment_id's docstring on
    directionality).

    Parameters
    ----------
    graph : networkx.MultiDiGraph
    node : Any
        A node that must already exist in `graph` (typically the output
        of find_nearest_node()).

    Returns
    -------
    list of (u, v, key)
        Every incident edge triple, exactly as it exists in the graph.
        Never hand-constructed.

    Raises
    ------
    JunctionNodeNotFoundError
        If `node` is not present in `graph`.
    """
    if node not in graph:
        raise JunctionNodeNotFoundError(
            f"Node {node!r} is not present in the supplied graph; cannot "
            "list incident segments."
        )

    out_edges = list(graph.out_edges(node, keys=True))
    in_edges = list(graph.in_edges(node, keys=True))
    return out_edges + in_edges


def build_junction_report(
    graph: nx.MultiDiGraph,
    junction_id: str,
) -> JunctionMappingReport:
    """
    Build a full, inspectable mapping report for one PROTOTYPE junction:
    simulated coordinate -> nearest real graph node -> real incident
    edges.

    This function is read-only with respect to `graph` — it never
    mutates it, and it never invents a node or edge that the graph does
    not actually contain.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The loaded road-network graph (e.g. via graph_utils.load_graph).
    junction_id : str
        One of the keys in PROTOTYPE_JUNCTION_COORDINATES.

    Returns
    -------
    JunctionMappingReport

    Raises
    ------
    UnknownJunctionError
        If junction_id is not a known prototype junction.
    JunctionNodeNotFoundError
        If graph_utils.find_nearest_node() cannot resolve a node for the
        junction's simulated coordinate (invalid coordinate, or an empty
        graph — find_nearest_node returns None in these cases, per its
        own docstring).
    """
    latitude, longitude = get_junction_coordinate(junction_id)

    node = find_nearest_node(graph, latitude, longitude)
    if node is None:
        raise JunctionNodeNotFoundError(
            f"graph_utils.find_nearest_node() could not resolve a graph "
            f"node for junction {junction_id!r} at "
            f"(lat={latitude}, lon={longitude}). See prior log output "
            "from find_nearest_node for the specific reason."
        )

    node_data = graph.nodes[node]
    node_lat = float(node_data["y"])
    node_lon = float(node_data["x"])
    distance_m = _haversine_distance_meters(latitude, longitude, node_lat, node_lon)

    segments = get_incident_segments(graph, node)

    return JunctionMappingReport(
        junction_id=junction_id,
        latitude=latitude,
        longitude=longitude,
        nearest_graph_node=node,
        distance_to_node_meters=distance_m,
        selected_segments=segments,
    )


def build_all_junction_reports(
    graph: nx.MultiDiGraph,
    junction_ids: Optional[List[str]] = None,
) -> List[JunctionMappingReport]:
    """
    Build mapping reports for multiple (default: all) prototype
    junctions.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
    junction_ids : list of str or None
        Defaults to every key in PROTOTYPE_JUNCTION_COORDINATES, in
        insertion order (junction_1, junction_2, junction_3).

    Returns
    -------
    list of JunctionMappingReport
    """
    if junction_ids is None:
        junction_ids = list(PROTOTYPE_JUNCTION_COORDINATES.keys())
    return [build_junction_report(graph, jid) for jid in junction_ids]


# ---------------------------------------------------------------------------
# Pretty-printing (for manual inspection, per Prompt 4 instructions)
# ---------------------------------------------------------------------------

def print_junction_report(report: JunctionMappingReport) -> None:
    """Print one junction's mapping report in a human-inspectable form."""
    print(f"\n=== {report.junction_id} (PROTOTYPE/SIMULATED mapping) ===")
    print(f"  latitude:                 {report.latitude}")
    print(f"  longitude:                {report.longitude}")
    print(f"  nearest_graph_node:       {report.nearest_graph_node}")
    print(f"  distance_to_node_meters:  {report.distance_to_node_meters:.2f}")
    print(f"  selected_segments ({len(report.selected_segments)}):")
    for seg_id in report.selected_segment_ids():
        print(f"    - {seg_id}")


def print_all_junction_reports(graph: nx.MultiDiGraph) -> List[JunctionMappingReport]:
    """Build and print reports for all prototype junctions. Returns the
    reports for further programmatic use (e.g. by tests)."""
    print(
        "PROTOTYPE/SIMULATED junction -> OSM segment mapping.\n"
        "These coordinates are demo/simulated locations, NOT verified "
        "real-world camera positions. Every node and edge below was "
        "read live from the actual loaded graph."
    )
    reports = build_all_junction_reports(graph)
    for report in reports:
        print_junction_report(report)
    return reports


if __name__ == "__main__":
    from src.graph_utils import load_graph

    graph = load_graph()
    print_all_junction_reports(graph)