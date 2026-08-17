"""
ml_dynamic_graph.py

Applies already-computed NovaRoute ML congestion_factor values to a COPY
of the Ruthvesh road-network graph, producing a 'dynamic_travel_time'
edge attribute on EVERY edge — WITHOUT touching Dijkstra/routing.py,
without touching the original graph, and without re-deriving
congestion_factor from vehicle counts.

Pipeline this module wires together (all pre-existing, unmodified):

    ML observation JSON (ml_integration.load_ml_observation /
    extract_ml_fields)
        -> congestion_factor (read verbatim, never recomputed)
    junction_id
        -> junction_mapping.build_junction_report()
        -> verified nearest graph node + real incident OSM segments
           (junction_mapping.PROTOTYPE_JUNCTION_COORDINATES; live
           graph.out_edges/in_edges lookup, nothing hand-typed)
    graph.copy()
        -> for EVERY edge (baseline pass):
               dynamic_travel_time = travel_time
        -> then, for every mapped/ML-covered segment (override pass):
               dynamic_travel_time = travel_time * congestion_factor
           (via congestion.calculate_dynamic_travel_time(), reused
           unmodified — this module does not reimplement the multiply)

Why every edge, not just mapped ones (Prompt 6A)
--------------------------------------------------
Earlier, only ML-covered edges received dynamic_travel_time at all. Since
NetworkX's dijkstra_path silently defaults a missing weight attribute to
1 (see networkx.algorithms.shortest_paths.weighted._weight_function),
weight_mode="dynamic" routing over a graph where most edges have no
dynamic_travel_time effectively minimizes HOP COUNT through the
uncovered majority of the graph, not real travel time — a routing
correctness bug, not a congestion effect. The fix is a two-pass
initialization: every edge in the graph COPY first gets
dynamic_travel_time := travel_time (physically meaningful baseline for
edges with no ML observation yet), and only afterwards do ML-covered
edges get that baseline overwritten with travel_time * congestion_factor.
This guarantees every edge in the returned graph has a real,
physically-meaningful dynamic_travel_time, so weight_mode="dynamic"
Dijkstra never falls back to NetworkX's implicit weight=1 for any edge
that has a travel_time to begin with.

Design notes
------------
- This module deliberately does NOT use
  ml_integration.JUNCTION_TO_SEGMENTS. That dict is intentionally empty
  in ml_integration.py ("Option B": a human-verified static mapping that
  has not yet been populated). Instead, this module uses
  junction_mapping.py's live nearest-node + incident-edges lookup
  ("Option A"-style), which is the mapping that has actually been
  verified so far (Prompt 4). If JUNCTION_TO_SEGMENTS is populated in
  the future, that is a deliberate, separate decision — not something
  this module silently falls back to.
- junction_3 is a demo/coverage junction with no real ML observation
  (see junction_mapping.py's module docstring: it snaps ~6.5 km away,
  far outside the dense network). Callers simply never pass an
  observation for it; this module never fabricates a congestion factor
  for it — its edges get only the baseline dynamic_travel_time ==
  travel_time, exactly like every other unmapped edge, and it never
  iterates PROTOTYPE_JUNCTION_COORDINATES itself — only junction ids
  actually present in the caller-supplied observations get an ML-derived
  override.
- This module does NOT modify NovaRoute_AI, routing.py, Dijkstra, YOLO,
  ByteTrack, vehicle_counter.py, or risk_scorer.py. It does not connect
  to cost_matrix.py or Dijkstra at all — that is a later, separate
  stage.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, Union

import networkx as nx

try:
    from src.config import WEIGHT_ATTRIBUTE
    from src.congestion import (
        DYNAMIC_WEIGHT_ATTRIBUTE,
        calculate_dynamic_travel_time,
        get_segment_id,
    )
    from src.junction_mapping import (
        PROTOTYPE_JUNCTION_COORDINATES,
        build_junction_report,
    )
    from src.ml_integration import (
        MLObservationError,
        MLObservationFields,
        extract_ml_fields,
        load_ml_observation,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import WEIGHT_ATTRIBUTE
    from src.congestion import (
        DYNAMIC_WEIGHT_ATTRIBUTE,
        calculate_dynamic_travel_time,
        get_segment_id,
    )
    from src.junction_mapping import (
        PROTOTYPE_JUNCTION_COORDINATES,
        build_junction_report,
    )
    from src.ml_integration import (
        MLObservationError,
        MLObservationFields,
        extract_ml_fields,
        load_ml_observation,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
# All subclass ValueError, matching the rest of the codebase's error
# handling style (congestion.py / travel_time.py / ml_integration.py).

class UnmappedJunctionError(ValueError):
    """Raised when an ML observation refers to a junction_id that has no
    entry in junction_mapping.PROTOTYPE_JUNCTION_COORDINATES. This module
    never guesses a mapping for an unrecognized junction."""


class JunctionObservationMismatchError(ValueError):
    """Raised when the junction_id a caller says they are loading an
    observation for does not match the road_segment_id actually inside
    that observation's JSON. Catches copy/paste mistakes (e.g. loading
    junction_2's file but labeling it junction_1) early and loudly."""


class SegmentCongestionConflictError(ValueError):
    """Raised when two different junctions' verified incident-segment
    sets both include the same directed edge with different
    congestion_factor values. Prevents applying a factor twice with two
    different values to the same edge."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class DynamicGraphResult:
    """Everything a caller needs to inspect what this module did, without
    having to re-derive it themselves."""

    graph: nx.MultiDiGraph
    junction_ids_applied: List[str]
    updated_segment_ids_by_junction: Dict[str, List[str]] = field(default_factory=dict)
    congestion_factor_by_junction: Dict[str, float] = field(default_factory=dict)

    @property
    def total_edges_updated(self) -> int:
        return sum(len(v) for v in self.updated_segment_ids_by_junction.values())


# ---------------------------------------------------------------------------
# Step 1-2: Load ML observations and extract (junction_id, congestion_factor)
# ---------------------------------------------------------------------------

def load_junction_observations(
    sources: Mapping[str, Union[str, Path, Dict[str, Any]]],
) -> Dict[str, MLObservationFields]:
    """
    Load and validate one real ML observation per junction id.

    Parameters
    ----------
    sources : mapping of junction_id -> path (str/Path) or already-parsed
        dict, e.g.:
            {
                "junction_1": r"C:\\...\\junction_1_latest.json",
                "junction_2": r"C:\\...\\junction_2_latest.json",
            }
        junction_3 is deliberately never included here — it currently
        has no real ML observation (see module docstring).

    Returns
    -------
    dict of junction_id -> MLObservationFields
        congestion_factor is read verbatim from the JSON — never
        recomputed from vehicle_counts.

    Raises
    ------
    ml_integration.MLObservationError
        If a file is missing, not valid JSON, or fails schema
        validation (reused unmodified from ml_integration.py).
    JunctionObservationMismatchError
        If the caller-supplied junction_id key does not match the
        'road_segment_id' actually found inside that observation's JSON.
    """
    observations: Dict[str, MLObservationFields] = {}

    for junction_id, source in sources.items():
        obs = load_ml_observation(source)
        fields = extract_ml_fields(obs)

        if fields.road_segment_id != junction_id:
            raise JunctionObservationMismatchError(
                f"Observation loaded under junction_id={junction_id!r} "
                f"actually has road_segment_id={fields.road_segment_id!r} "
                "inside its JSON. Refusing to apply an observation to a "
                "junction it does not claim to belong to — check that "
                "the file paths/keys line up."
            )

        observations[junction_id] = fields
        logger.info(
            "Loaded real ML observation for %s: congestion_factor=%.9f "
            "(risk_level=%s).",
            junction_id, fields.congestion_factor, fields.risk_level,
        )

    return observations


# ---------------------------------------------------------------------------
# Step 3: Resolve each junction's verified incident segments (junction_mapping)
# ---------------------------------------------------------------------------

def _segment_congestion_map(
    graph: nx.MultiDiGraph,
    junction_observations: Mapping[str, MLObservationFields],
) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    """
    Build {segment_id: congestion_factor}, using junction_mapping's
    verified nearest-node -> live incident-edges lookup for each
    junction present in `junction_observations`.

    Returns
    -------
    (segment_congestion, segment_ids_by_junction)
        segment_congestion : dict of segment_id -> congestion_factor
        segment_ids_by_junction : dict of junction_id -> list of
            segment_id, for per-junction reporting.

    Raises
    ------
    UnmappedJunctionError
        If a junction_id in `junction_observations` has no entry in
        junction_mapping.PROTOTYPE_JUNCTION_COORDINATES.
    SegmentCongestionConflictError
        If two junctions' verified segment sets overlap on the same
        directed edge with different congestion_factor values.
    """
    segment_congestion: Dict[str, float] = {}
    segment_source: Dict[str, str] = {}
    segment_ids_by_junction: Dict[str, List[str]] = {}

    for junction_id, fields in junction_observations.items():
        if junction_id not in PROTOTYPE_JUNCTION_COORDINATES:
            raise UnmappedJunctionError(
                f"Junction {junction_id!r} has a real ML observation but "
                "no entry in junction_mapping.PROTOTYPE_JUNCTION_COORDINATES "
                f"(known junctions: {sorted(PROTOTYPE_JUNCTION_COORDINATES)}). "
                "Refusing to guess a coordinate/mapping for it."
            )

        # junction_mapping.build_junction_report() re-derives the nearest
        # node and incident segments live from the graph every call — the
        # verified Prompt-4 mapping, not a hand-typed/cached list.
        report = build_junction_report(graph, junction_id)

        seg_ids_this_junction: List[str] = []
        for u, v, key in report.selected_segments:
            segment_id = get_segment_id(u, v, key)
            existing = segment_congestion.get(segment_id)

            if existing is not None and existing != fields.congestion_factor:
                raise SegmentCongestionConflictError(
                    f"Segment {segment_id!r} was assigned "
                    f"congestion_factor={existing!r} by junction "
                    f"{segment_source[segment_id]!r} and "
                    f"congestion_factor={fields.congestion_factor!r} by "
                    f"junction {junction_id!r}. These junctions' verified "
                    "incident-segment sets overlap with conflicting ML "
                    "readings; refusing to apply either silently."
                )

            segment_congestion[segment_id] = fields.congestion_factor
            segment_source[segment_id] = junction_id
            seg_ids_this_junction.append(segment_id)

        segment_ids_by_junction[junction_id] = seg_ids_this_junction

    return segment_congestion, segment_ids_by_junction


# ---------------------------------------------------------------------------
# Step 4-9: Build the dynamic graph copy
# ---------------------------------------------------------------------------

def apply_ml_congestion_to_graph(
    graph: nx.MultiDiGraph,
    junction_observations: Mapping[str, MLObservationFields],
    base_weight_attribute: str = WEIGHT_ATTRIBUTE,
    dynamic_weight_attribute: str = DYNAMIC_WEIGHT_ATTRIBUTE,
) -> DynamicGraphResult:
    """
    Build a COPY of `graph` where EVERY edge has 'dynamic_travel_time'
    set: baseline dynamic_travel_time == travel_time for every edge,
    then overwritten with travel_time * congestion_factor for edges
    junction_mapping verifies as incident to a junction present in
    `junction_observations`.

    The original `graph` object is never mutated (graph.copy() is used,
    consistent with congestion.apply_congestion_to_graph() and
    travel_time.add_travel_times_to_graph()'s existing pattern in this
    codebase). Every edge's existing `base_weight_attribute`
    ("travel_time") is left completely untouched.

    congestion_factor is taken verbatim from each junction's ML
    observation — this function does not call
    congestion.calculate_congestion_factor() and does not look at
    vehicle_counts at all.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The processed Nagpur road graph (must already have
        base_weight_attribute set on its edges).
    junction_observations : dict of junction_id -> MLObservationFields
        As returned by load_junction_observations(). Only junctions
        present here are touched; e.g. omitting "junction_3" means its
        incident edges are left completely unmodified.
    base_weight_attribute : str
        Edge attribute holding free-flow travel time (defaults to
        config.WEIGHT_ATTRIBUTE, "travel_time"). Never overwritten.
    dynamic_weight_attribute : str
        Edge attribute to store the congestion-adjusted travel time
        under (defaults to congestion.DYNAMIC_WEIGHT_ATTRIBUTE,
        "dynamic_travel_time").

    Returns
    -------
    DynamicGraphResult

    Raises
    ------
    UnmappedJunctionError
        If an observation's junction has no verified mapping.
    SegmentCongestionConflictError
        If two junctions' segments overlap with conflicting factors.
    """
    segment_congestion, segment_ids_by_junction = _segment_congestion_map(
        graph, junction_observations
    )

    # Map segment_id -> (u, v, key) once, from the ORIGINAL graph, purely
    # to locate which edge each segment_id refers to. This does not
    # mutate `graph` — it is a read-only lookup.
    edge_lookup: Dict[str, Tuple[Any, Any, Any]] = {
        get_segment_id(u, v, k): (u, v, k) for u, v, k in graph.edges(keys=True)
    }

    graph_copy = graph.copy()

    # Baseline initialization: EVERY edge in the copy gets
    # dynamic_travel_time = travel_time first. This is what makes
    # weight_mode="dynamic" routing physically meaningful for the whole
    # graph rather than only the currently ML-covered fraction of it —
    # without this, NetworkX's dijkstra_path silently defaults any edge
    # missing the requested weight attribute to 1 (see
    # networkx.algorithms.shortest_paths.weighted._weight_function),
    # which turns "dynamic" routing into hop-count minimization over
    # every unmapped edge instead of travel-time minimization. Only the
    # copy is touched — base_weight_attribute ("travel_time") itself is
    # never read as a target, only as a source value, and the original
    # `graph` object is never touched by this loop.
    missing_base_travel_time_edges: List[Tuple[Any, Any, Any]] = []
    for u, v, key, edge_data in graph_copy.edges(keys=True, data=True):
        base_travel_time = edge_data.get(base_weight_attribute)
        if base_travel_time is None:
            # No base travel_time at all on this edge — cannot derive a
            # meaningful baseline dynamic_travel_time. Leave
            # dynamic_travel_time unset here (as before) rather than
            # inventing a number; tracked so it can be surfaced, not
            # silently swallowed.
            missing_base_travel_time_edges.append((u, v, key))
            continue
        edge_data[dynamic_weight_attribute] = base_travel_time

    if missing_base_travel_time_edges:
        logger.warning(
            "%d edge(s) have no '%s' at all and therefore received no "
            "baseline '%s' either (e.g. %s...); weight_mode='dynamic' "
            "routing will still fall back to NetworkX's implicit "
            "weight=1 for exactly these edges.",
            len(missing_base_travel_time_edges), base_weight_attribute,
            dynamic_weight_attribute, missing_base_travel_time_edges[:3],
        )

    # segment_id -> junction_id, so a single update pass can attribute
    # each successfully-updated segment back to its owning junction
    # without a second graph traversal.
    junction_of_segment: Dict[str, str] = {
        sid: jid for jid, sids in segment_ids_by_junction.items() for sid in sids
    }
    updated_segment_ids_by_junction: Dict[str, List[str]] = {
        jid: [] for jid in junction_observations
    }

    # Second pass: OVERWRITE dynamic_travel_time, but only for the
    # segments verified as incident to an ML-covered junction. Every
    # other edge keeps the baseline (== travel_time) set above.
    for segment_id, factor in segment_congestion.items():
        u, v, key = edge_lookup[segment_id]  # guaranteed present: segment
        # ids came from junction_mapping, which reads live from this same
        # graph (see junction_mapping.get_incident_segments's guarantee).
        edge_data = graph_copy[u][v][key]

        base_travel_time = edge_data.get(base_weight_attribute)
        if base_travel_time is None:
            logger.warning(
                "Skipping ML dynamic_travel_time override for edge "
                "(%s, %s, %s): missing '%s'.", u, v, key, base_weight_attribute,
            )
            continue

        edge_data[dynamic_weight_attribute] = calculate_dynamic_travel_time(
            base_travel_time, factor
        )
        updated_segment_ids_by_junction[junction_of_segment[segment_id]].append(segment_id)

    congestion_factor_by_junction = {
        jid: fields.congestion_factor for jid, fields in junction_observations.items()
    }

    logger.info(
        "Applied ML congestion to %d segment(s) across %d junction(s): %s",
        sum(len(v) for v in updated_segment_ids_by_junction.values()),
        len(junction_observations),
        {jid: len(v) for jid, v in updated_segment_ids_by_junction.items()},
    )

    return DynamicGraphResult(
        graph=graph_copy,
        junction_ids_applied=list(junction_observations.keys()),
        updated_segment_ids_by_junction=updated_segment_ids_by_junction,
        congestion_factor_by_junction=congestion_factor_by_junction,
    )


# ---------------------------------------------------------------------------
# Convenience: files -> dynamic graph, in one call
# ---------------------------------------------------------------------------

def build_dynamic_graph_from_files(
    graph: nx.MultiDiGraph,
    junction_observation_files: Mapping[str, Union[str, Path]],
) -> DynamicGraphResult:
    """
    Convenience wrapper: load ML observation JSON files for one or more
    junctions and apply them to a copy of `graph` in one call.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The processed Nagpur road graph, already loaded (e.g. via
        graph_utils.load_graph()).
    junction_observation_files : mapping of junction_id -> file path
        e.g. {"junction_1": ".../junction_1_latest.json",
              "junction_2": ".../junction_2_latest.json"}
        junction_3 is intentionally omitted by callers until it has a
        real ML observation.

    Returns
    -------
    DynamicGraphResult
    """
    observations = load_junction_observations(junction_observation_files)
    return apply_ml_congestion_to_graph(graph, observations)


if __name__ == "__main__":
    from src.graph_utils import load_graph
    from src.config import PROCESSED_GRAPH_PATH

    # Edit these two paths to point at your real NovaRoute ML output
    # files before running this module directly. junction_3 has no real
    # ML observation yet, so it is intentionally not listed here.
    JUNCTION_OBSERVATION_FILES = {
        "junction_1": r"C:\Users\koush\OneDrive\Desktop\NovaRoute_AI\ml\outputs\metrics\junction_1_latest.json",
        "junction_2": r"C:\Users\koush\OneDrive\Desktop\NovaRoute_AI\ml\outputs\metrics\junction_2_latest.json",
    }

    base_graph = load_graph(PROCESSED_GRAPH_PATH)
    result = build_dynamic_graph_from_files(base_graph, JUNCTION_OBSERVATION_FILES)

    print(f"Junctions applied: {result.junction_ids_applied}")
    for jid in result.junction_ids_applied:
        print(
            f"  {jid}: congestion_factor="
            f"{result.congestion_factor_by_junction[jid]:.9f}, "
            f"edges_updated={len(result.updated_segment_ids_by_junction[jid])}"
        )
    print(f"Total edges updated: {result.total_edges_updated}")
    print(
        "Original graph unchanged: "
        f"{DYNAMIC_WEIGHT_ATTRIBUTE not in next(iter(base_graph.edges(data=True)))[2]}"
    )