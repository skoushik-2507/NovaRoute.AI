"""
congestion.py

Interface and calculation logic for dynamic, congestion-adjusted travel
times on the Nagpur road graph.

Architecture:

    base_travel_time  x  congestion_factor  =  dynamic_travel_time

This module defines HOW vehicle counts become a congestion factor, and
HOW that factor is applied to produce a new 'dynamic_travel_time' edge
attribute — separate from the existing 'travel_time' (base) attribute.

This module does NOT:
- Run YOLO / ByteTrack or any vehicle-detection code.
- Modify routing.py or the Dijkstra algorithm itself.

It only provides a clean data interface that a future traffic-detection
module (vehicle counts per road segment) can plug into, and a function
to turn those counts into adjusted edge weights that routing.py can
later be pointed at (by passing weight_attribute="dynamic_travel_time"
to shortest_path()).
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx

try:
    from src.config import WEIGHT_ATTRIBUTE, DEFAULT_CONGESTION_FACTOR
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.config import WEIGHT_ATTRIBUTE, DEFAULT_CONGESTION_FACTOR

logger = logging.getLogger(__name__)

# Name of the new edge attribute that stores congestion-adjusted travel
# time. Kept separate from WEIGHT_ATTRIBUTE ("travel_time") so the base,
# free-flow travel time is never overwritten.
DYNAMIC_WEIGHT_ATTRIBUTE = "dynamic_travel_time"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CongestionConfig:
    """
    Configuration for converting a vehicle count into a congestion
    factor.

    The congestion factor is calculated using a simplified version of
    the well-known BPR (Bureau of Public Roads) congestion function:

        factor = 1 + alpha * (vehicle_count / capacity) ** beta

    - When vehicle_count = 0            -> factor = 1.0 (no congestion)
    - When vehicle_count = capacity     -> factor = 1 + alpha
    - When vehicle_count > capacity     -> factor grows superlinearly
                                           (if beta > 1), modeling
                                           increasingly severe slowdowns

    The result is always clamped to [min_factor, max_factor] so a single
    bad or extreme reading cannot produce an unrealistic travel time.

    Attributes
    ----------
    capacity : int
        Approximate number of vehicles a road segment can carry under
        free-flow conditions before congestion starts to build up.
        Should be tuned per road type in a future iteration (e.g. a
        residential street has a much lower capacity than a highway).
    alpha : float
        Controls how strongly congestion increases travel time once
        vehicle_count approaches/exceeds capacity.
    beta : float
        Controls how sharply the congestion factor accelerates as
        vehicle_count grows relative to capacity. beta > 1 makes the
        slowdown accelerate quickly once past capacity.
    min_factor : float
        The lowest allowed congestion factor. Fixed at 1.0 so congestion
        can only ever slow traffic down, never speed it up.
    max_factor : float
        The highest allowed congestion factor, to prevent extreme or
        faulty vehicle-count readings from producing unrealistically
        large travel times.
    """

    capacity: int = 50
    alpha: float = 1.0
    beta: float = 2.0
    min_factor: float = DEFAULT_CONGESTION_FACTOR
    max_factor: float = 5.0

    def __post_init__(self):
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}.")
        if self.min_factor < 1.0:
            raise ValueError(
                f"min_factor must be >= 1.0 (1.0 = normal traffic), "
                f"got {self.min_factor}."
            )
        if self.max_factor < self.min_factor:
            raise ValueError(
                f"max_factor ({self.max_factor}) must be >= "
                f"min_factor ({self.min_factor})."
            )


DEFAULT_CONGESTION_CONFIG = CongestionConfig()


# ---------------------------------------------------------------------------
# Segment ID <-> graph edge mapping
# ---------------------------------------------------------------------------
# A "road segment" corresponds to one edge (u, v, key) in the graph.
# We give each edge a stable, human-readable string ID so external
# systems (like a future YOLO/ByteTrack module) can report vehicle
# counts without needing to know about NetworkX internals.

def get_segment_id(u: Any, v: Any, key: Any) -> str:
    """
    Build a stable string identifier for a road segment (graph edge).

    Parameters
    ----------
    u : Any
        Source node ID of the edge.
    v : Any
        Target node ID of the edge.
    key : Any
        NetworkX MultiDiGraph parallel-edge key.

    Returns
    -------
    str
        A segment ID of the form "u_v_key", e.g. "12345_67890_0".
    """
    return f"{u}_{v}_{key}"


def get_all_segment_ids(graph: nx.MultiDiGraph) -> List[str]:
    """
    List every road segment ID present in the graph.

    Intended for a future traffic-detection module to discover which
    segment IDs it should be reporting vehicle counts for.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The road network graph.

    Returns
    -------
    list of str
        All segment IDs in the graph, e.g. ["12345_67890_0", ...].
    """
    return [
        get_segment_id(u, v, key)
        for u, v, key in graph.edges(keys=True)
    ]


def get_segment_id_mapping(graph: nx.MultiDiGraph) -> Dict[str, Tuple[Any, Any, Any]]:
    """
    Build a lookup from segment ID back to its (u, v, key) edge tuple.

    Useful for a future module that needs to go from a segment ID (e.g.
    received from a traffic camera / detection pipeline) back to the
    specific graph edge it refers to.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The road network graph.

    Returns
    -------
    dict
        Mapping of segment_id -> (u, v, key).
    """
    return {
        get_segment_id(u, v, key): (u, v, key)
        for u, v, key in graph.edges(keys=True)
    }


# ---------------------------------------------------------------------------
# Congestion factor calculation
# ---------------------------------------------------------------------------

def calculate_congestion_factor(
    vehicle_count: int,
    config: CongestionConfig = DEFAULT_CONGESTION_CONFIG,
) -> float:
    """
    Convert a vehicle count into a congestion factor.

    factor = 1 + alpha * (vehicle_count / capacity) ** beta,
    clamped to [config.min_factor, config.max_factor].

    A factor of 1.0 represents normal (free-flow) traffic. Factors
    greater than 1.0 represent congestion, and the higher the factor,
    the more travel time is inflated.

    Parameters
    ----------
    vehicle_count : int
        Number of vehicles currently observed on the road segment.
        Must be >= 0.
    config : CongestionConfig
        Configuration controlling how strongly vehicle count affects
        the resulting factor.

    Returns
    -------
    float
        Congestion factor, always within
        [config.min_factor, config.max_factor].

    Raises
    ------
    ValueError
        If vehicle_count is negative or not numeric.
    """
    try:
        vehicle_count = float(vehicle_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"vehicle_count must be numeric, got {vehicle_count!r}."
        ) from exc

    if vehicle_count < 0:
        raise ValueError(
            f"vehicle_count cannot be negative, got {vehicle_count}."
        )

    if vehicle_count == 0:
        return config.min_factor

    raw_factor = 1.0 + config.alpha * (vehicle_count / config.capacity) ** config.beta

    clamped_factor = max(config.min_factor, min(raw_factor, config.max_factor))
    return clamped_factor


# ---------------------------------------------------------------------------
# Applying congestion to the graph (dynamic_travel_time)
# ---------------------------------------------------------------------------

def calculate_dynamic_travel_time(
    base_travel_time: float,
    congestion_factor: float,
) -> float:
    """
    Combine a base travel time with a congestion factor.

    dynamic_travel_time = base_travel_time * congestion_factor

    Parameters
    ----------
    base_travel_time : float
        Free-flow travel time in seconds (must be > 0).
    congestion_factor : float
        Multiplier representing current congestion (must be > 0; 1.0 =
        normal traffic).

    Returns
    -------
    float
        The congestion-adjusted travel time in seconds. Always > 0.

    Raises
    ------
    ValueError
        If base_travel_time or congestion_factor is missing, zero, or
        negative — dynamic_travel_time is never allowed to be zero or
        negative.
    """
    if base_travel_time is None:
        raise ValueError("base_travel_time is missing.")
    if congestion_factor is None:
        raise ValueError("congestion_factor is missing.")

    try:
        base_travel_time = float(base_travel_time)
        congestion_factor = float(congestion_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Non-numeric base_travel_time or congestion_factor "
            f"(base_travel_time={base_travel_time!r}, "
            f"congestion_factor={congestion_factor!r})."
        ) from exc

    if base_travel_time <= 0:
        raise ValueError(
            f"base_travel_time must be positive, got {base_travel_time}."
        )

    if congestion_factor <= 0:
        raise ValueError(
            f"congestion_factor must be positive, got {congestion_factor}."
        )

    return base_travel_time * congestion_factor


def apply_congestion_to_graph(
    graph: nx.MultiDiGraph,
    vehicle_counts: Dict[str, int],
    config: CongestionConfig = DEFAULT_CONGESTION_CONFIG,
    base_weight_attribute: str = WEIGHT_ATTRIBUTE,
    dynamic_weight_attribute: str = DYNAMIC_WEIGHT_ATTRIBUTE,
) -> nx.MultiDiGraph:
    """
    Apply congestion data to a graph, producing a new graph with a
    'dynamic_travel_time' attribute on every edge.

    The original graph is not modified — this returns a new graph
    (a copy) with the additional attribute added. The existing
    base_weight_attribute (e.g. 'travel_time') is left untouched, so
    routing.py can still be used with either the base or the dynamic
    weight, and the routing algorithm itself does not need to change.

    Edges with no reported vehicle count are treated as normal traffic
    (congestion_factor = 1.0), i.e. dynamic_travel_time == base
    travel_time for that edge. Edges missing a valid base travel time
    are skipped (not given a fabricated dynamic_travel_time), same as
    travel_time.py's behavior for missing/invalid base data.

    Parameters
    ----------
    graph : networkx.MultiDiGraph
        The processed road network graph (must already have
        base_weight_attribute set on its edges, e.g. via
        travel_time.add_travel_times_to_graph()).
    vehicle_counts : dict
        Mapping of segment_id (see get_segment_id / get_all_segment_ids)
        to the current observed vehicle count on that segment. This is
        the interface a future YOLO/ByteTrack module is expected to
        populate.
    config : CongestionConfig
        Configuration for converting vehicle counts into congestion
        factors.
    base_weight_attribute : str
        Edge attribute holding the free-flow base travel time (defaults
        to config.WEIGHT_ATTRIBUTE, i.e. "travel_time").
    dynamic_weight_attribute : str
        Edge attribute to store the congestion-adjusted travel time
        under (defaults to "dynamic_travel_time").

    Returns
    -------
    networkx.MultiDiGraph
        A new graph (copy of the input) with `dynamic_weight_attribute`
        set on every edge that had a valid base travel time.
    """
    graph_copy = graph.copy()

    num_congested = 0
    num_normal = 0
    num_skipped = 0

    for u, v, key, data in graph_copy.edges(keys=True, data=True):
        base_travel_time = data.get(base_weight_attribute)

        if base_travel_time is None:
            num_skipped += 1
            logger.warning(
                "Skipping dynamic_travel_time for edge (%s, %s, %s): "
                "missing '%s'.", u, v, key, base_weight_attribute,
            )
            continue

        segment_id = get_segment_id(u, v, key)
        vehicle_count = vehicle_counts.get(segment_id)

        if vehicle_count is None:
            # No traffic data reported for this segment -> assume normal
            # traffic (factor = 1.0), per requirement #4.
            congestion_factor = config.min_factor
            num_normal += 1
        else:
            try:
                congestion_factor = calculate_congestion_factor(vehicle_count, config)
                num_congested += 1
            except ValueError as exc:
                logger.warning(
                    "Invalid vehicle_count for segment '%s': %s. "
                    "Falling back to normal traffic (factor=1.0).",
                    segment_id, exc,
                )
                congestion_factor = config.min_factor
                num_normal += 1

        try:
            data[dynamic_weight_attribute] = calculate_dynamic_travel_time(
                base_travel_time, congestion_factor
            )
        except ValueError as exc:
            num_skipped += 1
            num_congested = max(num_congested - (vehicle_count is not None), 0)
            logger.warning(
                "Skipping dynamic_travel_time for edge (%s, %s, %s): %s",
                u, v, key, exc,
            )

    logger.info(
        "Congestion applied: %d segment(s) with live traffic data, "
        "%d assumed normal, %d skipped (missing/invalid base data).",
        num_congested, num_normal, num_skipped,
    )

    return graph_copy