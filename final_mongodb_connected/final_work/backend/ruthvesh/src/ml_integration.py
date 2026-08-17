"""
ml_integration.py

Adapter between NovaRoute.AI's ML pipeline output (traffic_data.json
contract, schema_version "1.1.0") and Ruthvesh's road-network graph.

This is Option B: a junction's ML-computed congestion_factor is applied
verbatim to the OSM road segments that have been explicitly, manually
verified to belong to that junction (see JUNCTION_TO_SEGMENTS below).

This module does NOT:
- Run YOLO, ByteTrack, vehicle counting, or risk scoring (ml/src/**).
  Those stay entirely inside NovaRoute_AI and are treated as already
  correct.
- Compute, re-derive, or duplicate a congestion factor. The ML system's
  congestion_factor is read and used as-is. See
  congestion.calculate_congestion_factor() for the (deliberately
  untouched) formula this module does NOT call.
- Implement or modify Dijkstra (routing.py) or the cost matrix
  (cost_matrix.py).
- Invent OSM segment (u, v, key) identifiers. Segment ids only ever come
  from two places: (a) the explicit JUNCTION_TO_SEGMENTS mapping in this
  file, which must be populated with real (u, v, key) triples verified
  against the actual loaded graph (see analyze_segments.py-style
  inspection), and (b) congestion.get_segment_id(), reused unmodified so
  the id string format ("u_v_key") is generated in exactly one place in
  the whole codebase.

What this module produces (build_segment_congestion_map) is a plain
dict: {segment_id: congestion_factor}. Turning that into an actual
'dynamic_travel_time' edge attribute on the graph is a separate,
not-yet-requested step (it would reuse
congestion.calculate_dynamic_travel_time(), which is already the
correct multiply-through function and does not need to change either).
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

try:
    from src.congestion import get_segment_id
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.congestion import get_segment_id

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
# All subclass ValueError so existing call sites that already catch
# ValueError (matching the rest of this codebase's error-handling style,
# e.g. congestion.py / travel_time.py) keep working without change.


class MLObservationError(ValueError):
    """Raised when an ML JSON observation is missing a required field, or
    a field has the wrong type / an out-of-range value. Never raised for
    junction-mapping problems (see UnknownJunctionError)."""


class UnknownJunctionError(ValueError):
    """Raised when an ML observation's road_segment_id / junction_id has
    no entry in JUNCTION_TO_SEGMENTS. This module never guesses a mapping
    for an unrecognized junction id."""


class SegmentCongestionConflictError(ValueError):
    """Raised when two different ML observations map to the same OSM
    segment (u, v, key) but disagree on the congestion_factor to apply to
    it. Silently picking one would hide a real data problem (e.g. two
    junctions' verified segment lists overlap, or a duplicate/stale
    observation was passed in)."""


# ---------------------------------------------------------------------------
# Explicit junction -> OSM segment mapping (Option B)
# ---------------------------------------------------------------------------
# Each value is a list of verified (u, v, key) edge triples, exactly as
# they appear in the loaded ruthvesh graph (see
# congestion.get_segment_id_mapping() to go the other direction).
#
# INTENTIONALLY EMPTY. Populating this requires a human to confirm, for
# each ML junction id, the real-world intersection the camera observes
# and which OSM edges around that intersection represent it. That has
# not been confirmed yet (see prior audit: the only coordinates found
# anywhere for "junction_1"/"junction_2" were unverified placeholder
# values in src/pipeline_demo.py, and "junction_3" has no coordinate at
# all). Do NOT populate this dict with guessed/nearest-node segments -
# every entry must be a verified mapping, at which point map_junction_
# to_segments() below will start resolving it correctly with no further
# code changes needed.
#
# Example of the expected shape once verified (NOT filled in - illustrative
# only):
#   JUNCTION_TO_SEGMENTS: Dict[str, List[Tuple[int, int, int]]] = {
#       "junction_1": [(3750261536, 3750261542, 0), (3750261542, 3750261536, 0)],
#   }
JUNCTION_TO_SEGMENTS: Dict[str, List[Tuple[int, int, int]]] = {}


# ---------------------------------------------------------------------------
# Parsed observation type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MLObservationFields:
    """Typed view of the specific fields this adapter consumes from an ML
    traffic_data.json observation. Not a full mirror of the schema -
    fields this adapter never reads (e.g. peak_vehicles, road_capacity,
    risk_factor_scores) are intentionally left out."""

    road_segment_id: str
    congestion_factor: float
    risk_score: float
    risk_level: str
    vehicle_counts: Dict[str, int]
    timestamp: str
    schema_version: str


_VALID_RISK_LEVELS = ("low", "moderate", "high", "critical")
_VEHICLE_CLASS_KEYS = ("car", "motorcycle", "bus", "truck", "total")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_ml_observation(obs: Any) -> None:
    """
    Validate that a parsed ML observation has the fields this adapter
    depends on, with correct types and sane values.

    This is deliberately NOT full JSON-Schema validation against
    integration/schemas/traffic_data.json (that would require adding
    jsonschema as a ruthvesh dependency, which was intentionally kept
    out for now). It validates every field this adapter reads:
    schema_version, road_segment_id, congestion_factor, risk_score,
    risk_level, vehicle_counts, timestamp.

    Parameters
    ----------
    obs : Any
        The parsed JSON observation (expected to be a dict).

    Raises
    ------
    MLObservationError
        If `obs` is not a dict, or any required field is missing, has
        the wrong type, or an out-of-range/invalid value.
    """
    if not isinstance(obs, dict):
        raise MLObservationError(
            f"ML observation must be a JSON object, got {type(obs).__name__}."
        )

    # --- schema_version -----------------------------------------------
    schema_version = obs.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise MLObservationError(
            "ML observation is missing a valid 'schema_version' string."
        )
    if not schema_version.startswith("1."):
        raise MLObservationError(
            f"Unsupported ML observation schema_version {schema_version!r}; "
            "this adapter was built against the '1.x' traffic_data.json contract."
        )

    # --- road_segment_id -------------------------------------------------
    road_segment_id = obs.get("road_segment_id")
    if not isinstance(road_segment_id, str) or not road_segment_id.strip():
        raise MLObservationError(
            "ML observation is missing a valid non-empty 'road_segment_id' string."
        )

    # --- timestamp ----------------------------------------------------
    timestamp = obs.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise MLObservationError(
            "ML observation is missing a valid 'timestamp' string."
        )
    _parse_timestamp(timestamp)  # raises MLObservationError if unparseable

    # --- congestion_factor ----------------------------------------------
    congestion_factor = obs.get("congestion_factor")
    if (
        isinstance(congestion_factor, bool)
        or not isinstance(congestion_factor, (int, float))
        or not isfinite(float(congestion_factor))
    ):
        raise MLObservationError(
            f"ML observation 'congestion_factor' must be a finite number, "
            f"got {congestion_factor!r}."
        )
    if float(congestion_factor) < 1.0:
        raise MLObservationError(
            "ML observation 'congestion_factor' must be >= 1.0 "
            f"(1.0 = free-flow, no congestion); got {congestion_factor!r}."
        )

    # --- risk_score -----------------------------------------------------
    risk_score = obs.get("risk_score")
    if (
        isinstance(risk_score, bool)
        or not isinstance(risk_score, (int, float))
        or not isfinite(float(risk_score))
    ):
        raise MLObservationError(
            f"ML observation 'risk_score' must be a finite number, got {risk_score!r}."
        )
    if not (0.0 <= float(risk_score) <= 100.0):
        raise MLObservationError(
            f"ML observation 'risk_score' must be within [0, 100], got {risk_score!r}."
        )

    # --- risk_level -------------------------------------------------------
    risk_level = obs.get("risk_level")
    if not isinstance(risk_level, str) or risk_level not in _VALID_RISK_LEVELS:
        raise MLObservationError(
            f"ML observation 'risk_level' must be one of {_VALID_RISK_LEVELS}, "
            f"got {risk_level!r}."
        )

    # --- vehicle_counts ---------------------------------------------------
    vehicle_counts = obs.get("vehicle_counts")
    if not isinstance(vehicle_counts, dict):
        raise MLObservationError(
            f"ML observation 'vehicle_counts' must be an object, "
            f"got {type(vehicle_counts).__name__}."
        )
    for key in _VEHICLE_CLASS_KEYS:
        if key not in vehicle_counts:
            raise MLObservationError(
                f"ML observation 'vehicle_counts' is missing required key {key!r}."
            )
        value = vehicle_counts[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MLObservationError(
                f"ML observation 'vehicle_counts.{key}' must be a non-negative "
                f"integer, got {value!r}."
            )


def _parse_timestamp(timestamp: str) -> datetime:
    """Parse an ISO-8601 timestamp string (accepting a trailing 'Z').

    Raises
    ------
    MLObservationError
        If the timestamp cannot be parsed.
    """
    text = timestamp.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise MLObservationError(
            f"ML observation 'timestamp' is not a valid ISO-8601 datetime: "
            f"{timestamp!r} ({exc})."
        ) from exc


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_ml_observation(source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Load and validate a single NovaRoute ML JSON observation.

    Parameters
    ----------
    source : str | Path | dict
        Either a path to a traffic_data.json-shaped file on disk, or an
        already-parsed dict (useful for tests / in-memory pipelines that
        skip the filesystem).

    Returns
    -------
    dict
        The parsed observation, validated by validate_ml_observation().
        Returned as a plain dict (not a copy of an internal object) so
        callers can also read fields this adapter doesn't extract
        (e.g. traffic_density) if they need to.

    Raises
    ------
    MLObservationError
        If the file does not exist, is not valid JSON, is not a JSON
        object, or fails validate_ml_observation().
    """
    if isinstance(source, dict):
        obs = source
    else:
        path = Path(source)
        if not path.exists():
            raise MLObservationError(f"ML observation file not found: '{path}'.")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise MLObservationError(
                f"Could not read ML observation file '{path}': {exc}."
            ) from exc
        try:
            obs = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MLObservationError(
                f"ML observation file '{path}' is not valid JSON: {exc}."
            ) from exc

    validate_ml_observation(obs)
    logger.info(
        "Loaded ML observation for road_segment_id=%r (congestion_factor=%.4f, "
        "risk_level=%s).",
        obs["road_segment_id"], float(obs["congestion_factor"]), obs["risk_level"],
    )
    return obs


def extract_ml_fields(obs: Dict[str, Any]) -> MLObservationFields:
    """
    Extract the specific fields this adapter needs from a validated ML
    observation into a typed, immutable MLObservationFields.

    Parameters
    ----------
    obs : dict
        A parsed ML observation. Re-validated internally (cheap, and
        makes this function safe to call directly without going through
        load_ml_observation first).

    Returns
    -------
    MLObservationFields

    Raises
    ------
    MLObservationError
        If `obs` fails validate_ml_observation().
    """
    validate_ml_observation(obs)
    return MLObservationFields(
        road_segment_id=obs["road_segment_id"],
        congestion_factor=float(obs["congestion_factor"]),
        risk_score=float(obs["risk_score"]),
        risk_level=obs["risk_level"],
        vehicle_counts=dict(obs["vehicle_counts"]),
        timestamp=obs["timestamp"],
        schema_version=obs["schema_version"],
    )


# ---------------------------------------------------------------------------
# Junction -> segment mapping
# ---------------------------------------------------------------------------

def map_junction_to_segments(
    road_segment_id: str,
    junction_to_segments: Optional[Mapping[str, Iterable[Tuple[Any, Any, Any]]]] = None,
) -> List[Tuple[Any, Any, Any]]:
    """
    Resolve a junction id to its verified list of OSM (u, v, key) edge
    triples using the explicit JUNCTION_TO_SEGMENTS mapping.

    This function never invents, guesses, or falls back to a
    nearest-node lookup - it only ever returns triples that are already
    present in the mapping.

    Parameters
    ----------
    road_segment_id : str
        The ML observation's road_segment_id (junction id), e.g.
        "junction_1".
    junction_to_segments : mapping or None
        Overrides the module-level JUNCTION_TO_SEGMENTS (mainly for
        tests). Defaults to JUNCTION_TO_SEGMENTS when not given.

    Returns
    -------
    list of (u, v, key)
        The verified OSM edges mapped to this junction. Can legitimately
        be an empty list if the junction is deliberately registered with
        zero segments so far - that is different from the junction id
        not being registered at all (see Raises below).

    Raises
    ------
    UnknownJunctionError
        If `road_segment_id` has no entry in the mapping at all. This is
        a hard stop by design: routing on an unmapped junction would
        otherwise silently apply no congestion adjustment anywhere,
        which looks identical to "confirmed no congestion" and would
        hide a real integration gap.
    """
    mapping = JUNCTION_TO_SEGMENTS if junction_to_segments is None else junction_to_segments

    if road_segment_id not in mapping:
        raise UnknownJunctionError(
            f"Unknown junction id {road_segment_id!r}: no entry in "
            "JUNCTION_TO_SEGMENTS. Refusing to guess a mapping - add a "
            "verified (u, v, key) segment list for this junction to "
            "ml_integration.JUNCTION_TO_SEGMENTS (see module docstring) "
            "before this observation can be applied to the graph."
        )

    segments = list(mapping[road_segment_id])
    for i, edge in enumerate(segments):
        if not (isinstance(edge, tuple) and len(edge) == 3):
            raise MLObservationError(
                f"JUNCTION_TO_SEGMENTS[{road_segment_id!r}][{i}] must be a "
                f"(u, v, key) tuple, got {edge!r}."
            )
    return segments


# ---------------------------------------------------------------------------
# Building the segment -> congestion_factor map
# ---------------------------------------------------------------------------

def build_segment_congestion_map(
    observations: Iterable[Dict[str, Any]],
    junction_to_segments: Optional[Mapping[str, Iterable[Tuple[Any, Any, Any]]]] = None,
    on_unknown_junction: str = "raise",
) -> Dict[str, float]:
    """
    Turn one or more ML observations into a single
    {segment_id: congestion_factor} map, ready to be applied to the
    graph (e.g. by a future step that calls
    congestion.calculate_dynamic_travel_time() per mapped edge - not
    implemented here).

    The congestion_factor for each segment is taken verbatim from the ML
    observation that owns the junction it belongs to - this function
    never computes or re-derives a congestion factor itself.

    Parameters
    ----------
    observations : iterable of dict
        Parsed ML observations (as returned by load_ml_observation()).
        Each is validated internally.
    junction_to_segments : mapping or None
        Overrides the module-level JUNCTION_TO_SEGMENTS (mainly for
        tests).
    on_unknown_junction : str
        "raise" (default): raise UnknownJunctionError immediately on the
        first observation whose junction isn't in the mapping.
        "skip": log a warning and skip that observation's segments
        instead of raising. Use only when it's expected/acceptable that
        some junctions aren't mapped yet.

    Returns
    -------
    dict
        Mapping of segment_id (the "u_v_key" string produced by
        congestion.get_segment_id) to the congestion_factor that should
        be applied to that edge.

    Raises
    ------
    ValueError
        If `on_unknown_junction` is not "raise" or "skip".
    UnknownJunctionError
        If on_unknown_junction="raise" and an observation's junction id
        is not in the mapping.
    SegmentCongestionConflictError
        If two observations map to the same OSM segment with different
        congestion_factor values.
    MLObservationError
        If any observation fails validation.
    """
    if on_unknown_junction not in ("raise", "skip"):
        raise ValueError(
            f"on_unknown_junction must be 'raise' or 'skip', got {on_unknown_junction!r}."
        )

    segment_congestion: Dict[str, float] = {}
    # Tracks which junction each segment came from, purely to produce a
    # clear conflict error message (which two junctions disagreed).
    segment_source: Dict[str, str] = {}

    for obs in observations:
        fields = extract_ml_fields(obs)

        try:
            segments = map_junction_to_segments(
                fields.road_segment_id, junction_to_segments=junction_to_segments
            )
        except UnknownJunctionError:
            if on_unknown_junction == "skip":
                logger.warning(
                    "Skipping ML observation for unknown junction %r "
                    "(on_unknown_junction='skip').", fields.road_segment_id,
                )
                continue
            raise

        for u, v, key in segments:
            segment_id = get_segment_id(u, v, key)
            existing = segment_congestion.get(segment_id)

            if existing is not None and existing != fields.congestion_factor:
                raise SegmentCongestionConflictError(
                    f"Segment {segment_id!r} was assigned congestion_factor="
                    f"{existing!r} by junction {segment_source[segment_id]!r} "
                    f"and congestion_factor={fields.congestion_factor!r} by "
                    f"junction {fields.road_segment_id!r}. Two junctions' "
                    "verified segment lists overlap with conflicting ML "
                    "readings - resolve JUNCTION_TO_SEGMENTS or the input "
                    "observations before applying congestion to the graph."
                )

            segment_congestion[segment_id] = fields.congestion_factor
            segment_source[segment_id] = fields.road_segment_id

    logger.info(
        "Built segment congestion map: %d segment(s) from %d distinct junction(s).",
        len(segment_congestion), len(set(segment_source.values())),
    )
    return segment_congestion