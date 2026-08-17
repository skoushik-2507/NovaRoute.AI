"""
risk_priority.py

Small adapter connecting NovaRoute_AI's real ML risk_score (0-100 scale)
to coverage.py's existing `risk_scores` parameter, for junction
priority / uncovered-junction ordering.

Pipeline (Prompt 7B):

    NovaRoute_AI ML observation JSON
        -> ml_integration.load_ml_observation() / extract_ml_fields()
        -> real risk_score (0-100 scale, read verbatim, never rescaled)
        -> junction_id -> risk_score mapping (this module)
        -> coverage.py's `risk_scores` parameter (already existed, unmodified)
        -> junction priority / uncovered-junction ordering

This module does NOT:
- Compute or re-derive a risk_score. ml/src/risk/** inside NovaRoute_AI
  does that; it is treated as already correct, mirroring
  ml_integration.py's own stance on congestion_factor (see that module's
  docstring).
- Rescale risk_score from its real 0-100 range to the old 0-1 demo scale
  used in pipeline_demo.py. There is no documented architectural reason
  to do so: coverage.py's `risk_scores` parameter is scale-agnostic (it
  only pass-throughs the value into per-junction reports and uses it as
  a descending sort key - see coverage.py::analyze_coverage_from_cost_
  matrix and Prompt 7A's test_coverage.py), so passing the real 0-100
  value straight through is both the smaller change and the more
  faithful one.
- Touch congestion_factor, travel_time, or dynamic_travel_time in any
  way. Those live entirely on the congestion side (congestion.py /
  ml_dynamic_graph.py), which this module does not import from and does
  not modify.
- Fabricate a risk_score for a junction with no real ML observation
  (e.g. junction_3, as of this writing). A junction_id not present in
  the `sources` mapping passed to build_risk_priority_map() is simply
  absent from the returned map - never defaulted, interpolated, or
  guessed.
- Modify coverage.py's public API. analyze_coverage_from_cost_matrix()
  already accepts an optional `risk_scores: Dict[str, Any]` and already
  does exactly what's needed with it. This module only builds a
  correctly-shaped dict to feed into that existing parameter.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Union

try:
    from src.ml_integration import extract_ml_fields, load_ml_observation
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.ml_integration import extract_ml_fields, load_ml_observation

logger = logging.getLogger(__name__)


class RiskJunctionMismatchError(ValueError):
    """Raised when an ML observation is supplied under one junction_id
    key but its JSON 'road_segment_id' field claims a different
    junction. This is an "unknown junction" / mapping problem, distinct
    from a malformed-observation problem (see ml_integration.
    MLObservationError, raised by load_ml_observation()/extract_ml_
    fields() for missing files, invalid JSON, or missing/invalid
    fields such as risk_score). This module never guesses which
    junction an observation belongs to when the two disagree."""


def build_risk_priority_map(
    sources: Mapping[str, Union[str, Path, Dict[str, Any]]],
) -> Dict[str, float]:
    """
    Build a {junction_id: risk_score} mapping from real NovaRoute_AI ML
    observations, shaped for use directly as coverage.py's
    `risk_scores` parameter.

    Parameters
    ----------
    sources : mapping of junction_id -> path (str/Path) or an
        already-parsed observation dict, e.g.:
            {
                "junction_1": ".../junction_1_latest.json",
                "junction_2": ".../junction_2_latest.json",
            }
        Only junctions that have a real ML observation should be
        included here. A junction with no real observation (e.g.
        junction_3, as of this writing) must simply be omitted from
        `sources` - never included with a placeholder/guessed path.

    Returns
    -------
    dict of junction_id -> float
        risk_score, read verbatim from the ML observation, on its
        original 0-100 scale. Never rescaled, never recomputed, and
        never fabricated for a junction that wasn't in `sources`.

    Raises
    ------
    ml_integration.MLObservationError
        If a source file/dict is missing, not valid JSON/not a JSON
        object, or fails ml_integration's schema validation. This
        single error type covers both the "missing observation"
        (file-not-found) case and the "invalid observation"
        (malformed / out-of-range field, including a missing
        risk_score) case - see load_ml_observation() and
        validate_ml_observation() in ml_integration.py, reused here
        unmodified.
    RiskJunctionMismatchError
        If the caller-supplied junction_id key does not match the
        'road_segment_id' actually found inside that observation's
        JSON (the "unknown junction" case).
    """
    risk_scores: Dict[str, float] = {}

    for junction_id, source in sources.items():
        obs = load_ml_observation(source)
        fields = extract_ml_fields(obs)

        if fields.road_segment_id != junction_id:
            raise RiskJunctionMismatchError(
                f"Observation supplied under junction_id={junction_id!r} "
                f"actually has road_segment_id={fields.road_segment_id!r} "
                "inside its JSON. Refusing to assign this risk_score to a "
                "junction it does not claim to belong to - check that the "
                "file paths/keys line up."
            )

        risk_scores[junction_id] = fields.risk_score
        logger.info(
            "Loaded real ML risk_score for %s: %.4f (risk_level=%s).",
            junction_id, fields.risk_score, fields.risk_level,
        )

    return risk_scores