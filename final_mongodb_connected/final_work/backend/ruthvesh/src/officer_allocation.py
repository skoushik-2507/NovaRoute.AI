"""
officer_allocation.py

Officer -> high-risk-junction assignment layer (Prompt 10).

This module is COMPOSITION + OPTIMIZATION ONLY over data that already
exists. It consumes:

    - cost_matrix_result  -> cost_matrix.build_cost_matrix()'s return
                              value (matrix, officer_ids, junction_ids,
                              routes, time_unit). Read-only input; never
                              mutated.
    - risk_scores          -> risk_priority.build_risk_priority_map()'s
                              return value ({junction_id: risk_score},
                              0-100 scale, real ML data). Read-only
                              input; never mutated.

It does NOT:
    - Implement or call Dijkstra (routing.py is never imported).
    - Recompute travel time or congestion (congestion.py, travel_time.py,
      ml_dynamic_graph.py are never imported).
    - Snap coordinates to graph nodes or touch the graph at all
      (graph_utils.py is never imported; this module never sees lat/lon).
    - Recompute a route. Every route in the output is looked up verbatim
      from cost_matrix_result["routes"], keyed by the same (officer_id,
      junction_id) tuple cost_matrix.py already uses.
    - Modify cost_matrix_result or risk_scores in place.

Assignment model
-----------------
Each officer may be assigned to at most one junction, and each junction
may receive at most one officer (one-to-one bipartite assignment). This
is solved with the Hungarian algorithm
(scipy.optimize.linear_sum_assignment), which finds the cost-minimizing
one-to-one matching in polynomial time.

risk_score determines PRIORITY, never cost. The two are kept
architecturally separate, exactly as the rest of this project already
treats them (see emergency_response_pipeline.py's docstring: congestion/
travel-time and risk_score "remain architecturally independent"):

    1. Junctions are grouped into priority tiers by their risk_score
       value: every distinct score is its own tier, tied scores share a
       tier, and junctions with no ML risk_score form the lowest-priority
       tier (never fabricated, never defaulted to 0).
    2. Tiers are processed strictly highest-risk-score first.
    3. Within a tier, Hungarian optimization runs over the remaining
       (not-yet-assigned) officers against that tier's junctions,
       minimizing total response time for that tier only — risk_score is
       never added to, subtracted from, or multiplied with the cost.
    4. Officers assigned in one tier are removed from the pool available
       to every subsequent (lower-priority) tier.

Unreachable pairs (UNREACHABLE_COST / inf in the matrix, exactly as
cost_matrix.py already represents them) are never assigned, even if the
Hungarian solver would otherwise be forced to pick one to complete a
matching. A junction that cannot be reached by any remaining officer
becomes unassigned rather than crashing or fabricating an assignment.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

ASSIGNMENT_METHOD = "hungarian_per_risk_tier"

SECONDS_PER_MINUTE = 60.0

# Required keys on cost_matrix_result (the dict returned by
# cost_matrix.build_cost_matrix()) for this module to operate on it.
_REQUIRED_COST_MATRIX_KEYS = (
    "matrix",
    "officer_ids",
    "junction_ids",
    "routes",
    "time_unit",
)

_VALID_TIME_UNITS = ("minutes", "seconds")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
# Both subclass ValueError, matching this codebase's existing
# error-handling style (see congestion.py, ml_integration.py,
# risk_priority.py, ml_dynamic_graph.py).

class InvalidCostMatrixError(ValueError):
    """Raised when cost_matrix_result is missing required keys, has a
    matrix shape inconsistent with officer_ids/junction_ids, or contains
    duplicate officer/junction ids. This module never guesses a shape or
    silently drops a duplicate — it refuses ambiguous input clearly."""


class InvalidRiskScoreError(ValueError):
    """Raised when a risk_scores value is present but not a valid
    real-valued score on the project's 0-100 ML risk scale (see
    integration/schemas/traffic_data.json's risk_score field and
    risk_priority.py's docstring). A junction_id simply absent from
    risk_scores (or explicitly mapped to None) is NOT an error — that is
    the documented "no real ML observation yet" case and is treated as
    lowest priority, never fabricated."""


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_cost_matrix_result(cost_matrix_result: Any) -> Tuple[List[str], List[str], np.ndarray, Mapping]:
    if not isinstance(cost_matrix_result, dict):
        raise InvalidCostMatrixError(
            f"cost_matrix_result must be a dict (as returned by "
            f"cost_matrix.build_cost_matrix()), got "
            f"{type(cost_matrix_result).__name__}."
        )

    missing = [k for k in _REQUIRED_COST_MATRIX_KEYS if k not in cost_matrix_result]
    if missing:
        raise InvalidCostMatrixError(
            f"cost_matrix_result is missing required key(s): {missing}. "
            "Expected the dict returned by cost_matrix.build_cost_matrix()."
        )

    officer_ids = cost_matrix_result["officer_ids"]
    junction_ids = cost_matrix_result["junction_ids"]
    routes = cost_matrix_result["routes"]
    time_unit = cost_matrix_result["time_unit"]

    if not isinstance(officer_ids, (list, tuple)):
        raise InvalidCostMatrixError("cost_matrix_result['officer_ids'] must be a list.")
    if not isinstance(junction_ids, (list, tuple)):
        raise InvalidCostMatrixError("cost_matrix_result['junction_ids'] must be a list.")
    if not isinstance(routes, Mapping):
        raise InvalidCostMatrixError("cost_matrix_result['routes'] must be a mapping.")
    if time_unit not in _VALID_TIME_UNITS:
        raise InvalidCostMatrixError(
            f"cost_matrix_result['time_unit'] must be one of {_VALID_TIME_UNITS}, "
            f"got {time_unit!r}."
        )

    officer_ids = list(officer_ids)
    junction_ids = list(junction_ids)

    if len(set(officer_ids)) != len(officer_ids):
        raise InvalidCostMatrixError(
            "cost_matrix_result['officer_ids'] contains duplicate ids; "
            "refusing to assign ambiguously."
        )
    if len(set(junction_ids)) != len(junction_ids):
        raise InvalidCostMatrixError(
            "cost_matrix_result['junction_ids'] contains duplicate ids; "
            "refusing to assign ambiguously."
        )

    try:
        matrix = np.array(cost_matrix_result["matrix"], dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise InvalidCostMatrixError(
            f"cost_matrix_result['matrix'] could not be interpreted as a "
            f"numeric array: {exc}"
        ) from exc

    expected_shape = (len(officer_ids), len(junction_ids))
    if matrix.shape != expected_shape:
        raise InvalidCostMatrixError(
            f"cost_matrix_result['matrix'] has shape {matrix.shape}, but "
            f"officer_ids x junction_ids implies shape {expected_shape}."
        )

    return officer_ids, junction_ids, matrix, routes


def _validate_risk_scores(risk_scores: Optional[Mapping[str, Any]]) -> Dict[str, float]:
    """Return a NEW dict of junction_id -> float risk_score, never
    mutating the caller's `risk_scores`. Junctions mapped to None (or
    simply absent) are omitted here, not defaulted to any number."""
    if risk_scores is None:
        return {}
    if not isinstance(risk_scores, Mapping):
        raise InvalidRiskScoreError(
            f"risk_scores must be a dict (as returned by "
            f"risk_priority.build_risk_priority_map()) or None, got "
            f"{type(risk_scores).__name__}."
        )

    validated: Dict[str, float] = {}
    for junction_id, score in risk_scores.items():
        if score is None:
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise InvalidRiskScoreError(
                f"risk_score for junction {junction_id!r} must be numeric "
                f"or None, got {score!r}."
            )
        score_f = float(score)
        if not np.isfinite(score_f):
            raise InvalidRiskScoreError(
                f"risk_score for junction {junction_id!r} must be finite, "
                f"got {score!r}."
            )
        if not (0.0 <= score_f <= 100.0):
            raise InvalidRiskScoreError(
                f"risk_score for junction {junction_id!r} must be within "
                f"[0, 100] (the project's real ML risk scale), got {score!r}."
            )
        validated[junction_id] = score_f

    return validated


# ---------------------------------------------------------------------------
# Priority tiering
# ---------------------------------------------------------------------------

def _build_priority_tiers(
    junction_ids: Sequence[str],
    risk_scores: Dict[str, float],
) -> List[List[str]]:
    """Group junction_ids into priority tiers: every distinct risk_score
    value is its own tier (tied scores share a tier), ordered strictly
    highest-score-first, with every junction that has no valid risk_score
    collected into a single lowest-priority tier at the end. Order within
    a tier is deterministic (sorted by junction_id) but does not affect
    the Hungarian result, which is order-independent."""
    scored_by_value: Dict[float, List[str]] = {}
    unscored: List[str] = []

    for junction_id in junction_ids:
        score = risk_scores.get(junction_id)
        if score is None:
            unscored.append(junction_id)
        else:
            scored_by_value.setdefault(score, []).append(junction_id)

    tiers = [
        sorted(scored_by_value[score])
        for score in sorted(scored_by_value.keys(), reverse=True)
    ]
    if unscored:
        tiers.append(sorted(unscored))
    return tiers


# ---------------------------------------------------------------------------
# Hungarian solve for a single tier, respecting unreachable (inf) pairs
# ---------------------------------------------------------------------------

def _solve_tier(
    submatrix: np.ndarray,
) -> List[Tuple[int, int]]:
    """Solve one tier's officer x junction submatrix with the Hungarian
    algorithm, guaranteed never to return a pair whose original cost was
    infinite (unreachable), and never to raise even when the submatrix
    contains unreachable pairs that would otherwise make a complete
    matching infeasible for scipy.

    Returns a list of (row_index, col_index) pairs, indices local to
    `submatrix`, containing only finite-cost (reachable) pairs.
    """
    finite_mask = np.isfinite(submatrix)
    if not finite_mask.any():
        return []

    # scipy's linear_sum_assignment supports np.inf directly, but raises
    # ValueError("cost matrix is infeasible") whenever no COMPLETE
    # matching exists without using at least one inf cell (e.g. two
    # officers that can only reach the same single junction, leaving a
    # second junction with no finite option at all). That is a normal,
    # expected outcome here (some junctions simply end up unassigned),
    # not an error condition, so inf is replaced with a large-but-finite
    # sentinel before solving, and any resulting pair whose ORIGINAL cost
    # was inf is filtered out afterward instead of ever reaching the
    # caller as a real assignment.
    finite_values = submatrix[finite_mask]
    sentinel = (float(finite_values.max()) + 1.0) * 1_000.0
    solve_matrix = np.where(finite_mask, submatrix, sentinel)

    row_indices, col_indices = linear_sum_assignment(solve_matrix)

    return [
        (int(r), int(c))
        for r, c in zip(row_indices, col_indices)
        if np.isfinite(submatrix[r, c])
    ]


def _to_minutes(value: float, time_unit: str) -> float:
    if time_unit == "seconds":
        return value / SECONDS_PER_MINUTE
    return value


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assign_officers(
    cost_matrix_result: Dict[str, Any],
    risk_scores: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assign officers to high-risk junctions (one-to-one) using Hungarian
    optimization within strict risk-priority tiers.

    Parameters
    ----------
    cost_matrix_result : dict
        The dict returned by cost_matrix.build_cost_matrix(). Must
        contain "matrix", "officer_ids", "junction_ids", "routes", and
        "time_unit". Never mutated by this function.
    risk_scores : dict or None
        The dict returned by risk_priority.build_risk_priority_map()
        ({junction_id: risk_score}, 0-100 scale), or None. A junction_id
        absent from this mapping (or explicitly None) is treated as
        having no real ML risk score and is placed in the lowest-priority
        tier — never fabricated, never defaulted to 0. Never mutated by
        this function.

    Returns
    -------
    dict
        {
            "assignments": [
                {
                    "officer_id": str,
                    "junction_id": str,
                    "response_time_minutes": float,
                    "risk_score": float or None,
                    "route": dict or None,
                        # looked up verbatim from
                        # cost_matrix_result["routes"][(officer_id,
                        # junction_id)]; never recomputed.
                },
                ...
            ],
            "unassigned_junction_ids": [...],
                # highest-priority (risk_score descending) first, then
                # the no-score tier; never fabricated an assignment for
                # these.
            "unassigned_officer_ids": [...],
                # officers left over after every tier has been solved,
                # in their original cost_matrix_result["officer_ids"]
                # order.
            "assignment_method": "hungarian_per_risk_tier",
        }

    Raises
    ------
    InvalidCostMatrixError
        If cost_matrix_result is malformed (missing keys, wrong types,
        shape mismatch, or duplicate officer/junction ids).
    InvalidRiskScoreError
        If risk_scores contains a non-numeric, non-finite, or
        out-of-[0, 100]-range value for some junction.
    """
    officer_ids, junction_ids, matrix, routes = _validate_cost_matrix_result(cost_matrix_result)
    time_unit = cost_matrix_result["time_unit"]
    validated_risk_scores = _validate_risk_scores(risk_scores)

    tiers = _build_priority_tiers(junction_ids, validated_risk_scores)

    officer_col_index = {officer_id: i for i, officer_id in enumerate(officer_ids)}
    junction_col_index = {junction_id: j for j, junction_id in enumerate(junction_ids)}

    remaining_officer_ids: List[str] = list(officer_ids)
    assignments: List[Dict[str, Any]] = []
    unassigned_junction_ids: List[str] = []

    for tier_junction_ids in tiers:
        if not remaining_officer_ids:
            unassigned_junction_ids.extend(tier_junction_ids)
            continue

        row_positions = [officer_col_index[oid] for oid in remaining_officer_ids]
        col_positions = [junction_col_index[jid] for jid in tier_junction_ids]
        submatrix = matrix[np.ix_(row_positions, col_positions)]

        local_pairs = _solve_tier(submatrix)

        assigned_officer_ids_this_tier = set()
        assigned_junction_ids_this_tier = set()

        for local_row, local_col in local_pairs:
            officer_id = remaining_officer_ids[local_row]
            junction_id = tier_junction_ids[local_col]
            raw_cost = float(submatrix[local_row, local_col])

            assignments.append({
                "officer_id": officer_id,
                "junction_id": junction_id,
                "response_time_minutes": _to_minutes(raw_cost, time_unit),
                "risk_score": validated_risk_scores.get(junction_id),
                "route": routes.get((officer_id, junction_id)),
            })
            assigned_officer_ids_this_tier.add(officer_id)
            assigned_junction_ids_this_tier.add(junction_id)

        unassigned_junction_ids.extend(
            jid for jid in tier_junction_ids if jid not in assigned_junction_ids_this_tier
        )
        remaining_officer_ids = [
            oid for oid in remaining_officer_ids if oid not in assigned_officer_ids_this_tier
        ]

    logger.info(
        "Officer allocation complete: %d assignment(s), %d unassigned junction(s), "
        "%d unassigned officer(s).",
        len(assignments), len(unassigned_junction_ids), len(remaining_officer_ids),
    )

    return {
        "assignments": assignments,
        "unassigned_junction_ids": unassigned_junction_ids,
        "unassigned_officer_ids": remaining_officer_ids,
        "assignment_method": ASSIGNMENT_METHOD,
    }