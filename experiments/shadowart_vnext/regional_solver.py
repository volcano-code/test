"""MILP selection of finite regional ray-footprint unions.

This module operates only on already-rendered binary ray footprints.  It makes
no geometric connectivity, manufacturing, or printability claim.
"""
from numbers import Real

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


_FLOOR_SCALE = 1_000_000
_INTEGRAL_TOLERANCE = 1e-6
_CAVEAT = (
    "Any infeasibility is only for this finite candidate family; it is not a "
    "global geometric or manufacturing infeasibility claim."
)
_OBJECTIVE_CAVEAT = (
    "The objective is normalized foreground gain minus base-IoU-weighted "
    "background gain; it is not exact mean-IoU optimization."
)


def _binary_copy(value, name):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a regular binary array") from exc
    if array.dtype.kind not in "biuf" or (
        array.dtype.kind == "f" and not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must contain only finite binary values")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only binary values")
    return np.array(array, dtype=bool, copy=True, order="C")


def _validated_inputs(
    base_predictions, targets, additions, floors, max_selected, time_limit
):
    base = _binary_copy(base_predictions, "base_predictions")
    target = _binary_copy(targets, "targets")
    candidates = _binary_copy(additions, "additions")
    if (
        base.ndim != 3
        or any(size <= 0 for size in base.shape)
        or base.shape[1] != base.shape[2]
    ):
        raise ValueError("base_predictions must have nonempty square shape [V,H,H]")
    if target.shape != base.shape:
        raise ValueError("targets must match base_predictions shape [V,H,H]")
    if candidates.ndim != 4 or candidates.shape[1:] != base.shape:
        raise ValueError("additions must have shape [K,V,H,H] matching the base")
    candidate_count = candidates.shape[0]
    if candidate_count > 256:
        raise ValueError("at most 256 additions are supported")

    try:
        raw_floors = np.asarray(floors)
    except (TypeError, ValueError) as exc:
        raise ValueError("floors must be a finite numeric vector") from exc
    if raw_floors.dtype.kind not in "biuf":
        raise ValueError("floors must be a finite numeric vector")
    floor_array = np.array(raw_floors, dtype=np.float64, copy=True)
    if (
        floor_array.shape != (base.shape[0],)
        or not np.isfinite(floor_array).all()
        or np.any(floor_array < 0)
        or np.any(floor_array > 1)
    ):
        raise ValueError("floors must have shape [V] with finite values in [0,1]")
    if (
        isinstance(max_selected, (bool, np.bool_))
        or not isinstance(max_selected, (int, np.integer))
        or max_selected < 0
        or max_selected > candidate_count
    ):
        raise ValueError("max_selected must be an integer in [0,K]")
    if (
        isinstance(time_limit, (bool, np.bool_))
        or not isinstance(time_limit, Real)
        or not np.isfinite(time_limit)
        or time_limit <= 0
    ):
        raise ValueError("time_limit must be positive and finite")
    return (
        base,
        target,
        candidates,
        floor_array,
        int(max_selected),
        float(time_limit),
    )


def _base_counts(base, target):
    positives = target.sum(axis=(1, 2), dtype=np.int64)
    true_positives = np.logical_and(base, target).sum(axis=(1, 2), dtype=np.int64)
    false_positives = np.logical_and(base, ~target).sum(
        axis=(1, 2), dtype=np.int64
    )
    denominator = positives + false_positives
    ious = np.divide(
        true_positives,
        denominator,
        out=np.ones(base.shape[0], dtype=np.float64),
        where=denominator > 0,
    )
    return positives, true_positives, false_positives, ious


def _compressed_patterns(base, target, additions):
    """Aggregate base-unseen pixels by (view, candidate incidence pattern)."""
    candidate_count, view_count = additions.shape[:2]
    patterns = []
    for view in range(view_count):
        unseen = ~base[view].reshape(-1)
        if not np.any(unseen):
            continue
        footprints = additions[:, view].reshape(candidate_count, -1)[:, unseen]
        packed = np.packbits(footprints, axis=0, bitorder="little").T
        _, first_indices, inverse = np.unique(
            packed, axis=0, return_index=True, return_inverse=True
        )
        unseen_target = target[view].reshape(-1)[unseen]
        pattern_count = len(first_indices)
        foreground_counts = np.bincount(
            inverse[unseen_target], minlength=pattern_count
        )
        background_counts = np.bincount(
            inverse[~unseen_target], minlength=pattern_count
        )
        for pattern_index, pixel_index in enumerate(first_indices):
            members = np.flatnonzero(footprints[:, pixel_index])
            if members.size == 0:
                continue
            patterns.append(
                {
                    "view": view,
                    "members": tuple(int(index) for index in members),
                    "foreground": int(foreground_counts[pattern_index]),
                    "background": int(background_counts[pattern_index]),
                }
            )
    return patterns


def _effective_floor_numerators(floors):
    # np.ceil is deliberately applied after multiplication: each requested
    # threshold is conservatively represented on the required millionth grid.
    return np.ceil(floors * _FLOOR_SCALE).astype(np.int64)


def _report_template(floors, floor_numerators, time_limit, pattern_count):
    return {
        "backend": "scipy.optimize.milp",
        "backend_version": scipy.__version__,
        "status": None,
        "solver_status_code": None,
        "solver_message": None,
        "selected": None,
        "exact_ious": None,
        "requested_floors": floors.tolist(),
        "effective_floors": (floor_numerators / _FLOOR_SCALE).tolist(),
        "floor_rule": "ceil(requested_floor*1e6)/1e6",
        "compressed_pattern_count": int(pattern_count),
        "time_limit_seconds": time_limit,
        "objective_caveat": _OBJECTIVE_CAVEAT,
        "caveat": _CAVEAT,
        "printability_claim": False,
    }


def _recompute(base, target, additions, selected, floor_numerators):
    prediction = base.copy()
    for index in selected:
        prediction |= additions[index]
    positives, true_positives, false_positives, ious = _base_counts(
        prediction, target
    )
    floor_met = (
        _FLOOR_SCALE * true_positives - floor_numerators * false_positives
        >= floor_numerators * positives
    )
    return ious.tolist(), floor_met


def _finish_without_variables(
    base, target, additions, floors, floor_numerators, time_limit
):
    report = _report_template(floors, floor_numerators, time_limit, 0)
    exact_ious, floor_met = _recompute(
        base, target, additions, [], floor_numerators
    )
    report["exact_ious"] = exact_ious
    report["solver_message"] = "No candidate variables; base union evaluated directly."
    if np.all(floor_met):
        report.update(status="optimal", selected=[])
        return [], report
    report["status"] = "infeasible_for_finite_candidate_family"
    return None, report


def solve_union(
    base_predictions,
    targets,
    additions,
    floors,
    *,
    max_selected=4,
    time_limit=5.0,
):
    """Select at most ``max_selected`` additions under exact per-view IoU floors.

    The candidate footprints are unioned as booleans, so overlapping pixels are
    counted once.  A returned incumbent has been independently replayed and
    checked against the conservative millionth-grid floor constraints.
    """
    (
        base,
        target,
        candidates,
        floor_array,
        selection_limit,
        limit_seconds,
    ) = _validated_inputs(
        base_predictions, targets, additions, floors, max_selected, time_limit
    )
    candidate_count = candidates.shape[0]
    floor_numerators = _effective_floor_numerators(floor_array)
    if candidate_count == 0:
        return _finish_without_variables(
            base,
            target,
            candidates,
            floor_array,
            floor_numerators,
            limit_seconds,
        )

    positives, base_tp, base_fp, base_ious = _base_counts(base, target)
    patterns = _compressed_patterns(base, target, candidates)
    pattern_count = len(patterns)
    variable_count = candidate_count + pattern_count

    # scipy.optimize.milp minimizes. Pattern coefficients are therefore the
    # negative of normalized foreground gain plus the base-IoU-weighted
    # normalized background cost. A minute x cost makes otherwise exact ties
    # prefer fewer and then lower-index candidates without driving the metric.
    objective = np.zeros(variable_count, dtype=np.float64)
    objective[:candidate_count] = (
        1e-9 * (np.arange(candidate_count, dtype=np.float64) + 1)
        / (candidate_count + 1)
    )
    for offset, pattern in enumerate(patterns):
        view = pattern["view"]
        normalizer = max(int(positives[view]), 1) * base.shape[0]
        objective[candidate_count + offset] = (
            base_ious[view] * pattern["background"] - pattern["foreground"]
        ) / normalizer

    rows = []
    columns = []
    values = []
    lower = []
    upper = []

    def add_constraint(coefficients, lower_bound, upper_bound):
        row = len(lower)
        for column, coefficient in coefficients:
            if coefficient:
                rows.append(row)
                columns.append(column)
                values.append(float(coefficient))
        lower.append(float(lower_bound))
        upper.append(float(upper_bound))

    add_constraint(
        ((index, 1.0) for index in range(candidate_count)),
        -np.inf,
        selection_limit,
    )
    patterns_by_view = [[] for _ in range(base.shape[0])]
    for offset, pattern in enumerate(patterns):
        y_column = candidate_count + offset
        patterns_by_view[pattern["view"]].append((y_column, pattern))
        for member in pattern["members"]:
            add_constraint(((member, 1.0), (y_column, -1.0)), -np.inf, 0.0)
        add_constraint(
            [(y_column, 1.0)]
            + [(member, -1.0) for member in pattern["members"]],
            -np.inf,
            0.0,
        )
    for view, numerator in enumerate(floor_numerators):
        coefficients = []
        for y_column, pattern in patterns_by_view[view]:
            coefficients.append(
                (
                    y_column,
                    _FLOOR_SCALE * pattern["foreground"]
                    - int(numerator) * pattern["background"],
                )
            )
        constant = _FLOOR_SCALE * int(base_tp[view]) - int(numerator) * int(
            base_fp[view]
        )
        required = int(numerator) * int(positives[view]) - constant
        add_constraint(coefficients, required, np.inf)

    matrix = coo_matrix(
        (values, (rows, columns)), shape=(len(lower), variable_count)
    ).tocsr()
    constraints = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    report = _report_template(
        floor_array, floor_numerators, limit_seconds, pattern_count
    )
    result = milp(
        objective,
        integrality=np.ones(variable_count, dtype=np.uint8),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=constraints,
        options={"time_limit": limit_seconds, "presolve": True},
    )
    status_code = getattr(result, "status", None)
    report["solver_status_code"] = (
        int(status_code) if isinstance(status_code, (int, np.integer)) else None
    )
    report["solver_message"] = str(getattr(result, "message", ""))
    if status_code == 2:
        report["status"] = "infeasible_for_finite_candidate_family"
        return None, report
    if status_code not in (0, 1):
        report["status"] = "solver_terminated_without_valid_incumbent"
        return None, report

    solution = getattr(result, "x", None)
    if solution is None:
        report["status"] = "invalid_solver_result_no_incumbent"
        return None, report
    solution = np.asarray(solution, dtype=np.float64)
    if (
        solution.shape != (variable_count,)
        or not np.isfinite(solution).all()
        or np.any(solution < -_INTEGRAL_TOLERANCE)
        or np.any(solution > 1 + _INTEGRAL_TOLERANCE)
        or np.any(np.abs(solution - np.rint(solution)) > _INTEGRAL_TOLERANCE)
    ):
        report["status"] = "invalid_solver_result_nonintegral"
        return None, report
    selected = np.flatnonzero(np.rint(solution[:candidate_count])).astype(int).tolist()
    if len(selected) > selection_limit:
        report["status"] = "invalid_solver_result_cardinality"
        return None, report
    exact_ious, floor_met = _recompute(
        base, target, candidates, selected, floor_numerators
    )
    report["exact_ious"] = exact_ious
    if not np.all(floor_met):
        report["status"] = "invalid_solver_result_floor_violation"
        return None, report

    report["selected"] = selected
    report["status"] = "optimal" if status_code == 0 else "time_limit_valid_incumbent"
    return selected, report


__all__ = ["solve_union"]
