"""Engineering-gate validators for soft_voxel_v4 auxiliary audits."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from .losses import V4Profile, auxiliary_slot, resolve_profile
from .trainer_step import AuxiliarySlotAudit


_FULL_PROFILES = {"v4_full", "v4_no_l1", "v4_no_l2"}
_ALLOWED_PROFILES = {"dic_only", "dts_only", "v4_full", "v4_no_l1", "v4_no_l2"}
_ALLOWED_RATIOS = {0.04, 0.08, 0.12}


@dataclass(frozen=True)
class GradientEngineeringGate:
    status: str
    eligible_slots: int
    fully_projected_zero_slots: int
    fully_projected_zero_fraction: float
    failure_codes: tuple[str, ...]


@dataclass(frozen=True)
class RedundancyGate:
    status: str
    eligible_slots: int
    redundant_slots: int
    failure_codes: tuple[str, ...]


def _validate_profile(profile: object) -> V4Profile:
    if not isinstance(profile, V4Profile):
        raise ValueError("profile must be a V4Profile")
    registered = resolve_profile(profile.name)
    if profile is not registered:
        raise ValueError("profile must be registered")
    return profile


def _validate_total_steps(total_steps: object) -> int:
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    return total_steps


def _eligible_sequence(profile: V4Profile, total_steps: int) -> list[tuple[int, object]]:
    expected: list[tuple[int, object]] = []
    for step in range(total_steps):
        progress = (step + 1) / float(total_steps)
        slot = auxiliary_slot(step, progress, profile)
        if slot is not None:
            expected.append((step, slot))
    return expected


def _strict_float(name: str, value: object) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite Python float")
    return value


def _strict_nonnegative_float(name: str, value: object) -> float:
    scalar = _strict_float(name, value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _strict_fraction(name: str, value: object) -> float:
    scalar = _strict_float(name, value)
    if scalar < 0.0 or scalar > 1.0:
        raise ValueError(f"{name} must be within [0,1]")
    return scalar


def _strict_str(name: str, value: object) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be an exact str")
    return value


def _strict_cosine(name: str, value: object) -> float:
    scalar = _strict_float(name, value)
    if scalar < -1.0 or scalar > 1.0:
        raise ValueError(f"{name} must be clamped to [-1,1]")
    return scalar


def _strict_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _strict_int(name: str, value: object) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an exact int")
    return value


def _validate_single_or_full_profile_name(audit: AuxiliarySlotAudit) -> None:
    _strict_str("profile", audit.profile)
    if audit.profile not in _ALLOWED_PROFILES:
        raise ValueError("audit profile is invalid")


def _validate_full_diagnostics(audit: AuxiliarySlotAudit) -> None:
    if audit.diagnostic_dic_raw_gradient_norm is None:
        raise ValueError("full diagnostics require diagnostic DIC norm")
    if audit.diagnostic_dts_004_raw_gradient_norm is None or audit.diagnostic_dts_006_raw_gradient_norm is None:
        raise ValueError("full diagnostics require both diagnostic DTS norms")
    dic_norm = _strict_nonnegative_float(
        "diagnostic_dic_raw_gradient_norm",
        audit.diagnostic_dic_raw_gradient_norm,
    )
    dts_004_norm = _strict_nonnegative_float(
        "diagnostic_dts_004_raw_gradient_norm",
        audit.diagnostic_dts_004_raw_gradient_norm,
    )
    dts_006_norm = _strict_nonnegative_float(
        "diagnostic_dts_006_raw_gradient_norm",
        audit.diagnostic_dts_006_raw_gradient_norm,
    )
    _validate_relation(
        prefix="dic_dts_004",
        left_norm=dic_norm,
        right_norm=dts_004_norm,
        cosine=audit.dic_dts_004_cosine,
        cosine_invalid=audit.dic_dts_004_cosine_invalid,
        support_iou=audit.dic_dts_004_support_iou,
        support_iou_invalid=audit.dic_dts_004_support_iou_invalid,
        intersection=audit.dic_dts_004_support_intersection,
        union=audit.dic_dts_004_support_union,
    )
    _validate_relation(
        prefix="dic_dts_006",
        left_norm=dic_norm,
        right_norm=dts_006_norm,
        cosine=audit.dic_dts_006_cosine,
        cosine_invalid=audit.dic_dts_006_cosine_invalid,
        support_iou=audit.dic_dts_006_support_iou,
        support_iou_invalid=audit.dic_dts_006_support_iou_invalid,
        intersection=audit.dic_dts_006_support_intersection,
        union=audit.dic_dts_006_support_union,
    )
    if audit.component == "dic":
        if audit.raw_component_gradient_norm != audit.diagnostic_dic_raw_gradient_norm:
            raise ValueError("selected raw norm mismatches full diagnostics")
    elif audit.endpoint == 0.04:
        if audit.raw_component_gradient_norm != audit.diagnostic_dts_004_raw_gradient_norm:
            raise ValueError("selected raw norm mismatches full diagnostics")
    else:
        if audit.raw_component_gradient_norm != audit.diagnostic_dts_006_raw_gradient_norm:
            raise ValueError("selected raw norm mismatches full diagnostics")


def _validate_relation(
    *,
    prefix: str,
    left_norm: float,
    right_norm: float,
    cosine: float | None,
    cosine_invalid: bool,
    support_iou: float | None,
    support_iou_invalid: bool,
    intersection: int | None,
    union: int | None,
) -> None:
    _strict_bool(f"{prefix}_cosine_invalid", cosine_invalid)
    _strict_bool(f"{prefix}_support_iou_invalid", support_iou_invalid)
    if intersection is None or union is None:
        raise ValueError("full diagnostics require support counts")
    _strict_int(f"{prefix}_support_intersection", intersection)
    _strict_int(f"{prefix}_support_union", union)
    if intersection < 0 or union < 0 or intersection > union:
        raise ValueError("support counts must satisfy 0 <= intersection <= union")
    zero_participant = left_norm == 0.0 or right_norm == 0.0
    if cosine_invalid:
        if not zero_participant:
            raise ValueError("invalid cosine requires a zero participating norm")
        if cosine is not None:
            raise ValueError("invalid cosine must be reported as None")
    else:
        if zero_participant:
            raise ValueError("zero participating norm requires an invalid cosine flag")
        _strict_cosine(f"{prefix}_cosine", cosine)
    if support_iou_invalid:
        if union != 0 or support_iou is not None:
            raise ValueError("invalid support IoU must be substantiated by an empty union")
    else:
        value = _strict_float(f"{prefix}_support_iou", support_iou)
        if union == 0:
            raise ValueError("valid support IoU requires a non-empty union")
        if value != intersection / union:
            raise ValueError("support IoU must equal intersection/union")


def _validate_single_diagnostics(audit: AuxiliarySlotAudit) -> None:
    if (
        audit.diagnostic_dic_raw_gradient_norm is not None
        or audit.diagnostic_dts_004_raw_gradient_norm is not None
        or audit.diagnostic_dts_006_raw_gradient_norm is not None
        or audit.dic_dts_004_support_intersection is not None
        or audit.dic_dts_004_support_union is not None
        or audit.dic_dts_006_support_intersection is not None
        or audit.dic_dts_006_support_union is not None
        or audit.dic_dts_004_cosine is not None
        or audit.dic_dts_006_cosine is not None
        or audit.dic_dts_004_support_iou is not None
        or audit.dic_dts_006_support_iou is not None
    ):
        raise ValueError("single-component audits must structurally omit cross diagnostics")
    _strict_bool("dic_dts_004_cosine_invalid", audit.dic_dts_004_cosine_invalid)
    _strict_bool("dic_dts_006_cosine_invalid", audit.dic_dts_006_cosine_invalid)
    _strict_bool("dic_dts_004_support_iou_invalid", audit.dic_dts_004_support_iou_invalid)
    _strict_bool("dic_dts_006_support_iou_invalid", audit.dic_dts_006_support_iou_invalid)
    if not (
        audit.dic_dts_004_cosine_invalid
        and audit.dic_dts_006_cosine_invalid
        and audit.dic_dts_004_support_iou_invalid
        and audit.dic_dts_006_support_iou_invalid
    ):
        raise ValueError("single-component audits must mark omitted cross diagnostics invalid")


def _validate_audit(audit: object, expected_step: int, expected_slot: object, profile: V4Profile) -> AuxiliarySlotAudit:
    if not isinstance(audit, AuxiliarySlotAudit):
        raise ValueError("audits must use the exact AuxiliarySlotAudit schema")
    _validate_single_or_full_profile_name(audit)
    _strict_int("global_step_before", audit.global_step_before)
    if audit.global_step_before != expected_step:
        raise ValueError("reordered or mismatched audit step")
    if audit.profile != profile.name:
        raise ValueError("mismatched audit profile")
    _strict_str("component", audit.component)
    if audit.component != expected_slot.component or audit.endpoint != expected_slot.endpoint:
        raise ValueError("mismatched audit slot")
    _strict_float("multiplier", audit.multiplier)
    if audit.multiplier != expected_slot.multiplier:
        raise ValueError("mismatched audit multiplier")
    _strict_bool("cross_diagnostics_expected", audit.cross_diagnostics_expected)
    if audit.cross_diagnostics_expected != (profile.name in _FULL_PROFILES):
        raise ValueError("mismatched cross-diagnostics expectation")
    if audit.component not in {"dic", "dts"}:
        raise ValueError("component must be dic or dts")
    if audit.component == "dic":
        if audit.endpoint is not None:
            raise ValueError("DIC audit endpoint must be None")
    else:
        if type(audit.endpoint) is not float:
            raise ValueError("DTS endpoint must be exact")
        if audit.endpoint not in (0.04, 0.06):
            raise ValueError("DTS endpoint must be exact")
    unit_coefficient = _strict_float("unit_coefficient", audit.unit_coefficient)
    if unit_coefficient <= 0.0:
        raise ValueError("unit_coefficient must be positive")
    _strict_float("max_ratio", audit.max_ratio)
    if audit.max_ratio not in _ALLOWED_RATIOS:
        raise ValueError("max_ratio must be registered")
    _strict_float("loss_auxiliary", audit.loss_auxiliary)
    _strict_nonnegative_float("raw_component_gradient_norm", audit.raw_component_gradient_norm)
    _strict_nonnegative_float("weighted_auxiliary_gradient_norm", audit.weighted_auxiliary_gradient_norm)
    _strict_nonnegative_float("projected_auxiliary_gradient_norm", audit.projected_auxiliary_gradient_norm)
    _strict_nonnegative_float("final_auxiliary_gradient_norm", audit.final_auxiliary_gradient_norm)
    _strict_cosine("anchor_cosine_before", audit.anchor_cosine_before)
    _strict_cosine("anchor_cosine_after", audit.anchor_cosine_after)
    _strict_fraction("removed_conflict_fraction", audit.removed_conflict_fraction)
    _strict_nonnegative_float("auxiliary_base_ratio", audit.auxiliary_base_ratio)
    _strict_bool("cap_applied", audit.cap_applied)
    _strict_bool("fully_projected_zero", audit.fully_projected_zero)
    if audit.fully_projected_zero != (
        audit.raw_component_gradient_norm > 0.0 and audit.projected_auxiliary_gradient_norm == 0.0
    ):
        raise ValueError("fully_projected_zero must exactly reflect raw/projected norms")
    if audit.cross_diagnostics_expected:
        _validate_full_diagnostics(audit)
    else:
        _validate_single_diagnostics(audit)
    return audit


def _validated_sequence(
    audits: Sequence[AuxiliarySlotAudit],
    *,
    profile: V4Profile,
    total_steps: int,
) -> list[AuxiliarySlotAudit]:
    expected = _eligible_sequence(profile, total_steps)
    if not expected:
        raise ValueError("eligible auxiliary evidence is required")
    if len(audits) < len(expected):
        raise ValueError("missing eligible audit records")
    if len(audits) > len(expected):
        raise ValueError("duplicate or unexpected eligible audit records")
    validated: list[AuxiliarySlotAudit] = []
    seen_steps: set[int] = set()
    for audit, (expected_step, expected_slot) in zip(audits, expected):
        checked = _validate_audit(audit, expected_step, expected_slot, profile)
        if checked.global_step_before in seen_steps:
            raise ValueError("duplicate eligible audit step")
        seen_steps.add(checked.global_step_before)
        validated.append(checked)
    return validated


def validate_gradient_engineering_gate(
    audits: Sequence[AuxiliarySlotAudit],
    *,
    profile: V4Profile,
    total_steps: int,
) -> GradientEngineeringGate:
    checked_profile = _validate_profile(profile)
    checked_total_steps = _validate_total_steps(total_steps)
    validated = _validated_sequence(audits, profile=checked_profile, total_steps=checked_total_steps)
    fully_projected_zero_slots = sum(1 for audit in validated if audit.fully_projected_zero)
    eligible_slots = len(validated)
    fraction = fully_projected_zero_slots / eligible_slots
    if fraction > 0.10:
        return GradientEngineeringGate(
            status="STOP",
            eligible_slots=eligible_slots,
            fully_projected_zero_slots=fully_projected_zero_slots,
            fully_projected_zero_fraction=float(fraction),
            failure_codes=("EXCESS_PROJECTION_ZERO",),
        )
    return GradientEngineeringGate(
        status="PASS",
        eligible_slots=eligible_slots,
        fully_projected_zero_slots=fully_projected_zero_slots,
        fully_projected_zero_fraction=float(fraction),
        failure_codes=(),
    )


def validate_redundancy_gate(
    audits: Sequence[AuxiliarySlotAudit],
    *,
    profile: V4Profile,
    total_steps: int,
) -> RedundancyGate:
    checked_profile = _validate_profile(profile)
    if checked_profile.name not in _FULL_PROFILES:
        raise ValueError("redundancy evidence requires both auxiliaries enabled")
    checked_total_steps = _validate_total_steps(total_steps)
    validated = _validated_sequence(audits, profile=checked_profile, total_steps=checked_total_steps)
    redundant_slots = 0
    for audit in validated:
        redundant = (
            audit.dic_dts_004_cosine is not None
            and audit.dic_dts_004_support_iou is not None
            and audit.dic_dts_006_cosine is not None
            and audit.dic_dts_006_support_iou is not None
            and abs(audit.dic_dts_004_cosine) > 0.95
            and audit.dic_dts_004_support_iou > 0.95
            and abs(audit.dic_dts_006_cosine) > 0.95
            and audit.dic_dts_006_support_iou > 0.95
        )
        if redundant:
            redundant_slots += 1
    if redundant_slots:
        return RedundancyGate(
            status="STOP",
            eligible_slots=len(validated),
            redundant_slots=redundant_slots,
            failure_codes=("REDUNDANT_AUXILIARY_DESIGN",),
        )
    return RedundancyGate(
        status="PASS",
        eligible_slots=len(validated),
        redundant_slots=0,
        failure_codes=(),
    )


__all__ = [
    "GradientEngineeringGate",
    "RedundancyGate",
    "validate_gradient_engineering_gate",
    "validate_redundancy_gate",
]
