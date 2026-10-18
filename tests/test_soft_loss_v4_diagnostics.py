import json
import math
from dataclasses import FrozenInstanceError, asdict, fields

import numpy as np
import pytest

from shadowforge_pipeline.soft_voxel_v4 import auxiliary_slot, resolve_profile
import shadowforge_pipeline.soft_voxel_v4.diagnostics as diagnostics_module
from shadowforge_pipeline.soft_voxel_v4.diagnostics import (
    GradientEngineeringGate,
    RedundancyGate,
    validate_gradient_engineering_gate,
    validate_redundancy_gate,
)
from shadowforge_pipeline.soft_voxel_v4.trainer_step import AuxiliarySlotAudit


def _eligible_slots(profile_name: str, total_steps: int) -> list[tuple[int, object]]:
    profile = resolve_profile(profile_name)
    slots: list[tuple[int, object]] = []
    for step in range(total_steps):
        progress = (step + 1) / float(total_steps)
        slot = auxiliary_slot(step, progress, profile)
        if slot is not None:
            slots.append((step, slot))
    return slots


def _audit(
    step: int,
    *,
    profile_name: str,
    component: str,
    endpoint: float | None,
    multiplier: float,
    fully_projected_zero: bool = False,
    raw_norm: float = 2.0,
    projected_norm: float = 1.0,
    final_norm: float = 1.0,
    cap_applied: bool = False,
    full_relations: bool | None = None,
    dts_004_cosine: float | None = 0.5,
    dts_006_cosine: float | None = -0.5,
    dts_004_iou: float | None = 0.25,
    dts_006_iou: float | None = 0.5,
    dts_004_invalid: bool = False,
    dts_006_invalid: bool = False,
    dts_004_iou_invalid: bool = False,
    dts_006_iou_invalid: bool = False,
    dts_004_union: int | None = 8,
    dts_006_union: int | None = 8,
    dts_004_intersection: int | None = 2,
    dts_006_intersection: int | None = 4,
) -> AuxiliarySlotAudit:
    cross_expected = (
        profile_name in {"v4_full", "v4_no_l1", "v4_no_l2"}
        if full_relations is None
        else full_relations
    )
    if component == "dic":
        diagnostic_dic = raw_norm
        diagnostic_dts_004 = 1.5
        diagnostic_dts_006 = 1.0
    elif endpoint == 0.04:
        diagnostic_dic = 1.25
        diagnostic_dts_004 = raw_norm
        diagnostic_dts_006 = 1.0
    else:
        diagnostic_dic = 1.25
        diagnostic_dts_004 = 1.5
        diagnostic_dts_006 = raw_norm

    if not cross_expected:
        diagnostic_dic = None
        diagnostic_dts_004 = None
        diagnostic_dts_006 = None
        dts_004_union = None
        dts_006_union = None
        dts_004_intersection = None
        dts_006_intersection = None
        dts_004_cosine = None
        dts_006_cosine = None
        dts_004_iou = None
        dts_006_iou = None
        dts_004_invalid = True
        dts_006_invalid = True
        dts_004_iou_invalid = True
        dts_006_iou_invalid = True

    return AuxiliarySlotAudit(
        global_step_before=step,
        profile=profile_name,
        cross_diagnostics_expected=cross_expected,
        component=component,
        endpoint=endpoint,
        multiplier=float(multiplier),
        unit_coefficient=2.0 if component == "dic" else 3.0,
        max_ratio=0.04 if component == "dic" else 0.08,
        loss_auxiliary=0.75,
        raw_component_gradient_norm=float(raw_norm),
        weighted_auxiliary_gradient_norm=0.5,
        projected_auxiliary_gradient_norm=float(projected_norm),
        final_auxiliary_gradient_norm=float(final_norm),
        anchor_cosine_before=-0.25,
        anchor_cosine_after=0.25,
        removed_conflict_fraction=0.5,
        auxiliary_base_ratio=0.04,
        cap_applied=cap_applied,
        fully_projected_zero=fully_projected_zero,
        diagnostic_dic_raw_gradient_norm=diagnostic_dic,
        diagnostic_dts_004_raw_gradient_norm=diagnostic_dts_004,
        diagnostic_dts_006_raw_gradient_norm=diagnostic_dts_006,
        dic_dts_004_support_intersection=dts_004_intersection,
        dic_dts_004_support_union=dts_004_union,
        dic_dts_006_support_intersection=dts_006_intersection,
        dic_dts_006_support_union=dts_006_union,
        dic_dts_004_cosine=dts_004_cosine,
        dic_dts_004_cosine_invalid=dts_004_invalid,
        dic_dts_006_cosine=dts_006_cosine,
        dic_dts_006_cosine_invalid=dts_006_invalid,
        dic_dts_004_support_iou=dts_004_iou,
        dic_dts_004_support_iou_invalid=dts_004_iou_invalid,
        dic_dts_006_support_iou=dts_006_iou,
        dic_dts_006_support_iou_invalid=dts_006_iou_invalid,
    )


def _audits(profile_name: str, total_steps: int) -> list[AuxiliarySlotAudit]:
    result: list[AuxiliarySlotAudit] = []
    for step, slot in _eligible_slots(profile_name, total_steps):
        result.append(
            _audit(
                step,
                profile_name=profile_name,
                component=slot.component,
                endpoint=slot.endpoint,
                multiplier=slot.multiplier,
            )
        )
    return result


def _replace_dic_max_ratio(
    audits: list[AuxiliarySlotAudit],
    *,
    max_ratio: float,
) -> list[AuxiliarySlotAudit]:
    updated: list[AuxiliarySlotAudit] = []
    for audit in audits:
        payload = asdict(audit)
        if payload["component"] == "dic":
            payload["max_ratio"] = max_ratio
        updated.append(AuxiliarySlotAudit(**payload))
    return updated


def test_public_gate_exports_are_wired() -> None:
    import shadowforge_pipeline.soft_voxel_v4 as v4

    assert v4.validate_gradient_engineering_gate is validate_gradient_engineering_gate
    assert v4.validate_redundancy_gate is validate_redundancy_gate
    assert v4.GradientEngineeringGate is GradientEngineeringGate
    assert v4.RedundancyGate is RedundancyGate


def test_gate_dataclasses_are_exact_frozen_and_json_safe() -> None:
    engineering = GradientEngineeringGate("PASS", 4, 0, 0.0, ())
    redundancy = RedundancyGate("STOP", 4, 1, ("REDUNDANT_AUXILIARY_DESIGN",))

    assert [field.name for field in fields(GradientEngineeringGate)] == [
        "status",
        "eligible_slots",
        "fully_projected_zero_slots",
        "fully_projected_zero_fraction",
        "failure_codes",
    ]
    assert [field.name for field in fields(RedundancyGate)] == [
        "status",
        "eligible_slots",
        "redundant_slots",
        "failure_codes",
    ]
    engineering_payload = json.loads(json.dumps(asdict(engineering), allow_nan=False))
    redundancy_payload = json.loads(json.dumps(asdict(redundancy), allow_nan=False))
    assert engineering_payload == {
        "status": "PASS",
        "eligible_slots": 4,
        "fully_projected_zero_slots": 0,
        "fully_projected_zero_fraction": 0.0,
        "failure_codes": [],
    }
    assert redundancy_payload == {
        "status": "STOP",
        "eligible_slots": 4,
        "redundant_slots": 1,
        "failure_codes": ["REDUNDANT_AUXILIARY_DESIGN"],
    }
    with pytest.raises(FrozenInstanceError):
        engineering.status = "STOP"
    with pytest.raises(FrozenInstanceError):
        redundancy.redundant_slots = 0


def test_projection_gate_reconstructs_exact_eligible_sequence_before_arithmetic() -> None:
    total_steps = 20
    audits = _audits("v4_full", total_steps)

    gate = validate_gradient_engineering_gate(
        audits,
        profile=resolve_profile("v4_full"),
        total_steps=total_steps,
    )

    assert gate == GradientEngineeringGate("PASS", len(audits), 0, 0.0, ())


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda audits: audits[1:], "missing"),
        (lambda audits: audits + [audits[-1]], "duplicate"),
        (lambda audits: [audits[1], audits[0], *audits[2:]], "reordered"),
        (
            lambda audits: [
                _audit(
                    audits[0].global_step_before,
                    profile_name="v4_full",
                    component="dts" if audits[0].component == "dic" else "dic",
                    endpoint=0.04 if audits[0].component == "dic" else None,
                    multiplier=audits[0].multiplier,
                ),
                *audits[1:],
            ],
            "mismatched",
        ),
    ],
)
def test_projection_gate_rejects_missing_duplicate_reordered_and_mismatched_audits(
    mutator,
    message: str,
) -> None:
    audits = _audits("v4_full", 20)

    with pytest.raises(ValueError, match=message):
        validate_gradient_engineering_gate(
            mutator(audits),
            profile=resolve_profile("v4_full"),
            total_steps=20,
        )


def test_single_profile_projection_gate_accepts_structural_absence_of_cross_diagnostics() -> None:
    audits = _audits("dic_only", 20)

    gate = validate_gradient_engineering_gate(
        audits,
        profile=resolve_profile("dic_only"),
        total_steps=20,
    )

    assert gate.status == "PASS"
    assert gate.eligible_slots == len(audits)
    assert gate.fully_projected_zero_slots == 0
    assert gate.failure_codes == ()


@pytest.mark.parametrize("profile_name", ["dic_only", "v4_full"])
@pytest.mark.parametrize("dic_max_ratio", [0.08, 0.12])
def test_registered_dic_max_ratios_remain_valid_gate_evidence(
    profile_name: str,
    dic_max_ratio: float,
) -> None:
    total_steps = 20
    audits = _replace_dic_max_ratio(_audits(profile_name, total_steps), max_ratio=dic_max_ratio)

    gate = validate_gradient_engineering_gate(
        audits,
        profile=resolve_profile(profile_name),
        total_steps=total_steps,
    )

    assert gate.status == "PASS"
    if profile_name == "v4_full":
        redundancy = validate_redundancy_gate(
            audits,
            profile=resolve_profile(profile_name),
            total_steps=total_steps,
        )

        assert redundancy.status == "PASS"


def test_projection_gate_treats_exact_ten_percent_as_pass_and_next_float_above_as_stop() -> None:
    total_steps = 50
    audits = _audits("v4_full", total_steps)
    tenth = len(audits) // 10
    for index in range(tenth):
        audits[index] = _audit(
            audits[index].global_step_before,
            profile_name="v4_full",
            component=audits[index].component,
            endpoint=audits[index].endpoint,
            multiplier=audits[index].multiplier,
            fully_projected_zero=True,
            raw_norm=2.0,
            projected_norm=0.0,
            final_norm=0.0,
        )

    pass_gate = validate_gradient_engineering_gate(
        audits,
        profile=resolve_profile("v4_full"),
        total_steps=total_steps,
    )

    assert pass_gate.status == "PASS"
    assert pass_gate.fully_projected_zero_fraction == 0.10

    extra = audits[tenth]
    audits[tenth] = _audit(
        extra.global_step_before,
        profile_name="v4_full",
        component=extra.component,
        endpoint=extra.endpoint,
        multiplier=extra.multiplier,
        fully_projected_zero=True,
        raw_norm=math.nextafter(0.0, 1.0),
        projected_norm=0.0,
        final_norm=0.0,
    )
    stop_gate = validate_gradient_engineering_gate(
        audits,
        profile=resolve_profile("v4_full"),
        total_steps=total_steps,
    )

    assert stop_gate.status == "STOP"
    assert stop_gate.failure_codes == ("EXCESS_PROJECTION_ZERO",)
    assert stop_gate.fully_projected_zero_fraction > math.nextafter(0.10, 1.0)


def test_projection_gate_does_not_count_budget_cap_to_zero_as_projection_zero() -> None:
    audits = _audits("v4_full", 20)
    audits[0] = _audit(
        audits[0].global_step_before,
        profile_name="v4_full",
        component=audits[0].component,
        endpoint=audits[0].endpoint,
        multiplier=audits[0].multiplier,
        raw_norm=2.0,
        projected_norm=0.5,
        final_norm=0.0,
        cap_applied=True,
        fully_projected_zero=False,
    )

    gate = validate_gradient_engineering_gate(
        audits,
        profile=resolve_profile("v4_full"),
        total_steps=20,
    )

    assert gate.status == "PASS"
    assert gate.fully_projected_zero_slots == 0


def test_redundancy_gate_requires_both_auxiliary_components_and_exact_strict_thresholds() -> None:
    total_steps = 20
    audits = _audits("v4_full", total_steps)
    exact = math.nextafter(0.95, 0.0)
    audits[0] = _audit(
        audits[0].global_step_before,
        profile_name="v4_full",
        component=audits[0].component,
        endpoint=audits[0].endpoint,
        multiplier=audits[0].multiplier,
        dts_004_cosine=exact,
        dts_006_cosine=-exact,
        dts_004_iou=19.0 / 20.0,
        dts_006_iou=19.0 / 20.0,
        dts_004_union=20,
        dts_006_union=20,
        dts_004_intersection=19,
        dts_006_intersection=19,
    )

    gate = validate_redundancy_gate(
        audits,
        profile=resolve_profile("v4_full"),
        total_steps=total_steps,
    )

    assert gate.status == "PASS"
    assert gate.redundant_slots == 0

    above = math.nextafter(0.95, 1.0)
    audits[0] = _audit(
        audits[0].global_step_before,
        profile_name="v4_full",
        component=audits[0].component,
        endpoint=audits[0].endpoint,
        multiplier=audits[0].multiplier,
        dts_004_cosine=above,
        dts_006_cosine=-above,
        dts_004_iou=20.0 / 21.0,
        dts_006_iou=20.0 / 21.0,
        dts_004_union=21,
        dts_006_union=21,
        dts_004_intersection=20,
        dts_006_intersection=20,
    )
    stop_gate = validate_redundancy_gate(
        audits,
        profile=resolve_profile("v4_full"),
        total_steps=total_steps,
    )

    assert stop_gate.status == "STOP"
    assert stop_gate.redundant_slots == 1
    assert stop_gate.failure_codes == ("REDUNDANT_AUXILIARY_DESIGN",)


def test_redundancy_gate_treats_zero_norm_and_empty_union_relations_as_legal_nonredundant() -> None:
    audits = _audits("v4_full", 20)
    audits[0] = _audit(
        audits[0].global_step_before,
        profile_name="v4_full",
        component=audits[0].component,
        endpoint=audits[0].endpoint,
        multiplier=audits[0].multiplier,
        dts_004_cosine=None,
        dts_006_cosine=None,
        dts_004_iou=None,
        dts_006_iou=None,
        dts_004_invalid=True,
        dts_006_invalid=True,
        dts_004_iou_invalid=True,
        dts_006_iou_invalid=True,
        dts_004_union=0,
        dts_006_union=0,
        dts_004_intersection=0,
        dts_006_intersection=0,
        raw_norm=0.0,
    )
    payload = asdict(audits[0])
    payload["diagnostic_dts_004_raw_gradient_norm"] = 0.0
    payload["diagnostic_dts_006_raw_gradient_norm"] = 0.0
    audits[0] = AuxiliarySlotAudit(**payload)

    gate = validate_redundancy_gate(
        audits,
        profile=resolve_profile("v4_full"),
        total_steps=20,
    )

    assert gate.status == "PASS"
    assert gate.redundant_slots == 0


def test_redundancy_gate_fails_closed_on_incomplete_full_diagnostics() -> None:
    audits = _audits("v4_full", 20)
    broken = asdict(audits[0])
    broken["diagnostic_dts_006_raw_gradient_norm"] = None
    incomplete = AuxiliarySlotAudit(**broken)

    with pytest.raises(ValueError, match="full diagnostics"):
        validate_redundancy_gate(
            [incomplete, *audits[1:]],
            profile=resolve_profile("v4_full"),
            total_steps=20,
        )


@pytest.mark.parametrize(
    ("index", "patch"),
    [
        (0, {"profile": np.str_("v4_full")}),
        (0, {"component": np.str_("dic")}),
        (1, {"endpoint": np.float64(0.04)}),
    ],
)
def test_validators_require_exact_builtin_profile_component_and_endpoint_types(
    index: int,
    patch: dict[str, object],
) -> None:
    audits = _audits("v4_full", 20)
    payload = asdict(audits[index])
    payload.update(patch)
    malformed = AuxiliarySlotAudit(**payload)

    with pytest.raises(ValueError):
        validate_gradient_engineering_gate(
            [*audits[:index], malformed, *audits[index + 1 :]],
            profile=resolve_profile("v4_full"),
            total_steps=20,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dic_dts_004_cosine_invalid", 1),
        ("dic_dts_004_cosine_invalid", np.bool_(True)),
        ("dic_dts_006_support_iou_invalid", 1),
        ("dic_dts_006_support_iou_invalid", np.bool_(False)),
    ],
)
def test_validators_reject_non_builtin_invalid_flags(field: str, value: object) -> None:
    audits = _audits("v4_full", 20)
    payload = asdict(audits[0])
    payload[field] = value
    malformed = AuxiliarySlotAudit(**payload)

    with pytest.raises(ValueError):
        validate_gradient_engineering_gate(
            [malformed, *audits[1:]],
            profile=resolve_profile("v4_full"),
            total_steps=20,
        )


@pytest.mark.parametrize("profile_name", ["dic_only", "dts_only"])
@pytest.mark.parametrize(
    "field",
    [
        "dic_dts_004_cosine_invalid",
        "dic_dts_006_cosine_invalid",
        "dic_dts_004_support_iou_invalid",
        "dic_dts_006_support_iou_invalid",
    ],
)
@pytest.mark.parametrize("value", [1, np.bool_(True)])
def test_single_component_validators_reject_truthy_non_builtin_invalid_flags(
    profile_name: str,
    field: str,
    value: object,
) -> None:
    """Break caught: single-component omitted relation flags must be exact bools, not just truthy."""
    audits = _audits(profile_name, 20)

    gate = validate_gradient_engineering_gate(
        audits,
        profile=resolve_profile(profile_name),
        total_steps=20,
    )

    assert gate.status == "PASS"

    payload = asdict(audits[0])
    payload[field] = value
    malformed = AuxiliarySlotAudit(**payload)

    with pytest.raises(ValueError):
        validate_gradient_engineering_gate(
            [malformed, *audits[1:]],
            profile=resolve_profile(profile_name),
            total_steps=20,
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda audit: {
            **audit,
            "dic_dts_004_cosine": None,
            "dic_dts_004_cosine_invalid": True,
        },
        lambda audit: {
            **audit,
            "diagnostic_dts_004_raw_gradient_norm": 0.0,
            "dic_dts_004_cosine": 0.0,
            "dic_dts_004_cosine_invalid": False,
        },
    ],
)
def test_validators_require_cosine_invalid_flags_to_match_participating_norms(mutator) -> None:
    audits = _audits("v4_full", 20)
    payload = asdict(audits[0])
    malformed = AuxiliarySlotAudit(**mutator(payload))

    with pytest.raises(ValueError):
        validate_gradient_engineering_gate(
            [malformed, *audits[1:]],
            profile=resolve_profile("v4_full"),
            total_steps=20,
        )


@pytest.mark.parametrize("profile_name", ["l12_bal", "dic_only", "dts_only"])
def test_redundancy_gate_rejects_profiles_without_both_auxiliaries(profile_name: str) -> None:
    audits = _audits("dic_only" if profile_name == "l12_bal" else profile_name, 20)

    with pytest.raises(ValueError, match="both auxiliaries"):
        validate_redundancy_gate(
            audits,
            profile=resolve_profile(profile_name),
            total_steps=20,
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda audit: {**audit, "profile": 1},
        lambda audit: {**audit, "multiplier": True},
        lambda audit: {**audit, "max_ratio": 0.05},
        lambda audit: {**audit, "raw_component_gradient_norm": float("nan")},
        lambda audit: {**audit, "dic_dts_004_support_intersection": 9, "dic_dts_004_support_union": 8},
        lambda audit: {**audit, "dic_dts_004_cosine": 0.2, "dic_dts_004_cosine_invalid": True},
        lambda audit: {**audit, "dic_dts_004_support_iou": None, "dic_dts_004_support_iou_invalid": False},
        lambda audit: {**audit, "fully_projected_zero": True, "raw_component_gradient_norm": 0.0, "projected_auxiliary_gradient_norm": 0.0},
        lambda audit: {**audit, "unit_coefficient": -0.1},
        lambda audit: {**audit, "raw_component_gradient_norm": -0.1},
        lambda audit: {**audit, "weighted_auxiliary_gradient_norm": -0.1},
        lambda audit: {**audit, "projected_auxiliary_gradient_norm": -0.1},
        lambda audit: {**audit, "final_auxiliary_gradient_norm": -0.1},
        lambda audit: {**audit, "diagnostic_dic_raw_gradient_norm": -0.1},
        lambda audit: {**audit, "removed_conflict_fraction": -0.1},
        lambda audit: {**audit, "removed_conflict_fraction": 1.1},
        lambda audit: {**audit, "auxiliary_base_ratio": -0.1},
        lambda audit: {**audit, "dic_dts_004_cosine": 1.0000001},
    ],
)
def test_validators_reject_nonexact_nonfinite_and_inconsistent_audits(mutator) -> None:
    audits = _audits("v4_full", 20)
    payload = asdict(audits[0])
    malformed = AuxiliarySlotAudit(**mutator(payload))

    with pytest.raises(ValueError):
        validate_gradient_engineering_gate(
            [malformed, *audits[1:]],
            profile=resolve_profile("v4_full"),
            total_steps=20,
        )


def test_projection_gate_rejects_l12_bal_as_non_evidence() -> None:
    with pytest.raises(ValueError, match="eligible"):
        validate_gradient_engineering_gate(
            (),
            profile=resolve_profile("l12_bal"),
            total_steps=20,
        )


def test_diagnostics_module_scope_is_v4_only() -> None:
    source = diagnostics_module.__file__
    assert source.endswith("shadowforge_pipeline/soft_voxel_v4/diagnostics.py")
