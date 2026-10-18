"""Public Task 1 exports for soft_voxel_v4."""

from .diagnostics import (
    GradientEngineeringGate,
    RedundancyGate,
    validate_gradient_engineering_gate,
    validate_redundancy_gate,
)
from .gradients import GuardedAuxiliaryGradient, guard_auxiliary_gradient
from .losses import (
    AuxiliarySlot,
    DirectionalDICResult,
    V4Profile,
    asymmetric_dts_loss,
    auxiliary_slot,
    balanced_lp_loss,
    compute_baseline_branch,
    directional_dic_loss,
    directional_dic_terms,
    registered_profiles,
    resolve_profile,
)
from .trainer_step import AuxiliarySlotAudit, V4StepConfig, V4StepResult, train_one_v4_update


__all__ = [
    "AuxiliarySlot",
    "AuxiliarySlotAudit",
    "DirectionalDICResult",
    "GradientEngineeringGate",
    "GuardedAuxiliaryGradient",
    "RedundancyGate",
    "V4StepConfig",
    "V4Profile",
    "V4StepResult",
    "asymmetric_dts_loss",
    "auxiliary_slot",
    "balanced_lp_loss",
    "compute_baseline_branch",
    "directional_dic_loss",
    "directional_dic_terms",
    "guard_auxiliary_gradient",
    "registered_profiles",
    "resolve_profile",
    "train_one_v4_update",
    "validate_gradient_engineering_gate",
    "validate_redundancy_gate",
]
