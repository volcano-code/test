"""Public physical-registration and auxiliary-gradient API for soft-loss v7."""

from .core import (
    build_registered_target_stack,
    build_registered_visual_hull_logits,
    compose_v7_gradient,
    coverage_hit_loss,
    dhw_to_xyz,
    leakage_barrier_loss,
    v7_auxiliary_multiplier,
    v7_auxiliary_slot,
    v7_learning_rate,
    xyz_to_dhw,
)
from .protocol import RegisteredEARProtocol, registered_camera_payload


__all__ = [
    "RegisteredEARProtocol",
    "build_registered_target_stack",
    "build_registered_visual_hull_logits",
    "compose_v7_gradient",
    "coverage_hit_loss",
    "dhw_to_xyz",
    "leakage_barrier_loss",
    "registered_camera_payload",
    "v7_auxiliary_multiplier",
    "v7_auxiliary_slot",
    "v7_learning_rate",
    "xyz_to_dhw",
]
