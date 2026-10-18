"""Public v6 soft-voxel core API."""

from importlib import import_module

from .core import (
    CANONICAL_VIEW_DESCRIPTORS,
    FIXED_CAMERA_AXES,
    PROFILE_FLAGS,
    PROFILE_ORDER,
    TASK_ORDER,
    background_hard_negative_tail_loss,
    build_fixed_visual_hull_logits,
    compose_v6_gradient,
    delivery_lovasz_loss,
    fixed_camera_payload,
    map_suite_views,
    v6_auxiliary_multiplier,
    v6_component_budgets,
    v6_learning_rate,
)


_TRAINER_EXPORTS = frozenset(
    {
        "CompletedRun",
        "ResumableRun",
        "StepTelemetry",
        "train_cell",
        "validate_completed_run",
        "validate_resumable_run",
    }
)


def __getattr__(name: str):
    """Load trainer exports lazily so the runner module remains warning-free."""

    if name not in _TRAINER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    trainer = import_module(f"{__name__}.trainer")
    value = getattr(trainer, name)
    globals()[name] = value
    return value

__all__ = [
    "CANONICAL_VIEW_DESCRIPTORS",
    "FIXED_CAMERA_AXES",
    "PROFILE_FLAGS",
    "PROFILE_ORDER",
    "TASK_ORDER",
    "background_hard_negative_tail_loss",
    "build_fixed_visual_hull_logits",
    "compose_v6_gradient",
    "delivery_lovasz_loss",
    "fixed_camera_payload",
    "map_suite_views",
    "v6_auxiliary_multiplier",
    "v6_component_budgets",
    "v6_learning_rate",
]
