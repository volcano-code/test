"""Isolated soft-only loss v3 package."""

from importlib import import_module

from .diagnostics import ComponentGradientSample, measure_component_gradients
from .losses import (
    V3Profile,
    balanced_lp_loss,
    compute_baseline_branch,
    compute_dic_branch,
    compute_dts_branch,
    dic_margin_loss,
    dic_schedule,
    dts_endpoint_loss,
    dts_schedule,
    registered_profiles,
    resolve_profile,
)
from .lovasz import lovasz_hinge_per_view
from .provenance import canonical_hash, implementation_source_manifest


_TRAINER_EXPORTS = frozenset(
    {
        "ResumeState",
        "SoftTrainingStepError",
        "StepRecord",
        "build_parser",
        "run",
        "train_one_update",
    }
)


def __getattr__(name: str):
    """Load trainer exports lazily so its ``python -m`` CLI executes once."""

    if name not in _TRAINER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    trainer = import_module(f"{__name__}.trainer")
    value = getattr(trainer, name)
    globals()[name] = value
    return value

__all__ = [
    "ComponentGradientSample",
    "ResumeState",
    "SoftTrainingStepError",
    "StepRecord",
    "V3Profile",
    "balanced_lp_loss",
    "build_parser",
    "canonical_hash",
    "compute_baseline_branch",
    "compute_dic_branch",
    "compute_dts_branch",
    "dic_margin_loss",
    "dic_schedule",
    "dts_endpoint_loss",
    "dts_schedule",
    "lovasz_hinge_per_view",
    "implementation_source_manifest",
    "measure_component_gradients",
    "registered_profiles",
    "resolve_profile",
    "run",
    "train_one_update",
]
