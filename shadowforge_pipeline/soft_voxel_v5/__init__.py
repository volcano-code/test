"""Public seams for the minimal soft-loss v5 ablation harness."""

from .quick_ablation import (
    INFERENCE_SCOPE,
    PROFILE_ORDER,
    TASK_ORDER,
    analyze_ablation,
    background_hard_negative_tail_loss,
    build_matrix,
    classify_summary,
    compose_v5_gradient,
    delivery_lovasz_loss,
    evaluate_binary_masks,
    one_sided_sign_flip_pvalue,
    run_ablation,
    train_cell,
    v5_ramp,
    validate_v4_base_cell,
)

__all__ = [
    "INFERENCE_SCOPE",
    "PROFILE_ORDER",
    "TASK_ORDER",
    "analyze_ablation",
    "background_hard_negative_tail_loss",
    "build_matrix",
    "classify_summary",
    "compose_v5_gradient",
    "delivery_lovasz_loss",
    "evaluate_binary_masks",
    "one_sided_sign_flip_pvalue",
    "run_ablation",
    "train_cell",
    "v5_ramp",
    "validate_v4_base_cell",
]
