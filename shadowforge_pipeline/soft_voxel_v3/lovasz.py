"""Stable row-major Lovasz hinge for soft-loss v3."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_logits_and_labels(logits: torch.Tensor, labels: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise ValueError("logits must be a torch.Tensor")
    if not isinstance(labels, torch.Tensor):
        raise ValueError("labels must be a torch.Tensor")
    if not logits.is_floating_point():
        raise ValueError("logits must use a floating-point dtype")
    if labels.dtype != torch.bool:
        raise ValueError("labels must have boolean dtype")
    if logits.ndim != 3 or any(size == 0 for size in logits.shape):
        raise ValueError("logits must have non-empty shape [views,H,W]")
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have the same shape")
    if logits.device != labels.device:
        raise ValueError("logits and labels must share a device")
    if not torch.isfinite(logits).all().item():
        raise ValueError("logits must be finite")


def lovasz_hinge_per_view(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return stable row-major binary Lovasz hinge, one scalar per view."""

    _validate_logits_and_labels(logits, labels)
    flat_logits = logits.reshape(logits.shape[0], -1)
    flat_labels = labels.reshape(labels.shape[0], -1)
    signs = 2.0 * flat_labels.to(dtype=flat_logits.dtype) - 1.0
    errors = 1.0 - flat_logits * signs
    order = torch.argsort(errors, dim=1, descending=True, stable=True)
    sorted_errors = torch.gather(errors, 1, order)
    sorted_labels = torch.gather(flat_labels, 1, order).to(dtype=flat_logits.dtype)

    intersection = sorted_labels.sum(dim=1, keepdim=True) - sorted_labels.cumsum(dim=1)
    union = sorted_labels.sum(dim=1, keepdim=True) + (1.0 - sorted_labels).cumsum(dim=1)
    jaccard = 1.0 - intersection / union
    lovasz_gradient = torch.cat(
        (jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]), dim=1
    )
    return (F.relu(sorted_errors) * lovasz_gradient).sum(dim=1)
