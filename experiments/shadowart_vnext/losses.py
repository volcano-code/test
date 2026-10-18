"""Small, view-equal objectives on physical silhouette probabilities.

Lovasz hinge and class-balanced BCE are established objectives, not new losses.
No loss here performs camera registration or per-image maximum normalization.
"""
from dataclasses import asdict, dataclass, replace
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    core: str = "l2_l1"
    auxiliary: str = "none"
    aux_weight: float = 0.0
    control: str | None = None
    l1_weight: float | None = None
    l2_weight: float = 1.0

    def __post_init__(self):
        if self.core not in {"none", "l1", "l2", "l2_l1"}:
            raise ValueError("unknown core loss")
        if self.auxiliary not in {"none", "lovasz", "ray_bce"}:
            raise ValueError("at most one registered auxiliary is permitted")
        if self.control not in {None, "current_proposal"}:
            raise ValueError("unknown historical control")
        if self.l1_weight is None:
            object.__setattr__(self, "l1_weight", .35 if self.core == "l2_l1" else 1.0)
        for value in (self.l1_weight, self.l2_weight, self.aux_weight):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("loss weights must be finite and nonnegative")
        if self.auxiliary == "none" and self.aux_weight != 0:
            raise ValueError("disabled auxiliary must have zero weight")
        if self.auxiliary != "none" and self.aux_weight <= 0:
            raise ValueError("enabled auxiliary needs a positive frozen coefficient")
        if self.control is not None and (self.auxiliary != "none" or self.core != "l2_l1" or self.l1_weight != .35 or self.l2_weight != 1):
            raise ValueError("historical control has an exact frozen formula")

    def to_dict(self):
        return asdict(self)

    def without(self, term):
        """Deletion keeps every remaining coefficient fixed, including lone L1."""
        if self.control is not None:
            raise ValueError("historical control is not a newly claimed objective")
        if term == self.auxiliary and term != "none":
            return replace(self, auxiliary="none", aux_weight=0.)
        if term == "l1" and self.core in {"l1", "l2_l1"}:
            return replace(self, core="l2" if self.core == "l2_l1" else "none")
        if term == "l2" and self.core in {"l2", "l2_l1"}:
            return replace(self, core="l1" if self.core == "l2_l1" else "none")
        raise ValueError("cannot delete an absent term")


def auxiliary_ramp(progress):
    if isinstance(progress, bool) or not math.isfinite(float(progress)) or not 0 <= progress <= 1:
        raise ValueError("progress must lie in [0,1]")
    return min(1., max(0., (float(progress) - .2) / .1))


def _validate(preds, targets):
    if not isinstance(preds, torch.Tensor) or not preds.is_floating_point() or preds.dtype not in (torch.float32, torch.float64):
        raise ValueError("predictions must be float32/float64 torch tensors")
    if not isinstance(targets, torch.Tensor) or preds.shape != targets.shape or preds.ndim != 3 or min(preds.shape) == 0:
        raise ValueError("predictions and targets need equal nonempty [V,H,W] shapes")
    if preds.device != targets.device:
        raise ValueError("predictions and targets must share device")
    if not torch.isfinite(preds).all() or not torch.isfinite(targets).all():
        raise ValueError("nonfinite projection or target")
    if (preds < 0).any() or (preds > 1).any() or ((targets != 0) & (targets != 1)).any():
        raise ValueError("physical probabilities must be [0,1] and targets binary")


def _lovasz_per_view(preds, targets):
    result = []
    for pred, target in zip(preds, targets):
        truth = target.flatten()
        scores = 2. * pred.flatten() - 1.
        errors = 1. - scores * (2. * truth - 1.)
        errors, order = torch.sort(errors, descending=True, stable=True)
        sorted_truth = truth[order]
        foreground = sorted_truth.sum()
        intersection = foreground - sorted_truth.cumsum(0)
        union = foreground + (1. - sorted_truth).cumsum(0)
        jaccard = 1. - intersection / union.clamp_min(1.)
        gradient = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
        result.append(torch.dot(F.relu(errors), gradient))
    return torch.stack(result)


def _balanced_bce_per_view(preds, targets):
    # Built-in BCE has finite endpoint values/gradients; no per-image rescaling.
    pixel = F.binary_cross_entropy(preds, targets, reduction="none")
    fg = targets.sum((-2, -1))
    bg = (1. - targets).sum((-2, -1))
    fg_loss = (pixel * targets).sum((-2, -1)) / fg.clamp_min(1.)
    bg_loss = (pixel * (1. - targets)).sum((-2, -1)) / bg.clamp_min(1.)
    classes = (fg > 0).to(preds.dtype) + (bg > 0).to(preds.dtype)
    return (fg_loss + bg_loss) / classes


def _legacy_ssim(preds, targets):
    pred, truth = preds[:, None], targets[:, None]
    pool = lambda value: F.avg_pool2d(value, 11, 1, 5)
    mp, mt = pool(pred), pool(truth)
    vp, vt = pool(pred * pred) - mp.square(), pool(truth * truth) - mt.square()
    covariance = pool(pred * truth) - mp * mt
    ssim = ((2 * mp * mt + .01**2) * (2 * covariance + .03**2) /
            ((mp.square() + mt.square() + .01**2) * (vp + vt + .03**2) + 1e-8))
    return 1. - ssim.mean((1, 2, 3))


def compute_loss(preds, targets, config=LossConfig(), progress=1.):
    """Return one differentiable scalar and detached named diagnostics."""
    if not isinstance(config, LossConfig):
        raise ValueError("config must be LossConfig")
    ramp = auxiliary_ramp(progress)
    _validate(preds, targets)
    truth = targets.to(preds.dtype)
    l1 = (preds - truth).abs().mean((-2, -1))
    l2 = (preds - truth).square().mean((-2, -1))
    total = preds.sum((-2, -1)) * 0.
    if config.core in {"l1", "l2_l1"}:
        total = total + config.l1_weight * l1
    if config.core in {"l2", "l2_l1"}:
        total = total + config.l2_weight * l2
    diagnostics = {"l1": l1.mean().item(), "l2": l2.mean().item(), "aux_ramp": ramp}
    if config.control:
        dice = 1. - (2. * (preds * truth).sum((-2, -1)) + 1.) / (preds.sum((-2, -1)) + truth.sum((-2, -1)) + 1.)
        fp = (preds * (1. - truth)).mean((-2, -1))
        ssim = _legacy_ssim(preds, truth)
        total = total + .8 * dice + .35 * fp + .18 * ssim
        diagnostics.update(dice=dice.mean().item(), false_positive=fp.mean().item(), ssim=ssim.mean().item(), ssim_method="uniform11_zero_padding_legacy")
    if config.auxiliary != "none":
        aux = _lovasz_per_view(preds, truth) if config.auxiliary == "lovasz" else _balanced_bce_per_view(preds, truth)
        total = total + config.aux_weight * ramp * aux
        diagnostics[config.auxiliary] = aux.mean().item()
    result = total.mean()
    if not torch.isfinite(result):
        raise ValueError("nonfinite total loss")
    diagnostics["loss"] = result.item()
    return result, diagnostics
