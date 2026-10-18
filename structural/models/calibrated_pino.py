"""Calibrated PINO wrapper for OOD (dense Stage A) geometry.

The base PINO surrogate (v24/v25) was trained on shadow-art voxels — thin/sparse
geometry with corr(vol_frac, log_vm_max) = -0.4. On dense reconstructed voxels
the relationship is reversed and PINO's scale-head log_vm_max prediction is
useless (LOO |err| ~= 2.9, near-zero correlation with ground truth).

This wrapper composes:
  - The base PINO model (used for the field branch / per-sample stress shape)
  - A post-hoc OOD calibrator (Huber regression on structural features) that
    overrides log_vm_max when the input looks OOD.

The calibrator was fit on 38 FEM-verified Stage A samples
(modelnet40 + thingi10k) and achieves LOO |err| = 0.36 vs 2.9 for raw PINO.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


_DEFAULT_CALIB_PATH = Path(__file__).resolve().parents[2] / "ood_calibrator.pkl"


def _structural_features_np(density: np.ndarray) -> np.ndarray:
    """Compute the 6 structural features used by the calibrator (numpy)."""
    d = (density > 0.3).astype(np.float32)
    if d.ndim == 4:
        d = d[0]
    N = d.shape[0]

    vol_frac = float(d.mean())

    # 6-connected surface count
    pad = np.pad(d, 1, mode="constant")
    surface = (
        np.abs(d - pad[2:, 1:-1, 1:-1]) +
        np.abs(d - pad[:-2, 1:-1, 1:-1]) +
        np.abs(d - pad[1:-1, 2:, 1:-1]) +
        np.abs(d - pad[1:-1, :-2, 1:-1]) +
        np.abs(d - pad[1:-1, 1:-1, 2:]) +
        np.abs(d - pad[1:-1, 1:-1, :-2])
    )
    n_surface = float((surface > 0.5).sum())
    n_solid = max(float(d.sum()), 1.0)
    sv_ratio = n_surface / n_solid

    cx = d.sum(axis=(1, 2))
    cy = d.sum(axis=(0, 2))
    cz = d.sum(axis=(0, 1))
    min_cx = float(cx[cx > 0].min() / (N * N)) if (cx > 0).any() else 0.0
    min_cy = float(cy[cy > 0].min() / (N * N)) if (cy > 0).any() else 0.0
    min_cz = float(cz[cz > 0].min() / (N * N)) if (cz > 0).any() else 0.0

    n_slices_x = float((cx > 0).sum()) / N
    return np.array([vol_frac, sv_ratio, min_cx, min_cy, min_cz, n_slices_x],
                    dtype=np.float64)


def _force_features_np(force: Optional[np.ndarray], density: np.ndarray) -> dict[str, float]:
    """Compute optional force features for force-aware calibrators."""
    if force is None:
        # Standing-load default used by LoadGenerator with rho=1240, g=9.81,
        # cell_size=1.0. This keeps legacy density-only calls compatible.
        return {"log_force_l1_per_solid": float(np.log10(12164.4))}
    f = np.asarray(force, dtype=np.float64)
    if f.ndim == 5:
        f = f[0]
    d = np.asarray(density)
    if d.ndim == 4:
        d = d[0]
    solid = d > 0.3
    force_l1 = np.abs(f).sum(axis=0) if f.ndim == 4 and f.shape[0] == 3 else np.abs(f)
    if force_l1.shape != solid.shape:
        return {"log_force_l1_per_solid": float(np.log10(max(np.abs(f).mean(), 1e-12)))}
    per_solid = float(force_l1[solid].mean()) if solid.any() else float(force_l1.mean())
    return {"log_force_l1_per_solid": float(np.log10(max(per_solid, 1e-12)))}


def _ood_score(density: np.ndarray) -> float:
    """Heuristic OOD score in [0, 1]. Higher = more OOD (denser than training).

    Training distribution has vol_frac mostly < 0.08; test distribution has
    vol_frac in [0.10, 0.30]. We use vol_frac as the gating feature.
    """
    feats = _structural_features_np(density)
    vol_frac = feats[0]
    # Smooth gate: 0 below 0.08, 1 above 0.12
    return float(np.clip((vol_frac - 0.08) / 0.04, 0.0, 1.0))


class CalibratedPINO:
    """Wrap a loaded UFNODualBranch3d model with an OOD log_vm_max calibrator.

    Usage:
        cp = CalibratedPINO(pino_model)
        log_vm_max = cp.predict_log_vm_max(density_np)        # calibrated scalar
        sf = cp.predict_safety_factor(density_np)             # yield/vm_max
        fields = cp.predict_fields(density_tensor, force_tensor)
    """

    def __init__(
        self,
        pino,
        calibrator_path: str | Path = _DEFAULT_CALIB_PATH,
        yield_stress: float = 250e6,
    ):
        self.pino = pino
        self.yield_stress = yield_stress
        with open(calibrator_path, "rb") as f:
            blob = pickle.load(f)
        self.cal_model = blob["model"]
        self.cal_scaler = blob["scaler"]
        self.cal_feat_names = list(blob["feat_names"])
        self.log_idx = list(blob["log_transform_idx"])

    # ---------- calibrator ----------
    def _calibrate(self, density: np.ndarray, force: Optional[np.ndarray] = None) -> float:
        feats = _structural_features_np(density)  # [vol, sv, mcx, mcy, mcz, conn]
        feat_map = {
            "vol_frac": feats[0],
            "sv_ratio": feats[1],
            "min_cx": feats[2],
            "min_cy": feats[3],
            "min_cz": feats[4],
            "connect": feats[5],
            "n_slices_x": feats[5],
        }
        feat_map.update(_force_features_np(force, density))
        x = np.array([feat_map[name] for name in self.cal_feat_names], dtype=np.float64)
        for i in self.log_idx:
            x[i] = np.log10(x[i] + 1e-6)
        xs = self.cal_scaler.transform(x.reshape(1, -1))
        return float(self.cal_model.predict(xs)[0])

    # ---------- public API ----------
    def predict_log_vm_max(
        self,
        density: np.ndarray,
        x_input: torch.Tensor | None = None,
        force: np.ndarray | None = None,
    ) -> float:
        """Return calibrated log10(vm_max) for the given density.

        For OOD inputs (vol_frac > ~0.10) returns the structural-feature
        calibrator output. For in-distribution inputs returns the PINO
        scale_head prediction (if `x_input` was supplied) or falls back to
        the calibrator.
        """
        ood = _ood_score(density)
        cal_pred = self._calibrate(density, force=force)
        if ood > 0.5 or x_input is None:
            return cal_pred
        # In-distribution: use PINO directly
        with torch.no_grad():
            if not hasattr(self.pino, "forward_with_scale"):
                return cal_pred
            _, log_vm = self.pino.forward_with_scale(x_input)
            pino_pred = float(log_vm.squeeze().cpu())
        # Smoothly blend toward cal_pred as ood -> 1
        return (1.0 - ood) * pino_pred + ood * cal_pred

    def predict_safety_factor(
        self,
        density: np.ndarray,
        x_input: torch.Tensor | None = None,
        force: np.ndarray | None = None,
    ) -> float:
        log_vm = self.predict_log_vm_max(density, x_input=x_input, force=force)
        vm_max = 10.0 ** log_vm
        return float(self.yield_stress / max(vm_max, 1.0))

    def predict_fields(
        self,
        x_input: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through the field branch only. The wrapper does not
        recalibrate the per-voxel stress field — only the scalar log_vm_max."""
        with torch.no_grad():
            return self.pino(x_input)


def load_calibrated_pino(
    base_ckpt: str,
    device: torch.device | str = "cuda:0",
    calibrator_path: str | Path = _DEFAULT_CALIB_PATH,
):
    from structural.models.load_model import load_surrogate
    pino, _ = load_surrogate(base_ckpt, device=device)
    pino.eval()
    for p in pino.parameters():
        p.requires_grad_(False)
    return CalibratedPINO(pino, calibrator_path=calibrator_path)
