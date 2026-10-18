"""D23 calibrated PINO: v2_weighted scale head + isotonic post-calibration.

Bypasses the OOD gate — for in-distribution shadow art, PINO raw + isotonic
outperforms the structural-feature fallback. Calibration fitted on 5930 d23
FEM-verified samples spanning mn40/thingi/shadowart geometries.

D22 n=43 eval: logMAE=0.723, agree=97.7%, FP=1, FN=0, bias=+0.09
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from scripts.train_v26_shadowart_scale_sf import build_x as _build_x

_DEFAULT_CAL = Path(__file__).resolve().parents[2] / "checkpoints/v2_cal_d23_full.pkl"
_DEFAULT_V2_CKPT = Path(__file__).resolve().parents[2] / "checkpoints/scale_head_v2_weighted.bak/best_scale_head_v2_weighted.pt"


class CalibratedPINOD23:
    """v2_weighted + isotonic post-cal on log_vm_max."""

    def __init__(self, pino, cal_path=_DEFAULT_CAL, yield_stress=50e6):
        self.pino = pino
        self.yield_stress = yield_stress
        blob = pickle.load(open(cal_path, "rb"))
        self.isotonic = blob["isotonic"]

    def build_input(self, density, force, device=None):
        """Build normalized 5-ch input matching scale-head training format."""
        if device is None:
            device = next(self.pino.parameters()).device
        x = _build_x(density, force).unsqueeze(0).to(device)
        return x

    def predict_log_vm_max(self, density, x_input=None, force=None):
        if x_input is None:
            if force is None:
                raise ValueError("Either x_input or force required")
            x_input = self.build_input(density, force)
        with torch.no_grad():
            xg = self._add_grid(x_input)
            log_vm_raw = float(self.pino._scale_forward(xg).squeeze().cpu())
        return float(self.isotonic.predict([log_vm_raw])[0])

    def predict_safety_factor(self, density, x_input=None, force=None):
        log_vm = self.predict_log_vm_max(density, x_input=x_input, force=force)
        vm_max = 10.0 ** log_vm
        return float(self.yield_stress / max(vm_max, 1.0))

    def predict_fields(self, x_input):
        with torch.no_grad():
            return self.pino(x_input)

    def _add_grid(self, x):
        b, c, nx, ny, nz = x.shape
        grid = self.pino._get_grid((nx, ny, nz), x.device).expand(b, -1, -1, -1, -1)
        return torch.cat([x, grid], dim=1)


def load_calibrated_pino_d23(
    base_ckpt: str,
    device="cuda:0",
    scale_ckpt=_DEFAULT_V2_CKPT,
    cal_path=_DEFAULT_CAL,
    yield_stress=50e6,
):
    from structural.models.load_model import load_surrogate
    pino, _ = load_surrogate(base_ckpt, device=device)
    pino.enable_enhanced_scale_head()
    ft = torch.load(scale_ckpt, map_location=device, weights_only=False)
    pino.scale_head_v2.load_state_dict(ft["scale_head_v2_state_dict"])
    if ft.get("scale_branch_state_dict"):
        for name, sd in ft["scale_branch_state_dict"].items():
            getattr(pino, name).load_state_dict(sd)
    pino.eval()
    for p in pino.parameters():
        p.requires_grad_(False)
    return CalibratedPINOD23(pino, cal_path=cal_path, yield_stress=yield_stress)
