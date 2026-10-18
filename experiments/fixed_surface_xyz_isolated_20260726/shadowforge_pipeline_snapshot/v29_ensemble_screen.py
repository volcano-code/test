"""Frozen v29 ensemble and v27/v28 disagreement screen."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts.train_v26_shadowart_scale_sf import build_x
from structural.models.load_model import load_surrogate


class V29EnsembleScreen:
    def __init__(self, root: Path, device: torch.device, artifact_path: Path):
        self.root = Path(root)
        self.device = device
        self.artifact_path = Path(artifact_path)
        self.artifact = json.loads(self.artifact_path.read_text())
        self.models = {
            name: self._load_model(self.root / relative_path)
            for name, relative_path in self.artifact["checkpoints"].items()
        }

    def _load_model(self, checkpoint_path: Path):
        base_path = self.root / self.artifact["base_checkpoint"]
        model, _ = load_surrogate(str(base_path), device=self.device)
        model.enable_enhanced_scale_head()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.scale_head_v2.load_state_dict(checkpoint["scale_head_v2_state_dict"])
        for name, state in checkpoint["scale_branch_state_dict"].items():
            getattr(model, name).load_state_dict(state)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    @torch.no_grad()
    def _predict(self, model, density: np.ndarray, force: np.ndarray) -> float:
        x = build_x(density.astype(np.float32), force.astype(np.float32)).unsqueeze(0).to(self.device)
        grid = model._get_grid(tuple(x.shape[-3:]), x.device).expand(x.shape[0], -1, -1, -1, -1)
        return float(model._scale_forward(torch.cat([x, grid], dim=1)).reshape(-1)[0].cpu())

    def evaluate(self, density: np.ndarray, force: np.ndarray, sf_threshold: float) -> dict:
        prediction = {
            name: self._predict(model, density, force)
            for name, model in self.models.items()
        }
        mean_log_vm = float(np.mean([prediction[name] for name in ("v29a", "v29b", "v29c")]))
        disagreement = abs(prediction["v27"] - prediction["v28"])
        yield_stress = float(self.artifact["yield_stress"])
        lower_sf = yield_stress / max(10.0 ** (mean_log_vm + float(self.artifact["q_safe"])), 1e-12)
        upper_sf = yield_stress / max(10.0 ** (mean_log_vm + float(self.artifact["q_unsafe"])), 1e-12)
        predicted_safe = lower_sf >= sf_threshold
        predicted_unsafe = upper_sf < sf_threshold
        trigger = disagreement >= float(self.artifact["disagreement_threshold"])
        ambiguous = not predicted_safe and not predicted_unsafe
        needs_fem = bool((trigger and not predicted_unsafe) or ambiguous)
        return {
            "safe": bool(predicted_safe and not needs_fem),
            "predicted_unsafe": bool(predicted_unsafe),
            "ambiguous": bool(ambiguous),
            "needs_fem": needs_fem,
            "mean_log_vm": mean_log_vm,
            "sf_pino": float(yield_stress / max(10.0 ** mean_log_vm, 1e-12)),
            "acs_lower_sf": float(lower_sf),
            "acs_upper_sf": float(upper_sf),
            "disagreement": float(disagreement),
            "member_log_vm": prediction,
        }
