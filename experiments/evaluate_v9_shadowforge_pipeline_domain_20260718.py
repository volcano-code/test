#!/usr/bin/env python3
"""Fixed-policy v9 audit on candidates emitted by the current ShadowForge pipeline.

This is intentionally not a recalibration run.  It reuses the frozen v9
alpha=0.1 ACS quantiles and evaluates delivered candidates for which the
pipeline already recorded voxel-FEM ground truth.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from structural.analysis.load_generator import LoadGenerator
from structural.models.pino import PINO3d, UFNO3d, UFNODualBranch3d
from structural.training.pretrain_pino_v3 import FEMDatasetV3, compute_von_mises


CHECKPOINT = ROOT / "checkpoints/strict_physics_residual_decision_pino_v9_stable_20260715/best_model.pt"
PIPELINE_ROOT = ROOT / "results/fem_two_wall_print_candidate_search_20260714"
OUTPUT = ROOT / "results/v9_fixed_alpha01_shadowforge_pipeline_domain_20260718.json"
SF_THRESHOLD = 1.3
YIELD_STRESS = 50e6
Q_SAFE = 0.1952986028043909
Q_UNSAFE = 0.31987571058142183


def load_model(device: torch.device) -> torch.nn.Module:
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    model_cls = {"pino": PINO3d, "ufno": UFNO3d, "ufno_dual": UFNODualBranch3d}[state["arch"]]
    model = model_cls(**state["config"]).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.eval()


def make_input(volume: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    density = np.asarray(volume, dtype=np.float32)
    load = LoadGenerator().standing_load(density)
    force = load["force"].astype(np.float32)
    fixed = load["fixed_mask"].astype(np.float32)
    solid = density > FEMDatasetV3.SOLID_THRESHOLD
    if solid.any() and (~solid).any():
        sdf = distance_transform_edt(solid) - distance_transform_edt(~solid)
        sdf = sdf.astype(np.float32) / (density.shape[0] / 2.0)
    else:
        sdf = np.zeros_like(density)
    x = np.concatenate([
        density[None], force / FEMDatasetV3.FORCE_SCALE, sdf[None], fixed
    ], axis=0)
    return torch.from_numpy(x), torch.from_numpy(density[None])


def orient(volume: np.ndarray, name: str) -> np.ndarray:
    transforms = {
        "display_z_up": lambda v: v,
        "display_z_down": lambda v: v[:, :, ::-1],
        "display_x_up": lambda v: np.transpose(v, (1, 2, 0)),
        "display_x_down": lambda v: np.transpose(v, (1, 2, 0))[:, :, ::-1],
        "display_y_up": lambda v: np.transpose(v, (0, 2, 1)),
        "display_y_down": lambda v: np.transpose(v, (0, 2, 1))[:, :, ::-1],
    }
    return transforms[name](volume)


def confidence_counts(rows: list[dict]) -> dict:
    n = len(rows)
    safe = [r for r in rows if r["truth_safe"]]
    unsafe = [r for r in rows if not r["truth_safe"]]
    auto_safe = [r for r in rows if r["decision"] == "SAFE"]
    auto_unsafe = [r for r in rows if r["decision"] == "UNSAFE"]
    defer = [r for r in rows if r["decision"] == "DEFER"]
    fs = [r for r in auto_safe if not r["truth_safe"]]
    fu = [r for r in auto_unsafe if r["truth_safe"]]
    decided = auto_safe + auto_unsafe
    return {
        "n": n,
        "n_truth_safe": len(safe),
        "n_truth_unsafe": len(unsafe),
        "n_auto_safe": len(auto_safe),
        "n_auto_unsafe": len(auto_unsafe),
        "n_defer": len(defer),
        "false_safe": len(fs),
        "false_unsafe": len(fu),
        "overall_autonomous_coverage": len(decided) / n if n else None,
        "safe_auto_pass_rate": sum(r["decision"] == "SAFE" for r in safe) / len(safe) if safe else None,
        "unsafe_auto_reject_rate": sum(r["decision"] == "UNSAFE" for r in unsafe) / len(unsafe) if unsafe else None,
        "auto_safe_precision": (len(auto_safe) - len(fs)) / len(auto_safe) if auto_safe else None,
        "autonomous_accuracy": (len(decided) - len(fs) - len(fu)) / len(decided) if decided else None,
    }


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    samples: list[dict] = []
    pending: list[tuple[dict, torch.Tensor, torch.Tensor]] = []
    for manifest_path in sorted(PIPELINE_ROOT.glob("**/MANIFEST.json")):
        manifest = json.loads(manifest_path.read_text())
        evidence_root = manifest_path.parent / "candidate_evidence/delivery_candidates"
        for row in manifest.get("candidates", []):
            sf_fem = row.get("structural_screen", {}).get("sf_fem")
            if sf_fem is None:
                continue
            candidate = row["candidate"]
            volume_path = evidence_root / candidate.replace(":", "_") / "volume.npy"
            if not volume_path.exists():
                raise FileNotFoundError(volume_path)
            delivered = np.load(volume_path, mmap_mode="r")
            oriented = np.ascontiguousarray(orient(delivered, row["print_orientation"]), dtype=np.float32)
            dense = torch.from_numpy(oriented)[None, None].to(device)
            proxy = torch.nn.functional.adaptive_max_pool3d(dense, (48, 48, 48))[0, 0].cpu().numpy() >= 0.5
            del dense
            x, density = make_input(proxy)
            pending.append(({
                "case": str(manifest_path.parent.relative_to(PIPELINE_ROOT)),
                "candidate": candidate,
                "print_orientation": row["print_orientation"],
                "sf_fem": float(sf_fem),
                "truth_safe": bool(sf_fem >= SF_THRESHOLD),
            }, x, density))

    for start in range(0, len(pending), 4):
        chunk = pending[start:start + 4]
        xb = torch.stack([x for _, x, _ in chunk]).to(device)
        pred, log_vm_max = model.forward_with_scale(xb)
        vm = compute_von_mises(pred[:, 3:].float())[:, 0]
        for i, (record, _, density) in enumerate(chunk):
            mask = density[0].to(device) > FEMDatasetV3.SOLID_THRESHOLD
            vm_shape_peak = float(vm[i][mask].max().item()) if mask.any() else 1e-12
            scale = 10.0 ** float(log_vm_max[i, 0].item())
            sf_pred = YIELD_STRESS / max(vm_shape_peak * scale, 1e-12)
            p = math.log10(sf_pred)
            lower, upper = p - Q_SAFE, p + Q_UNSAFE
            decision = "SAFE" if lower >= math.log10(SF_THRESHOLD) else (
                "UNSAFE" if upper < math.log10(SF_THRESHOLD) else "DEFER"
            )
            record.update({"sf_pred": float(sf_pred), "decision": decision})
            samples.append(record)
        print(f"predicted {min(start + 4, len(pending))}/{len(pending)}", flush=True)

    result = {
        "scope": "current ShadowForge full-pipeline delivered candidates with recorded voxel-FEM labels",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "policy": {"alpha": 0.1, "q_safe": Q_SAFE, "q_unsafe": Q_UNSAFE, "sf_threshold": SF_THRESHOLD},
        "protocol": "saved Vdel,256 -> saved print orientation -> conservative max-pool C48 -> normalized standing self-weight",
        "metrics": confidence_counts(samples),
        "by_case": {
            case: confidence_counts([r for r in samples if r["case"] == case])
            for case in sorted({r["case"] for r in samples})
        },
        "samples": samples,
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["metrics"], indent=2))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
