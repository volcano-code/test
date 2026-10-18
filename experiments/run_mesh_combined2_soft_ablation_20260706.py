#!/usr/bin/env python3
"""Run loss ablations on the strongest local mesh_combined2 soft branch.

The source project is copied per variant before patching so the historical
project under /data is not modified.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

try:
    import trimesh
except Exception:  # pragma: no cover
    trimesh = None


SRC = Path("/data/xvxinyao/xvxinyao/total_project/mesh_combined2")
OUT = Path("/home/lin/xvxinyao/ShadowForge/results/mesh_combined2_best_soft_ablation_20260706")
PY = "/home/lin/anaconda3/envs/shadowart-revisited/bin/python"

BASE_CMD = [
    PY,
    "train.py",
    "cuda:0",
    "{out_dir}",
    "250",
    "0.05",
    "0",
    "-sdlist",
    "Spider-Man.png",
    "superman.png",
    "-mr",
    "0",
    "-tsf",
    "1",
    "-tsf4",
    "0",
    "-cam",
    "ortho",
    "-imsz",
    "128",
    "-res",
    "128",
    "-vh",
    "1",
    "-eres",
    "192",
    "--post_export_bundle",
    "0",
]


VARIANTS = {
    "full": {},
    "no_visual_hull": {"cli": {"-vh": "0"}},
    "no_projection_losses": {"patch": "zero_projection_function"},
    "no_export_structure": {"patch": "zero_export_structure_calls"},
    "no_export_distill": {"patch": "zero_export_distill_weights"},
    "no_iou_dice": {"patch": "zero_iou_dice_functions"},
    "no_asym_fn": {"patch": "zero_asym_fn_functions"},
    "no_boundary_tversky": {"patch": "zero_boundary_tversky_functions"},
}


def replace_cli(cmd: list[str], key: str, value: str) -> list[str]:
    out = list(cmd)
    idx = out.index(key)
    out[idx + 1] = value
    return out


def patch_train(project: Path, mode: str | None) -> None:
    if not mode:
        return
    train = project / "train.py"
    losses = project / "losses.py"
    text = train.read_text(encoding="utf-8")
    loss_text = losses.read_text(encoding="utf-8")

    if mode == "zero_projection_function":
        text = re.sub(
            r"def compute_tfs_projection_losses\(densities, fp_weight=4\.0\):",
            "def compute_tfs_projection_losses(densities, fp_weight=4.0):\n"
            "        zero = densities.sum() * 0.0\n"
            "        return zero, zero, zero, zero\n"
            "    def _disabled_compute_tfs_projection_losses(densities, fp_weight=4.0):",
            text,
            count=1,
        )
    elif mode == "zero_export_structure_calls":
        text = text.replace(
            "export_consistency, support_reg, main_struct_reg, manuf_reg = compute_export_structure_losses(densities)",
            "zero_struct = densities.sum() * 0.0\n"
            "        export_consistency, support_reg, main_struct_reg, manuf_reg = zero_struct, zero_struct, zero_struct, zero_struct",
        )
    elif mode == "zero_export_distill_weights":
        text = text.replace("(n_export_distill / total_w) * (0.18 + 0.22 * ramp)", "(n_export_distill / total_w) * 0.0")
        text = text.replace("(p2_export_distill / total_w) * 0.90", "(p2_export_distill / total_w) * 0.0")
        text = text.replace("(p3_export_distill / total_w) * 1.35", "(p3_export_distill / total_w) * 0.0")
        text = text.replace("(p4_export_distill / total_w) * 1.45", "(p4_export_distill / total_w) * 0.0")
    elif mode == "zero_iou_dice_functions":
        loss_text = re.sub(
            r"def differentiable_iou_loss\(pred, gt, smooth=1e-6\):",
            "def differentiable_iou_loss(pred, gt, smooth=1e-6):\n"
            "    return pred.sum() * 0.0\n"
            "def _disabled_differentiable_iou_loss(pred, gt, smooth=1e-6):",
            loss_text,
            count=1,
        )
        loss_text = re.sub(
            r"def differentiable_dice_loss\(pred, gt, smooth=1e-6\):",
            "def differentiable_dice_loss(pred, gt, smooth=1e-6):\n"
            "    return pred.sum() * 0.0\n"
            "def _disabled_differentiable_dice_loss(pred, gt, smooth=1e-6):",
            loss_text,
            count=1,
        )
    elif mode == "zero_asym_fn_functions":
        loss_text = re.sub(
            r"def asymmetric_silhouette_loss\(silh_pred, silh_gt, fp_weight=3\.0\):",
            "def asymmetric_silhouette_loss(silh_pred, silh_gt, fp_weight=3.0):\n"
            "    return silh_pred.sum() * 0.0\n"
            "def _disabled_asymmetric_silhouette_loss(silh_pred, silh_gt, fp_weight=3.0):",
            loss_text,
            count=1,
        )
        loss_text = re.sub(
            r"def false_negative_only_loss\(silh_pred, silh_gt\):",
            "def false_negative_only_loss(silh_pred, silh_gt):\n"
            "    return silh_pred.sum() * 0.0\n"
            "def _disabled_false_negative_only_loss(silh_pred, silh_gt):",
            loss_text,
            count=1,
        )
    elif mode == "zero_boundary_tversky_functions":
        loss_text = re.sub(
            r"def focal_tversky_loss\(pred, gt, alpha=0\.3, beta=0\.7, gamma=2\.0, smooth=1e-6\):",
            "def focal_tversky_loss(pred, gt, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-6):\n"
            "    return pred.sum() * 0.0\n"
            "def _disabled_focal_tversky_loss(pred, gt, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-6):",
            loss_text,
            count=1,
        )
        loss_text = re.sub(
            r"def boundary_loss\(pred, gt, device='cuda'\):",
            "def boundary_loss(pred, gt, device='cuda'):\n"
            "    return pred.sum() * 0.0\n"
            "def _disabled_boundary_loss(pred, gt, device='cuda'):",
            loss_text,
            count=1,
        )
    else:
        raise ValueError(mode)

    train.write_text(text, encoding="utf-8")
    losses.write_text(loss_text, encoding="utf-8")


def copy_project(dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns("output", "__pycache__", "*.pyc", ".git")
    shutil.copytree(SRC, dst, ignore=ignore)


def load_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics_history.json"
    if not path.exists():
        return {}
    final = json.loads(path.read_text(encoding="utf-8")).get("final_metrics", {})
    out = {}
    for key in ("renderer_metrics", "export_metrics"):
        m = final.get(key, {})
        out[key] = {
            "mean_iou": m.get("mean_iou"),
            "min_iou": m.get("min_iou"),
            "mean_dice": m.get("mean_dice"),
            "component_count": m.get("component_count"),
            "largest_component_ratio": m.get("largest_component_ratio"),
            "occupied_voxels": m.get("occupied_voxels"),
        }
    for name, m in final.get("export_candidates", {}).items():
        if name in {"learned", "learned_clean", "learned_clip_clean", "learned_density_clip_balanced"}:
            out[f"candidate_{name}"] = {
                "mean_iou": m.get("mean_iou"),
                "min_iou": m.get("min_iou"),
                "mean_dice": m.get("mean_dice"),
                "component_count": m.get("component_count"),
                "largest_component_ratio": m.get("largest_component_ratio"),
                "occupied_voxels": m.get("occupied_voxels"),
            }
    vox = run_dir / "sample_0_final_voxels.npy"
    obj = run_dir / "sample_0_output.obj"
    out["files"] = {"voxels": str(vox) if vox.exists() else None, "obj": str(obj) if obj.exists() else None}
    if vox.exists():
        arr = np.load(vox).astype(bool)
        out["voxel_shape"] = list(arr.shape)
        out["voxel_occupied"] = int(arr.sum())
    if obj.exists() and trimesh is not None:
        try:
            mesh = trimesh.load(obj, force="mesh")
            out["mesh_watertight"] = bool(mesh.is_watertight)
            out["mesh_split_components"] = int(len(mesh.split(only_watertight=False)))
        except Exception as exc:
            out["mesh_error"] = str(exc)
    return out


def run_variant(name: str, spec: dict) -> dict:
    project = OUT / "projects" / name
    run_dir = OUT / "runs" / name
    copy_project(project)
    patch_train(project, spec.get("patch"))
    cmd = [x.format(out_dir=str(run_dir)) for x in BASE_CMD]
    for key, value in spec.get("cli", {}).items():
        cmd = replace_cli(cmd, key, value)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(json.dumps({"variant": name, "cmd": cmd, "spec": spec}, indent=2), encoding="utf-8")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=project, text=True, capture_output=True)
    log = proc.stdout + proc.stderr
    (run_dir / "run.log").write_text(log, encoding="utf-8", errors="ignore")
    return {
        "variant": name,
        "returncode": proc.returncode,
        "seconds": round(time.time() - t0, 2),
        "run_dir": str(run_dir),
        "metrics": load_metrics(run_dir),
        "tail": "\n".join(log.splitlines()[-20:]),
    }


def flatten(row: dict) -> dict:
    m = row.get("metrics", {})
    learned = m.get("candidate_learned_clip_clean") or m.get("candidate_learned_density_clip_balanced") or m.get("candidate_learned") or {}
    export = m.get("export_metrics", {})
    return {
        "variant": row["variant"],
        "rc": row["returncode"],
        "sec": row["seconds"],
        "learned_iou": learned.get("mean_iou"),
        "learned_min_iou": learned.get("min_iou"),
        "learned_comp": learned.get("component_count"),
        "learned_occ": learned.get("occupied_voxels"),
        "export_iou": export.get("mean_iou"),
        "export_min_iou": export.get("min_iou"),
        "export_comp": export.get("component_count"),
        "export_occ": export.get("occupied_voxels"),
        "mesh_watertight": m.get("mesh_watertight"),
        "mesh_split_components": m.get("mesh_split_components"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, spec in VARIANTS.items():
        row = run_variant(name, spec)
        rows.append(row)
        (OUT / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        flat = [flatten(r) for r in rows]
        (OUT / "summary.json").write_text(json.dumps(flat, indent=2), encoding="utf-8")
        with (OUT / "summary.tsv").open("w", encoding="utf-8") as f:
            keys = list(flat[0].keys())
            f.write("\t".join(keys) + "\n")
            for rec in flat:
                f.write("\t".join(str(rec.get(k, "")) for k in keys) + "\n")
        print(json.dumps(flat[-1], ensure_ascii=False), flush=True)
    print(f"saved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
