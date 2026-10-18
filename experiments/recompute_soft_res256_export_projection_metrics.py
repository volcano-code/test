#!/usr/bin/env python3
"""Recompute foreground projection metrics for exported v13 soft runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT = Path("/data/xvxinyao/xvxinyao/total_project/mesh_combined_fusion_generalizable_ready_20260528")
RUN_ROOT = PROJECT / "runs_codex_v13_soft_ablation_printability_33rows_res256_20260705"
OUT_DIR = Path("/home/lin/xvxinyao/ShadowForge/results/soft_res256_export_projection_metrics_20260706")


def foreground_mask(path: Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    arr = np.asarray(img)
    dark = arr < 128
    bright = arr >= 128
    # The soft-v13 silhouettes are encoded as either dark-on-light or light-on-dark.
    # Use the smaller side as the foreground to avoid scoring background overlap.
    return bright if bright.mean() <= dark.mean() else dark


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    denom = int(pred.sum() + target.sum())
    return {
        "iou": float(inter / union) if union else 1.0,
        "dice": float((2 * inter) / denom) if denom else 1.0,
        "pred_foreground_ratio": float(pred.mean()),
        "target_foreground_ratio": float(target.mean()),
    }


def main() -> None:
    summary = json.loads((RUN_ROOT / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for row in summary["rows"]:
        run_dir = Path(row["run_dir"])
        command_path = run_dir / "command.json"
        if not command_path.exists():
            continue
        command = json.loads(command_path.read_text(encoding="utf-8"))
        silhouettes = command["silhouettes"]
        view_metrics = []
        missing = []
        for i, sil_name in enumerate(silhouettes):
            target_path = PROJECT / sil_name
            pred_path = run_dir / f"sample_0_view_{i}.png"
            if not target_path.exists() or not pred_path.exists():
                missing.append({"view": i, "target": str(target_path), "pred": str(pred_path)})
                continue
            target = foreground_mask(target_path)
            pred = foreground_mask(pred_path)
            m = metrics(pred, target)
            m.update({"view": i, "target": sil_name, "pred": str(pred_path)})
            view_metrics.append(m)
        if view_metrics:
            ious = [m["iou"] for m in view_metrics]
            dices = [m["dice"] for m in view_metrics]
            mean_iou = float(np.mean(ious))
            min_iou = float(np.min(ious))
            mean_dice = float(np.mean(dices))
        else:
            mean_iou = min_iou = mean_dice = None
        rows.append(
            {
                "record_id": row.get("record_id"),
                "case": row.get("case"),
                "ablation": row.get("name"),
                "run_dir": str(run_dir),
                "mean_iou": mean_iou,
                "worst_view_iou": min_iou,
                "mean_dice": mean_dice,
                "view_metrics": view_metrics,
                "missing": missing,
                "component_count_6": row.get("component_count_6"),
                "mesh_is_watertight": row.get("mesh_is_watertight"),
                "printability_proxy_pass": row.get("printability_proxy_pass"),
                "timed_out": row.get("timed_out"),
                "obj_exists": row.get("obj_exists"),
            }
        )

    by_ablation: dict[str, list[dict]] = {}
    for row in rows:
        by_ablation.setdefault(row["ablation"], []).append(row)

    aggregate = []
    for ablation, group in sorted(by_ablation.items()):
        valid = [r for r in group if r["mean_iou"] is not None]
        aggregate.append(
            {
                "ablation": ablation,
                "n": len(group),
                "metric_n": len(valid),
                "mean_iou": float(np.mean([r["mean_iou"] for r in valid])) if valid else None,
                "worst_view_iou": float(np.min([r["worst_view_iou"] for r in valid])) if valid else None,
                "mean_dice": float(np.mean([r["mean_dice"] for r in valid])) if valid else None,
                "single_body": f"{sum(r.get('component_count_6') == 1 for r in group)}/{len(group)}",
                "watertight": f"{sum(bool(r.get('mesh_is_watertight')) for r in group)}/{len(group)}",
                "printable_proxy": f"{sum(bool(r.get('printability_proxy_pass')) for r in group)}/{len(group)}",
                "obj_exists": f"{sum(bool(r.get('obj_exists')) for r in group)}/{len(group)}",
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "soft_res256_export_projection_metrics_v1",
        "source_summary": str(RUN_ROOT / "summary.json"),
        "foreground_rule": "threshold at 128, smaller binary side is foreground",
        "rows": rows,
        "aggregate": aggregate,
    }
    (OUT_DIR / "soft_res256_export_projection_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    with (OUT_DIR / "soft_res256_export_projection_metrics.tsv").open("w", encoding="utf-8") as f:
        f.write(
            "Ablation\tN\tMetric N\tMean IoU\tWorst-view IoU\tMean Dice\t"
            "Single-body\tWatertight\tPrintable proxy\tOBJ exists\n"
        )
        for r in aggregate:
            f.write(
                f"{r['ablation']}\t{r['n']}\t{r['metric_n']}\t"
                f"{r['mean_iou'] if r['mean_iou'] is not None else 'NA'}\t"
                f"{r['worst_view_iou'] if r['worst_view_iou'] is not None else 'NA'}\t"
                f"{r['mean_dice'] if r['mean_dice'] is not None else 'NA'}\t"
                f"{r['single_body']}\t{r['watertight']}\t{r['printable_proxy']}\t{r['obj_exists']}\n"
            )
    print(OUT_DIR / "soft_res256_export_projection_metrics.tsv")


if __name__ == "__main__":
    main()
