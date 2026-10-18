#!/usr/bin/env bash
# Pure Stage-A soft-loss ablation on a complex three-silhouette TSF task.
#
# Protocol:
#   silhouettes = Spider-Man / Superman / Batman
#   tsf=1, mr=1 -> three axis-aligned views plus their three mirrors
#   no Stage-B repair, bridge, carving, delivery shell, selector, or structural screen
#
# The summary intentionally reports raw_soft_iou rather than the regularized
# proposal selected from morphology diagnostics.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
TRAIN=/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel/train.py
INPUT=/home/lin/xvxinyao/shadow/normal_shadow
OUT="$ROOT/results/soft_tsf1_mr1_heroes_ablation_20260717"
SEEDS_CSV=${SEEDS_CSV:-17,23,42}
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

VARIANTS=(
  "full|volume"
  "l1_l2_only|dice ssim false_positive edge smooth laplacian connectivity spurious wall spectral deviation volume"
  # Individual image-loss removals are required to attribute usefulness.
  "w_o_dice|dice volume"
  "w_o_ssim|ssim volume"
  "w_o_false_positive|false_positive volume"
  # Compact objective recommended for the paper: L2 + Dice + FP + structure.
  "compact_no_l1_ssim|l1 ssim volume"
  "w_o_l1|l1 volume"
  "w_o_l2|l2 volume"
  "w_o_l1_l2|l1 l2 volume"
  "w_o_image_aux|dice ssim false_positive volume"
  "w_o_structure_all|edge smooth laplacian connectivity spurious wall spectral deviation volume"
  "w_o_connectivity_group|connectivity spurious wall spectral volume"
)

mkdir -p "$OUT"

for seed in "${SEEDS[@]}"; do
  for item in "${VARIANTS[@]}"; do
    IFS='|' read -r name disabled <<< "$item"
    run="$OUT/$name/seed_$seed"
    if [[ -f "$run/metrics.json" ]]; then
      echo "skip completed $name seed=$seed"
      continue
    fi
    mkdir -p "$run"
    echo "===== $name seed=$seed disabled=[$disabled] ====="
    # shellcheck disable=SC2086
    "$PY" "$TRAIN" cuda:0 "$run" 2000 0.007 1 \
      -sdlist \
        "$INPUT/Spider-Man.png" \
        "$INPUT/superman.png" \
        "$INPUT/batman.png" \
      -mr 1 -tsf 1 -tsf4 0 -cam ortho -imsz 256 \
      --res 256 -eres 256 --control-res 0 --save-every 100 \
      --proposal-mode regularized \
      --connectivity-start-iou 0.60 --fidelity-gate 0.85 \
      --structure-warmup-steps 300 --structure-ramp-steps 300 \
      --no-cleanup --export-final-step --seed "$seed" \
      --disable-losses $disabled \
      2>&1 | tee "$run/train.log"
  done
done

"$PY" - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return image >= 128


def projection_diagnostics(run_dir: Path) -> dict:
    per_view = []
    for pred_path in sorted(run_dir.glob("pred_view_*.png")):
        suffix = pred_path.stem.split("_")[-1]
        gt_path = run_dir / f"gt_view_{suffix}.png"
        if not gt_path.exists():
            continue
        pred = load_mask(pred_path)
        target = load_mask(gt_path)
        fp = int(np.count_nonzero(pred & ~target))
        tp = int(np.count_nonzero(pred & target))
        background = int(np.count_nonzero(~target))
        target_area = int(np.count_nonzero(target))
        per_view.append({
            "view": int(suffix),
            "false_positive_pixels": fp,
            "background_false_positive_rate": fp / max(background, 1),
            "foreground_leakage_ratio": fp / max(target_area, 1),
            "projection_precision": tp / max(tp + fp, 1),
        })
    return {
        "per_view": per_view,
        "background_false_positive_rate_mean": float(np.mean(
            [v["background_false_positive_rate"] for v in per_view]
        )) if per_view else float("nan"),
        "foreground_leakage_ratio_mean": float(np.mean(
            [v["foreground_leakage_ratio"] for v in per_view]
        )) if per_view else float("nan"),
        "projection_precision_mean": float(np.mean(
            [v["projection_precision"] for v in per_view]
        )) if per_view else float("nan"),
    }

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/seed_*/metrics.json")):
    data = json.loads(path.read_text())
    history = data.get("history", [])
    final = history[-1] if history else {}
    views = data.get("views", [])
    raw_view_ious = [float(v["iou"]) for v in views]
    diagnostics = projection_diagnostics(path.parent)
    rows.append({
        "variant": path.parts[-3],
        "seed": int(path.parts[-2].split("_")[-1]),
        "disabled_losses": data.get("disabled_losses", []),
        "raw_soft_iou": float(data["raw_soft_iou"]),
        "raw_min_view_iou": float(min(raw_view_ious)),
        "raw_view_ious": raw_view_ious,
        **diagnostics,
        "raw_components": int(data["raw_soft_components"]),
        "largest_component_fraction": float(final.get("largest_component_fraction", np.nan)),
        "occupancy_fraction": float(final.get("occupancy_fraction", np.nan)),
        "final_soft_train_iou": float(final.get("soft_train_iou", np.nan)),
        "final_binary_train_iou": float(final.get("binary_train_iou", np.nan)),
        "best_binary_train_iou": float(data.get("best_binary_train_iou", np.nan)),
        "best_step": int(data.get("best_step", -1)),
        "diagnostic_selected_proposal_iou": float(data.get("soft_proposal_iou", np.nan)),
    })

full_by_seed = {r["seed"]: r for r in rows if r["variant"] == "full"}
for row in rows:
    full = full_by_seed.get(row["seed"])
    if full:
        row["delta_raw_iou_vs_full"] = row["raw_soft_iou"] - full["raw_soft_iou"]
        row["delta_components_vs_full"] = row["raw_components"] - full["raw_components"]

aggregate = []
for variant in sorted({r["variant"] for r in rows}):
    group = [r for r in rows if r["variant"] == variant]
    aggregate.append({
        "variant": variant,
        "n_seeds": len(group),
        "seeds": [r["seed"] for r in group],
        "raw_soft_iou_mean": float(np.mean([r["raw_soft_iou"] for r in group])),
        "raw_soft_iou_std": float(np.std([r["raw_soft_iou"] for r in group], ddof=1)) if len(group) > 1 else 0.0,
        "raw_min_view_iou_mean": float(np.mean([r["raw_min_view_iou"] for r in group])),
        "background_false_positive_rate_mean": float(np.mean(
            [r["background_false_positive_rate_mean"] for r in group]
        )),
        "foreground_leakage_ratio_mean": float(np.mean(
            [r["foreground_leakage_ratio_mean"] for r in group]
        )),
        "projection_precision_mean": float(np.mean(
            [r["projection_precision_mean"] for r in group]
        )),
        "raw_components_mean": float(np.mean([r["raw_components"] for r in group])),
        "largest_component_fraction_mean": float(np.mean(
            [r["largest_component_fraction"] for r in group]
        )),
        "delta_raw_iou_vs_full_mean": float(np.mean([r.get("delta_raw_iou_vs_full", np.nan) for r in group])),
    })

payload = {
    "schema": "soft_tsf1_mr1_heroes_loss_ablation_v1",
    "protocol": {
        "base_silhouettes": ["Spider-Man.png", "superman.png", "batman.png"],
        "n_base_views": 3,
        "mirror": True,
        "n_supervised_views": 6,
        "tsf": 1,
        "mr": 1,
        "resolution": 256,
        "iterations": 2000,
        "seeds": sorted({r["seed"] for r in rows}),
        "evaluation": "raw thresholded soft occupancy only; morphology proposal IoU is diagnostic only",
        "stage_b_geometry": False,
    },
    "rows": rows,
    "aggregate_by_variant": aggregate,
}
(root / "aggregate_raw.json").write_text(json.dumps(payload, indent=2) + "\n")

header = "| Variant | Seed | Raw mean IoU | Raw min-view IoU | BG FP rate | Leakage/target | Precision | Components | Largest comp. frac. | Delta IoU |"
lines = [header, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for row in rows:
    lines.append(
        f"| {row['variant']} | {row['seed']} | {row['raw_soft_iou']:.6f} | {row['raw_min_view_iou']:.6f} | "
        f"{row['background_false_positive_rate_mean']:.6f} | "
        f"{row['foreground_leakage_ratio_mean']:.6f} | "
        f"{row['projection_precision_mean']:.6f} | "
        f"{row['raw_components']} | {row['largest_component_fraction']:.6f} | "
        f"{row.get('delta_raw_iou_vs_full', float('nan')):+.6f} |"
    )
(root / "table_raw.md").write_text("\n".join(lines) + "\n")
print(json.dumps(payload, indent=2))
PY

"$PY" "$ROOT/experiments/summarize_soft_threshold_stability_20260718.py" \
  --root "$OUT" --canonical-threshold 0.05 --thresholds 0.03,0.04,0.05,0.06,0.07
