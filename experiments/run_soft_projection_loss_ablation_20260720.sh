#!/usr/bin/env bash
# Projection-only soft ablation. No repair, carving, delivery shell, mesh QA,
# PINO, or FEM is invoked. All variants export the common final optimizer step.
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
VOXEL=${VOXEL:-/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
OUT=${OUT:-$ROOT/results/soft_projection_loss_ablation_20260720}
ITERATIONS=${ITERATIONS:-2000}
SEEDS=(${SEEDS:-17 29 43})
CASE_LIMIT=${CASE_LIMIT:-0}

CASES=(
  "H01_future_mech|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H01_future_mech/inputs"
  "H02_sword_fox|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H02_sword_fox/inputs"
  "H03_street_mecha|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H03_street_mecha/inputs"
  "H04_mage_warrior|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H04_mage_warrior/inputs"
  "H05_archer_heroine|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H05_archer_heroine/inputs"
  "H06_knight_mage|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H06_knight_mage/inputs"
  "L01_parrot_snail|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L01_parrot_snail/inputs"
  "L02_owl_turtle|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L02_owl_turtle/inputs"
  "L03_frog_beetle|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L03_frog_beetle/inputs"
  "L04_armadillo_bunny|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L04_armadillo_bunny/inputs"
  "L05_puma_teddy|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L05_puma_teddy/inputs"
  "L06_lobster_duck|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L06_lobster_duck/inputs"
)

# These are the only added projection terms retained by the current proposal.
VARIANTS=(
  "full|"
  "w_o_Ldice|dice"
  "w_o_Lout|false_positive"
  "w_o_Lssim|ssim"
)

mkdir -p "$OUT"
case_index=0
for case in "${CASES[@]}"; do
  (( CASE_LIMIT > 0 && case_index >= CASE_LIMIT )) && break
  ((case_index += 1))
  IFS='|' read -r task inputs <<< "$case"
  left="$inputs/left_wall_x_target.png"
  right="$inputs/right_wall_y_target.png"
  for variant in "${VARIANTS[@]}"; do
    IFS='|' read -r name removed <<< "$variant"
    for seed in "${SEEDS[@]}"; do
      run="$OUT/$task/$name/seed_$seed"
      if [[ -f "$run/metrics.json" ]]; then
        echo "skip completed $task $name seed=$seed"
        continue
      fi
      mkdir -p "$run"
      disabled=(volume)
      [[ -n "$removed" ]] && disabled+=("$removed")
      "$PY" "$VOXEL/train.py" cuda:0 "$run" "$ITERATIONS" 0.007 1 \
        -sdlist "$left" "$right" \
        -mr 1 -tsf 1 -tsf4 0 -cam ortho -imsz 256 \
        --res 256 -eres 256 --control-res 0 \
        --proposal-mode proposal --save-every 100 \
        --no-cleanup --export-final-step --seed "$seed" \
        --disable-losses "${disabled[@]}"
    done
  done
done

"$PY" - "$OUT" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/*/seed_*/metrics.json")):
    data = json.loads(path.read_text())
    task, variant, seed = path.parts[-4:-1]
    leaks, boundary_f1 = [], []
    for pred_path in sorted(path.parent.glob("pred_view_*.png")):
        index = pred_path.stem.rsplit("_", 1)[1]
        gt_path = path.parent / f"gt_view_{index}.png"
        pred = np.asarray(Image.open(pred_path).convert("L")) >= 128
        gt = np.asarray(Image.open(gt_path).convert("L")) >= 128
        leaks.append(float(np.logical_and(pred, ~gt).mean()))
        pred_edge = pred ^ binary_erosion(pred)
        gt_edge = gt ^ binary_erosion(gt)
        precision = np.logical_and(pred_edge, binary_dilation(gt_edge, iterations=1)).sum() / max(pred_edge.sum(), 1)
        recall = np.logical_and(gt_edge, binary_dilation(pred_edge, iterations=1)).sum() / max(gt_edge.sum(), 1)
        boundary_f1.append(float(2 * precision * recall / max(precision + recall, 1e-12)))
    rows.append({
        "task": task, "variant": variant, "seed": seed,
        "mean_iou": float(data["mean_iou"]),
        "worst_view_iou": float(min(view["iou"] for view in data["views"])),
        "mean_dice": float(data["mean_dice"]),
        "fp_leak": float(np.mean(leaks)),
        "boundary_f1": float(np.mean(boundary_f1)),
    })

groups = defaultdict(list)
for row in rows:
    groups[row["variant"]].append(row)
summary = []
for variant, items in sorted(groups.items()):
    out = {"variant": variant, "n": len(items)}
    for metric in ("mean_iou", "worst_view_iou", "mean_dice", "fp_leak", "boundary_f1"):
        values = [item[metric] for item in items]
        out[f"{metric}_mean"] = float(np.mean(values))
        out[f"{metric}_std"] = float(np.std(values))
    summary.append(out)
(root / "rows.json").write_text(json.dumps(rows, indent=2) + "\n")
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(root / "summary.json")
PY
