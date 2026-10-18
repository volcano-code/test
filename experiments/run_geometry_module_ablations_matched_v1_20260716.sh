#!/usr/bin/env bash
# Stage-B deterministic geometry ablation on the exact Stage-A task set.
# Each task reuses one frozen full/seed_17 soft field for every variant.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
SOFT="$ROOT/results/soft_novel_loss_ablations_v3_20260715"
OUT="$ROOT/results/geometry_module_ablations_matched_v1_20260716"

CASES=(
  "future_mech|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H01_future_mech/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H01_future_mech/inputs/right_wall_y_target.png"
  "sword_fox|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H02_sword_fox/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H02_sword_fox/inputs/right_wall_y_target.png"
  "parrot_snail|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L01_parrot_snail/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L01_parrot_snail/inputs/right_wall_y_target.png"
  "owl_turtle|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L02_owl_turtle/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L02_owl_turtle/inputs/right_wall_y_target.png"
)

VARIANTS=(
  "full|"
  "w_o_ray_restoration|--disable-ray-restoration"
  "w_o_bridge_repair|--disable-bridge"
  "w_o_guarded_carving|--disable-guarded-carve"
  "w_o_delivery_shell|--disable-delivery-shell"
)

run_case() {
  local task="$1" left="$2" right="$3" variant="$4" flags="$5"
  local frozen="$SOFT/$task/full/seed_17"
  local output="$OUT/$task/$variant"
  while [[ ! -f "$frozen/metrics.json" ]]; do
    echo "waiting for frozen soft artifact: $frozen"
    sleep 60
  done
  if [[ -f "$output/MANIFEST.json" ]]; then
    echo "skip completed $task/$variant"
    return
  fi
  echo "===== geometry task=$task variant=$variant frozen=$frozen ====="
  # shellcheck disable=SC2086
  "$PY" "$ROOT/shadowforge_pipeline/unified_run.py" \
    --silhouettes "$left" "$right" \
    --branch soft-geometry --output "$output" \
    --tsf 0 --mr 0 \
    --soft-resolution 256 --soft-control-res 0 \
    --soft-artifact-dir "$frozen" \
    --soft-seed-thresholds 0.55 \
    --core-resolution 48 --delivery-shell-voxels 2 \
    --min-iou 0.0 --min-view-iou 0.0 \
    --min-saving 0.0 --max-saving 1.0 --min-erosion-survival 0.0 \
    --screen skip --device cpu $flags
}

mkdir -p "$OUT"
# The delivered 256^3 SDF/mesh QA has a high transient RAM cost.  Wait for the
# matched Stage-A GPU experiment to finish so both studies cannot contend for
# host memory while preserving their identical task set.
while [[ ! -f "$SOFT/aggregate.json" ]]; do
  echo "waiting for matched Stage-A aggregate: $SOFT/aggregate.json"
  sleep 60
done
for case in "${CASES[@]}"; do
  IFS='|' read -r task left right <<< "$case"
  for item in "${VARIANTS[@]}"; do
    IFS='|' read -r variant flags <<< "$item"
    run_case "$task" "$left" "$right" "$variant" "$flags"
  done
done

"$PY" - "$OUT" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
groups = defaultdict(list)
raw = []
for path in sorted(root.glob("*/*/MANIFEST.json")):
    data = json.loads(path.read_text())
    task, variant = path.parts[-3:-1]
    selected = data.get("selected")
    if selected is None:
        candidates = data.get("candidates", [])
        if not candidates:
            continue
        selected = max(candidates, key=lambda row: (row.get("mean_iou", 0.0), row.get("saving", 0.0)))
    row = {
        "task": task,
        "variant": variant,
        "mean_iou": float(selected["mean_iou"]),
        "mean_dice": float(selected["mean_dice"]),
        "saving": float(selected["saving"]),
        "single_body": int(selected.get("components", 0) == 1),
        "watertight": int(bool(selected.get("watertight", False))),
    }
    raw.append(row)
    groups[variant].append(row)

summary = []
for variant, rows in sorted(groups.items()):
    summary.append({
        "variant": variant,
        "n_tasks": len(rows),
        "mean_iou": float(np.mean([r["mean_iou"] for r in rows])),
        "std_iou": float(np.std([r["mean_iou"] for r in rows])),
        "mean_dice": float(np.mean([r["mean_dice"] for r in rows])),
        "std_dice": float(np.std([r["mean_dice"] for r in rows])),
        "mean_saving": float(np.mean([r["saving"] for r in rows])),
        "std_saving": float(np.std([r["saving"] for r in rows])),
        "single_body_rate": float(np.mean([r["single_body"] for r in rows])),
        "watertight_rate": float(np.mean([r["watertight"] for r in rows])),
    })
(root / "per_task.json").write_text(json.dumps(raw, indent=2) + "\n")
(root / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
