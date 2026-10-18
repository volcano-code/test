#!/usr/bin/env bash
# One frozen soft proposal, followed by isolated delivery-stage switches.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
BASE="$ROOT/results/dark_knight_current_pipeline_recheck_20260722/inputs"
L="$BASE/left_wall_x_target.png"
R="$BASE/right_wall_y_target.png"
OUT="$ROOT/results/table2b_bridge_delivery_dark_knight_20260723"
FROZEN="$OUT/frozen_soft"

cd "$ROOT"
mkdir -p "$OUT"

# This is the only learned optimization in this ablation.  Every variant below
# reads exactly this saved probability/proposal pair.
if [[ ! -f "$FROZEN/soft_training/occupancy_probability_core.npy" ]]; then
  "$PY" shadowforge_pipeline/unified_run.py \
    --silhouettes "$L" "$R" --output "$FROZEN" --branch soft \
    --tsf 0 --mr 0 --soft-resolution 256 --soft-control-res 0 \
    --soft-proposal-mode regularized --soft-iterations 2000 --soft-lr 0.007 \
    --soft-save-every 100 --soft-seed 17 --soft-export-final-step \
    --device cuda:0
fi

run_case() {
  local name="$1"; shift
  "$PY" shadowforge_pipeline/unified_run.py \
    --silhouettes "$L" "$R" --output "$OUT/$name" --branch soft-geometry \
    --tsf 0 --mr 0 --soft-resolution 256 --soft-artifact-dir "$FROZEN/soft_training" \
    --core-resolution 48 --soft-seed-thresholds 0.15 \
    --bridge-radius 1 --bridge-passes 16 --delivery-shell-voxels 2 \
    --min-iou 0.0 --min-view-iou 0.0 --min-saving 0.0 --max-saving 1.0 \
    --min-erosion-survival 0.0 --selection-iou-tolerance 0.0 \
    --selection-policy material --screen skip --save-candidate-evidence --device cuda:0 "$@"
}

run_case full
run_case w_o_ray_restoration --disable-ray-restoration
run_case w_o_bridge_repair --disable-bridge
run_case w_o_delivery_shell --disable-delivery-shell

"$PY" experiments/summarize_bridge_delivery_ablation_20260723.py \
  --root "$OUT" --output "$OUT/summary.json"
