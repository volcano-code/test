#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/lin/xvxinyao/ShadowForge
EXPERIMENT_PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
SOFT_RUNNER="$PROJECT_ROOT/experiments/fixed_surface_xyz_isolated_20260726/shadowforge_pipeline/unified_run.py"
DELIVERY_RUNNER="$PROJECT_ROOT/experiments/multiview_delivery_v2_20260728/run_multiview_delivery_v2.py"
MITRA_RUNNER="$PROJECT_ROOT/experiments/mitra_pauly2009_shadow_hull.py"

# Frozen Figure 4 inputs, in the declared X/Y/Z order:
# slender detective, the user-selected second silhouette, and spider.
INPUT_ROOT="$PROJECT_ROOT/results/fig4_correct_silhouettes_mitra_rerun_20260728"
INPUT_X="$INPUT_ROOT/target_view0_x.png"
INPUT_Y="$INPUT_ROOT/target_view1_y.png"
INPUT_Z="$INPUT_ROOT/target_view2_z.png"

SOFT_OUTPUT="$PROJECT_ROOT/results/fig4_slender_detective_aspect_soft_20260729"
DELIVERY_OUTPUT="$PROJECT_ROOT/results/fig4_slender_detective_aspect_multiview_v2_20260729"
MITRA_OUTPUT="$PROJECT_ROOT/results/fig4_slender_detective_aspect_mitra_20260729"
LOG_ROOT="$PROJECT_ROOT/logs"

mkdir -p "$LOG_ROOT"

for input_path in "$INPUT_X" "$INPUT_Y" "$INPUT_Z"; do
  test -f "$input_path"
done

"$EXPERIMENT_PY" -u "$SOFT_RUNNER" \
  --silhouettes "$INPUT_X" "$INPUT_Y" "$INPUT_Z" \
  --output "$SOFT_OUTPUT" \
  --branch soft \
  --tsf 1 \
  --mr 1 \
  --soft-resolution 256 \
  --soft-control-res 0 \
  --soft-proposal-mode proposal \
  --soft-iterations 2000 \
  --soft-lr 0.007 \
  --soft-save-every 100 \
  --soft-seed 42 \
  --soft-preserve-target-aspect \
  --device cuda:0 \
  | tee "$LOG_ROOT/fig4_slender_detective_aspect_soft_20260729.log"

"$EXPERIMENT_PY" -u "$MITRA_RUNNER" \
  --silhouettes \
    "$SOFT_OUTPUT/soft_training/gt_view_0.png" \
    "$SOFT_OUTPUT/soft_training/gt_view_1.png" \
    "$SOFT_OUTPUT/soft_training/gt_view_2.png" \
  --axes x y z \
  --resolution 256 \
  --output "$MITRA_OUTPUT" \
  | tee "$LOG_ROOT/fig4_slender_detective_aspect_mitra_20260729.log"

"$EXPERIMENT_PY" -u "$DELIVERY_RUNNER" \
  --soft-artifact-dir "$SOFT_OUTPUT/soft_training" \
  --output "$DELIVERY_OUTPUT" \
  --device cuda:0 \
  --print-mm 100 \
  --min-mean-iou 0.95 \
  --min-view-iou 0.95 \
  | tee "$LOG_ROOT/fig4_slender_detective_aspect_multiview_v2_20260729.log"
