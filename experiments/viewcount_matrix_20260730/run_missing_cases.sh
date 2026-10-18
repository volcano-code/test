#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
OUT_ROOT=${OUT_ROOT:-$ROOT/results/viewcount_matrix_anime_ordinary_mixed_20260730}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs/viewcount_matrix_20260730}
V2_RUNNER="$ROOT/experiments/multiview_delivery_v2_20260728/run_multiview_delivery_v2.py"
CKPT=${CKPT:-$ROOT/checkpoints/strict_physics_residual_pino_v16_shadowart_scale_20260718/best_model.pt}
POLICY=${POLICY:-$ROOT/results/pino_ranked_fem_taskdisjoint_v16_20260720/frozen_policy.json}

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_ordinary_three_view() {
  local output="$OUT_ROOT/ordinary_3v_beetle_parrot_eagle"
  local log="$LOG_ROOT/ordinary_3v_beetle_parrot_eagle.log"
  if [[ -f "$output/MANIFEST.json" ]]; then
    printf '[skip] %s already has a manifest\n' "$output"
    return
  fi
  "$PY" -u "$V2_RUNNER" \
    --soft-artifact-dir \
      "$ROOT/results/fig4_normal_beetle_parrot_eagle_pcraft_20260724/soft_training" \
    --output "$output" \
    --device cuda:0 \
    --print-mm 100 \
    --min-mean-iou 0.95 \
    --min-view-iou 0.90 \
    --carve-checkpoints 0 \
    --sdf-sigmas 0.0,0.2,0.3,0.5,0.8 \
    --sdf-dilations 0,1 \
    --max-surface-fraction 0.45 \
    >"$log" 2>&1
}

run_mixed_two_view() {
  local output="$OUT_ROOT/mixed_2v_slender_detective_spider"
  local log="$LOG_ROOT/mixed_2v_slender_detective_spider.log"
  local input_root="$ROOT/results/fig4_correct_silhouettes_mitra_rerun_20260728"
  if [[ -f "$output/MANIFEST.json" ]]; then
    printf '[skip] %s already has a manifest\n' "$output"
    return
  fi
  "$PY" -u "$ROOT/shadowforge_pipeline/unified_run.py" \
    --silhouettes \
      "$input_root/target_view0_x.png" \
      "$input_root/target_view2_z.png" \
    --output "$output" \
    --branch soft-geometry \
    --tsf 0 \
    --mr 0 \
    --soft-resolution 256 \
    --soft-control-res 0 \
    --soft-proposal-mode proposal \
    --soft-disable-losses volume \
    --soft-iterations 2000 \
    --soft-lr 0.007 \
    --soft-save-every 100 \
    --soft-seed 42 \
    --soft-seed-thresholds 0.05,0.10,0.15,0.20,0.25,0.35,0.45,0.55,0.65 \
    --bridge-radius 2 \
    --bridge-passes 48 \
    --core-resolution 48 \
    --delivery-shell-voxels 2 \
    --min-iou 0.95 \
    --min-view-iou 0.90 \
    --min-saving 0.000001 \
    --max-saving 0.85 \
    --min-erosion-survival 0.08 \
    --selection-iou-tolerance 0.01 \
    --selection-policy material \
    --print-mm 100 \
    --screen pino_ranked_fem \
    --structural-resolution 48 \
    --sf-threshold 1.3 \
    --pino-checkpoint "$CKPT" \
    --pino-ranked-policy "$POLICY" \
    --pino-top-k 3 \
    --pino-top-k-fallback expand \
    --save-candidate-evidence \
    --device cuda:0 \
    >"$log" 2>&1
}

run_ordinary_three_view &
ordinary_pid=$!

# Keep the high-memory soft optimizer separate from the V256 delivery process.
wait "$ordinary_pid"
run_mixed_two_view

"$PY" "$ROOT/experiments/viewcount_matrix_20260730/summarize.py"
