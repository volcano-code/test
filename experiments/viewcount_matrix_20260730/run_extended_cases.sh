#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
OUT_ROOT=${OUT_ROOT:-$ROOT/results/viewcount_matrix_anime_ordinary_mixed_20260730}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs/viewcount_matrix_20260730}
SOFT_RUNNER="$ROOT/experiments/fixed_surface_xyz_isolated_20260726/shadowforge_pipeline/unified_run.py"
DELIVERY_RUNNER="$ROOT/experiments/multiview_delivery_v2_20260728/run_multiview_delivery_v2.py"
TWO_WALL_RUNNER="$ROOT/shadowforge_pipeline/unified_run.py"
CKPT=${CKPT:-$ROOT/checkpoints/strict_physics_residual_pino_v16_shadowart_scale_20260718/best_model.pt}
POLICY=${POLICY:-$ROOT/results/pino_ranked_fem_taskdisjoint_v16_20260720/frozen_policy.json}

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_three_view_case() {
  local case_id=$1
  local input_x=$2
  local input_y=$3
  local input_z=$4
  local output="$OUT_ROOT/$case_id"
  local soft_output="$OUT_ROOT/${case_id}_soft"
  local soft_log="$LOG_ROOT/${case_id}_soft.log"
  local delivery_log="$LOG_ROOT/${case_id}_delivery.log"

  if [[ -f "$output/MANIFEST.json" ]]; then
    printf '[skip] %s already has a manifest\n' "$output"
    return
  fi

  if [[ ! -f "$soft_output/soft_training/metrics.json" ]]; then
    "$PY" -u "$SOFT_RUNNER" \
      --silhouettes "$input_x" "$input_y" "$input_z" \
      --output "$soft_output" \
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
      >"$soft_log" 2>&1
  fi

  "$PY" -u "$DELIVERY_RUNNER" \
    --soft-artifact-dir "$soft_output/soft_training" \
    --output "$output" \
    --device cuda:0 \
    --print-mm 100 \
    --min-mean-iou 0.95 \
    --min-view-iou 0.90 \
    --carve-checkpoints 0 \
    --sdf-sigmas 0.0,0.2,0.3,0.5,0.8 \
    --sdf-dilations 0,1 \
    --max-surface-fraction 0.45 \
    >"$delivery_log" 2>&1
}

run_mixed_two_view() {
  local output="$OUT_ROOT/mixed_2v_street_mecha_beetle"
  local log="$LOG_ROOT/mixed_2v_street_mecha_beetle.log"
  local anime_x="/home/lin/xvxinyao/shadow/anime_shadow_processed_black_bg_light_simplified/A010_010_未来街头英雄轮廓.png"
  local ordinary_y="$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L03_frog_beetle/inputs/right_wall_y_target.png"

  if [[ -f "$output/MANIFEST.json" ]]; then
    printf '[skip] %s already has a manifest\n' "$output"
    return
  fi

  "$PY" -u "$TWO_WALL_RUNNER" \
    --silhouettes "$anime_x" "$ordinary_y" \
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

anime_old_root="$ROOT/results/anime30_diverse_fixed_50_60_10cm_256_20260707/diverse5_thick1_13_21_23/inputs"
run_three_view_case \
  "anime_3v_diverse_13_21_23" \
  "$anime_old_root/input0_anime30_13.png" \
  "$anime_old_root/input1_anime30_21.png" \
  "$anime_old_root/input2_anime30_23.png"

run_mixed_two_view

connected_root="$ROOT/results/fixed3_mr1_connected10_pending_20260722/inputs"
run_three_view_case \
  "mixed_3v_street_mecha_connected09" \
  "/home/lin/xvxinyao/shadow/anime_shadow_processed_black_bg_light_simplified/A010_010_未来街头英雄轮廓.png" \
  "$connected_root/connected_09_view_y.png" \
  "$connected_root/connected_09_view_z.png"

"$PY" "$ROOT/experiments/viewcount_matrix_20260730/summarize.py"
