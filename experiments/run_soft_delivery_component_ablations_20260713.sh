#!/usr/bin/env bash
# Controlled ablation of non-learned delivery components.  The soft optimizer
# remains fixed in the deployed configuration (regularized, volume disabled).
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
OUT="$ROOT/results/ablation_soft_delivery_components_dark_knight_20260713"
L="$ROOT/results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709/dark_knight_left__magic_scholar_right/inputs/left_wall_x_view1_target.png"
R="$ROOT/results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709/dark_knight_left__magic_scholar_right/inputs/right_wall_y_view2_target.png"

mkdir -p "$OUT"
cd "$ROOT"

run_case() {
  local name="$1"
  shift
  echo "===== ${name} ====="
  "$PY" shadowforge_pipeline/unified_run.py \
    --silhouettes "$L" "$R" \
    --branch soft-geometry \
    --output "$OUT/$name" \
    --tsf 0 --mr 1 \
    --soft-resolution 256 \
    --soft-control-res 0 \
    --soft-proposal-mode regularized \
    --soft-iterations 2000 \
    --soft-lr 0.007 \
    --soft-save-every 100 \
    --soft-connectivity-start-iou 0.60 \
    --soft-fidelity-gate 0.85 \
    --soft-structure-warmup-steps 300 \
    --soft-structure-ramp-steps 300 \
    --soft-seed-only \
    --soft-seed-thresholds 0.15,0.25,0.35,0.45,0.55 \
    --core-resolution 48 \
    --delivery-shell-voxels 2 \
    --min-iou 0.975 \
    --min-view-iou 0.95 \
    --min-saving 0.30 \
    --max-saving 0.85 \
    --selection-iou-tolerance 0.01 \
    --soft-disable-losses volume \
    --device cuda:0 \
    "$@"
}

run_case full_deployed
run_case w_o_ray_restoration --disable-ray-restoration
run_case w_o_bridge --disable-bridge
run_case w_o_delivery_shell --disable-delivery-shell
