#!/usr/bin/env bash
# Controlled one-loss-at-a-time ablation of the voxel trainer's actual losses.
# Runs serially on one GPU.  The source is soft-only; deterministic repair,
# carving, and 48^3-to-256^3 delivery remain fixed across all rows.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
OUT="$ROOT/results/ablation_soft_code_losses_dark_knight_20260713"
L="$ROOT/results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709/dark_knight_left__magic_scholar_right/inputs/left_wall_x_view1_target.png"
R="$ROOT/results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709/dark_knight_left__magic_scholar_right/inputs/right_wall_y_view2_target.png"

mkdir -p "$OUT"
cd "$ROOT"

run_case() {
  local name="$1"
  local disabled="$2"
  echo "===== ${name}: disabled=${disabled:-<none>} ====="
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
    --soft-disable-losses "$disabled" \
    --device cuda:0
}

# All losses implemented in voxel/train.py are active in this controlled row.
run_case full_all_losses ""

# Projection losses.
run_case w_o_l2 l2
run_case w_o_l1 l1
run_case w_o_dice dice
run_case w_o_ssim ssim
run_case w_o_false_positive false_positive

# Voxel regularizers.
run_case w_o_edge edge
run_case w_o_smooth smooth
run_case w_o_laplacian laplacian
run_case w_o_connectivity connectivity
run_case w_o_spurious spurious
run_case w_o_wall wall
run_case w_o_spectral spectral
run_case w_o_deviation deviation
run_case w_o_volume volume

# This is the actual deployed ShadowForge setting: volume loss remains off,
# because guarded carving plus the selector performs material reduction.
run_case deployed_configuration volume
