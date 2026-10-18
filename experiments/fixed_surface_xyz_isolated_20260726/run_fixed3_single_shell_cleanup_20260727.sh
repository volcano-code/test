#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
SOURCE="$ROOT/results/fig4_anime_fixed3_minimal_delivery_isolated_20260726"
SOFT="$ROOT/results/fig4_anime_fixed_xyz_minimal_mr1_20260726/soft_training"
OUT="$ROOT/results/fig4_anime_fixed3_single_shell_sdf_20260727"

X="$ROOT/results/fig4_anime_curated_original_v1_20260724/inputs/adventurer_schoolgirl_fighter_view_x.png"
Y="$ROOT/results/fig4_anime_curated_original_v1_20260724/inputs/adventurer_schoolgirl_fighter_view_y.png"
Z="$ROOT/results/fig4_anime_curated_original_v1_20260724/inputs/adventurer_schoolgirl_fighter_view_z.png"

cd "$ROOT"

"$PY" -u \
  experiments/fixed_surface_xyz_isolated_20260726/clean_single_shell_v256_20260727.py \
  --input-volume "$SOURCE/delivered_volume_256.npy" \
  --soft-artifact-dir "$SOFT" \
  --silhouettes "$X" "$Y" "$Z" \
  --output "$OUT" \
  --mr 1 \
  --device cuda:0 \
  --print-mm 100 \
  --closing-radii 0 1 2 \
  --smooth-sigmas 0.55 0.80 \
  --min-mean-iou 0.95 \
  --min-view-iou 0.90

