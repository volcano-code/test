#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
GINO_PATTERN='pretrain_pino_v3.py.*--arch gino_official'

X="$ROOT/results/fig4_anime_curated_original_v1_20260724/inputs/adventurer_schoolgirl_fighter_view_x.png"
Y="$ROOT/results/fig4_anime_curated_original_v1_20260724/inputs/adventurer_schoolgirl_fighter_view_y.png"
Z="$ROOT/results/fig4_anime_curated_original_v1_20260724/inputs/adventurer_schoolgirl_fighter_view_z.png"
SOFT="$ROOT/results/fig4_anime_curated_original_v1_20260724/pcraft/adventurer_schoolgirl_fighter_original/seed_17/soft_training"
OUT="$ROOT/results/fig4_anime_curated_protocol_unified_20260725/adventurer_schoolgirl_fighter_original/seed_17"

cd "$ROOT"

# Geometry delivery still uses CUDA for TFS rendering. Wait for the official
# GINO run to release GPU memory rather than risking a second OOM.
while pgrep -f "$GINO_PATTERN" >/dev/null; do
  echo "[$(date '+%F %T')] Waiting for official GINO training to release cuda:0"
  sleep 120
done

"$PY" shadowforge_pipeline/unified_run.py \
  --silhouettes "$X" "$Y" "$Z" \
  --output "$OUT" \
  --branch soft-geometry \
  --tsf 1 --mr 1 \
  --soft-artifact-dir "$SOFT" \
  --soft-resolution 256 \
  --core-resolution 48 \
  --soft-seed-thresholds 0.05,0.15,0.25,0.35,0.45,0.55,0.65 \
  --bridge-radius 2 --bridge-passes 48 \
  --delivery-shell-voxels 2 \
  --delivery-shortlist-size 1 \
  --min-iou 0.95 --min-view-iou 0.90 \
  --min-saving 0.000001 --max-saving 0.85 \
  --min-erosion-survival 0.08 \
  --selection-iou-tolerance 0.01 --selection-policy material \
  --screen skip --save-candidate-evidence --device cuda:0
