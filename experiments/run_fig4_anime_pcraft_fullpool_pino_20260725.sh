#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
CKPT=${CKPT:-$ROOT/checkpoints/strict_physics_residual_pino_v16_shadowart_scale_20260718/best_model.pt}
INPUTS=${INPUTS:-$ROOT/results/fig4_anime_curated_original_v1_20260724/inputs}
SOFT=${SOFT:-$ROOT/results/fig4_anime_curated_original_v1_20260724/pcraft/adventurer_schoolgirl_fighter_original/seed_17/soft_training}
OUT=${OUT:-$ROOT/results/fig4_anime_pcraft_fullpool_pino_20260725/adventurer_schoolgirl_fighter}

cd "$ROOT"

"$PY" shadowforge_pipeline/unified_run.py \
  --silhouettes \
    "$INPUTS/adventurer_schoolgirl_fighter_view_x.png" \
    "$INPUTS/adventurer_schoolgirl_fighter_view_y.png" \
    "$INPUTS/adventurer_schoolgirl_fighter_view_z.png" \
  --output "$OUT" \
  --branch soft-geometry \
  --tsf 1 --mr 1 \
  --soft-artifact-dir "$SOFT" \
  --soft-resolution 256 \
  --core-resolution 48 \
  --soft-seed-thresholds 0.05,0.10,0.15,0.20,0.25,0.35,0.45,0.55,0.65 \
  --bridge-radius 2 \
  --bridge-passes 48 \
  --delivery-shell-voxels 2 \
  --delivery-bridge-passes 16 \
  --delivery-shortlist-size 0 \
  --delivery-prescreen-iou-tolerance 0.015 \
  --min-iou 0.95 \
  --min-view-iou 0.90 \
  --min-saving 0.000001 \
  --max-saving 0.85 \
  --min-erosion-survival 0.08 \
  --selection-iou-tolerance 0.01 \
  --selection-policy material \
  --allow-best-effort-fidelity-fallback \
  --screen pino_rollback_fem \
  --pino-checkpoint "$CKPT" \
  --pino-stop-margin 0.05 \
  --pino-max-rollbacks -1 \
  --structural-resolution 48 \
  --sf-threshold 1.3 \
  --print-mm 100 \
  --save-candidate-evidence \
  --device cuda:0
