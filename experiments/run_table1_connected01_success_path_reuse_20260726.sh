#!/usr/bin/env bash
set -u

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
SOFT="$ROOT/results/fixed3_mr1_connected10_pcraft_20260722/connected_01/seed_17/soft_training"
OUT="$ROOT/results/table1_connected01_success_path_reuse_t065_20260726"

V0="$ROOT/results/fixed3_mr1_connected10_pending_20260722/inputs/connected_01_view_x.png"
V1="$ROOT/results/fixed3_mr1_connected10_pending_20260722/inputs/connected_01_view_y.png"
V2="$ROOT/results/fixed3_mr1_connected10_pending_20260722/inputs/connected_01_view_z.png"

"$PY" "$ROOT/shadowforge_pipeline/unified_run.py" \
  --silhouettes "$V0" "$V1" "$V2" \
  --output "$OUT" \
  --branch soft-geometry \
  --tsf 1 \
  --mr 1 \
  --soft-resolution 256 \
  --soft-control-res 0 \
  --soft-proposal-mode proposal \
  --soft-artifact-dir "$SOFT" \
  --core-resolution 48 \
  --soft-seed-thresholds 0.65 \
  --bridge-radius 2 \
  --bridge-passes 48 \
  --disable-guarded-carve \
  --delivery-shortlist-size 0 \
  --delivery-shell-voxels 2 \
  --min-iou 0.95 \
  --min-view-iou 0.90 \
  --min-saving 0.000001 \
  --max-saving 0.85 \
  --selection-iou-tolerance 0.01 \
  --selection-policy material \
  --screen skip \
  --print-mm 100 \
  --save-candidate-evidence \
  --device cuda:0

printf '%s\n' "$?" > "$OUT/EXIT_CODE"
