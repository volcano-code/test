#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
INPUTS=${INPUTS:-$ROOT/paper_aaai_submission/symbol_silhouettes_20260721}
OUT=${OUT:-$ROOT/results/aaai_maple_printsafe_display_100mm_r1_20260722}

cd "$ROOT"

"$PY" shadowforge_pipeline/unified_run.py \
  --silhouettes "$INPUTS/aaai_triangle_silhouette.png" "$INPUTS/maple_leaf_silhouette.png" \
  --output "$OUT" \
  --branch soft-geometry \
  --tsf 0 --mr 0 \
  --soft-resolution 256 --soft-control-res 0 --core-resolution 48 \
  --soft-proposal-mode proposal --soft-disable-losses volume \
  --soft-iterations 2000 --soft-lr 0.007 --soft-save-every 100 --soft-seed 43 \
  --soft-seed-thresholds 0.15,0.25,0.35,0.45 \
  --bridge-radius 1 --bridge-passes 16 \
  --delivery-shell-voxels 3 \
  --min-iou 0.975 --min-view-iou 0.95 \
  --min-saving 0.15 --max-saving 0.55 --min-erosion-survival 0.55 \
  --selection-iou-tolerance 0.005 --selection-policy fdm \
  --max-unsupported-ratio 1.0 --max-support-islands 1000000000 --min-bed-area-mm2 0.0 \
  --screen fem_only --structural-resolution 48 --sf-threshold 1.3 \
  --print-mm 100 --display-fixture --save-candidate-evidence --device cuda:0
