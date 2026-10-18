#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
ASSET_DIR="$ROOT/data/silhouettes/curated_symbolic_v1"
OUT_ROOT="$ROOT/results/fig1_symbol_pairs_pcraft_proposal_20260722"

mkdir -p "$OUT_ROOT"
cd "$ROOT"

run_pair() {
  local name="$1"
  local left="$2"
  local right="$3"

  "$PY" shadowforge_pipeline/unified_run.py \
    --silhouettes "$ASSET_DIR/$left" "$ASSET_DIR/$right" \
    --branch soft-geometry \
    --output "$OUT_ROOT/$name" \
    --tsf 0 --mr 0 \
    --soft-resolution 256 \
    --soft-control-res 48 \
    --soft-proposal-mode proposal \
    --soft-iterations 2000 \
    --soft-lr 0.007 \
    --soft-save-every 100 \
    --soft-seed-thresholds 0.15,0.25,0.35,0.45,0.55 \
    --delivery-shell-voxels 2 \
    --bridge-radius 1 \
    --bridge-passes 16 \
    --min-iou 0.975 \
    --min-view-iou 0.950 \
    --min-saving 0.00 \
    --max-saving 0.85 \
    --selection-iou-tolerance 0.01 \
    --screen fem_only \
    --structural-resolution 48 \
    --sf-threshold 1.3 \
    --print-mm 70 \
    --save-candidate-evidence \
    --device cuda:0
}

# The first and third Figure 1(a) examples both use A--2, so it is generated once.
run_pair A_2 aaai_A.png aaai_2.png
run_pair A_0 aaai_A.png aaai_0.png
run_pair I_7 aaai_I.png aaai_7.png
