#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
MANIFEST="$ROOT/results/fixed3_mr1_connected10_pending_20260722/benchmark_manifest.json"
MITRA="$ROOT/results/fixed3_mr1_connected10_mitra_20260722"
SAD="$ROOT/results/fixed3_mr1_connected10_sadekar_20260722"
PCRAFT="$ROOT/results/fixed3_mr1_connected10_pcraft_20260722"
FINAL="$ROOT/results/fixed3_mr1_connected10_comparison_20260722"

cd "$ROOT"

echo '[1/6] Mitra-style fixed-three-view baseline'
for index in $(seq -w 1 10); do
  case_id="connected_${index}"
  echo "  Mitra: ${case_id}"
  "$PY" experiments/run_mitra_stratified_manifest_20260719.py \
    --manifest "$MANIFEST" --output "$MITRA" --resolution 256 \
    --case "$case_id" --reuse --skip-obj
done
"$PY" experiments/run_mitra_stratified_manifest_20260719.py \
  --manifest "$MANIFEST" --output "$MITRA" --resolution 256 --reuse

echo '[2/6] Sadekar author voxel and mesh baselines, TFS fixed3 mr1'
"$PY" experiments/run_fixed3_author_tsf_baselines_20260716.py \
  --manifest "$MANIFEST" \
  --out-tag fixed3_mr1_connected10_sadekar_20260722 \
  --keep-input-frame

echo '[3/6] Common-mask Sadekar evaluation'
"$PY" experiments/evaluate_stratified_sadekar_same_protocol_20260717.py \
  --manifest "$MANIFEST" \
  --run-summary "$SAD/baseline_run_summary.json" \
  --output "$SAD/unified_eval"

echo '[4/6] P-CRAFT soft-to-delivery pipeline'
"$PY" experiments/run_fixed3_mr1_shadowforge_20260721.py \
  --manifest "$MANIFEST" --output "$PCRAFT"

echo '[5/6] Common aggregation'
"$PY" experiments/aggregate_fixed3_mr1_challenge_20260721.py \
  --manifest "$MANIFEST" \
  --mitra "$MITRA" \
  --sadekar "$SAD/unified_eval/unified_metrics.json" \
  --shadowforge "$PCRAFT" \
  --output "$FINAL"

echo '[6/6] Figure 4 evidence packaging'
"$PY" experiments/select_fig4_fixed3_mr1_evidence_20260721.py \
  --manifest "$MANIFEST" \
  --table "$FINAL/table1_real.json" \
  --sadekar "$SAD/unified_eval/unified_metrics.json" \
  --shadowforge "$PCRAFT" \
  --output "$FINAL/fig4_evidence"

echo "complete: $FINAL"
