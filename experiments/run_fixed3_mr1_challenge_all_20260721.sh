#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
MANIFEST="$ROOT/results/fixed3_mr1_challenge_frozen_20260721/benchmark_manifest.json"
MITRA="$ROOT/results/fixed3_mr1_challenge_mitra_20260721"
SAD="$ROOT/results/fixed3_mr1_challenge_sadekar_20260721"
SF="$ROOT/results/fixed3_mr1_challenge_shadowforge_20260721"
FINAL="$ROOT/results/fixed3_mr1_challenge_comparison_20260721"

cd "$ROOT"
mkdir -p "$ROOT/logs"

echo '[1/6] frozen input-only challenge manifest'
"$PY" experiments/build_fixed3_mr1_challenge_benchmark_20260721.py --output "$MANIFEST"

echo '[2/6] Mitra-style fixed3 baseline'
if [[ -f "$MITRA/results.json" ]]; then
  echo "reuse: $MITRA/results.json"
else
  "$PY" experiments/run_mitra_stratified_manifest_20260719.py --manifest "$MANIFEST" --output "$MITRA" --resolution 256
fi

echo '[3/6] Sadekar author voxel and mesh, TFS fixed3 mr1, three seeds'
"$PY" experiments/run_fixed3_author_tsf_baselines_20260716.py \
  --manifest "$MANIFEST" --out-tag fixed3_mr1_challenge_sadekar_20260721 --keep-input-frame

echo '[4/6] common-mask Sadekar evaluation'
"$PY" experiments/evaluate_stratified_sadekar_same_protocol_20260717.py \
  --manifest "$MANIFEST" --run-summary "$SAD/baseline_run_summary.json" --output "$SAD/unified_eval"

echo '[5/6] formal ShadowForge proposal + deterministic delivery, three seeds'
"$PY" experiments/run_fixed3_mr1_shadowforge_20260721.py \
  --manifest "$MANIFEST" --output "$SF"

echo '[6/6] common aggregation and Figure 4 evidence packaging'
"$PY" experiments/aggregate_fixed3_mr1_challenge_20260721.py \
  --manifest "$MANIFEST" --mitra "$MITRA" --sadekar "$SAD/unified_eval/unified_metrics.json" \
  --shadowforge "$SF" --output "$FINAL"
"$PY" experiments/select_fig4_fixed3_mr1_evidence_20260721.py \
  --manifest "$MANIFEST" --table "$FINAL/table1_real.json" \
  --sadekar "$SAD/unified_eval/unified_metrics.json" --shadowforge "$SF" \
  --output "$FINAL/fig4_evidence"

echo "complete: $FINAL"
