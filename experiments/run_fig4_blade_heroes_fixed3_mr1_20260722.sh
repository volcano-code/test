#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
MANIFEST="$ROOT/results/fig4_blade_heroes_fixed3_mr1_20260722/benchmark_manifest.json"
MITRA="$ROOT/results/fig4_blade_heroes_mitra_20260722"
SAD="$ROOT/results/fig4_blade_heroes_sadekar_20260722"
PCRAFT="$ROOT/results/fig4_blade_heroes_pcraft_20260722"

cd "$ROOT"
"$PY" experiments/build_fig4_blade_heroes_manifest_20260722.py

# Serialize this one-case qualitative suite behind the active Table 2 pilot.
while systemctl --user is-active --quiet pcraft-table2a-anime-fixed3-pilot-20260722.service; do
  sleep 15
done

"$PY" experiments/run_mitra_stratified_manifest_20260719.py \
  --manifest "$MANIFEST" --output "$MITRA" --resolution 256

"$PY" experiments/run_fixed3_author_tsf_baselines_20260716.py \
  --manifest "$MANIFEST" --out-tag fig4_blade_heroes_sadekar_20260722 \
  --keep-input-frame --seed 17

"$PY" experiments/evaluate_stratified_sadekar_same_protocol_20260717.py \
  --manifest "$MANIFEST" --run-summary "$SAD/baseline_run_summary.json" \
  --output "$SAD/unified_eval"

"$PY" experiments/run_fixed3_mr1_shadowforge_20260721.py \
  --manifest "$MANIFEST" --output "$PCRAFT" --seed 17

echo "Figure 4 single-case evidence completed."
