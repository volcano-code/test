#!/usr/bin/env bash
# Fixed pool first; selection is performed only after all four methods finish.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
STAMP=20260723
SUITE="$ROOT/results/fig4_anime_pool_fixed3_mr1_${STAMP}"
MANIFEST="$SUITE/benchmark_manifest.json"
MITRA="$ROOT/results/fig4_anime_pool_mitra_${STAMP}"
PCR="$ROOT/results/fig4_anime_pool_pcraft_${STAMP}"
SAD_TAG="fig4_anime_pool_sadekar_${STAMP}"
SAD="$ROOT/results/$SAD_TAG"
AUDIT="$ROOT/results/fig4_anime_pool_sadekar_audit_${STAMP}.json"
OUT="$ROOT/results/fig4_anime_pool_aggregate_${STAMP}"

cd "$ROOT"
mkdir -p "$ROOT/logs"
"$PY" experiments/build_fig4_anime_pool_20260723.py

# Mitra is CPU-only, so it can overlap with the first GPU P-CRAFT run.
"$PY" experiments/run_mitra_stratified_manifest_20260719.py \
  --manifest "$MANIFEST" --resolution 256 --output "$MITRA" \
  > "logs/fig4_anime_mitra_${STAMP}.log" 2>&1 &
MITRA_PID=$!

"$PY" experiments/run_fixed3_mr1_shadowforge_20260721.py \
  --manifest "$MANIFEST" --output "$PCR" --seed 17 \
  > "logs/fig4_anime_pcraft_${STAMP}.log" 2>&1

# Official Sadekar implementations.  Do not launch this alongside P-CRAFT:
# both occupy cuda:0.  The original input frame is retained for common scoring.
"$PY" experiments/run_fixed3_author_tsf_baselines_20260716.py \
  --manifest "$MANIFEST" --out-tag "$SAD_TAG" --seed 17 --keep-input-frame \
  > "logs/fig4_anime_sadekar_${STAMP}.log" 2>&1

wait "$MITRA_PID" || true

"$PY" experiments/evaluate_stratified_sadekar_same_protocol_20260717.py \
  --manifest "$MANIFEST" --run-summary "$SAD/baseline_run_summary.json" \
  --output "$AUDIT" \
  > "logs/fig4_anime_sadekar_audit_${STAMP}.log" 2>&1

"$PY" experiments/aggregate_fixed3_mr1_challenge_20260721.py \
  --manifest "$MANIFEST" --mitra "$MITRA" --sadekar "$AUDIT" \
  --shadowforge "$PCR" --output "$OUT"

"$PY" experiments/select_fig4_eligible_case_20260723.py \
  --table "$OUT/table1_real.json" --pcraft-summary "$PCR/SUMMARY.json" \
  --output "$OUT/fig4_selection.json"
