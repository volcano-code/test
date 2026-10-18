#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
RUNNER="$ROOT/experiments/run_fixed3_mr1_shadowforge_20260721.py"
ANIME_MANIFEST="$ROOT/experiments/manifests/table1_exact_anime3_20260725.json"
TABLE1_MANIFEST="$ROOT/results/fixed3_mr1_connected10_pending_20260722/benchmark_manifest.json"
OUT="$ROOT/results/table1_exact_reproduction_no_pino_20260725"

mkdir -p "$OUT"

echo "[1/3] Exact Table-1 P-CRAFT protocol on the three anime silhouettes (seed 17)"
"$PY" "$RUNNER" \
  --manifest "$ANIME_MANIFEST" \
  --output "$OUT/anime" \
  --seed 17 \
  --force

echo "[2/3] Exact Table-1 P-CRAFT protocol on connected_01 (seed 17)"
"$PY" "$RUNNER" \
  --manifest "$TABLE1_MANIFEST" \
  --output "$OUT/connected_01_rerun" \
  --case connected_01 \
  --seed 17 \
  --force

echo "[3/3] Compare the connected_01 rerun with the archived Table-1 evidence"
"$PY" "$ROOT/experiments/compare_table1_exact_rerun_20260725.py" \
  --reference "$ROOT/results/fixed3_mr1_connected10_pcraft_20260722/connected_01" \
  --rerun "$OUT/connected_01_rerun/connected_01" \
  --output "$OUT/connected_01_comparison.json"

echo "Complete: $OUT"
