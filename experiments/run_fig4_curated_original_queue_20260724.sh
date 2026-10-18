#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
SUITE="$ROOT/results/fig4_anime_curated_original_v1_20260724"
MANIFEST="$SUITE/benchmark_manifest.json"
MITRA="$SUITE/mitra"
PCRAFT="$SUITE/pcraft"
SAD_TAG="fig4_anime_curated_original_v1_sadekar_20260724"
SAD="$ROOT/results/$SAD_TAG"
AUDIT="$SUITE/sadekar_same_protocol.json"
OUT="$SUITE/four_method_comparison"

cd "$ROOT"
mkdir -p "$ROOT/logs" "$OUT"

stamp() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

stamp "Fig. 4 curated experiment queued."
while pgrep -f 'pretrain_pino_v3.py.*--arch gino_official' >/dev/null; do
  sleep 60
done

stamp "GINO released cuda:0; starting P-CRAFT."
"$PY" experiments/run_fixed3_mr1_shadowforge_20260721.py \
  --manifest "$MANIFEST" \
  --output "$PCRAFT" \
  --seed 17 \
  > "$ROOT/logs/fig4_curated_pcraft_20260724.log" 2>&1

stamp "Starting official Sadekar mesh and voxel baselines."
"$PY" experiments/run_fixed3_author_tsf_baselines_20260716.py \
  --manifest "$MANIFEST" \
  --out-tag "$SAD_TAG" \
  --seed 17 \
  --keep-input-frame \
  > "$ROOT/logs/fig4_curated_sadekar_20260724.log" 2>&1

stamp "Evaluating Sadekar artifacts with the common renderer."
"$PY" experiments/evaluate_stratified_sadekar_same_protocol_20260717.py \
  --manifest "$MANIFEST" \
  --run-summary "$SAD/baseline_run_summary.json" \
  --output "$AUDIT" \
  > "$ROOT/logs/fig4_curated_sadekar_audit_20260724.log" 2>&1

stamp "Aggregating four-method scores."
while [[ ! -s "$MITRA/results.json" ]]; do
  stamp "Waiting for the CPU Mitra baseline."
  sleep 30
done

"$PY" experiments/aggregate_fixed3_mr1_challenge_20260721.py \
  --manifest "$MANIFEST" \
  --mitra "$MITRA" \
  --sadekar "$AUDIT" \
  --shadowforge "$PCRAFT" \
  --output "$OUT" \
  > "$ROOT/logs/fig4_curated_aggregate_20260724.log" 2>&1

"$PY" experiments/select_fig4_eligible_case_20260723.py \
  --table "$OUT/table1_real.json" \
  --pcraft-summary "$PCRAFT/SUMMARY.json" \
  --output "$OUT/fig4_selection.json" \
  > "$ROOT/logs/fig4_curated_selection_20260724.log" 2>&1

stamp "Fig. 4 curated comparison complete: $OUT/fig4_selection.json"
