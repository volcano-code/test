#!/usr/bin/env bash
# Wait for the active Table 4b component ablation, then run controlled pilots.
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
PINO_OUT=${PINO_OUT:-$ROOT/results/pino_carve_stopping_pilot_20260721}
SOFT_OUT=${SOFT_OUT:-$ROOT/results/soft_projection_loss_ablation_pilot_20260721}
FIG_OUT=${FIG_OUT:-$ROOT/results/new_ablation_figures_pilot_20260721}

cd "$ROOT"
while pgrep -f '[r]un_pino_component_ablations_20260720.sh' >/dev/null; do
  echo "$(date -Is) waiting for Table 4b component ablation"
  sleep 60
done

echo "$(date -Is) starting PINO trajectory-stopping pilot"
CASE_LIMIT=1 OUT="$PINO_OUT" bash experiments/run_pino_carve_stopping_20260720.sh

echo "$(date -Is) starting soft projection-loss pilot"
CASE_LIMIT=1 SEEDS="43" ITERATIONS=1000 OUT="$SOFT_OUT" \
  bash experiments/run_soft_projection_loss_ablation_20260720.sh

echo "$(date -Is) rendering pilot tables and figures"
"$PY" experiments/render_pino_soft_ablation_results_20260721.py \
  --pino-dir "$PINO_OUT" --soft-dir "$SOFT_OUT" --out-dir "$FIG_OUT"
echo "$(date -Is) queue completed"
