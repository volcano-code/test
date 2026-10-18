#!/usr/bin/env bash
# Sequential queue for remaining publication experiments.  It intentionally
# waits for current GPU studies so new soft optimization cannot contend with
# PINO training or the already-running complex soft ablation.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
LOG="$ROOT/logs/supplemental_experiments_queue_20260718.log"

exec >>"$LOG" 2>&1
echo "queue started $(date -Is)"

while pgrep -f 'structural/training/pretrain_pino_v3.py|run_soft_tsf1_mr1_heroes_ablation_20260717.sh' >/dev/null; do
  echo "waiting for active PINO/soft jobs $(date -Is)"
  sleep 60
done

echo "starting current unified-pipeline 12-task comparison $(date -Is)"
"$PY" "$ROOT/experiments/run_stratified_shadowforge_same_tasks_unified_20260718.py"

echo "aggregating aligned comparison $(date -Is)"
"$PY" "$ROOT/experiments/aggregate_stratified_same_task_comparison_20260717.py" \
  --sadekar "$ROOT/results/mitra_stratified_low6_high6_unified_eval_20260718/sadekar_reaudit/unified_metrics.json" \
  --shadowforge "$ROOT/results/mitra_stratified_low6_high6_shadowforge_unified_20260718/SUMMARY.json" \
  --output "$ROOT/results/mitra_stratified_low6_high6_unified_eval_20260718"

echo "queue completed $(date -Is)"
