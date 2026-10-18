#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
LOG="$ROOT/logs/mitra_stratified_low6_high6_shadowforge_20260717.log"

cd "$ROOT"
while pgrep -f "strict_physics_residual_pino_v14_false_safe_20260717" >/dev/null; do
  sleep 60
done

"$PY" experiments/run_stratified_shadowforge_same_tasks_20260717.py \
  --device cuda:0 \
  --soft-iters 2000 \
  >> "$LOG" 2>&1

"$PY" experiments/aggregate_stratified_same_task_comparison_20260717.py \
  >> "$LOG" 2>&1
