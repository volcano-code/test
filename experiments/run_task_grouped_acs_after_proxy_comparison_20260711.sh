#!/usr/bin/env bash
set -euo pipefail

# Do not compete with the controlled proxy-model benchmark for the GPU.
while systemctl --user is-active --quiet shadowforge-pino-proxy-comparison-20260711.service; do
  sleep 60
done

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
OUT=$ROOT/results/acs_task_level_risk_coverage_20260711/rescored_best85
cd "$ROOT"
"$PY" experiments/rescore_task_triples_best_pino_20260711.py --device cuda --batch-size 16
"$PY" experiments/acs_task_level_risk_coverage_20260711.py \
  --free "$OUT/free_best85_rescored.json" \
  --fixed "$OUT/fixed_best85_rescored.json"
