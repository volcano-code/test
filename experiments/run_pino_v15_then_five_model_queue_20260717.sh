#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/lin/xvxinyao/ShadowForge
cd "$ROOT"
bash experiments/run_strict_physics_pino_v15_tail_20260717.sh \
  2>&1 | tee "$ROOT/logs/strict_physics_residual_pino_v15_tail_20260717.log"
bash experiments/run_taskdisjoint_five_model_comparator_20260717.sh \
  2>&1 | tee "$ROOT/logs/taskdisjoint_five_model_comparator_20260717.log"
