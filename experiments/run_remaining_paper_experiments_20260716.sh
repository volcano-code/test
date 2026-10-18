#!/usr/bin/env bash
# Sequential GPU queue. The controlled Sadekar run is launched separately and
# must finish before this queue advances to the remaining experiments.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
cd "$ROOT"

while pgrep -f "experiments/run_fixed3_author_tsf_baselines_20260716.py" >/dev/null; do
  sleep 60
done

bash experiments/run_soft_novel_loss_ablations_v3_20260715.sh \
  > logs/soft_novel_loss_ablations_v3_20260716.log 2>&1

bash experiments/run_effective_geometry_module_pilot_ablations_20260714.sh \
  > logs/geometry_module_ablations_20260716.log 2>&1

bash experiments/run_pino_comparator_v9_protocol_20260716.sh \
  > logs/pino_comparator_v9_protocol_20260716.log 2>&1

bash experiments/run_strict_physics_pino_v11_20260716.sh \
  > logs/strict_physics_pino_v11_queue_20260716.log 2>&1
