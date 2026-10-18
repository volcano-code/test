#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
INDEX="$ROOT/data/taskgrouped_strict_pino_v13_union_20260716/index.json"
LOG="$ROOT/logs/strict_physics_residual_pino_v13_queue_20260717.log"

mkdir -p "$ROOT/logs"
printf '%s waiting for v13 union index\n' "$(date -Is)" | tee -a "$LOG"
while [[ ! -s "$INDEX" ]]; do
  sleep 60
done

printf '%s v13 index ready; waiting for soft ablation GPU release\n' "$(date -Is)" | tee -a "$LOG"
while systemctl --user is-active --quiet shadowforge-soft-ablation-v3-resume.service; do
  sleep 60
done

printf '%s launching v13 PINO training\n' "$(date -Is)" | tee -a "$LOG"
exec bash "$ROOT/experiments/run_strict_physics_pino_v13_20260717.sh"
