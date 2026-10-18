#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lin/xvxinyao/ShadowForge"
STATUS="$ROOT/results/pino_decision_background_status.json"
cd "$ROOT"

trap 'code=$?; printf "{\"status\":\"failed\",\"exit_code\":%d,\"pid\":%d}\n" "$code" "$$" > "$STATUS"' EXIT

printf '{"status":"running","stage":"external_acs_baseline","pid":%d}\n' "$$" > "$STATUS"
PYTHONPATH=. python experiments/evaluate_field_acs_external.py

printf '{"status":"running","stage":"decision_oriented_training","pid":%d}\n' "$$" > "$STATUS"
PYTHONPATH=. python experiments/train_candidate_pino_field.py \
  --epochs 6 \
  --decision-weight 16 \
  --unsafe-margin-log 0.10 \
  --device cuda \
  --output-dir checkpoints/pino_candidate_decision_20260703 \
  --result results/pino_candidate_decision_20260703.json

printf '{"status":"running","stage":"external_acs_final","pid":%d}\n' "$$" > "$STATUS"
PYTHONPATH=. python experiments/evaluate_field_acs_external.py \
  --delta checkpoints/pino_candidate_decision_20260703/field_tail_delta.pt \
  --output results/pino_decision_acs_external_20260703.json

trap - EXIT
printf '{"status":"complete","stage":"finished","pid":%d}\n' "$$" > "$STATUS"
