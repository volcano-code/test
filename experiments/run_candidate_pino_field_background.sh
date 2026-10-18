#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lin/xvxinyao/ShadowForge"
STATUS="$ROOT/results/pino_candidate_field_background_status.json"

cd "$ROOT"
printf '{"status":"running","stage":"fem_field_generation","pid":%d}\n' "$$" > "$STATUS"
trap 'code=$?; printf "{\"status\":\"failed\",\"exit_code\":%d,\"pid\":%d}\n" "$code" "$$" > "$STATUS"' EXIT

if [[ ! -f data/candidate_vm_fields_grouped_20260703.npz ]]; then
  PYTHONPATH=. python experiments/generate_candidate_vm_fields.py
fi
printf '{"status":"running","stage":"field_training","pid":%d}\n' "$$" > "$STATUS"
PYTHONPATH=. python experiments/train_candidate_pino_field.py --epochs 6 --device cuda

trap - EXIT
printf '{"status":"complete","stage":"finished","pid":%d}\n' "$$" > "$STATUS"
