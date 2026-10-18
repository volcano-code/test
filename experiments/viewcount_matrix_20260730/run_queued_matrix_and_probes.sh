#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
WAIT_FOR_PID=${WAIT_FOR_PID:-}
POLL_SECONDS=${POLL_SECONDS:-20}

if [[ -n "$WAIT_FOR_PID" ]]; then
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done
fi

bash "$ROOT/experiments/viewcount_matrix_20260730/run_missing_cases.sh"
bash "$ROOT/experiments/viewcount_matrix_20260730/run_threeview_contribution_probes.sh"
bash "$ROOT/experiments/viewcount_matrix_20260730/run_extended_cases.sh"
bash "$ROOT/experiments/viewcount_matrix_20260730/run_threeview_probes.sh"
"$ROOT"/experiments/viewcount_matrix_20260730/summarize.py >/dev/null
"$ROOT"/experiments/viewcount_matrix_20260730/render_evidence_gallery.py
