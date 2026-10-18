#!/usr/bin/env bash
# Replay genuine repaired-to-carved trajectories under FEM and PINO screening.
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
BENCH=${BENCH:-$ROOT/results/fixed3_mr1_connected10_pending_20260722/benchmark_manifest.json}
OUT=${OUT:-$ROOT/results/true_state_trajectory_replay_20260724}
PRINT_MM=${PRINT_MM:-210}
CASE_LIMIT=${CASE_LIMIT:-0}
CASE_EXCLUDE=${CASE_EXCLUDE:-}

mkdir -p "$OUT"
STATUS_FILE="$OUT/case_status.tsv"
: > "$STATUS_FILE"
mapfile -t CASES < <("$PY" - "$BENCH" "$CASE_LIMIT" "$CASE_EXCLUDE" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
cases = [row["case_id"] for row in payload["cases"]]
limit = int(sys.argv[2])
excluded = {item.strip() for item in sys.argv[3].split(",") if item.strip()}
cases = [case_id for case_id in cases if case_id not in excluded]
if limit > 0:
    cases = cases[:limit]
print("\n".join(cases))
PY
)

for case_id in "${CASES[@]}"; do
  echo "[trajectory] $case_id"
  if CASE_ID="$case_id" \
    BENCH="$BENCH" \
    OUT="$OUT/$case_id" \
    PRINT_MM="$PRINT_MM" \
    bash "$ROOT/experiments/run_connected06_boundary_material_trajectory_20260723.sh"; then
    printf '%s\tSUCCESS\n' "$case_id" >> "$STATUS_FILE"
  else
    printf '%s\tNO_ADMISSIBLE_TRAJECTORY\n' "$case_id" >> "$STATUS_FILE"
    echo "[trajectory] $case_id has no state satisfying all retained gates; continuing." >&2
  fi
done

"$PY" "$ROOT/experiments/evaluate_true_state_trajectory_replay_20260724.py" \
  --root "$OUT" \
  --output "$OUT/summary" \
  --sf-threshold 1.3 \
  --pino-margin 0.05
