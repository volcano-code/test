#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/lin/xvxinyao/ShadowForge}
PY=${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}
OUT_ROOT=${OUT_ROOT:-$ROOT/results/viewcount_matrix_anime_ordinary_mixed_20260730/probes}
LOG_ROOT=${LOG_ROOT:-$ROOT/logs/viewcount_matrix_20260730/probes}
RUNNER="$ROOT/experiments/multiview_delivery_v2_20260728/run_multiview_delivery_v2.py"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

run_probe() {
  local name=$1
  local source=$2
  local mean_gate=$3
  local view_gate=$4
  local output="$OUT_ROOT/$name"
  local log="$LOG_ROOT/$name.log"
  if [[ -f "$output/MANIFEST.json" ]]; then
    printf '[skip] %s already has a manifest\n' "$output"
    return
  fi
  "$PY" -u "$RUNNER" \
    --soft-artifact-dir "$source" \
    --output "$output" \
    --device cuda:0 \
    --print-mm 100 \
    --min-mean-iou "$mean_gate" \
    --min-view-iou "$view_gate" \
    --carve-checkpoints 0 \
    --max-repair-components 128 \
    --max-repair-gap 9 \
    --max-repair-fraction 0.05 \
    --sdf-sigmas 0.0,0.15,0.25,0.35,0.5,0.8 \
    --sdf-dilations 0,1 \
    --max-surface-fraction 0.45 \
    >"$log" 2>&1
}

# Run sequentially: each case holds several 256^3 arrays during SDF search.
run_probe \
  anime_3v_bounded_repair \
  "$ROOT/results/fig4_anime_fixed3_protocol_recheck_20260727/soft_training" \
  0.987 0.974

run_probe \
  ordinary_3v_bounded_repair \
  "$ROOT/results/fig4_normal_beetle_parrot_eagle_pcraft_20260724/soft_training" \
  0.979 0.955

run_probe \
  mixed_3v_bounded_repair \
  "$ROOT/results/fig4_slender_detective_soft_20260729/soft_training" \
  0.985 0.975

"$PY" "$ROOT/experiments/viewcount_matrix_20260730/summarize.py"
