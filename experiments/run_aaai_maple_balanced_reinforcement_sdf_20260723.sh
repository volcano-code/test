#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
SRC="$ROOT/results/physical_print_aaai_maple_connected_pcraft_20260722/selected_display_voxels256.npy"
BASE="$ROOT/results/physical_print_aaai_maple_connected_pcraft_20260722/balanced_reinforcement_sdf_20260723"

mkdir -p "$BASE"
for budget in 0.25 0.23 0.20; do
  tag=${budget/./p}
  "$PY" "$ROOT/experiments/build_projection_guarded_local_reinforcement.py" \
    --volume "$SRC" \
    --output-dir "$BASE/min_saving_$tag" \
    --target-mm 100 \
    --min-saving "$budget" \
    > "$BASE/min_saving_$tag.log" 2>&1
done
