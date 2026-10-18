#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
VOXEL=/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel
INPUTS="$ROOT/results/table2_anime_fixed3_inputs_20260722"
OUT="${OUT:-$ROOT/results/joint_view_formal_128_3tasks_3seeds_20260727}"
JOINT_ITERS="${JOINT_ITERS:-300}"
N_VIEWS=6
SEQ_EQUAL_UPDATE_ITERS=$((JOINT_ITERS / N_VIEWS))

TASKS=(arcane_casters blade_heroes stage_dancers)
SEEDS=(17 23 42)

cd "$ROOT"
mkdir -p "$OUT"
"$PY" experiments/prepare_table2_anime_fixed3_20260722.py

run_one() {
  local task="$1"
  local seed="$2"
  local name="$3"
  local mode="$4"
  local iterations="$5"
  local run="$OUT/$task/$name/seed_$seed"
  local timing="$run/timing.json"

  if [[ -f "$run/metrics.json" && -f "$timing" ]]; then
    return
  fi

  mkdir -p "$run"
  local started finished
  started=$(date +%s)
  "$PY" "$VOXEL/train.py" cuda:0 "$run" "$iterations" 0.007 1 \
    -sdlist \
      "$INPUTS/$task/view_0.png" \
      "$INPUTS/$task/view_1.png" \
      "$INPUTS/$task/view_2.png" \
    -mr 1 -tsf 1 -tsf4 0 -cam ortho -imsz 256 \
    --res 128 -eres 128 --control-res 0 --sadekar-zdist 1.7 \
    --proposal-mode proposal --view-update-mode "$mode" \
    --save-every "$iterations" --no-cleanup --export-final-step \
    --seed "$seed" --disable-losses volume
  finished=$(date +%s)

  "$PY" - "$timing" "$mode" "$iterations" "$N_VIEWS" "$((finished - started))" <<'PY'
import json
import sys
from pathlib import Path

path, mode, iterations, n_views, elapsed = sys.argv[1:]
iterations = int(iterations)
n_views = int(n_views)
payload = {
    "mode": mode,
    "outer_iterations": iterations,
    "declared_views": n_views,
    "optimizer_steps": iterations if mode == "joint" else iterations * n_views,
    "training_view_exposures": iterations * n_views,
    "elapsed_s": int(elapsed),
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

for task in "${TASKS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "$task" "$seed" joint_300 joint "$JOINT_ITERS"
    run_one "$task" "$seed" sequential_equal_view_exposure sequential "$JOINT_ITERS"
    run_one "$task" "$seed" sequential_equal_optimizer_steps sequential \
      "$SEQ_EQUAL_UPDATE_ITERS"
  done
done

"$PY" experiments/summarize_joint_view_formal_20260727.py --root "$OUT"
echo "Formal joint-view audit complete: $OUT/summary.json"
