#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
VOXEL=/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel
INPUTS="$ROOT/results/table2_anime_fixed3_inputs_20260722"
OUT="${OUT:-$ROOT/results/table2a_loss_only_formal_128_3tasks_3seeds_20260727}"

TASKS=(arcane_casters blade_heroes stage_dancers)
SEEDS=(17 23 42)
VARIANTS=(
  "full|"
  "w_o_Ldice|dice"
  "w_o_Lssim|ssim"
  "w_o_Lout|false_positive"
)

mkdir -p "$OUT"
cd "$ROOT"
"$PY" experiments/prepare_table2_anime_fixed3_20260722.py

# Reuse the exact seed-17 runs from the completed controlled pilot. This is
# computation reuse only; paths, protocol, iterations, and disabled terms are
# identical to the formal experiment below.
PILOT="$ROOT/results/table2a_anime_fixed3_mr1_true128_seed17_20260723"
for item in "${VARIANTS[@]}"; do
  IFS='|' read -r variant disabled_loss <<< "$item"
  for task in "${TASKS[@]}"; do
    src="$PILOT/$task/$variant/seed_17"
    dst="$OUT/$task/$variant/seed_17"
    if [[ -f "$src/metrics.json" && ! -f "$dst/metrics.json" ]]; then
      mkdir -p "$(dirname "$dst")"
      cp -a "$src" "$dst"
    fi
  done
done

for item in "${VARIANTS[@]}"; do
  IFS='|' read -r variant disabled_loss <<< "$item"
  for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run="$OUT/$task/$variant/seed_$seed"
      [[ -f "$run/metrics.json" ]] && continue
      mkdir -p "$run"
      disabled=(volume)
      [[ -n "$disabled_loss" ]] && disabled+=("$disabled_loss")
      "$PY" "$VOXEL/train.py" cuda:0 "$run" 2000 0.007 1 \
        -sdlist \
          "$INPUTS/$task/view_0.png" \
          "$INPUTS/$task/view_1.png" \
          "$INPUTS/$task/view_2.png" \
        -mr 1 -tsf 1 -tsf4 0 -cam ortho -imsz 256 \
        --res 128 -eres 128 --control-res 0 --sadekar-zdist 1.7 \
        --proposal-mode proposal --view-update-mode joint \
        --save-every 200 --no-cleanup --export-final-step \
        --seed "$seed" --disable-losses "${disabled[@]}"
    done
  done
done

"$PY" experiments/summarize_table2_pilot_20260722.py \
  --root "$OUT" --soft-only --tasks "${TASKS[@]}"

echo "Table 2a loss-only experiment complete: $OUT/pilot_summary.json"
