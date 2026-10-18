#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
VOXEL=/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel
INPUTS="$ROOT/results/table2_anime_fixed3_inputs_20260722"
OUT="$ROOT/results/table2a_anime_fixed3_mr1_true256_seed17_20260722"
SEED=17

TASKS=(arcane_casters blade_heroes stage_dancers)
VARIANTS=(
  "full|joint|"
  "w_o_joint_view|sequential|"
  "w_o_Ldice|joint|dice"
  "w_o_Lssim|joint|ssim"
  "w_o_Lout|joint|false_positive"
)

mkdir -p "$OUT"
cd "$ROOT"
"$PY" experiments/prepare_table2_anime_fixed3_20260722.py

for item in "${VARIANTS[@]}"; do
  IFS='|' read -r variant update_mode disabled_loss <<< "$item"
  for task in "${TASKS[@]}"; do
    run="$OUT/$task/$variant/seed_$SEED"
    [[ -f "$run/metrics.json" ]] && continue
    mkdir -p "$run"
    disabled=(volume)
    [[ -n "$disabled_loss" ]] && disabled+=("$disabled_loss")
    "$PY" "$VOXEL/train.py" cuda:0 "$run" 2000 0.007 1 \
      -sdlist "$INPUTS/$task/view_0.png" "$INPUTS/$task/view_1.png" "$INPUTS/$task/view_2.png" \
      -mr 1 -tsf 1 -tsf4 0 -cam ortho -imsz 256 \
      --res 256 -eres 256 --control-res 0 --sadekar-zdist 1.7 \
      --proposal-mode proposal --view-update-mode "$update_mode" \
      --save-every 100 --no-cleanup --export-final-step --seed "$SEED" \
      --disable-losses "${disabled[@]}"
  done
done

"$PY" experiments/summarize_table2_pilot_20260722.py \
  --root "$OUT" --soft-only --tasks "${TASKS[@]}"
