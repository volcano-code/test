#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
VOXEL=/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel
OUT="$ROOT/results/table2_two_stage_pilot_true256_3tasks_seed17_20260722"
SEED=17

CASES=(
  "H01_future_mech|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H01_future_mech/inputs"
  "H04_mage_warrior|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H04_mage_warrior/inputs"
  "L03_frog_beetle|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L03_frog_beetle/inputs"
)

SOFT_VARIANTS=(
  "full|joint|"
  "w_o_joint_view|sequential|"
  "w_o_Ldice|joint|dice"
  "w_o_Lssim|joint|ssim"
  "w_o_Lout|joint|false_positive"
)

GEOMETRY_VARIANTS=(
  "full|"
  "w_o_ray_restoration|--disable-ray-restoration"
  "w_o_bridge_repair|--disable-bridge"
  "w_o_guarded_carving|--disable-guarded-carve"
  "w_o_core_shell_delivery|--disable-delivery-shell"
)

mkdir -p "$OUT/soft" "$OUT/geometry"
cd "$ROOT"

# Avoid overlapping two 256^3 P-CRAFT soft optimizers.  The independent
# Sadekar baseline may continue because its observed memory footprint is small.
while systemctl --user is-active --quiet pcraft-fig1-symbol-pairs-final-20260722.service; do
  echo "waiting for Figure-1 symbol delivery to finish"
  sleep 30
done

for item in "${SOFT_VARIANTS[@]}"; do
  IFS='|' read -r variant update_mode disabled_loss <<< "$item"
  for case in "${CASES[@]}"; do
    IFS='|' read -r task inputs <<< "$case"
    run="$OUT/soft/$task/$variant/seed_$SEED"
    [[ -f "$run/metrics.json" ]] && continue
    mkdir -p "$run"
    disabled=(volume)
    [[ -n "$disabled_loss" ]] && disabled+=("$disabled_loss")
    "$PY" "$VOXEL/train.py" cuda:0 "$run" 2000 0.007 1 \
      -sdlist "$inputs/left_wall_x_target.png" "$inputs/right_wall_y_target.png" \
      -mr 0 -tsf 0 -tsf4 0 -cam ortho -imsz 256 \
      --res 256 -eres 256 --control-res 0 \
      --proposal-mode proposal --view-update-mode "$update_mode" \
      --save-every 100 --no-cleanup --export-final-step --seed "$SEED" \
      --disable-losses "${disabled[@]}"
  done
done

for item in "${GEOMETRY_VARIANTS[@]}"; do
  IFS='|' read -r variant flags <<< "$item"
  for case in "${CASES[@]}"; do
    IFS='|' read -r task inputs <<< "$case"
    frozen="$OUT/soft/$task/full/seed_$SEED"
    run="$OUT/geometry/$task/$variant"
    [[ -f "$run/MANIFEST.json" ]] && continue
    # shellcheck disable=SC2086
    "$PY" shadowforge_pipeline/unified_run.py \
      --silhouettes "$inputs/left_wall_x_target.png" "$inputs/right_wall_y_target.png" \
      --branch soft-geometry --output "$run" \
      --tsf 0 --mr 0 \
      --soft-resolution 256 --soft-control-res 0 \
      --soft-proposal-mode proposal --soft-artifact-dir "$frozen" \
      --soft-seed-thresholds 0.15,0.25,0.35,0.45,0.55 \
      --core-resolution 48 --delivery-shell-voxels 2 \
      --min-iou 0.0 --min-view-iou 0.0 \
      --min-saving 0.0 --max-saving 1.0 --min-erosion-survival 0.0 \
      --selection-iou-tolerance 0.01 --screen skip --device cpu $flags
  done
done

"$PY" experiments/summarize_table2_pilot_20260722.py --root "$OUT"
