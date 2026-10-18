#!/usr/bin/env bash
# Geometry-delivery module ablation. Run only after the PINO job releases cuda:0.
# It holds the soft objective fixed and makes each hard module independently removable.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
OUT="$ROOT/results/geometry_delivery_ablation_pilot_20260714"
SEED=42

TASKS=(armadillo_bunny dark_knight future_mech parrot_snail)
LEFTS=(
  "$ROOT/../shadow/normal_shadow/armadillo_0.png"
  "$ROOT/results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709/dark_knight_left__magic_scholar_right/inputs/left_wall_x_view1_target.png"
  "$ROOT/../shadow/anime_shadow_processed_black_bg_light_simplified/A005_005_未来战士的剪影.png"
  "$ROOT/../shadow/normal_shadow/birds_parrot.png"
)
RIGHTS=(
  "$ROOT/../shadow/normal_shadow/bunny_0.png"
  "$ROOT/results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709/dark_knight_left__magic_scholar_right/inputs/right_wall_y_view2_target.png"
  "$ROOT/../shadow/anime_shadow_processed_black_bg_light_simplified/A014_014_未来战士的机械轮廓.png"
  "$ROOT/../shadow/normal_shadow/sea_snail.png"
)

run_case() {
  local task="$1" left="$2" right="$3" variant="$4"; shift 4
  echo "===== ${task}/${variant} ====="
  "$PY" shadowforge_pipeline/unified_run.py \
    --silhouettes "$left" "$right" --branch soft-geometry --output "$OUT/$task/$variant" \
    --tsf 0 --mr 0 --soft-resolution 256 --soft-control-res 0 \
    --soft-proposal-mode regularized --soft-iterations 2000 --soft-lr 0.007 \
    --soft-save-every 100 --soft-seed "$SEED" --soft-export-final-step \
    --soft-connectivity-start-iou 0.60 --soft-fidelity-gate 0.85 \
    --soft-structure-warmup-steps 300 --soft-structure-ramp-steps 300 \
    --soft-disable-losses volume --soft-seed-only \
    --soft-seed-thresholds 0.15,0.25,0.35,0.45,0.55 \
    --core-resolution 48 --delivery-shell-voxels 2 \
    --min-iou 0.0 --min-view-iou 0.0 --min-saving 0.0 --max-saving 1.0 \
    --min-erosion-survival 0.0 --screen skip --device cuda:0 "$@"
}

for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"; left="${LEFTS[$i]}"; right="${RIGHTS[$i]}"
  run_case "$task" "$left" "$right" full
  run_case "$task" "$left" "$right" w_o_ray_restoration --disable-ray-restoration
  run_case "$task" "$left" "$right" w_o_bridge_repair --disable-bridge
  run_case "$task" "$left" "$right" w_o_guarded_carving --disable-guarded-carve
  run_case "$task" "$left" "$right" w_o_delivery_shell --disable-delivery-shell
done
