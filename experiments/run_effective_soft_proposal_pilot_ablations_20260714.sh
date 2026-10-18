#!/usr/bin/env bash
# Soft-proposal ablation, deliberately before all deterministic geometry.
# Do not run while PINO occupies cuda:0. This pilot uses one fixed seed; rerun
# only the variants with a measurable effect at seeds 41/42/43 afterwards.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
VOXEL=/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel/train.py
OUT="$ROOT/results/soft_proposal_ablation_pilot_20260714"
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
  local disabled=(volume "$@")  # Volume is not a deployed soft objective.
  echo "===== ${task}/${variant}; disabled=${disabled[*]} ====="
  "$PY" "$VOXEL" cuda:0 "$OUT/$task/$variant/seed_$SEED" 2000 0.007 1 \
    -sdlist "$left" "$right" -mr 0 -tsf 0 -tsf4 0 -cam ortho -imsz 256 \
    --res 256 -eres 256 --save-every 100 --proposal-mode regularized \
    --fidelity-gate 0.85 --connectivity-start-iou 0.60 \
    --structure-warmup-steps 300 --structure-ramp-steps 300 \
    --soft-proposal-min-iou 0.95 --no-cleanup --export-final-step --seed "$SEED" \
    --disable-losses "${disabled[@]}"
}

for i in "${!TASKS[@]}"; do
  task="${TASKS[$i]}"; left="${LEFTS[$i]}"; right="${RIGHTS[$i]}"
  # Five interpretable groups. Projection losses are fixed in every row.
  run_case "$task" "$left" "$right" full
  run_case "$task" "$left" "$right" w_o_Lconn connectivity spurious spectral
  run_case "$task" "$left" "$right" w_o_Lwall wall
  run_case "$task" "$left" "$right" w_o_Lsurface edge smooth laplacian
  run_case "$task" "$left" "$right" w_o_Lguard false_positive deviation
done

