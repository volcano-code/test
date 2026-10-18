#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lin/xvxinyao/ShadowForge}"
PY="${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}"
INDEX="${INDEX:-$ROOT/data/taskgrouped_strict_pino_v13_union_20260716/index.json}"
EPOCHS="${EPOCHS:-60}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-8}"

cd "$ROOT"
mkdir -p logs checkpoints

run_model() {
  local arch="$1"
  local width="$2"
  local tag="official_${arch}_formal_${EPOCHS}e_w${WARMUP_EPOCHS}_20260723"

  "$PY" structural/training/pretrain_pino_v3.py \
    --index_file "$INDEX" \
    --target_res 48 \
    --include_fixed_mask \
    --arch "$arch" \
    --epochs "$EPOCHS" \
    --batch_size 1 \
    --accum_steps 4 \
    --width "$width" \
    --n_layers 6 \
    --modes 16 \
    --lr 1e-5 \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --no_amp \
    --disable_train_augment \
    --lambda_disp 1.0 \
    --lambda_stress_log 3.0 \
    --lambda_vm_asym 2.0 \
    --lambda_vm_peak 1.0 \
    --lambda_bc 1.0 \
    --lambda_pde 0.001 \
    --lambda_constitutive 0.02 \
    --lambda_scale 4.0 \
    --lambda_decision 1.0 \
    --lambda_decision_margin 2.0 \
    --decision_temperature 0.12 \
    --decision_margin_log 0.06 \
    --underestimate_weight 6.0 \
    --overestimate_weight 0.5 \
    --output_dir "checkpoints/$tag" \
    --version_tag "$tag"
}

# Widths give approximately matched parameter counts:
# GINO 8.65M, Transolver 7.16M, existing Full PINO 7.88M.
# Run sequentially to keep peak GPU memory deterministic.
run_model gino_official 96
run_model transolver_official 144
