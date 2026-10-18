#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/lin/xvxinyao/ShadowForge}"
PY="${PY:-/home/lin/anaconda3/envs/shadowart-revisited/bin/python}"
INDEX="${INDEX:-$ROOT/data/operator_baseline_smoke_v13_20260723/index.json}"
EPOCHS="${EPOCHS:-3}"

cd "$ROOT"
mkdir -p logs checkpoints

run_model() {
  local arch="$1"
  local width="$2"
  local layers="$3"
  local tag="official_${arch}_pilot_20260723"

  "$PY" structural/training/pretrain_pino_v3.py \
    --index_file "$INDEX" \
    --target_res 48 \
    --include_fixed_mask \
    --arch "$arch" \
    --epochs "$EPOCHS" \
    --batch_size 1 \
    --accum_steps 4 \
    --width "$width" \
    --n_layers "$layers" \
    --modes 12 \
    --lr 1e-4 \
    --warmup_epochs 1 \
    --no_amp \
    --disable_train_augment \
    --lambda_disp 1.0 \
    --lambda_stress_log 3.0 \
    --lambda_vm_asym 2.0 \
    --lambda_vm_peak 1.0 \
    --lambda_bc 1.0 \
    --lambda_pde 0.03 \
    --lambda_constitutive 0.03 \
    --lambda_scale 3.0 \
    --underestimate_weight 6.0 \
    --overestimate_weight 0.3 \
    --output_dir "checkpoints/$tag" \
    --version_tag "$tag"
}

# Sequential execution avoids cross-model GPU-memory interference.
run_model gino_official 24 2
run_model transolver_official 32 2
