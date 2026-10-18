#!/usr/bin/env bash
# Calibration-selected v17 refinement from v16.  This run is exploratory until
# evaluated on a fresh task-disjoint sealed split; it never trains on sealed rows.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
INDEX="$ROOT/data/taskgrouped_strict_pino_v13_union_20260716/index.json"
TAG=strict_physics_residual_pino_v17_safe90_20260718
CKPT="$ROOT/checkpoints/$TAG"

mkdir -p "$CKPT" "$ROOT/logs"
cd "$ROOT"

exec "$PY" structural/training/pretrain_pino_v3.py \
  --index_file "$INDEX" --target_res 48 --include_fixed_mask --arch ufno_dual \
  --epochs 50 --batch_size 1 --accum_steps 4 --lr 1e-7 --warmup_epochs 3 \
  --no_amp --disable_train_augment \
  --resume_from "$ROOT/checkpoints/strict_physics_residual_pino_v16_shadowart_scale_20260718/best_model.pt" \
  --hard_case_replay_ratio 1.10 --boundary_replay_ratio 1.50 --boundary_log_band 0.16 \
  --lambda_disp 0.5 --lambda_stress_log 2.0 --lambda_vm_asym 4.0 \
  --lambda_vm_peak 3.0 --lambda_bc 0.5 --lambda_pde 0.0001 \
  --lambda_constitutive 0.01 --lambda_scale 10.0 --lambda_decision 8.0 \
  --lambda_decision_margin 10.0 --decision_temperature 0.08 \
  --decision_margin_log 0.10 --underestimate_weight 12.0 --overestimate_weight 0.35 \
  --checkpoint_selection decision --seed 20260722 \
  --output_dir "$CKPT" --version_tag "$TAG"
