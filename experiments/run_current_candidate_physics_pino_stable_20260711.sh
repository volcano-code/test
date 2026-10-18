#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
INDEX="$ROOT/data/current_candidate_fullfield_fem_20260711/index.json"
OUT="$ROOT/checkpoints/current_candidate_physics_pino_stable_v2_120epoch_20260711"
LOG="$ROOT/results/current_candidate_physics_pino_stable_v2_120epoch_20260711.log"

mkdir -p "$OUT"
exec "$PY" "$ROOT/structural/training/pretrain_pino_v3.py" \
  --index_file "$INDEX" --output_dir "$OUT" \
  --epochs 120 --batch_size 2 --accum_steps 4 --lr 2e-6 --warmup_epochs 20 --disable_train_augment \
  --resume_from "$ROOT/checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt" \
  --arch ufno_dual --modes 20 --width 96 --n_layers 8 --target_res 48 \
  --lambda_disp 1.0 --lambda_stress_log 3.0 --lambda_vm_asym 1.0 --lambda_vm_peak 0.2 \
  --lambda_bc 1.0 --lambda_pde 0.02 --lambda_constitutive 0.02 --lambda_scale 2.0 \
  --underestimate_weight 4.0 --overestimate_weight 0.5 \
  --version_tag current_candidate_physics_pino_stable_v2_120epoch_20260711 \
  > "$LOG" 2>&1
