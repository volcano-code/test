#!/usr/bin/env bash
# ShadowArt-domain scale/boundary refinement from the current strict v15-tail model.
# The index is task-disjoint and contains delivered ShadowArt candidate FEM labels.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
INDEX="$ROOT/data/taskgrouped_strict_pino_v13_union_20260716/index.json"
TAG=strict_physics_residual_pino_v16_shadowart_scale_20260718
CKPT="$ROOT/checkpoints/$TAG"
ACS="$ROOT/results/${TAG}_dualside_acs"
mkdir -p "$CKPT" "$ACS" "$ROOT/logs"
cd "$ROOT"

"$PY" structural/training/pretrain_pino_v3.py \
  --index_file "$INDEX" --target_res 48 --include_fixed_mask --arch ufno_dual \
  --epochs 70 --batch_size 1 --accum_steps 4 --lr 2e-7 --warmup_epochs 3 \
  --no_amp --disable_train_augment \
  --resume_from "$ROOT/checkpoints/strict_physics_residual_pino_v15_tail_20260717/best_model.pt" \
  --hard_case_replay_ratio 0.90 --boundary_replay_ratio 1.20 --boundary_log_band 0.18 \
  --lambda_disp 0.5 --lambda_stress_log 2.0 --lambda_vm_asym 5.0 \
  --lambda_vm_peak 3.0 --lambda_bc 0.5 --lambda_pde 0.0001 \
  --lambda_constitutive 0.01 --lambda_scale 8.0 --lambda_decision 6.0 \
  --lambda_decision_margin 10.0 --decision_temperature 0.08 \
  --decision_margin_log 0.10 --underestimate_weight 20.0 --overestimate_weight 0.20 \
  --checkpoint_selection decision --seed 20260721 \
  --output_dir "$CKPT" --version_tag "$TAG"

"$PY" experiments/evaluate_strict_pino_dualside_acs_20260717.py \
  --index "$INDEX" --checkpoint "$CKPT/best_model.pt" \
  --safe-alphas 0.005,0.01,0.02,0.05,0.10 \
  --unsafe-alphas 0.02,0.05,0.10,0.15,0.20,0.30 \
  --target-res 48 --batch-size 4 --device cuda:0 \
  --prediction-cache "$ACS/predictions.npz" --out-dir "$ACS"
