#!/usr/bin/env bash
# Conservative v14 refinement: train-owned dangerous replay only.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
INDEX="$ROOT/data/taskgrouped_strict_pino_v13_union_20260716/index.json"
TAG=strict_physics_residual_pino_v14_false_safe_20260717
CKPT="$ROOT/checkpoints/$TAG"
ACS="$ROOT/results/${TAG}_dualside_acs"
mkdir -p "$CKPT" "$ACS" "$ROOT/logs"
cd "$ROOT"

"$PY" structural/training/pretrain_pino_v3.py \
  --index_file "$INDEX" \
  --target_res 48 \
  --include_fixed_mask \
  --arch ufno_dual \
  --epochs 100 \
  --batch_size 1 \
  --accum_steps 4 \
  --lr 1e-6 \
  --warmup_epochs 4 \
  --no_amp \
  --disable_train_augment \
  --resume_from "$ROOT/checkpoints/strict_physics_residual_pino_v13_dangerous_20260717/best_model.pt" \
  --hard_case_replay_ratio 0.70 \
  --boundary_replay_ratio 0.80 \
  --boundary_log_band 0.18 \
  --lambda_disp 1.0 \
  --lambda_stress_log 3.0 \
  --lambda_vm_asym 3.0 \
  --lambda_vm_peak 1.5 \
  --lambda_bc 1.0 \
  --lambda_pde 0.001 \
  --lambda_constitutive 0.02 \
  --lambda_scale 5.0 \
  --lambda_decision 2.5 \
  --lambda_decision_margin 5.0 \
  --decision_temperature 0.10 \
  --decision_margin_log 0.10 \
  --underestimate_weight 12.0 \
  --overestimate_weight 0.25 \
  --checkpoint_selection decision \
  --seed 20260718 \
  --output_dir "$CKPT" \
  --version_tag "$TAG"

"$PY" experiments/evaluate_strict_pino_dualside_acs_20260717.py \
  --index "$INDEX" \
  --checkpoint "$CKPT/best_model.pt" \
  --safe-alphas 0.005,0.01,0.02,0.05,0.10 \
  --unsafe-alphas 0.02,0.05,0.10,0.15,0.20,0.30 \
  --target-res 48 \
  --batch-size 4 \
  --device cuda:0 \
  --prediction-cache "$ACS/predictions.npz" \
  --out-dir "$ACS"
