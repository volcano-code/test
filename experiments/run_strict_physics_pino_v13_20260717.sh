#!/usr/bin/env bash
# Dangerous-candidate augmented, task-disjoint PINO training and frozen ACS evaluation.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
INDEX="$ROOT/data/taskgrouped_strict_pino_v13_union_20260716/index.json"
TAG=strict_physics_residual_pino_v13_dangerous_20260717
CKPT="$ROOT/checkpoints/$TAG"
ACS="$ROOT/results/${TAG}_acs"
mkdir -p "$CKPT" "$ACS" "$ROOT/logs"
cd "$ROOT"

"$PY" structural/training/pretrain_pino_v3.py \
  --index_file "$INDEX" \
  --target_res 48 \
  --include_fixed_mask \
  --arch ufno_dual \
  --epochs 180 \
  --batch_size 1 \
  --accum_steps 4 \
  --lr 3e-6 \
  --warmup_epochs 12 \
  --no_amp \
  --disable_train_augment \
  --resume_from "$ROOT/checkpoints/strict_physics_residual_decision_pino_v9_stable_20260715/best_model.pt" \
  --hard_case_replay_ratio 0.30 \
  --boundary_replay_ratio 0.50 \
  --boundary_log_band 0.15 \
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
  --checkpoint_selection decision \
  --seed 20260717 \
  --output_dir "$CKPT" \
  --version_tag "$TAG" \
  2>&1 | tee "$ROOT/logs/${TAG}.log"

"$PY" experiments/evaluate_strict_pino_acs_v5_20260713.py \
  --index "$INDEX" \
  --checkpoint "$CKPT/best_model.pt" \
  --target-res 48 \
  --batch-size 4 \
  --alphas 0.005,0.01,0.02,0.05,0.10,0.15,0.20,0.30 \
  --freeze-policy-on-calibration \
  --deployment-constraint zero_false_safe \
  --device cuda:0 \
  --out-dir "$ACS"
