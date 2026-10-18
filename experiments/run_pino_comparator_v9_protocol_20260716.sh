#!/usr/bin/env bash
# Task-disjoint surrogate comparison aligned with the best observed v9 model.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
SOURCE="$ROOT/data/taskgrouped_strict_pino_delivered_v6_balanced_20260713/index.json"
INDEX="$ROOT/results/pino_comparator_v9_protocol_20260716/manifest.json"
OUT="$ROOT/checkpoints/pino_comparator_v9_protocol_20260716"
mkdir -p "$OUT" "$(dirname "$INDEX")" "$ROOT/logs"
cd "$ROOT"

"$PY" experiments/prepare_pino_comparator_v7.py \
  --index "$SOURCE" --output "$INDEX" --target-res 48

"$PY" experiments/train_scalar_structural_baseline_v7.py --kind mlp --index "$INDEX" \
  --output "$OUT/geometry_mlp" --epochs 120 --batch-size 8 --target-res 48
"$PY" experiments/train_scalar_structural_baseline_v7.py --kind cnn --index "$INDEX" \
  --output "$OUT/cnn3d" --epochs 120 --batch-size 4 --target-res 48

"$PY" structural/training/pretrain_pino_v3.py --index_file "$INDEX" --target_res 48 \
  --include_fixed_mask --arch ufno --epochs 160 --batch_size 1 --accum_steps 4 --lr 1e-5 \
  --warmup_epochs 20 --no_amp --disable_train_augment --lambda_pde 0 --lambda_constitutive 0 \
  --lambda_scale 0 --output_dir "$OUT/ufno" --version_tag comparator_v9_ufno

"$PY" structural/training/pretrain_pino_v3.py --index_file "$INDEX" --target_res 48 \
  --include_fixed_mask --arch ufno_dual --epochs 160 --batch_size 1 --accum_steps 4 --lr 1e-5 \
  --warmup_epochs 20 --no_amp --disable_train_augment --lambda_pde 0 --lambda_constitutive 0 \
  --lambda_scale 4 --output_dir "$OUT/ufno_dual_no_physics" --version_tag comparator_v9_no_physics

# Full PINO uses the v9 objective, which is the current best observed model.
"$PY" structural/training/pretrain_pino_v3.py --index_file "$INDEX" --target_res 48 \
  --include_fixed_mask --arch ufno_dual --epochs 160 --batch_size 1 --accum_steps 4 --lr 1e-5 \
  --warmup_epochs 20 --no_amp --disable_train_augment \
  --lambda_disp 1 --lambda_stress_log 3 --lambda_vm_asym 2 --lambda_vm_peak 1 \
  --lambda_bc 1 --lambda_pde 0.0001 --lambda_constitutive 0.02 --lambda_scale 4 \
  --lambda_decision 0.5 --lambda_decision_margin 2 --decision_temperature 0.12 \
  --decision_margin_log 0.06 --underestimate_weight 6 --overestimate_weight 0.5 \
  --output_dir "$OUT/full_pino" --version_tag comparator_v9_full_pino
