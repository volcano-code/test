#!/usr/bin/env bash
# Unified comparator protocol. Run only after the v7 builder and the manifest
# preparer have completed. Every row uses the same task-disjoint index and C48.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
INDEX="$ROOT/results/pino_comparator_v7_20260716/manifest.json"
OUT="$ROOT/checkpoints/pino_comparator_v7_20260716"
mkdir -p "$OUT" "$ROOT/logs"
cd "$ROOT"

"$PY" experiments/train_scalar_structural_baseline_v7.py --kind mlp --index "$INDEX" \
  --output "$OUT/geometry_mlp" --epochs 120 --batch-size 8 --target-res 48
"$PY" experiments/train_scalar_structural_baseline_v7.py --kind cnn --index "$INDEX" \
  --output "$OUT/cnn3d" --epochs 120 --batch-size 4 --target-res 48

# Neural-operator baselines.  All retain the same FEM field supervision; the
# two no-physics rows explicitly set both PDE residual coefficients to zero.
"$PY" structural/training/pretrain_pino_v3.py --index_file "$INDEX" --target_res 48 \
  --include_fixed_mask --arch ufno --epochs 160 --batch_size 1 --accum_steps 4 --lr 1e-5 \
  --warmup_epochs 20 --no_amp --disable_train_augment --lambda_pde 0 --lambda_constitutive 0 \
  --lambda_scale 0 --output_dir "$OUT/ufno" --version_tag comparator_ufno
"$PY" structural/training/pretrain_pino_v3.py --index_file "$INDEX" --target_res 48 \
  --include_fixed_mask --arch ufno_dual --epochs 160 --batch_size 1 --accum_steps 4 --lr 1e-5 \
  --warmup_epochs 20 --no_amp --disable_train_augment --lambda_pde 0 --lambda_constitutive 0 \
  --lambda_scale 3 --output_dir "$OUT/ufno_dual_no_physics" --version_tag comparator_no_physics
"$PY" structural/training/pretrain_pino_v3.py --index_file "$INDEX" --target_res 48 \
  --include_fixed_mask --arch ufno_dual --epochs 180 --batch_size 1 --accum_steps 4 --lr 1e-5 \
  --warmup_epochs 20 --no_amp --disable_train_augment --hard_case_replay_ratio 0.35 \
  --boundary_replay_ratio 0.70 --boundary_log_band 0.12 --lambda_disp 1 --lambda_stress_log 3 \
  --lambda_vm_asym 2 --lambda_vm_peak 1 --lambda_bc 1 --lambda_pde 0.03 --lambda_constitutive 0.03 \
  --lambda_scale 3 --lambda_decision 1 --lambda_decision_margin 1 --decision_temperature 0.12 \
  --decision_margin_log 0.06 --underestimate_weight 6 --overestimate_weight 0.3 \
  --output_dir "$OUT/full_pino" --version_tag comparator_full_pino
