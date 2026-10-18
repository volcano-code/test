#!/usr/bin/env bash
# Five-model comparison on one immutable task-disjoint C48 protocol.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
INDEX="$ROOT/data/taskgrouped_strict_pino_v13_union_20260716/index.json"
TAG=${TAG:-taskdisjoint_five_model_comparator_20260718}
OUT="$ROOT/checkpoints/$TAG"
mkdir -p "$OUT" "$ROOT/logs"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PY" experiments/train_scalar_structural_baseline_v7.py --kind mlp --index "$INDEX" \
  --output "$OUT/geometry_mlp" --epochs 120 --batch-size 8 --target-res 48 --seed 20260720
"$PY" experiments/train_scalar_structural_baseline_v7.py --kind cnn --index "$INDEX" \
  --output "$OUT/cnn3d" --epochs 120 --batch-size 4 --target-res 48 --seed 20260720

# Single-trunk U-FNO with supervised scale head; no physics residuals.
"$PY" structural/training/pretrain_pino_v3.py --index_file "$INDEX" --target_res 48 \
  --include_fixed_mask --arch ufno --epochs 160 --batch_size 1 --accum_steps 4 \
  --lr 1e-5 --warmup_epochs 20 --no_amp --disable_train_augment \
  --lambda_pde 0 --lambda_constitutive 0 --lambda_scale 4 \
  --checkpoint_selection decision --seed 20260720 \
  --output_dir "$OUT/ufno" --version_tag comparator_taskdisjoint_ufno

# Same dual-branch architecture as PINO, but without PDE/constitutive residuals.
"$PY" structural/training/pretrain_pino_v3.py --index_file "$INDEX" --target_res 48 \
  --include_fixed_mask --arch ufno_dual --epochs 160 --batch_size 1 --accum_steps 4 \
  --lr 1e-5 --warmup_epochs 20 --no_amp --disable_train_augment \
  --lambda_disp 1 --lambda_stress_log 3 --lambda_vm_asym 2 --lambda_vm_peak 1 \
  --lambda_bc 1 --lambda_pde 0 --lambda_constitutive 0 --lambda_scale 4 \
  --lambda_decision 1 --lambda_decision_margin 2 --decision_temperature 0.12 \
  --decision_margin_log 0.06 --underestimate_weight 6 --overestimate_weight 0.5 \
  --checkpoint_selection decision --seed 20260720 \
  --output_dir "$OUT/ufno_dual_no_physics" --version_tag comparator_taskdisjoint_no_physics

# Full PINO: identical dual architecture and supervised losses plus physics residuals.
"$PY" structural/training/pretrain_pino_v3.py --index_file "$INDEX" --target_res 48 \
  --include_fixed_mask --arch ufno_dual --epochs 160 --batch_size 1 --accum_steps 4 \
  --lr 1e-5 --warmup_epochs 20 --no_amp --disable_train_augment \
  --hard_case_replay_ratio 0.30 --boundary_replay_ratio 0.50 --boundary_log_band 0.15 \
  --lambda_disp 1 --lambda_stress_log 3 --lambda_vm_asym 2 --lambda_vm_peak 1 \
  --lambda_bc 1 --lambda_pde 0.001 --lambda_constitutive 0.02 --lambda_scale 4 \
  --lambda_decision 1 --lambda_decision_margin 2 --decision_temperature 0.12 \
  --decision_margin_log 0.06 --underestimate_weight 6 --overestimate_weight 0.5 \
  --checkpoint_selection decision --seed 20260720 \
  --output_dir "$OUT/full_pino" --version_tag comparator_taskdisjoint_full_pino

"$PY" experiments/evaluate_taskdisjoint_five_model_comparator_20260717.py \
  --checkpoint-root "$OUT" --out-dir "$ROOT/results/$TAG"
