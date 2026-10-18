#!/usr/bin/env bash
# One-factor PINO component ablations on the immutable task-disjoint protocol.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
INDEX="$ROOT/data/taskgrouped_strict_pino_v13_union_20260716/index.json"
PARENT="$ROOT/checkpoints/strict_physics_residual_pino_v15_tail_20260717/best_model.pt"
FULL="$ROOT/checkpoints/strict_physics_residual_pino_v16_shadowart_scale_20260718/best_model.pt"
TAG=${TAG:-pino_component_ablations_20260720}
OUT="$ROOT/checkpoints/$TAG"
RESULT="$ROOT/results/$TAG"
SEED=${SEED:-20260720}
mkdir -p "$OUT" "$RESULT" "$ROOT/logs"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

COMMON=(
  --index_file "$INDEX" --target_res 48 --arch ufno_dual
  --epochs 70 --batch_size 1 --accum_steps 4 --lr 2e-7
  --warmup_epochs 3 --no_amp --disable_train_augment --resume_from "$PARENT"
  --hard_case_replay_ratio 0.90 --boundary_replay_ratio 1.20 --boundary_log_band 0.18
  --lambda_disp 0.5 --lambda_stress_log 2 --lambda_vm_asym 5 --lambda_vm_peak 3
  --lambda_bc 0.5 --lambda_pde 0.0001 --lambda_constitutive 0.01 --lambda_scale 8
  --lambda_decision 6 --lambda_decision_margin 10 --decision_temperature 0.08
  --decision_margin_log 0.10 --underestimate_weight 20 --overestimate_weight 0.20
  --checkpoint_selection decision --seed "$SEED"
)

# Remove equilibrium, constitutive, and support-boundary physics together.
"$PY" structural/training/pretrain_pino_v3.py "${COMMON[@]}" --include_fixed_mask \
  --lambda_bc 0 --lambda_pde 0 --lambda_constitutive 0 \
  --output_dir "$OUT/w_o_physics" --version_tag "${TAG}_w_o_physics"

# Keep absolute-scale supervision but attach it to the shared U-FNO trunk.
"$PY" structural/training/pretrain_pino_v3.py "${COMMON[@]}" --include_fixed_mask --arch ufno \
  --output_dir "$OUT/w_o_scale_fno" --version_tag "${TAG}_w_o_scale_fno"

# Symmetric under/over-estimation costs; every other loss remains unchanged.
"$PY" structural/training/pretrain_pino_v3.py "${COMMON[@]}" --include_fixed_mask \
  --underestimate_weight 1 --overestimate_weight 1 \
  --output_dir "$OUT/w_o_asymmetric_loss" --version_tag "${TAG}_w_o_asymmetric_loss"

# Retain occupancy and gravity only; remove both SDF and support-mask context.
"$PY" structural/training/pretrain_pino_v3.py "${COMMON[@]}" --disable_sdf_input \
  --output_dir "$OUT/w_o_structural_context" --version_tag "${TAG}_w_o_structural_context"

"$PY" experiments/evaluate_pino_component_ablations_20260720.py \
  --index "$INDEX" \
  --full "$FULL" \
  --no-physics "$OUT/w_o_physics/best_model.pt" \
  --no-scale-fno "$OUT/w_o_scale_fno/best_model.pt" \
  --no-asymmetric "$OUT/w_o_asymmetric_loss/best_model.pt" \
  --no-context "$OUT/w_o_structural_context/best_model.pt" \
  --out-dir "$RESULT" --device cuda:0
