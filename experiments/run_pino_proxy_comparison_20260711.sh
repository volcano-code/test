#!/usr/bin/env bash
set -euo pipefail

# One GPU worker intentionally runs the five models sequentially.  Every run
# consumes the immutable split manifest created by pino_proxy_comparison_20260711.py.
ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
OUT=$ROOT/results/pino_proxy_model_comparison_20260711
INDEX=$OUT/split_manifest.json

cd "$ROOT"
"$PY" experiments/pino_proxy_comparison_20260711.py prepare
"$PY" experiments/pino_proxy_comparison_20260711.py train-scalar --kind mlp --epochs 120 --batch 32 --lr 3e-4
"$PY" experiments/pino_proxy_comparison_20260711.py train-scalar --kind cnn --epochs 120 --batch 4 --lr 2e-4

# FNO baseline: U-shaped Fourier operator with ordinary FEM supervision only.
"$PY" structural/training/pretrain_pino_v3.py --index_file "$INDEX" \
  --output_dir "$OUT/fno" --arch ufno --epochs 100 --batch_size 4 --accum_steps 1 \
  --modes 8 --width 24 --n_layers 4 --lr 3e-4 --warmup_epochs 8 \
  --lambda_pde 0 --lambda_constitutive 0 --lambda_vm_asym 1 --lambda_vm_peak 1 \
  --seed 20260711 --version_tag fno_supervised

# Matched PINO architecture without physics terms.
"$PY" structural/training/pretrain_pino_v3.py --index_file "$INDEX" \
  --output_dir "$OUT/pino_no_physics" --arch ufno_dual --epochs 100 --batch_size 4 --accum_steps 1 \
  --modes 8 --width 24 --n_layers 4 --lr 3e-4 --warmup_epochs 8 \
  --lambda_pde 0 --lambda_constitutive 0 --lambda_vm_asym 1 --lambda_vm_peak 1 \
  --seed 20260711 --version_tag pino_no_physics

# Same PINO architecture and FEM labels, adding equilibrium and constitutive residuals.
"$PY" structural/training/pretrain_pino_v3.py --index_file "$INDEX" \
  --output_dir "$OUT/full_pino" --arch ufno_dual --epochs 100 --batch_size 4 --accum_steps 1 \
  --modes 8 --width 24 --n_layers 4 --lr 3e-4 --warmup_epochs 8 \
  --lambda_pde 0.05 --lambda_constitutive 0.05 --lambda_vm_asym 1 --lambda_vm_peak 1 \
  --seed 20260711 --version_tag full_pino
