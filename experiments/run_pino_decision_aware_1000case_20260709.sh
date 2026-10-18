#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
LOG_DIR="$ROOT/results/background_logs"
mkdir -p "$LOG_DIR"

DATA_DIR="$ROOT/results/candidate_selection_1000case_iou_dice_20260709"
DATA_JSON="$DATA_DIR/candidate_selection_100case_iou_dice.json"
TARGET_JSON="$ROOT/results/current_candidate_targeted_safe_manifest_1000case_20260709.json"
TRAIN_JSON="$ROOT/results/current_candidate_decision_aware_pino_1000case_120epoch_20260709.json"
CKPT_DIR="$ROOT/checkpoints/current_candidate_decision_aware_pino_1000case_20260709"

cd "$ROOT"

echo "[1/3] build 1000-case current candidate-domain dataset"
python experiments/candidate_selection_batch_with_dice_100case_20260708.py \
  --n_total 1000 \
  --resolution 48 \
  --seed 20260709 \
  --alpha 0.10 \
  --print_mm 100 \
  --nozzle_mm 0.4 \
  --wall_safety 2.0 \
  --output_dir "$DATA_DIR"

echo "[2/3] summarize near-boundary and ACS-margin safe rows"
python experiments/build_current_candidate_targeted_safe_manifest_20260708.py \
  --src "$DATA_JSON" \
  --out "$TARGET_JSON" \
  --near-log-width 0.35 \
  --acs-lower-margin 1.05

echo "[3/3] train decision-aware PINO on the 1000-case candidate domain"
python experiments/train_current_candidate_decision_aware_pino_20260708.py \
  --src "$DATA_JSON" \
  --epochs 120 \
  --batch-size 16 \
  --lr 3e-5 \
  --false-safe-weight 80 \
  --safe-miss-weight 3.0 \
  --threshold-margin 0.03 \
  --train-frac 0.60 \
  --cal-frac 0.20 \
  --device cuda \
  --out "$TRAIN_JSON" \
  --output-dir "$CKPT_DIR"

echo "[done] $TRAIN_JSON"
