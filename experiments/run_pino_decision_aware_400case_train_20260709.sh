#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
cd "$ROOT"

python experiments/train_current_candidate_decision_aware_pino_20260708.py \
  --src results/candidate_selection_400case_iou_dice_20260708/candidate_selection_100case_iou_dice.json \
  --epochs 120 \
  --batch-size 16 \
  --lr 3e-5 \
  --false-safe-weight 80 \
  --safe-miss-weight 3.0 \
  --threshold-margin 0.03 \
  --train-frac 0.60 \
  --cal-frac 0.20 \
  --device cuda \
  --out results/current_candidate_decision_aware_pino_400case_120epoch_20260709.json \
  --output-dir checkpoints/current_candidate_decision_aware_pino_400case_20260709
