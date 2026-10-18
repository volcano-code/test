#!/usr/bin/env bash
set -euo pipefail
cd /home/lin/xvxinyao/ShadowForge
python experiments/train_current_candidate_decision_aware_pino_20260708.py \
  --src results/candidate_selection_400case_iou_dice_20260708/candidate_selection_100case_iou_dice.json \
  --epochs 80 \
  --batch-size 32 \
  --lr 3e-5 \
  --false-safe-weight 50 \
  --safe-miss-weight 6.0 \
  --threshold-margins 0,0.002,0.005,0.01 \
  --threshold-mode stratified \
  --rank-weight 1.0 \
  --train-frac 0.60 \
  --cal-frac 0.20 \
  --device cuda \
  --out results/current_candidate_decision_aware_pino_stratified_rank_80epoch_20260709.json \
  --output-dir checkpoints/current_candidate_decision_aware_pino_stratified_rank_20260709
