#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lin/xvxinyao/ShadowForge"
cd "$ROOT"

mkdir -p logs checkpoints results

BASE="checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"
START="checkpoints/pino_safe_side_external_hardneg_cal240_20260703/safe_side_external_hardneg_scale.pt"

run_variant() {
  local tag="$1"
  local ext_cal="$2"
  local epochs="$3"
  local lr="$4"
  local hard_repeat="$5"
  local false_w="$6"
  local ext_false_w="$7"
  local safe_margin_w="$8"

  local result="results/${tag}.json"
  local ckpt_dir="checkpoints/${tag}"
  local audit="results/${tag}_threshold_audit.json"

  echo "=== ${tag} train $(date) ==="
  PYTHONUNBUFFERED=1 python experiments/train_pino_safe_side_external_hardneg.py \
    --base-ckpt "$BASE" \
    --start-ckpt "$START" \
    --external-calibration "$ext_cal" \
    --epochs "$epochs" \
    --lr "$lr" \
    --external-hard-repeat "$hard_repeat" \
    --false-safe-weight "$false_w" \
    --external-false-safe-weight "$ext_false_w" \
    --safe-margin-weight "$safe_margin_w" \
    --result "$result" \
    --output-dir "$ckpt_dir"

  echo "=== ${tag} threshold audit $(date) ==="
  PYTHONUNBUFFERED=1 python experiments/audit_pino_safe_side_thresholds.py \
    --checkpoint "${ckpt_dir}/safe_side_external_hardneg_scale.pt" \
    --base-ckpt "$BASE" \
    --external-calibration "$ext_cal" \
    --out "$audit"
}

run_variant pino_safe_accept90_cal240_m08_20260704 240 8 0.00003 4 12.0 14.0 0.8
run_variant pino_safe_accept90_cal300_m08_20260704 300 8 0.00003 4 12.0 14.0 0.8
run_variant pino_safe_accept90_cal240_m12_20260704 240 8 0.00002 5 14.0 16.0 1.2

echo "DONE $(date)"
