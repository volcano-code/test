#!/usr/bin/env bash
# Ordered queue requested on 2026-07-18:
#   1) five task-disjoint structural models and common evaluation
#   2) missing Soft ablation seeds and threshold-stability summary
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

TAG=taskdisjoint_five_model_comparator_20260718 \
  bash experiments/run_taskdisjoint_five_model_comparator_20260717.sh

# seed 17 is already complete and will be skipped; 23 and 42 are added.
SEEDS_CSV=17,23,42 \
  bash experiments/run_soft_tsf1_mr1_heroes_ablation_20260717.sh
