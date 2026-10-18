#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
MANIFEST="$ROOT/results/mitra_stratified_low6_high6_fixed3_20260716/benchmark_manifest.json"
TAG=mitra_stratified_low6_high6_author_tsf_20260716

cd "$ROOT"
while systemctl --user is-active --quiet shadowforge-pino-v10-v7.service; do
  echo "$(date -Is) waiting for PINO v10 GPU release"
  sleep 120
done

"$PY" experiments/run_fixed3_author_tsf_baselines_20260716.py \
  --manifest "$MANIFEST" \
  --out-tag "$TAG"
