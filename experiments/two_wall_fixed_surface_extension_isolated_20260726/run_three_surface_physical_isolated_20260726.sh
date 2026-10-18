#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
SNAPSHOT="$ROOT/experiments/two_wall_fixed_surface_extension_isolated_20260726"
SOFT="$ROOT/results/fig4_anime_curated_original_v1_20260724/pcraft/adventurer_schoolgirl_fighter_original/seed_17/soft_training"
OUT="$ROOT/results/two_wall_fixed_surface_extension_20260726/adventurer_schoolgirl_fighter_three_surface_physical"
STATUS="$OUT/TASK_STATUS.txt"

mkdir -p "$OUT"
printf 'QUEUED %s\n' "$(date --iso-8601=seconds)" > "$STATUS"

record_exit() {
  rc=$?
  if (( rc != 0 )); then
    printf 'FAILED exit_code=%s %s\n' "$rc" "$(date --iso-8601=seconds)" >> "$STATUS"
  fi
}
trap record_exit EXIT

# V256 repair can briefly require several GiB. Wait instead of forcing the
# current recovery/training workload into additional swap pressure.
while :; do
  available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if (( available_kib >= 10 * 1024 * 1024 )); then
    break
  fi
  printf 'WAITING_FOR_RAM available_kib=%s %s\n' \
    "$available_kib" "$(date --iso-8601=seconds)" >> "$STATUS"
  sleep 60
done

printf 'RUNNING %s\n' "$(date --iso-8601=seconds)" >> "$STATUS"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MALLOC_ARENA_MAX=2

nice -n 15 ionice -c 3 "$PY" \
  "$SNAPSHOT/shadowforge_pipeline/unified_run.py" \
  --silhouettes \
    "$ROOT/results/fig4_anime_curated_original_v1_20260724/inputs/adventurer_schoolgirl_fighter_view_x.png" \
    "$ROOT/results/fig4_anime_curated_original_v1_20260724/inputs/adventurer_schoolgirl_fighter_view_y.png" \
    "$ROOT/results/fig4_anime_curated_original_v1_20260724/inputs/adventurer_schoolgirl_fighter_view_z.png" \
  --output "$OUT" \
  --branch soft-geometry \
  --tsf 1 \
  --mr 1 \
  --soft-artifact-dir "$SOFT" \
  --soft-resolution 256 \
  --core-resolution 48 \
  --v256-geometry-owner \
  --soft-seed-thresholds 0.15 \
  --bridge-radius 2 \
  --bridge-passes 24 \
  --delivery-bridge-passes 12 \
  --delivery-shell-voxels 2 \
  --delivery-shortlist-size 1 \
  --carve-checkpoint-voxels 128 \
  --max-carve-checkpoints 64 \
  --min-iou 0.95 \
  --min-view-iou 0.90 \
  --min-saving 0.000001 \
  --max-saving 0.85 \
  --allow-best-effort-fidelity-fallback \
  --screen skip \
  --selection-policy material \
  --print-mm 100 \
  --save-candidate-evidence \
  --export-best-rejected \
  --device cpu

printf 'COMPLETED %s\n' "$(date --iso-8601=seconds)" >> "$STATUS"
trap - EXIT
