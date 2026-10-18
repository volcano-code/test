#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
INPUTS="$ROOT/results/fig4_anime_curated_original_v1_20260724/inputs"
X="$INPUTS/adventurer_schoolgirl_fighter_view_x.png"
Y="$INPUTS/adventurer_schoolgirl_fighter_view_y.png"
Z="$INPUTS/adventurer_schoolgirl_fighter_view_z.png"
STAMP="$(date +%Y%m%d_%H%M%S)"

OUT2="$ROOT/results/adventurer_schoolgirl_two_view_darkknight_protocol_$STAMP"
LOG2="$ROOT/logs/adventurer_schoolgirl_two_view_darkknight_protocol_$STAMP.log"
PID2_FILE="$ROOT/logs/adventurer_schoolgirl_two_view_darkknight_protocol_$STAMP.pid"
OUT3="$ROOT/results/adventurer_schoolgirl_fighter_three_view_twoview_flags_$STAMP"
LOG3="$ROOT/logs/adventurer_schoolgirl_fighter_three_view_twoview_flags_$STAMP.log"
QUEUE_LOG="$ROOT/logs/adventurer_schoolgirl_three_view_queue_$STAMP.log"
QUEUE_PID_FILE="$ROOT/logs/adventurer_schoolgirl_three_view_queue_$STAMP.pid"

COMMON_ARGS=(
  --branch soft-geometry
  --soft-resolution 256
  --core-resolution 48
  --soft-iterations 2000
  --soft-lr 0.007
  --soft-save-every 100
  --soft-seed 42
  --soft-seed-thresholds 0.15,0.25,0.35,0.45,0.55
  --bridge-radius 1
  --bridge-passes 16
  --delivery-bridge-passes 8
  --delivery-shell-voxels 2
  --delivery-shortlist-size 5
  --carve-checkpoint-voxels 0
  --max-carve-checkpoints 256
  --min-iou 0.975
  --min-view-iou 0.95
  --min-saving 0.0
  --max-saving 0.85
  --min-erosion-survival 0.08
  --screen pino_rollback_fem
  --structural-resolution 48
  --sf-threshold 1.3
  --pino-stop-margin 0.1
  --pino-max-rollbacks 0
  --selection-policy material
  --print-mm 100
  --tsf 0
  --mr 0
  --device cuda:0
  --save-candidate-evidence
)

mkdir -p "$ROOT/logs"

setsid "$PY" "$ROOT/shadowforge_pipeline/unified_run.py" \
  --silhouettes "$X" "$Y" \
  --output "$OUT2" \
  "${COMMON_ARGS[@]}" \
  >"$LOG2" 2>&1 </dev/null &
PID2=$!
echo "$PID2" >"$PID2_FILE"

setsid bash -c '
  set -euo pipefail
  pid="$1"
  py="$2"
  root="$3"
  x="$4"
  y="$5"
  z="$6"
  out="$7"
  log="$8"
  shift 8
  echo "waiting_for_two_view_pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
  echo "two_view_finished; starting_three_view=$(date --iso-8601=seconds)"
  exec "$py" "$root/shadowforge_pipeline/unified_run.py" \
    --silhouettes "$x" "$y" "$z" \
    --output "$out" \
    "$@" >"$log" 2>&1
' _ "$PID2" "$PY" "$ROOT" "$X" "$Y" "$Z" "$OUT3" "$LOG3" \
  "${COMMON_ARGS[@]}" >"$QUEUE_LOG" 2>&1 </dev/null &
QUEUE_PID=$!
echo "$QUEUE_PID" >"$QUEUE_PID_FILE"

cat <<EOF
two_view_pid=$PID2
two_view_output=$OUT2
two_view_log=$LOG2
queue_pid=$QUEUE_PID
three_view_output=$OUT3
three_view_log=$LOG3
queue_log=$QUEUE_LOG
EOF
