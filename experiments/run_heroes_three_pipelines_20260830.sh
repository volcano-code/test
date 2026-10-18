#!/usr/bin/env bash
set -uo pipefail

# Strictly serial comparison for the canonical heroes triplet.  The launcher
# waits for the already-running authenticated v7 campaign before acquiring the
# GPU for any of these jobs.  Every output root is create-new only.

WAIT_PID="${WAIT_PID:-3071070}"
PYTHON="/home/lin/anaconda3/envs/shadowart-revisited/bin/python3.10"
PCRAFT_ROOT="/home/lin/xvxinyao/P-CRAFT"
MESH2_ROOT="${PCRAFT_ROOT}/results/mesh_combined2_loss_ablation_2k_20260711/projects/heroes/full"
MESH1_ROOT="/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/mesh_combined"

SPIDER="${PCRAFT_ROOT}/generation/Spider-Man.png"
SUPERMAN="${PCRAFT_ROOT}/generation/superman.png"
BATMAN="${PCRAFT_ROOT}/generation/batman.png"

PCRAFT_OUT="/data/xvxinyao/pcraft_heroes_full_pipeline_20260830"
MESH2_OUT="/data/xvxinyao/mesh_combined2_heroes_full_2000_20260830"
MESH1_OUT="/data/xvxinyao/mesh_combined_heroes_mesh_15000_20260830"

PCRAFT_LOG="/data/xvxinyao/pcraft_heroes_full_pipeline_20260830.log"
MESH2_LOG="/data/xvxinyao/mesh_combined2_heroes_full_2000_20260830.log"
MESH1_LOG="/data/xvxinyao/mesh_combined_heroes_mesh_15000_20260830.log"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=17
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

log() {
  echo "[$(timestamp)] $*"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    log "FATAL missing required file: $1"
    exit 20
  fi
}

require_absent() {
  if [[ -e "$1" ]]; then
    log "FATAL refusing to overwrite existing path: $1"
    exit 21
  fi
}

wait_for_v7() {
  local ticks=0
  local args
  while [[ -r "/proc/${WAIT_PID}/cmdline" ]]; do
    args="$(tr '\0' ' ' < "/proc/${WAIT_PID}/cmdline")"
    if [[ "$args" != *"run_soft_loss_v7_hit_barrier_256.py"* ]]; then
      log "PID ${WAIT_PID} no longer belongs to the v7 runner; continuing."
      return
    fi
    if (( ticks % 10 == 0 )); then
      log "WAIT v7 PID ${WAIT_PID} is still active; no comparison job started."
    fi
    sleep 60
    ticks=$((ticks + 1))
  done
  log "v7 PID ${WAIT_PID} has exited."
}

wait_for_foreign_python_gpu() {
  local processes
  while true; do
    processes="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)"
    if ! grep -Eiq 'python' <<<"$processes"; then
      return
    fi
    log "WAIT another Python CUDA process is active: ${processes//$'\n'/; }"
    sleep 60
  done
}

run_stage() {
  local stage="$1"
  local stage_log="$2"
  local stage_cwd="$3"
  shift 3
  log "START ${stage}"
  (
    cd "$stage_cwd"
    "$@"
  ) 2>&1 | tee -a "$stage_log"
  local rc="${PIPESTATUS[0]}"
  log "END ${stage} returncode=${rc}"
  return "$rc"
}

for required in "$PYTHON" "$SPIDER" "$SUPERMAN" "$BATMAN" \
  "${PCRAFT_ROOT}/run_shadowforge.sh" "${MESH2_ROOT}/train.py" \
  "${MESH1_ROOT}/train.py"; do
  require_file "$required"
done

for output in "$PCRAFT_OUT" "$MESH2_OUT" "$MESH1_OUT"; do
  require_absent "$output"
done
for stage_log in "$PCRAFT_LOG" "$MESH2_LOG" "$MESH1_LOG"; do
  require_absent "$stage_log"
done

log "QUEUE canonical inputs: Spider-Man(x), superman(y), batman(z)."
sha256sum "$SPIDER" "$SUPERMAN" "$BATMAN"
wait_for_v7
wait_for_foreign_python_gpu

run_stage "P-CRAFT full free+fixed" "$PCRAFT_LOG" "$PCRAFT_ROOT" \
  "${PCRAFT_ROOT}/run_shadowforge.sh" \
  --silhouettes "$SPIDER" "$SUPERMAN" "$BATMAN" \
  --output "$PCRAFT_OUT" \
  --branches free fixed \
  --soft train \
  --soft-iters 15000 \
  --soft-lr 0.007 \
  --soft-resolution 256 \
  --cert-resolution 48 \
  --delivery-resolution 256 \
  --screen pino_acs_fem \
  --python "$PYTHON" \
  --device cuda:0
pcraft_rc=$?

wait_for_foreign_python_gpu
run_stage "mesh_combined2 heroes/full 2k" "$MESH2_LOG" "$MESH2_ROOT" \
  "$PYTHON" "${MESH2_ROOT}/train.py" cuda:0 "$MESH2_OUT" 2000 0.007 0 \
  -sdlist "$SPIDER" "$SUPERMAN" "$BATMAN" \
  -tsf 1 -mr 1 -tsf4 0 -cam ortho \
  -imsz 256 -res 128 -eres 256 -vh 1 -zdist 2.5 \
  --post_export_bundle 1 \
  --ablation_profile conservative \
  --enable_sparse_teacher 0
mesh2_rc=$?

wait_for_foreign_python_gpu
run_stage "mesh_combined mesh backend 15k" "$MESH1_LOG" "$MESH1_ROOT" \
  "$PYTHON" "${MESH1_ROOT}/train.py" cuda:0 "$MESH1_OUT" 15000 0.05 15901 \
  -sdlist "$SPIDER" "$SUPERMAN" "$BATMAN" \
  -mr 1 -tsf 1 -cam ortho -imsz 256 -zdist 2.5 \
  -vh 1 -vhr 128 --geometry_backend mesh
mesh1_rc=$?

log "SUMMARY pcraft=${pcraft_rc} mesh_combined2=${mesh2_rc} mesh_combined=${mesh1_rc}"
if (( pcraft_rc != 0 || mesh2_rc != 0 || mesh1_rc != 0 )); then
  exit 30
fi
log "ALL THREE PIPELINES COMPLETED"
