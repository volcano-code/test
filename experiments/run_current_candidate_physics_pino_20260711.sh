#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
INDEX="$ROOT/data/current_candidate_fullfield_fem_20260711/index.json"
OUT="$ROOT/checkpoints/current_candidate_physics_pino_120epoch_20260711"
LOG="$ROOT/results/current_candidate_physics_pino_120epoch_20260711.log"

# Full-field FEM labels are generated asynchronously. Do not start from an
# incomplete index, which would silently alter the train/validation protocol.
while true; do
  if "$PY" - "$INDEX" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
x = json.loads(p.read_text())
raise SystemExit(0 if len(x.get('train', [])) >= 160 and len(x.get('val', [])) >= 60 else 1)
PY
  then
    break
  fi
  sleep 30
done

mkdir -p "$OUT"
exec "$PY" "$ROOT/structural/training/pretrain_pino_v3.py" \
  --index_file "$INDEX" \
  --output_dir "$OUT" \
  --epochs 120 --batch_size 4 --accum_steps 2 --lr 1e-5 --warmup_epochs 10 \
  --resume_from "$ROOT/checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt" \
  --arch ufno_dual --modes 20 --width 96 --n_layers 8 --target_res 48 \
  --lambda_disp 1.0 --lambda_stress_log 3.0 --lambda_vm_asym 2.0 --lambda_vm_peak 1.0 \
  --lambda_bc 1.0 --lambda_pde 0.10 --lambda_constitutive 0.10 --lambda_scale 5.0 \
  --underestimate_weight 6.0 --overestimate_weight 0.3 \
  --version_tag current_candidate_physics_pino_120epoch_20260711 \
  > "$LOG" 2>&1
