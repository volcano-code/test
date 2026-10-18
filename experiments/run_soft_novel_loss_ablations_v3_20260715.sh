#!/usr/bin/env bash
# Stage-A differentiable proposal ablation.
#
# This script intentionally excludes L_l2 and L_l1, which are inherited
# projection-fitting terms. It also excludes L_volume because material
# reduction is evaluated in the downstream guarded-carving stage. No geometry
# repair, carving, shell delivery, or mesh QA is called here.
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
VOXEL=/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
OUT="$ROOT/results/soft_novel_loss_ablations_v3_20260715"

# All tasks use the fixed two-wall input convention. Keep the seed list frozen
# before running and do not tune variants per task.
CASES=(
  "future_mech|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H01_future_mech/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H01_future_mech/inputs/right_wall_y_target.png"
  "sword_fox|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H02_sword_fox/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/anime/H02_sword_fox/inputs/right_wall_y_target.png"
  "parrot_snail|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L01_parrot_snail/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L01_parrot_snail/inputs/right_wall_y_target.png"
  "owl_turtle|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L02_owl_turtle/inputs/left_wall_x_target.png|$ROOT/results/fem_two_wall_print_candidate_search_20260714/simple/L02_owl_turtle/inputs/right_wall_y_target.png"
)
SEEDS=(17 29 43)

# Every row starts from the same regularized soft objective. Only a newly added
# loss is removed in each ablation. L_volume remains disabled in every row.
VARIANTS=(
  "full|"
  "w_o_Ldice|dice"
  "w_o_Lssim|ssim"
  "w_o_Lfp|false_positive"
  "w_o_Ledge|edge"
  "w_o_Lsmooth|smooth"
  "w_o_Llap|laplacian"
  "w_o_Lconn|connectivity"
  "w_o_Lspur|spurious"
  "w_o_Lwall|wall"
  "w_o_Lspec|spectral"
  "w_o_Ldev|deviation"
)

run_one() {
  local task="$1" left="$2" right="$3" seed="$4" name="$5" disabled="$6"
  local dir="$OUT/$task/$name/seed_$seed"
  mkdir -p "$dir"
  if [[ -f "$dir/metrics.json" ]]; then
    echo "skip completed task=$task variant=$name seed=$seed"
    return
  fi
  echo "===== task=$task variant=$name seed=$seed disabled=${disabled:-<none>} ====="
  "$PY" "$VOXEL/train.py" cuda:0 "$dir" 2000 0.007 1 \
    -sdlist "$left" "$right" \
    -mr 0 -tsf 0 -tsf4 0 -cam ortho -imsz 256 \
    --res 256 -eres 256 --control-res 0 --save-every 100 \
    --proposal-mode regularized \
    --connectivity-start-iou 0.60 --fidelity-gate 0.85 \
    --structure-warmup-steps 300 --structure-ramp-steps 300 \
    --no-cleanup --export-final-step --seed "$seed" \
    --disable-losses volume ${disabled}
}

mkdir -p "$OUT"
for case in "${CASES[@]}"; do
  IFS='|' read -r task left right <<< "$case"
  for variant in "${VARIANTS[@]}"; do
    IFS='|' read -r name disabled <<< "$variant"
    for seed in "${SEEDS[@]}"; do
      run_one "$task" "$left" "$right" "$seed" "$name" "$disabled"
    done
  done
done

"$PY" - "$OUT" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
groups = defaultdict(list)
for path in sorted(root.glob('*/*/seed_*/metrics.json')):
    row = json.loads(path.read_text())
    task, variant, seed = path.parts[-4:-1]
    groups[variant].append({
        'task': task, 'seed': seed,
        'iou': float(row['mean_iou']), 'dice': float(row['mean_dice']),
        'raw_soft_iou': float(row.get('raw_soft_iou', row['mean_iou'])),
    })

summary = []
for variant, rows in sorted(groups.items()):
    summary.append({
        'variant': variant, 'n': len(rows),
        'mean_iou': float(np.mean([r['iou'] for r in rows])),
        'std_iou': float(np.std([r['iou'] for r in rows])),
        'mean_dice': float(np.mean([r['dice'] for r in rows])),
        'std_dice': float(np.std([r['dice'] for r in rows])),
        'mean_raw_soft_iou': float(np.mean([r['raw_soft_iou'] for r in rows])),
    })
(root / 'aggregate.json').write_text(json.dumps(summary, indent=2) + '\n')
print(root / 'aggregate.json')
PY
