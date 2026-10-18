#!/usr/bin/env bash
# Diagnostic experiment only, not a new production algorithm or an unmodified
# author-code reproduction. Equal camera/render/Adam-step budget isolates loss.
set -euo pipefail
ROOT=/home/lin/xvxinyao/P-CRAFT
VOXEL=/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python3.10
OUT="${1:?Provide a NEW absolute output directory}"
if [[ "$OUT" != /* || -e "$OUT" ]]; then
  echo 'Output must be a new absolute directory.' >&2
  exit 2
fi
mkdir "$OUT"
export PYTHONPATH="$ROOT:$VOXEL${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0
sha256sum "$VOXEL"/*.py "$ROOT"/generation/{Spider-Man.png,superman.png,batman.png} \
  | tee "$OUT/source_input_sha256.before.txt"
cd "$VOXEL"
for arm in sadekar_loss_joint current_proposal paper_formula reweighted_l1_l2; do
  mode=proposal
  disabled=(volume)
  case "$arm" in
    sadekar_loss_joint) mode=dense ;;
    paper_formula) disabled+=(ssim) ;;
    reweighted_l1_l2) disabled+=(dice ssim false_positive) ;;
  esac
  command=("$PY" -u train.py cuda:0 "$OUT/$arm" 2000 0.007 1
    -sdlist "$ROOT/generation/Spider-Man.png" "$ROOT/generation/superman.png" "$ROOT/generation/batman.png"
    -mr 1 -tsf 1 -tsf4 0 -cam ortho -imsz 256 --res 256 -eres 256 --control-res 0
    --proposal-mode "$mode" --view-update-mode joint --disable-losses "${disabled[@]}"
    --preserve-target-aspect --sadekar-zdist 1.7 --sadekar-volume-extent 1.7
    --sadekar-npts-per-ray 150 --sadekar-density-threshold 0.05
    --connectivity-start-iou 0.60 --fidelity-gate 0.85
    --structure-warmup-steps 300 --structure-ramp-steps 300
    --no-cleanup --export-final-step --save-every 100 --seed 42)
  printf '%q ' "${command[@]}" | tee "$OUT/$arm.command.txt"
  printf '\n'
  "${command[@]}" 2>&1 | tee "$OUT/$arm.log"
  sha256sum --check "$OUT/source_input_sha256.before.txt"
done
printf 'ALL_FOUR_SOFT_RUNS_COMPLETED; common postprocessing and author-native schedule comparison remain separate.\n'
