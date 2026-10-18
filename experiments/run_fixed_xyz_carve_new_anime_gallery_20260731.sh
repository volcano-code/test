#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lin/xvxinyao/ShadowForge
PY=/home/lin/anaconda3/envs/shadowart-revisited/bin/python
BLENDER=${BLENDER:-$(command -v blender || true)}
SOURCE='/home/lin/下载/original_anime_silhouettes_30(1) (2)/original_anime_silhouettes_30'
OUTPUT_ROOT="$ROOT/results/fixed_xyz_carve_new_anime_gallery_20260731"
LOG_ROOT="$ROOT/logs/fixed_xyz_carve_new_anime_gallery_20260731"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

run_case() {
  local name=$1
  local first=$2
  local second=$3
  local third=$4
  local output="$OUTPUT_ROOT/$name"
  local log="$LOG_ROOT/$name.log"

  rm -rf "$output"
  mkdir -p "$output"
  "$PY" -u "$ROOT/shadowforge_pipeline/run.py" \
    --pipeline-branch fixed_xyz_carve \
    --silhouettes "$SOURCE/$first" "$SOURCE/$second" "$SOURCE/$third" \
    --output "$output" \
    --fixed-xyz-resolution 256 \
    --fixed-xyz-candidate auto \
    --fixed-xyz-min-view-iou 0.950 \
    --min-iou 0.970 \
    --min-saving 0.40 \
    --max-saving 0.60 \
    --selection-saving-target 0.50 \
    --min-erosion-survival 0.15 \
    --print-mm 100 \
    --nozzle-mm 0.4 \
    --wall-safety 2.0 \
    --screen skip \
    >"$log" 2>&1

  if [[ -n "$BLENDER" ]]; then
    "$BLENDER" -b \
      --python "$ROOT/experiments/render_inspection_stl.py" \
      -- "$output/selected_display.stl" "$output/selected_render.png" \
      >>"$log" 2>&1
  fi
}

# Two cases run concurrently to reduce wall time without exhausting host memory.
run_case shadow_operatives \
  '04_忍者动态潜行姿势.png' \
  '07_黑暗气质人物剪影.png' \
  '19_孤独的流浪武士.png' &
pid_shadow=$!

run_case future_guardians \
  '06_英勇的指挥官轮廓.png' \
  '12_活力四射的机械少女.png' \
  '18_自信的未来战士.png' &
pid_future=$!

wait "$pid_shadow"
wait "$pid_future"

run_case mystic_archers \
  '05_巫女的仪式动作.png' \
  '09_神秘弓箭手的轮廓.png' \
  '29_风中独行的武士.png' &
pid_mystic=$!

run_case hero_companions \
  '10_黑色轮廓中的快乐少年.png' \
  '13_英雄与伙伴的轮廓.png' \
  '30_机巧少女与小机器人.png' &
pid_companions=$!

wait "$pid_mystic"
wait "$pid_companions"

"$PY" "$ROOT/experiments/assemble_fixed_xyz_carve_new_anime_gallery_20260731.py"

