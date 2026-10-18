#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

run_one() {
  local tag="$1"
  shift
  echo "[RUN] ${tag} $(date)" >&2
  PYTHONUNBUFFERED=1 /home/lin/anaconda3/bin/python -u experiments/train_pino_safe_side_external_hardneg.py \
    --start-ckpt checkpoints/pino_safe_accept90_cal300_m08_20260704/safe_side_external_hardneg_scale.pt \
    --external-calibration 300 \
    --boundary-limit 600 \
    --hardset-sf-values 1.5,2.0,3.0 \
    --hardset-calibration 0 \
    --hardset-unsafe-repeat 0 \
    --result "results/pino_safe_margin_light_${tag}_20260704.json" \
    --output-dir "checkpoints/pino_safe_margin_light_${tag}_20260704" \
    "$@" \
    > "results/pino_safe_margin_light_${tag}_20260704.log" 2>&1
}

run_one e6_hs1_m06 --epochs 6 --lr 2e-5 --external-hard-repeat 4 --hardset-safe-repeat 1 --safe-margin-weight 0.9 --hardset-safe-margin-weight 0.6 --safe-margin-log 0.06 --false-safe-weight 16 --external-false-safe-weight 18 --unsafe-buffer-log 0.02
run_one e6_hs2_m08 --epochs 6 --lr 2e-5 --external-hard-repeat 4 --hardset-safe-repeat 2 --safe-margin-weight 1.0 --hardset-safe-margin-weight 0.8 --safe-margin-log 0.08 --false-safe-weight 16 --external-false-safe-weight 18 --unsafe-buffer-log 0.02

/home/lin/anaconda3/bin/python - <<'PY'
import json
from pathlib import Path

rows = []
for path in sorted(Path("results").glob("pino_safe_margin_light_*_20260704.json")):
    data = json.loads(path.read_text())
    c = data.get("candidate_heldout_safe", {})
    e = data.get("external_heldout_safe", {})
    h = data.get("hardset_test_near_boundary_safe", {})
    rows.append({
        "file": str(path),
        "checkpoint": data.get("checkpoint"),
        "candidate_auto_safe_rate": c.get("auto_safe_rate"),
        "external_auto_safe_rate": e.get("auto_safe_rate"),
        "hardset_auto_safe_rate": h.get("auto_safe_rate"),
        "joint_false_pass": c.get("auto_false_pass", 0) + e.get("auto_false_pass", 0) + h.get("auto_false_pass", 0),
        "joint_auto_safe_true_main": c.get("auto_safe_true", 0) + e.get("auto_safe_true", 0),
        "promotion_eligible": data.get("promotion_eligible_for_fem_replacement_claim"),
    })
rows.sort(key=lambda r: (r["joint_false_pass"], -(r["joint_auto_safe_true_main"] or 0)))
Path("results/pino_safe_margin_light_search_summary_20260704.json").write_text(json.dumps({"schema": "pino_safe_margin_light_search_summary_v1", "rows": rows}, indent=2), encoding="utf-8")
print(json.dumps(rows, indent=2))
PY
