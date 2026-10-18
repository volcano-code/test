#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

run_one() {
  local tag="$1"
  shift
  echo "[RUN] ${tag}" >&2
  PYTHONUNBUFFERED=1 /home/lin/anaconda3/bin/python -u experiments/train_pino_safe_side_external_hardneg.py \
    --start-ckpt checkpoints/pino_safe_accept90_cal300_m08_20260704/safe_side_external_hardneg_scale.pt \
    --external-calibration 300 \
    --boundary-limit 600 \
    --hardset-sf-values 0.8,1.0,1.2,1.5,2.0,3.0 \
    --result "results/pino_near_safe_margin_${tag}_20260704.json" \
    --output-dir "checkpoints/pino_near_safe_margin_${tag}_20260704" \
    "$@" \
    > "results/pino_near_safe_margin_${tag}_20260704.log" 2>&1
}

run_one e5_safe4_m12 --epochs 5 --lr 4e-5 --hardset-safe-repeat 4 --hardset-unsafe-repeat 1 --safe-margin-weight 1.4 --hardset-safe-margin-weight 4.0 --safe-margin-log 0.12 --false-safe-weight 18 --external-false-safe-weight 18 --unsafe-buffer-log 0.02
run_one e6_safe6_m16 --epochs 6 --lr 3e-5 --hardset-safe-repeat 6 --hardset-unsafe-repeat 1 --safe-margin-weight 1.8 --hardset-safe-margin-weight 6.0 --safe-margin-log 0.16 --false-safe-weight 20 --external-false-safe-weight 20 --unsafe-buffer-log 0.02
run_one e6_safe8_m20 --epochs 6 --lr 3e-5 --hardset-safe-repeat 8 --hardset-unsafe-repeat 1 --safe-margin-weight 2.2 --hardset-safe-margin-weight 8.0 --safe-margin-log 0.20 --false-safe-weight 22 --external-false-safe-weight 22 --unsafe-buffer-log 0.015

/home/lin/anaconda3/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("results")
rows = []
for path in sorted(root.glob("pino_near_safe_margin_*_20260704.json")):
    if path.name == "pino_near_safe_margin_search_summary_20260704.json":
        continue
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
        "joint_auto_safe_true": c.get("auto_safe_true", 0) + e.get("auto_safe_true", 0) + h.get("auto_safe_true", 0),
        "promotion_eligible": data.get("promotion_eligible_for_fem_replacement_claim"),
    })
rows.sort(key=lambda r: (r["joint_false_pass"], -(r["joint_auto_safe_true"] or 0)))
(root / "pino_near_safe_margin_search_summary_20260704.json").write_text(json.dumps({"schema": "pino_near_safe_margin_search_summary_v1", "rows": rows}, indent=2), encoding="utf-8")
print(json.dumps(rows[:5], indent=2))
PY
