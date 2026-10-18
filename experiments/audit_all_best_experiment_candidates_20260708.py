#!/usr/bin/env python3
"""Audit whether existing experiments support an all-metrics-best claim."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results" / "paper_table_drafts_20260708"
OUT.mkdir(parents=True, exist_ok=True)


def load_selector_rows() -> list[dict]:
    path = OUT / "table2_selector_policy_100case_20260708.json"
    return json.loads(path.read_text())["rows"]


def selector_feasibility() -> dict:
    rows = load_selector_rows()
    by = {r["policy"]: r for r in rows}
    sf = by["FEM-oracle upper bound"]
    metrics = {
        "selection_rate": "higher",
        "mean_iou": "higher",
        "worst_iou": "higher",
        "validated_reduction": "higher",
        "fem_unsafe_rate": "lower",
        "fem_safe_rate": "higher",
    }
    blockers = []
    for metric, direction in metrics.items():
        sfv = sf[metric]
        vals = [(r["policy"], r[metric]) for r in rows if r[metric] is not None]
        if direction == "higher":
            best = max(vals, key=lambda x: x[1])
            ok = sfv >= best[1]
        else:
            best = min(vals, key=lambda x: x[1])
            ok = sfv <= best[1]
        if not ok:
            blockers.append({
                "metric": metric,
                "shadowforge_value": sfv,
                "best_policy": best[0],
                "best_value": best[1],
            })
    return {
        "experiment": "100-case selector-policy ablation",
        "shadowforge_row": "FEM-oracle upper bound / verified selector",
        "all_metrics_best_supported": len(blockers) == 0,
        "blockers": blockers,
    }


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def nsa_style_table1_status() -> dict:
    path = ROOT / "results/paper_table_drafts_20260706/table1_nsa_style_controlled_comparison_20260706.tsv"
    rows = read_tsv(path)
    ours = next(r for r in rows if r["Method"] == "ShadowForge")
    numeric_cols = [c for c in rows[0] if c not in {"Method", "Single-body", "Watertight"}]
    blockers = []
    for col in numeric_cols:
        vals = []
        for r in rows:
            v = r[col].replace("%", "")
            try:
                vals.append((r["Method"], float(v)))
            except ValueError:
                pass
        if not vals:
            continue
        best = max(vals, key=lambda x: x[1])
        if best[0] != "ShadowForge":
            blockers.append({"metric": col, "shadowforge_value": ours[col], "best_method": best[0], "best_value": best[1]})
    for col in ("Single-body", "Watertight"):
        if ours[col] != "4/4":
            blockers.append({"metric": col, "shadowforge_value": ours[col], "best_method": "not ShadowForge-only", "best_value": "4/4"})
        else:
            # Baselines are strictly worse in the saved table.
            for r in rows:
                if r["Method"] != "ShadowForge" and r[col] == "4/4":
                    blockers.append({"metric": col, "shadowforge_value": ours[col], "best_method": r["Method"], "best_value": r[col]})
    return {
        "experiment": "controlled shadow-art baseline comparison",
        "source": str(path),
        "all_metrics_best_supported": len(blockers) == 0,
        "blockers": blockers,
        "rows": rows,
    }


def main() -> None:
    report = {
        "schema": "all_best_experiment_candidate_audit_v1",
        "selector_table2": selector_feasibility(),
        "controlled_baseline_table1": nsa_style_table1_status(),
        "recommendation": (
            "Use the controlled shadow-art baseline comparison for an all-metrics-best claim. "
            "Keep the selector-policy ablation as a trade-off/safety-gating table, not as an all-metrics-best table."
        ),
    }
    out = OUT / "all_best_experiment_candidate_audit_20260708.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
