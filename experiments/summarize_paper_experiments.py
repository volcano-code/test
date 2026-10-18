#!/usr/bin/env python3
"""Create paper-facing experiment tables for the current StructShadow pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt(x, digits=3):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return "100%" if x else "0%"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if not np.isfinite(float(x)):
        return "--"
    return f"{float(x):.{digits}f}"


def pct(x, digits=1):
    if x is None:
        return "--"
    return f"{100.0 * float(x):.{digits}f}%"


def baseline_summary(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = read_json(path)
    rows = []
    for method, items in payload.items():
        valid = [r for r in items if isinstance(r, dict) and r.get("sf_fem") is not None]
        if not valid:
            continue
        rows.append(
            {
                "method": method,
                "n_cases": len(valid),
                "iou": float(np.mean([r.get("iou_3d", np.nan) for r in valid])),
                "volume_material": {
                    "mean_vol_ratio": float(np.mean([r.get("vol_ratio", np.nan) for r in valid])),
                    "material_saved_vs_dense": float(1.0 - np.mean([r.get("vol_ratio", np.nan) for r in valid])),
                },
                "safety_fem": {
                    "safe_rate": float(np.mean([bool(r.get("fem_safe", False)) for r in valid])),
                    "mean_sf": float(np.mean([r.get("sf_fem", np.nan) for r in valid])),
                    "failures": int(sum(1 for r in valid if r.get("failure_reason"))),
                },
                "runtime": {
                    "mean_elapsed_s": float(np.mean([r.get("elapsed_s", np.nan) for r in valid])),
                    "mean_fem_solve_time_s": float(np.mean([r.get("fem_solve_time_s", np.nan) for r in valid])),
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/paper_experiment_tables_20260623.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "results/baselines_fem20_20260623/results_combined.json")
    parser.add_argument("--baseline_fallback", type=Path, default=ROOT / "results/baselines_fem_light5_20260623/results_combined.json")
    args = parser.parse_args()

    generation = read_json(ROOT / "results/generation_pipeline_ablation.json")["aggregate"]
    resolution = read_json(ROOT / "results/generation_resolution_sensitivity.json")
    gap = read_json(ROOT / "results/gap_fill_stress_test.json")["aggregate"]
    acs = read_json(ROOT / "results/pino_accs_evaluation_20260623.json")
    runtime = read_json(ROOT / "results/pino_accs_runtime_20260623.json")
    dual_acs = read_json(ROOT / "results/pino_accs_dual_screen_20260624.json")
    selection = read_json(ROOT / "results/candidate_selection_summary_20260624.json")

    table1 = []
    for stage, row in generation.items():
        table1.append(
            {
                "stage": stage,
                "n_cases": row["n"],
                "iou": {"mean": row["mean_iou"], "min": row["min_iou"]},
                "volume_material": {"material_saved_vs_hull": row["mean_material_saved_vs_hull"]},
                "safety_fem": None,
                "runtime": {"mean_elapsed_s": row["mean_elapsed_s"]},
                "single_body_6_rate": row["single_body_6_rate"],
                "watertight_rate": row["watertight_rate"],
            }
        )

    table2_rows = baseline_summary(args.baseline)
    baseline_status = "full" if args.baseline.exists() and table2_rows and min(r["n_cases"] for r in table2_rows) >= 20 else "pilot"
    if not table2_rows:
        table2_rows = baseline_summary(args.baseline_fallback)
        baseline_status = "pilot_fallback_existing_5"

    table3 = {
        "schema_version": "paper_table_pino_acs_v1",
        "n_cases": acs["n_pairs"],
        "method": "PINO+ACS+FEA-fallback",
        "iou": None,
        "volume_material": None,
        "safety_fem": {
            "pino_spearman": acs["pino_raw"]["spearman_rho"],
            "raw_false_discovery_rate": acs["pino_raw"]["false_discovery_rate"],
            "acs_false_discovery_rate_mean": acs["component_ablation"]["accs"]["false_discovery_rate"]["mean"],
            "acs_safe_coverage_mean": acs["component_ablation"]["accs"]["safe_coverage"]["mean"],
        },
        "runtime": runtime["runtime"],
    }
    pure_fea_s = runtime["runtime"]["pure_fea_mean_s_per_case"]
    acs_decision_s = runtime["runtime"].get("acs_decision_mean_s", runtime["runtime"].get("accs_decision_mean_s", 0.0))
    pino_s = runtime["runtime"]["pino_inference_assumed_s"] + acs_decision_s
    screening_rows = [
        {
            "method": "pure_fea_screening",
            "n_cases": acs["n_pairs"],
            "false_discovery_rate": 0.0,
            "safe_coverage": 1.0,
            "fallback_rate": 1.0,
            "mean_time_s": pure_fea_s,
            "speedup_vs_pure_fea": 1.0,
        },
        {
            "method": "raw_pino_screening",
            "n_cases": acs["n_pairs"],
            "false_discovery_rate": acs["pino_raw"]["false_discovery_rate"],
            "safe_coverage": acs["pino_raw"]["safe_coverage"],
            "fallback_rate": 0.0,
            "mean_time_s": pino_s,
            "speedup_vs_pure_fea": (pure_fea_s / pino_s) if pure_fea_s is not None else None,
        },
        {
            "method": "acs_certified_with_fallback",
            "n_cases": acs["n_pairs"],
            "false_discovery_rate": acs["component_ablation"]["accs"]["false_discovery_rate"]["mean"],
            "safe_coverage": acs["component_ablation"]["accs"]["safe_coverage"]["mean"],
            "fallback_rate": runtime["runtime"]["fallback_rate_mean"],
            "mean_time_s": runtime["runtime"]["hybrid_mean_s_per_case"],
            "speedup_vs_pure_fea": runtime["runtime"]["estimated_speedup_vs_pure_fea"],
        },
        {
            "method": "dual_acs_boundary_fallback",
            "n_cases": dual_acs["n_cases"],
            "false_discovery_rate": dual_acs["metrics"]["safe_false_discovery_rate"]["mean"],
            "safe_coverage": dual_acs["metrics"]["safe_coverage"]["mean"],
            "fallback_rate": dual_acs["metrics"]["fallback_rate"]["mean"],
            "autonomous_rate": dual_acs["metrics"]["autonomous_rate"]["mean"],
            "mean_time_s": dual_acs["runtime"]["hybrid_mean_s_per_case"],
            "speedup_vs_pure_fea": dual_acs["runtime"]["estimated_speedup_vs_pure_fea"],
        },
    ]

    selection_rows = []
    for branch, branch_data in selection["branches"].items():
        for method, row in branch_data["policy_summary"].items():
            selection_rows.append({
                "branch": branch,
                "method": method,
                "n_cases": branch_data["n_cases"],
                "mean_candidates": branch_data["mean_candidates_per_case"],
                **row,
            })

    result = {
        "schema_version": "structshadow_paper_experiment_tables_v1",
        "notes": [
            "Only current pipeline results are included.",
            "Legacy structural-repair side-branch outputs are intentionally excluded.",
        ],
        "table1_generation_ablation": table1,
        "table1_generation_ablation_markdown": [
            "| Stage | IoU mean/min | Saved vs hull | 6-connected | Watertight | Time (s) |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| {r['stage']} | {fmt(r['iou']['mean'])}/{fmt(r['iou']['min'])} | "
                f"{pct(r['volume_material']['material_saved_vs_hull'])} | {pct(r['single_body_6_rate'])} | "
                f"{pct(r['watertight_rate'])} | {fmt(r['runtime']['mean_elapsed_s'])} |"
                for r in table1
            ],
        ],
        "table2_baseline_comparison_status": baseline_status,
        "table2_baseline_comparison": table2_rows,
        "table2_baseline_comparison_markdown": [
            "| Method | n | IoU | Vol. ratio | Safe rate | FEM-SF | Time (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {r['method']} | {r['n_cases']} | {fmt(r['iou'])} | "
                f"{fmt(r['volume_material']['mean_vol_ratio'])} | {pct(r['safety_fem']['safe_rate'])} | "
                f"{fmt(r['safety_fem']['mean_sf'])} | {fmt(r['runtime']['mean_elapsed_s'])} |"
                for r in table2_rows
            ],
        ],
        "table3_pino_acs_certification_runtime": table3,
        "table3_screening_baselines": screening_rows,
        "table3_pino_acs_markdown": [
            "| Method | n | FDR | Safe coverage | Fallback | Autonomous | Time (s) | Speedup |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {r['method']} | {r['n_cases']} | {pct(r['false_discovery_rate'])} | "
                f"{pct(r['safe_coverage'])} | {pct(r['fallback_rate'])} | "
                f"{pct(r.get('autonomous_rate', 1.0 - r['fallback_rate']))} | "
                f"{fmt(r['mean_time_s'])} | {fmt(r['speedup_vs_pure_fea'])}x |"
                for r in screening_rows
            ],
        ],
        "table4_candidate_selection_ablation": selection_rows,
        "table4_candidate_selection_markdown": [
            "| Branch | Policy | n | cand./case | select | safe | saving | FEM fallbacks |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {r['branch']} | {r['method']} | {r['n_cases']} | {fmt(r['mean_candidates'],1)} | "
                f"{pct(r['selection_rate'])} | {pct(r['safe_rate']) if r['safe_rate'] is not None else '--'} | "
                f"{pct(r['mean_saving']) if r['mean_saving'] is not None else '--'} | {fmt(r['mean_fem_fallbacks'])} |"
                for r in selection_rows
            ],
        ],
        "supporting_results": {
            "resolution_sensitivity": resolution["across_resolution"],
            "gap_fill_stress": gap,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
