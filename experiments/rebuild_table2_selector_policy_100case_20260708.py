#!/usr/bin/env python3
"""Rebuild Table 2 selector-policy ablation from the saved 100-case pool."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results" / "paper_table_drafts_20260708"
OUT.mkdir(parents=True, exist_ok=True)


def load_cases() -> list[dict]:
    rows = []
    for branch in ("free", "fixed"):
        path = ROOT / "results" / f"candidate_selection_batch_{branch}50_20260624.json"
        data = json.loads(path.read_text())
        for case in data["cases"]:
            case = dict(case)
            case["branch"] = branch
            rows.append(case)
    return rows


def candidate_ok(c: dict, *, iou=True, conn=True, wt=True, saving=True, fem=False) -> bool:
    if iou and float(c.get("mean_iou", -1)) < 0.95:
        return False
    if conn and int(c.get("component_count_6", 999999)) != 1:
        return False
    if wt and not bool(c.get("watertight", False)):
        return False
    if saving:
        s = float(c.get("material_saved_vs_hull", -999))
        if not (0.40 <= s <= 0.85):
            return False
    if fem and not bool(c.get("fem_safe", False)):
        return False
    return True


def by_name(case: dict, name: str) -> dict | None:
    return next((c for c in case["candidates"] if c["candidate"] == name), None)


def lightest(cands: list[dict]) -> dict | None:
    if not cands:
        return None
    return max(cands, key=lambda c: float(c.get("material_saved_vs_hull", -1e9)))


def policy_selected_candidate(case: dict, policy_name: str) -> dict | None:
    selected = case.get("policies", {}).get(policy_name, {}).get("selected")
    if selected is None:
        return None
    cand = by_name(case, selected)
    if cand is None:
        return None
    return cand


def summarize(label: str, cases: list[dict], selector) -> dict:
    selected = []
    fem_deferrals = []
    for case in cases:
        cand = selector(case)
        if cand is None:
            continue
        selected.append(cand)
        pol = case.get("policies", {}).get("dual_acs_fem_fallback", {})
        fem_deferrals.append(float(pol.get("fem_fallbacks", 0)))
    total = len(cases)
    n = len(selected)
    safe = sum(bool(c.get("fem_safe", False)) for c in selected)
    unsafe = n - safe
    single = sum(int(c.get("component_count_6", 999999)) == 1 for c in selected)
    watertight = sum(bool(c.get("watertight", False)) for c in selected)
    printable = sum(
        int(c.get("component_count_6", 999999)) == 1 and bool(c.get("watertight", False))
        for c in selected
    )
    return {
        "policy": label,
        "total_cases": total,
        "selection_rate": n / total if total else 0.0,
        "mean_iou": float(np.mean([float(c["mean_iou"]) for c in selected])) if selected else None,
        "worst_iou": float(np.min([float(c["mean_iou"]) for c in selected])) if selected else None,
        "validated_reduction": float(np.mean([float(c["material_saved_vs_hull"]) for c in selected])) if selected else None,
        "single_body_rate": single / n if n else None,
        "watertight_rate": watertight / n if n else None,
        "printable_success_rate": printable / n if n else None,
        "fem_safe_rate": safe / n if n else None,
        "fem_unsafe_rate": unsafe / n if n else None,
        "mean_fem_deferrals": float(np.mean(fem_deferrals)) if fem_deferrals else None,
        "selected_cases": n,
        "single_body_selected": single,
        "watertight_selected": watertight,
        "printable_selected": printable,
        "fem_safe_selected": safe,
        "fem_unsafe_selected": unsafe,
    }


def fmt_pct(x) -> str:
    return "N/A" if x is None else f"{100.0 * float(x):.1f}%"


def fmt_num(x) -> str:
    return "N/A" if x is None else f"{float(x):.3f}"


def main() -> None:
    cases = load_cases()
    policies = [
        ("Visual hull reference", lambda c: by_name(c, "visual_hull")),
        ("Terminal export only", lambda c: by_name(c, "watertight_export") if by_name(c, "watertight_export") is not None and candidate_ok(by_name(c, "watertight_export"), fem=True) else None),
        ("Lightest fabrication-valid", lambda c: policy_selected_candidate(c, "lightest_printable")),
        ("ACS-only selector", lambda c: policy_selected_candidate(c, "acs_safe")),
        ("Final selector", lambda c: policy_selected_candidate(c, "dual_acs_fem_fallback")),
        ("FEM-oracle upper bound", lambda c: policy_selected_candidate(c, "fem_oracle")),
    ]
    rows = [summarize(name, cases, fn) for name, fn in policies]

    json_path = OUT / "table2_selector_policy_100case_20260708.json"
    json_path.write_text(json.dumps({
        "schema": "table2_selector_policy_100case_v1",
        "source_files": [
            str(ROOT / "results/candidate_selection_batch_free50_20260624.json"),
            str(ROOT / "results/candidate_selection_batch_fixed50_20260624.json"),
        ],
        "n_cases": len(cases),
        "n_candidates": sum(len(c["candidates"]) for c in cases),
        "metrics": {
            "selection_rate": "selected cases / 100 total cases",
            "validated_reduction": "mean material reduction among selected outputs that pass the policy gates",
            "single_body_rate": "selected outputs with one 6-connected component / selected outputs",
            "watertight_rate": "selected outputs with watertight mesh proxy / selected outputs",
            "printable_success_rate": "selected outputs that are both single-body and watertight / selected outputs",
            "fem_unsafe_rate": "FEM-unsafe selected outputs / selected outputs; lower is better",
            "fem_safe_rate": "FEM-safe selected outputs / selected outputs; higher is better",
            "dice": "not recorded in the saved 100-case candidate pool",
        },
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")

    tsv_path = OUT / "table2_selector_policy_100case_20260708.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("Policy\tSelection rate ↑\tMean IoU ↑\tWorst IoU ↑\tValidated reduction ↑\tSingle-body ↑\tWatertight ↑\tPrintable success ↑\tFEM-unsafe rate ↓\tFEM-safe rate ↑\tMean FEM deferrals ↓\n")
        for r in rows:
            f.write(
                f"{r['policy']}\t{fmt_pct(r['selection_rate'])}\t{fmt_num(r['mean_iou'])}\t"
                f"{fmt_num(r['worst_iou'])}\t{fmt_pct(r['validated_reduction'])}\t"
                f"{fmt_pct(r['single_body_rate'])}\t{fmt_pct(r['watertight_rate'])}\t"
                f"{fmt_pct(r['printable_success_rate'])}\t"
                f"{fmt_pct(r['fem_unsafe_rate'])}\t{fmt_pct(r['fem_safe_rate'])}\t"
                f"{'N/A' if r['mean_fem_deferrals'] is None else f'{r['mean_fem_deferrals']:.2f}'}\n"
            )
    print(json.dumps({"json": str(json_path), "tsv": str(tsv_path), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
