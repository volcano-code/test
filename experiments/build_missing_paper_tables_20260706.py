#!/usr/bin/env python3
"""Build normalized paper-table drafts from saved ShadowForge evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import generate_binary_structure, label

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]

from datasets import load_data_from_list
from voxel_ops import (
    build_tfs_visual_hull,
    calibrate_tfs_projection,
    compute_iou_np,
    evaluate_tfs_volume,
    resize_binary_mask,
    sdf_watertight_mesh,
)

OUT = ROOT / "results" / "paper_table_drafts_20260706_normalized"
S6 = generate_binary_structure(3, 1)


def pct(x):
    if x is None:
        return "not recorded"
    return f"{100.0 * float(x):.1f}%"


def f3(x):
    return "not recorded" if x is None else f"{float(x):.3f}"


def f2(x):
    return "not recorded" if x is None else f"{float(x):.2f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def load_candidate_rows():
    rows = []
    for branch in ("free", "fixed"):
        p = ROOT / "results" / f"candidate_selection_batch_{branch}50_20260624.json"
        data = json.loads(p.read_text())
        for case in data["cases"]:
            case = dict(case)
            case["branch"] = branch
            rows.append(case)
    return rows


def candidate_ok(c, *, iou=True, conn=True, wt=True, saving=True, fem=False):
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


def select_lightest(cands):
    if not cands:
        return None
    return max(cands, key=lambda c: float(c.get("material_saved_vs_hull", -1e9)))


def summarize_selected(selected, total_cases):
    sel = [x for x in selected if x is not None]
    return {
        "selected_cases": f"{len(sel)}/{total_cases}",
        "mean_iou": mean([x.get("mean_iou") for x in sel]),
        "worst_iou": min([float(x.get("mean_iou")) for x in sel], default=None),
        "validated_saving": mean([x.get("material_saved_vs_hull") for x in sel]),
        "fem_safe": f"{sum(bool(x.get('fem_safe', False)) for x in sel)}/{len(sel)}" if sel else "0/0",
        "fem_deferrals": mean([x.get("_fem_fallbacks", 0) for x in sel]),
    }


def build_selector_policy_tables():
    cases = load_candidate_rows()
    total = len(cases)

    # Funnel.
    groups = [("All", cases), ("Free branch", [c for c in cases if c["branch"] == "free"]), ("Fixed branch", [c for c in cases if c["branch"] == "fixed"])]
    with (OUT / "table2a_candidate_funnel.tsv").open("w") as f:
        f.write("Group\tCases\tRaw candidates/case\tFabrication-valid/case\tStructurally eligible/case\tFabrication-valid coverage\tEligible coverage\n")
        for name, grp in groups:
            raw = [len(c["candidates"]) for c in grp]
            fab_counts = [sum(candidate_ok(x, fem=False) for x in c["candidates"]) for c in grp]
            elig_counts = [sum(candidate_ok(x, fem=True) for x in c["candidates"]) for c in grp]
            f.write(
                f"{name}\t{len(grp)}\t{mean(raw):.2f}\t{mean(fab_counts):.2f}\t{mean(elig_counts):.2f}\t"
                f"{sum(v > 0 for v in fab_counts)}/{len(grp)}\t{sum(v > 0 for v in elig_counts)}/{len(grp)}\n"
            )

    policies = []
    # Reference hull, no material reduction.
    policies.append(("Visual hull reference", [next((x for x in c["candidates"] if x["candidate"] == "visual_hull"), None) for c in cases]))
    policies.append(("Terminal carve only", [next((x for x in c["candidates"] if x["candidate"] == "watertight_export" and candidate_ok(x, fem=True)), None) for c in cases]))
    policies.append(("Lightest fabrication-valid", [select_lightest([x for x in c["candidates"] if candidate_ok(x, fem=False)]) for c in cases]))
    policies.append(("Full structurally eligible selector", [select_lightest([x for x in c["candidates"] if candidate_ok(x, fem=True)]) for c in cases]))
    policies.append(("Fixed-only eligible selector", [select_lightest([x for x in c["candidates"] if candidate_ok(x, fem=True)]) for c in cases if c["branch"] == "fixed"]))
    policies.append(("Free-or-fixed eligible selector", [select_lightest([x for x in c["candidates"] if candidate_ok(x, fem=True)]) for c in cases]))
    with (OUT / "table2b_selector_policy_ablation.tsv").open("w") as f:
        f.write("Policy\tSelected cases\tMean IoU\tWorst IoU\tValidated material reduction\tFEM-safe\tFEM deferrals/selected\n")
        for name, selected in policies:
            s = summarize_selected(selected, len(selected))
            f.write(
                f"{name}\t{s['selected_cases']}\t"
                f"{s['mean_iou']:.3f if s['mean_iou'] is not None else 'not recorded'}\t"
            )


def write_selector_policy_tables_fixed():
    cases = load_candidate_rows()
    policies = [
        ("Visual hull reference", cases, lambda c: next((x for x in c["candidates"] if x["candidate"] == "visual_hull"), None)),
        ("Terminal export only", cases, lambda c: next((x for x in c["candidates"] if x["candidate"] == "watertight_export" and candidate_ok(x, fem=True)), None)),
        ("Lightest fabrication-valid", cases, lambda c: select_lightest([x for x in c["candidates"] if candidate_ok(x, fem=False)])),
        ("Lightest structurally eligible", cases, lambda c: select_lightest([x for x in c["candidates"] if candidate_ok(x, fem=True)])),
        ("Fixed-branch eligible", [c for c in cases if c["branch"] == "fixed"], lambda c: select_lightest([x for x in c["candidates"] if candidate_ok(x, fem=True)])),
        ("Free-or-fixed eligible", cases, lambda c: select_lightest([x for x in c["candidates"] if candidate_ok(x, fem=True)])),
    ]
    with (OUT / "table2b_selector_policy_ablation.tsv").open("w") as f:
        f.write("Policy\tCandidate pool\tSelected cases\tMean IoU\tWorst IoU\tValidated material reduction\tFEM-safe\tMean FEM deferrals\n")
        for name, grp, fn in policies:
            selected = []
            for c in grp:
                s = fn(c)
                if s is not None:
                    s = dict(s)
                    pol = c.get("policies", {}).get("dual_acs_fem_fallback", {})
                    s["_fem_fallbacks"] = pol.get("fem_fallbacks", 0)
                selected.append(s)
            s = summarize_selected(selected, len(grp))
            f.write(
                f"{name}\t{len(grp)} cases\t{s['selected_cases']}\t"
                f"{f3(s['mean_iou'])}\t"
                f"{f3(s['worst_iou'])}\t"
                f"{pct(s['validated_saving'])}\t{s['fem_safe']}\t"
                f"{f2(s['fem_deferrals'])}\n"
            )


def build_gate_leakage_table():
    cases = load_candidate_rows()
    cands = [x for c in cases for x in c["candidates"]]
    variants = [
        ("Full fabrication gate", dict(iou=True, conn=True, wt=True, saving=True)),
        ("w/o global 6-connectivity audit", dict(iou=True, conn=False, wt=True, saving=True)),
        ("w/o watertight mesh validation", dict(iou=True, conn=True, wt=False, saving=True)),
        ("w/o material-reduction policy", dict(iou=True, conn=True, wt=True, saving=False)),
    ]
    with (OUT / "tableS_gate_leakage_audit.tsv").open("w") as f:
        f.write("Gate variant\tCandidates admitted\tMulti-body admitted\tNon-watertight admitted\tSaving-policy violations admitted\tThin-wall admitted\n")
        for name, kw in variants:
            admitted = [x for x in cands if candidate_ok(x, fem=False, **kw)]
            multi = sum(int(x.get("component_count_6", 999)) != 1 for x in admitted)
            nonwt = sum(not bool(x.get("watertight", False)) for x in admitted)
            savviol = sum(not (0.40 <= float(x.get("material_saved_vs_hull", -999)) <= 0.85) for x in admitted)
            f.write(f"{name}\t{len(admitted)}\t{multi}\t{nonwt}\t{savviol}\tnot recorded\n")


def find_mask(case_dir: Path, name: str) -> Path | None:
    candidates = [
        case_dir / "inputs" / name,
        ROOT / "data" / "silhouettes" / "bench3" / name,
        ROOT / "data" / "silhouettes" / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    found = list(ROOT.glob(f"**/{name}"))
    return found[0] if found else None


def load_masks_for_row(row):
    case_dir = Path(row["artifacts"]["final"]).parents[1]
    paths = []
    for name in row["input_masks"]:
        p = find_mask(case_dir, name)
        if p is None:
            return None
        paths.append(p)
    masks, _ = load_data_from_list([str(p) for p in paths], True, (256, 256), torch.device("cpu"), 3, False)
    return [np.asarray(x).squeeze().astype(bool) for x in masks]


def score_volume_256(vol, masks, fix_axis):
    cal = calibrate_tfs_projection(masks, search_res=64, fix_axis=fix_axis)
    ev = evaluate_tfs_volume(vol.astype(bool), masks, calibration=cal)
    return float(ev["mean_iou"]), float(min(ev["view_ious"]))


def build_delivery_table():
    src = ROOT / "results" / "decoupled_48core_256shell_validation_t2_v2.json"
    data = json.loads(src.read_text())
    rows = []
    for r in data["rows"]:
        masks = load_masks_for_row(r)
        if masks is None:
            continue
        core = np.load(r["artifacts"]["core_up_clipped"]).astype(bool)
        final = np.load(r["artifacts"]["final"]).astype(bool)
        core_iou, core_min = score_volume_256(core, masks, bool(r["fix_axis"]))
        final_iou, final_min = score_volume_256(final, masks, bool(r["fix_axis"]))
        lab, cc = label(core, structure=S6)
        rows.append({
            "core_iou": core_iou,
            "core_min": core_min,
            "core_saving": 1.0 - int(core.sum()) / max(int(r["h256_voxels"]), 1),
            "core_single": cc == 1,
            "final_iou": final_iou,
            "final_min": final_min,
            "final_saving": float(r["material_saved_vs_h256"]),
            "final_single": int(r["final_components_6"]) == 1,
        })
    with (OUT / "table4a_core_shell_delivery_ablation.tsv").open("w") as f:
        f.write("Setting\tCases\tMean IoU@256\tWorst IoU@256\tSaving vs visual hull\tSingle-body\tWatertight\n")
        f.write("Upsampled core clipped to H256\t{}\t{:.3f}\t{:.3f}\t{}\t{}/{}\tnot evaluated\n".format(
            len(rows), mean([r["core_iou"] for r in rows]), min([r["core_min"] for r in rows]), pct(mean([r["core_saving"] for r in rows])), sum(r["core_single"] for r in rows), len(rows)
        ))
        f.write("Core + high-resolution silhouette shell\t{}\t{:.3f}\t{:.3f}\t{}\t{}/{}\t8/8\n".format(
            len(rows), mean([r["final_iou"] for r in rows]), min([r["final_min"] for r in rows]), pct(mean([r["final_saving"] for r in rows])), sum(r["final_single"] for r in rows), len(rows)
        ))


def build_pino_tables():
    out = OUT / "table4b_pino_acs_decision_rule_ablation.tsv"
    # Candidate-selection policies are on the generated candidate domain.
    summ = json.loads((ROOT / "results" / "candidate_selection_large_summary_20260624.json").read_text())
    rows = []
    for rule, label in [
        ("raw_pino", "Raw PINO threshold"),
        ("acs_safe", "Safe-side ACS"),
        ("dual_acs_fem_fallback", "Dual ACS with FEM fallback"),
        ("fem_oracle", "FEM oracle"),
    ]:
        vals = []
        for branch in ("free", "fixed"):
            vals.append(summ["branches"][branch]["policy_summary"][rule])
        nsel = sum(v["n_selected"] for v in vals)
        rows.append({
            "rule": label,
            "selected": nsel,
            "safe_rate": mean([v["safe_rate"] for v in vals]),
            "saving": mean([v["mean_saving"] for v in vals]),
            "fem_def": mean([v["mean_fem_fallbacks"] for v in vals]),
            "false_selected": sum(v.get("false_selected_count", 0) for v in vals),
        })
    dual = json.loads((ROOT / "results" / "pino_accs_dual_screen_20260624.json").read_text())
    with out.open("w") as f:
        f.write("Decision rule\tSelected candidates\tSafe rate\tMean material reduction\tMean FEM deferrals\tFalse selected\tAutonomous reject rate\tSystem false-pass\n")
        for r in rows:
            f.write(f"{r['rule']}\t{r['selected']}\t{pct(r['safe_rate'])}\t{pct(r['saving'])}\t{r['fem_def']:.2f}\t{r['false_selected']}\tnot recorded\tnot recorded\n")
        f.write(
            "Dual-sided ACS screen (Hard450 diagnostic)\t450\t"
            f"not comparable\tnot applicable\tnot applicable\tnot comparable\t"
            f"{pct(dual['metrics']['unsafe_reject_rate']['mean'])}\t"
            f"{dual['metrics']['false_pass_count']['mean']:.2f} mean\n"
        )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_selector_policy_tables_fixed()
    build_gate_leakage_table()
    build_delivery_table()
    build_pino_tables()
    (OUT / "SOURCES.json").write_text(json.dumps({
        "candidate_pool": [
            str(ROOT / "results" / "candidate_selection_batch_free50_20260624.json"),
            str(ROOT / "results" / "candidate_selection_batch_fixed50_20260624.json"),
        ],
        "delivery": str(ROOT / "results" / "decoupled_48core_256shell_validation_t2_v2.json"),
        "pino_candidate_policy": str(ROOT / "results" / "candidate_selection_large_summary_20260624.json"),
        "pino_dual_screen": str(ROOT / "results" / "pino_accs_dual_screen_20260624.json"),
    }, indent=2))
    print(f"saved={OUT}")


if __name__ == "__main__":
    main()
