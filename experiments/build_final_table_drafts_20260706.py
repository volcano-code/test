#!/usr/bin/env python3
"""Build paper table drafts without modifying the paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import skimage.measure
import trimesh


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "results/paper_table_drafts_20260706"


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def copy_existing_tables() -> None:
    src = ROOT / "results/paper_table_drafts_20260705"
    # Keep yesterday's figure companion table untouched; create revised main-table copies.
    for name in [
        "table1_controlled_shadow_art_baselines_20260705.tsv",
        "table1_controlled_shadow_art_baselines_per_example_20260705.tsv",
        "table1_nsa_style_controlled_comparison_20260705.tsv",
    ]:
        text = (src / name).read_text(encoding="utf-8")
        (OUT / name.replace("20260705", "20260706")).write_text(text, encoding="utf-8")


def build_table2() -> None:
    data = json.loads((ROOT / "results/paper_experiment_tables_20260623.json").read_text())
    names = {
        "visual_hull": "Visual hull",
        "gap_fill": "Visual hull + gap filling",
        "carve_no_robust_bridge": "w/o connectivity guard / bridge",
        "carve_plus_bridge": "w/o watertight repair",
        "full_pipeline": "Full geometry pipeline",
    }
    rows = []
    for r in data["table1_generation_ablation"]:
        n = int(r["n_cases"])
        rows.append(
            [
                names.get(r["stage"], r["stage"]),
                n,
                f"{r['iou']['mean']:.3f}",
                f"{r['iou']['min']:.3f}",
                pct(r["volume_material"]["material_saved_vs_hull"]),
                f"{round(r['single_body_6_rate'] * n)}/{n}",
                f"{round(r['watertight_rate'] * n)}/{n}",
                f"{r['runtime']['mean_elapsed_s']:.3f}",
            ]
        )
    write_tsv(
        OUT / "table2_discrete_geometry_ablation_20260706.tsv",
        [
            "Variant",
            "Cases",
            "Mean IoU",
            "Worst-view IoU",
            "Saving",
            "Single-body",
            "Watertight",
            "Runtime / case (s)",
        ],
        rows,
    )


def selected_policy_row(case: dict, policy: str) -> dict | None:
    selected = case.get("policies", {}).get(policy, {}).get("selected")
    if not selected:
        return None
    for cand in case["candidates"]:
        if cand["candidate"] == selected:
            return cand
    return None


def build_table3() -> None:
    rows = []
    all_cases = []
    for branch in ["free", "fixed"]:
        data = json.loads((ROOT / f"results/candidate_selection_batch_{branch}50_20260624.json").read_text())
        cases = data["cases"]
        all_cases.extend((branch, c) for c in cases)
        rows.append(candidate_group_row(branch, cases, selected_source=f"{data['policy_summary']['dual_acs_fem_fallback']['n_selected']}/{data['n_cases']}"))
    all_data = []
    for branch, case in all_cases:
        copied = dict(case)
        copied["_branch"] = branch
        all_data.append(copied)
    rows.insert(0, candidate_group_row("All cases", all_data, selected_source="free 37, fixed 29"))
    write_tsv(
        OUT / "table3_candidate_pool_selector_audit_20260706.tsv",
        [
            "Case group",
            "Cases",
            "Raw cand./case",
            "Fabrication-valid/case",
            "Structurally eligible/case",
            "Selected mIoU",
            "Selected saving",
            "Fixed/free selected",
            "FEM deferred / selected",
        ],
        rows,
    )


def candidate_group_row(label: str, cases: list[dict], selected_source: str) -> list[object]:
    raw = []
    fab = []
    elig = []
    selected = []
    fem_deferred = []
    for c in cases:
        cands = c["candidates"]
        raw.append(len(cands))
        fab_cands = [x for x in cands if x.get("component_count_6") == 1 and bool(x.get("watertight"))]
        elig_cands = [x for x in fab_cands if bool(x.get("fem_safe"))]
        fab.append(len(fab_cands))
        elig.append(len(elig_cands))
        sel = selected_policy_row(c, "dual_acs_fem_fallback")
        if sel is not None:
            selected.append(sel)
            fem_deferred.append(c.get("policies", {}).get("dual_acs_fem_fallback", {}).get("fem_fallbacks", 0))
    return [
        label,
        len(cases),
        f"{np.mean(raw):.2f}",
        f"{np.mean(fab):.2f}",
        f"{np.mean(elig):.2f}",
        f"{np.mean([s['mean_iou'] for s in selected]):.3f}" if selected else "NA",
        pct(float(np.mean([s["material_saved_vs_hull"] for s in selected]))) if selected else "NA",
        selected_source,
        f"{np.mean(fem_deferred):.2f}" if fem_deferred else "NA",
    ]


def mesh_watertight_from_npy(path: Path) -> bool:
    vol = np.load(path).astype(bool)
    if vol.sum() == 0:
        return False
    padded = np.pad(vol, 1, mode="constant", constant_values=False)
    verts, faces, _, _ = skimage.measure.marching_cubes(padded.astype(np.float32), level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return bool(mesh.is_watertight)


def build_table4a() -> None:
    data = json.loads((ROOT / "results/decoupled_48core_256shell_validation_t2_v2.json").read_text())
    rows_data = data["rows"]
    watertight = []
    for r in rows_data:
        final_path = Path(r["artifacts"]["final"])
        watertight.append(mesh_watertight_from_npy(final_path))
    n = len(rows_data)
    row = [
        "48^3 core + 256^3 shell",
        n,
        "48^3/97^3 stored cores",
        "256^3",
        f"{np.mean([r['final_mean_iou_256'] for r in rows_data]):.3f}",
        f"{min(r['final_mean_iou_256'] for r in rows_data):.3f}",
        pct(float(np.mean([r["material_saved_vs_h256"] for r in rows_data]))),
        f"{sum(r['final_components_6'] == 1 for r in rows_data)}/{n}",
        f"{sum(r['shell_attached_rate'] >= 0.95 for r in rows_data)}/{n}",
        f"{sum(watertight)}/{n}",
    ]
    write_tsv(
        OUT / "table4a_resolution_decoupled_delivery_20260706.tsv",
        [
            "Setting",
            "Cases",
            "Structural core",
            "Delivered resolution",
            "IoU@256",
            "Worst case IoU@256",
            "Saving vs H256",
            "Single-body",
            "Shell attached",
            "Watertight",
        ],
        [row],
    )
    (OUT / "table4a_watertight_checks.json").write_text(
        json.dumps(
            {
                "source": str(ROOT / "results/decoupled_48core_256shell_validation_t2_v2.json"),
                "checks": [
                    {"case": r["case"], "final": r["artifacts"]["final"], "watertight": w}
                    for r, w in zip(rows_data, watertight)
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_table4b() -> None:
    src = ROOT / "results/paper_table_drafts_20260705/table4b_pino_acs_fem_triage_draft.tsv"
    lines = src.read_text(encoding="utf-8").splitlines()
    (OUT / "table4b_pino_acs_fem_triage_20260706.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def include_soft_table() -> None:
    src = ROOT / "results/soft_res256_export_projection_metrics_20260706/soft_res256_export_projection_metrics.tsv"
    dst = OUT / "appendix_soft_res256_loss_ablation_20260706.tsv"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copy_existing_tables()
    build_table2()
    build_table3()
    build_table4a()
    build_table4b()
    include_soft_table()
    sources = {
        "table1": str(ROOT / "results/paper_table_drafts_20260705/table1_controlled_shadow_art_baselines_20260705.tsv"),
        "table1_figure_companion_kept": str(ROOT / "results/paper_table_drafts_20260705/table1_nsa_style_controlled_comparison_20260705.tsv"),
        "table2": str(ROOT / "results/paper_experiment_tables_20260623.json"),
        "table3": [
            str(ROOT / "results/candidate_selection_batch_free50_20260624.json"),
            str(ROOT / "results/candidate_selection_batch_fixed50_20260624.json"),
        ],
        "table4a": str(ROOT / "results/decoupled_48core_256shell_validation_t2_v2.json"),
        "table4b": str(ROOT / "results/pino_safe_accept90_cal300_m08_20260704_threshold_audit.json"),
        "soft_appendix": str(ROOT / "results/soft_res256_export_projection_metrics_20260706/soft_res256_export_projection_metrics.tsv"),
    }
    (OUT / "SOURCES.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
