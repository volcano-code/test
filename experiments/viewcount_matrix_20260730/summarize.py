#!/usr/bin/env python3
"""Summarize audited and newly generated view-count evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
TASKS = json.loads((EXPERIMENT / "tasks.json").read_text())
OUTPUT = ROOT / "results/viewcount_matrix_anime_ordinary_mixed_20260730"
COMMON_MEAN_IOU_GATE = 0.98
COMMON_MIN_IOU_GATE = 0.96


def absolute(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def read_case(entry: dict[str, Any]) -> dict[str, Any]:
    result_dir = absolute(entry["result_dir"])
    manifest_path = result_dir / "MANIFEST.json"
    row = {
        **entry,
        "result_dir": str(result_dir),
        "status": "PENDING",
        "manifest": str(manifest_path),
    }
    if not manifest_path.is_file():
        return row
    data = json.loads(manifest_path.read_text())
    selected = data.get("selected") or {}
    volume_metrics = data.get("selected_volume_metrics") or {}
    delivery_qa = data.get("delivery_qa") or {}
    metrics = selected or volume_metrics or delivery_qa
    row.update(
        {
            "status": data.get("status", "MANIFEST_PRESENT"),
            "mean_iou": first_present(metrics, "mean_iou"),
            "min_iou": first_present(metrics, "min_iou"),
            "mean_dice": first_present(metrics, "mean_dice"),
            "saving": first_present(metrics, "saving"),
            "components": first_present(delivery_qa, "components")
            if delivery_qa
            else first_present(metrics, "components"),
            "watertight": first_present(delivery_qa, "watertight")
            if delivery_qa
            else first_present(metrics, "watertight"),
            "candidate": selected.get("candidate")
            or data.get("experiment"),
            "display_stl": data.get("display_stl"),
            "print_stl": data.get("print_stl"),
            "raw_soft": data.get("raw_soft"),
            "largest_component": data.get("largest_component")
            or data.get("main_component"),
            "bounded_repair": data.get("bounded_repair"),
            "structural_screen": selected.get("structural_screen"),
        }
    )
    mean_iou = row.get("mean_iou")
    min_iou = row.get("min_iou")
    row["common_fidelity_pass"] = (
        mean_iou is not None
        and min_iou is not None
        and float(mean_iou) >= COMMON_MEAN_IOU_GATE
        and float(min_iou) >= COMMON_MIN_IOU_GATE
    )
    raw_soft = row.get("raw_soft") or {}
    raw_mean_iou = first_present(raw_soft, "mean_iou")
    raw_min_iou = first_present(raw_soft, "min_iou")
    row["raw_mean_iou"] = raw_mean_iou
    row["raw_min_iou"] = raw_min_iou
    row["delivery_mean_iou_drop"] = (
        float(raw_mean_iou) - float(mean_iou)
        if raw_mean_iou is not None and mean_iou is not None
        else None
    )
    row["delivery_min_iou_drop"] = (
        float(raw_min_iou) - float(min_iou)
        if raw_min_iou is not None and min_iou is not None
        else None
    )
    return row


def read_probe(entry: dict[str, Any]) -> dict[str, Any]:
    row = read_case(
        {
            **entry,
            "family": entry["id"].split("_", 1)[0],
            "views": 3,
            "evidence_origin": "probe",
            "protocol": "isolated_v256_owned_multiview",
            "structural_backend": "not evaluated in fidelity probe",
        }
    )
    row["source_soft"] = str(absolute(entry["source_soft"]))
    return row


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": TASKS["experiment"],
        "common_fidelity_gate": {
            "mean_iou_at_least": COMMON_MEAN_IOU_GATE,
            "min_iou_at_least": COMMON_MIN_IOU_GATE,
        },
        "matrix": [read_case(entry) for entry in TASKS["matrix"]],
        "probes": [read_probe(entry) for entry in TASKS["probes"]],
        "contribution_probes": [
            read_probe(entry)
            for entry in TASKS.get("contribution_probes", [])
        ],
    }
    summary_path = OUTPUT / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    columns = [
        "id",
        "family",
        "views",
        "evidence_origin",
        "status",
        "mean_iou",
        "min_iou",
        "mean_dice",
        "saving",
        "components",
        "watertight",
        "common_fidelity_pass",
        "raw_mean_iou",
        "raw_min_iou",
        "delivery_mean_iou_drop",
        "delivery_min_iou_drop",
        "display_stl",
    ]
    lines = ["\t".join(columns)]
    for row in summary["matrix"]:
        lines.append(
            "\t".join(str(row.get(column, "")) for column in columns)
        )
    table_path = OUTPUT / "SUMMARY.tsv"
    table_path.write_text("\n".join(lines) + "\n")

    probe_lines = ["\t".join(columns)]
    for row in summary["probes"] + summary["contribution_probes"]:
        probe_lines.append(
            "\t".join(str(row.get(column, "")) for column in columns)
        )
    probe_table_path = OUTPUT / "PROBES.tsv"
    probe_table_path.write_text("\n".join(probe_lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
