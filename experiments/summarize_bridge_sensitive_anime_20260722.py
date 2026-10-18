#!/usr/bin/env python3
"""Build the bridge-sensitive anime pre-screen used before Table 2b."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
CASES = {
    "dark_knight": ROOT / "results/dark_knight_current_pipeline_recheck_20260722",
    "sword_fox": ROOT / "results/physical_print_soft_geometry_20260714/H02_sword_fox",
    "dark_mage_fox": ROOT / "results/physical_print_soft_geometry_20260714/H03_dark_mage_fox",
}


def load_stages(run: Path) -> dict[str, dict]:
    stages = {}
    for path in (run / "candidate_evidence").glob("*/stage.json"):
        row = json.loads(path.read_text())
        stages[row["stage"]] = row
    return stages


def main() -> None:
    rows = []
    for task, run in CASES.items():
        stages = load_stages(run)
        choices = []
        for name, raw in stages.items():
            if not name.endswith("_raw") or int(raw.get("components", 0)) <= 1:
                continue
            bridge = stages.get(name[:-4] + "_bridge6")
            if bridge is None or int(bridge.get("components", 0)) != 1:
                continue
            choices.append((float(bridge.get("mean_iou", 0.0)), name, raw, bridge))
        if not choices:
            raise RuntimeError(f"No bridge-sensitive seed found for {task}: {run}")
        _, name, raw, bridge = max(choices)
        rows.append({
            "task": task,
            "run": str(run),
            "seed": name.removesuffix("_raw"),
            "raw_components": int(raw["components"]),
            "bridged_components": int(bridge["components"]),
            "raw_iou": float(raw["mean_iou"]),
            "bridged_iou": float(bridge["mean_iou"]),
            "bridged_saving": float(bridge["saving"]),
        })
    output = ROOT / "results/table2b_bridge_sensitive_anime_prescreen_20260722.json"
    output.write_text(json.dumps({
        "purpose": "pre-screen only; identifies tasks where bridge repair is active",
        "rows": rows,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
