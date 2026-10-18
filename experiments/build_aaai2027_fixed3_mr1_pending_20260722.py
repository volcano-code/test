#!/usr/bin/env python3
"""Build four pending fixed3/mr1 character cases from A, I, 2, 0, and 7."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_fixed3_mr1_connected10_pending_20260722 import (
    S4_2D,
    component_count,
    load_mask,
    save_mask,
    triple_audit,
)


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SOURCE = ROOT / "data/silhouettes/curated_symbolic_v1"
OUTPUT = ROOT / "results/aaai2027_fixed3_mr1_special4_pending_20260722"
SYMBOLS = {name: SOURCE / f"aaai_{name}.png" for name in ("A", "I", "2", "0", "7")}


def make_sheet(cases: list[dict], destination: Path) -> None:
    tile, label_width, header = 180, 170, 42
    canvas = Image.new("RGB", (label_width + 3 * tile, header + len(cases) * tile), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, axis in enumerate(("X", "Y", "Z")):
        draw.text((label_width + column * tile + 75, 14), f"View {axis}", fill="black", font=font)
    for row, case in enumerate(cases):
        y = header + row * tile
        draw.text((10, y + 58), case["case_id"], fill="black", font=font)
        draw.text((10, y + 78), " / ".join(case["symbols_xyz"]), fill=(0, 105, 90), font=font)
        draw.text((10, y + 98), "inputs 1 comp; hull 1", fill=(0, 105, 90), font=font)
        for column, path in enumerate(case["target_paths_xyz"]):
            image = Image.open(path).convert("RGB").resize((tile - 12, tile - 12), Image.Resampling.NEAREST)
            panel = Image.new("RGB", (tile, tile), "black")
            panel.paste(image, (6, 6))
            canvas.paste(panel, (label_width + column * tile, y))
    canvas.save(destination)


def main() -> None:
    missing = [str(path) for path in SYMBOLS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Generate symbolic masks first: {missing}")
    rows = {
        name: {
            "path": str(path.resolve()),
            "symbol": name,
        }
        for name, path in SYMBOLS.items()
    }
    candidates = []
    for symbols in itertools.permutations(SYMBOLS, 3):
        triple = tuple(rows[name] for name in symbols)
        audit = triple_audit(triple, 256)
        if audit is not None:
            candidates.append((audit["min_iou"], audit["mean_iou"], symbols, audit))
    candidates.sort(reverse=True)

    selected = []
    used_symbol_sets = set()
    covered = set()
    while len(selected) < 4:
        best = None
        for candidate in candidates:
            _, _, symbols, _ = candidate
            signature = frozenset(symbols)
            if signature in used_symbol_sets:
                continue
            coverage_gain = len(set(symbols) - covered)
            score = (coverage_gain, candidate[0], candidate[1])
            if best is None or score > best[0]:
                best = (score, candidate)
        if best is None:
            raise RuntimeError("Could not select four distinct connected symbol triples")
        candidate = best[1]
        selected.append(candidate)
        used_symbol_sets.add(frozenset(candidate[2]))
        covered.update(candidate[2])

    inputs = OUTPUT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, (_, _, symbols, audit) in enumerate(selected, 1):
        case_id = f"aaai2027_{index:02d}"
        paths = []
        views = []
        for axis, symbol in zip(("x", "y", "z"), symbols, strict=True):
            mask = load_mask(SYMBOLS[symbol], 256)
            destination = inputs / f"{case_id}_view_{axis}_{symbol}.png"
            save_mask(mask, destination)
            paths.append(str(destination.resolve()))
            views.append({
                "axis": axis,
                "symbol": symbol,
                "components_4": component_count(mask, S4_2D),
                "mirrored_components_4": component_count(np.fliplr(mask), S4_2D),
            })
        cases.append({
            "case_id": case_id,
            "stratum": "AAAI-2027-connected-symbols",
            "symbols_xyz": list(symbols),
            "target_paths_xyz": paths,
            "sadekar_tfs_targets": [paths[1], paths[0], paths[2]],
            "input_connectivity_audit": views,
            "visual_hull_audit": audit,
        })

    payload = {
        "schema": "aaai2027_fixed3_mr1_special4_pending_v1",
        "status": "PENDING_USER_APPROVAL",
        "protocol": {
            "views": 3,
            "declared_rendered_views": 6,
            "axes": ["x", "y", "z"],
            "camera": "fixed orthographic TFS",
            "tsf": 1,
            "mr": 1,
            "zdist": 1.7,
            "resolution": 256,
            "seeds": [17, 29, 43],
        },
        "selection_policy": "Input-only connectedness and visual-hull compatibility; no method output used.",
        "cases": cases,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT / "benchmark_manifest.pending.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    make_sheet(cases, OUTPUT / "special4_contact_sheet.png")
    print(json.dumps({
        "output": str(OUTPUT),
        "status": payload["status"],
        "cases": [{
            "case_id": case["case_id"],
            "symbols_xyz": case["symbols_xyz"],
            "min_hull_iou": case["visual_hull_audit"]["min_iou"],
            "mean_hull_iou": case["visual_hull_audit"]["mean_iou"],
        } for case in cases],
    }, indent=2))


if __name__ == "__main__":
    main()
