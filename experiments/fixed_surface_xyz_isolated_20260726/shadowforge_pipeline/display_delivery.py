#!/usr/bin/env python3
"""Optional display-only delivery assets for an exported ShadowForge model.

The fixture is intentionally a separate part.  It is not part of the shadow
solid and must not be included in projection, fidelity, or material results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def _box(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=np.asarray(extents, dtype=float), transform=trimesh.transformations.translation_matrix(center))


def create_display_delivery(
    model_path: Path,
    output_dir: Path,
    margin_mm: float = 8.0,
    base_thickness_mm: float = 4.0,
    clearance_mm: float = 0.8,
    retainer_height_mm: float = 1.5,
) -> dict:
    """Create a separate plinth and a translated display model.

    The four small retaining tabs sit outside the model bounding box.  They
    prevent gross lateral slip but are not a substitute for a fitted clamp or
    a structural certification.  The model itself remains geometrically
    unchanged, apart from a rigid translation onto the plinth.
    """
    if margin_mm <= 0 or base_thickness_mm <= 0 or clearance_mm < 0 or retainer_height_mm < 0:
        raise ValueError("Fixture dimensions must be non-negative, with positive margin and base thickness")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = trimesh.load(model_path, force="mesh")
    if not isinstance(model, trimesh.Trimesh) or model.is_empty:
        raise ValueError(f"Could not load a non-empty mesh from {model_path}")

    model.remove_unreferenced_vertices()
    bounds = np.asarray(model.bounds, dtype=float)
    extents = bounds[1] - bounds[0]
    # Place the lowest model point on the top of the separate plinth.
    translation = np.array([-bounds[:, 0].mean(), -bounds[:, 1].mean(), base_thickness_mm - bounds[0, 2]])
    placed = model.copy()
    placed.apply_translation(translation)
    placed_bounds = np.asarray(placed.bounds, dtype=float)

    base_x = float(extents[0] + 2 * margin_mm)
    base_y = float(extents[1] + 2 * margin_mm)
    plinth_parts = [_box((base_x, base_y, base_thickness_mm), (0.0, 0.0, base_thickness_mm / 2.0))]

    # Four external tabs make the display model repeatably positionable while
    # avoiding any boolean union with the projection-verified object.
    tab_width = max(2.0, min(4.0, margin_mm * 0.5))
    tab_depth = max(2.0, min(6.0, margin_mm * 0.75))
    x0, y0 = placed_bounds[0, 0] - clearance_mm, placed_bounds[0, 1] - clearance_mm
    x1, y1 = placed_bounds[1, 0] + clearance_mm, placed_bounds[1, 1] + clearance_mm
    zc = base_thickness_mm + retainer_height_mm / 2.0
    for x, y, sx, sy in ((x0, y0, 1, 1), (x0, y1, 1, -1), (x1, y0, -1, 1), (x1, y1, -1, -1)):
        plinth_parts.append(_box((tab_width, tab_depth, retainer_height_mm),
                                 (x + sx * tab_width / 2.0, y + sy * tab_depth / 2.0, zc)))
    plinth = trimesh.util.concatenate(plinth_parts)

    model_out = output_dir / "display_model_on_plinth.stl"
    plinth_out = output_dir / "display_plinth_separate.stl"
    placed.export(model_out)
    plinth.export(plinth_out)
    record = {
        "schema": "shadowforge_display_delivery_v1",
        "purpose": "display only; fixture is excluded from projection and paper metrics",
        "source_model": str(model_path.resolve()),
        "display_model": str(model_out),
        "separate_plinth": str(plinth_out),
        "model_translation_mm": translation.tolist(),
        "model_extents_mm": extents.tolist(),
        "plinth_extents_mm": [base_x, base_y, base_thickness_mm + retainer_height_mm],
        "clearance_mm": float(clearance_mm),
        "notes": [
            "Print the model and plinth as separate parts.",
            "Do not include the plinth in wall-shadow evaluation.",
            "Perform a physical fit test before a public display; the tabs are locating stops, not a load-bearing clamp.",
        ],
    }
    (output_dir / "DISPLAY_DELIVERY.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--margin-mm", type=float, default=8.0)
    ap.add_argument("--base-thickness-mm", type=float, default=4.0)
    ap.add_argument("--clearance-mm", type=float, default=0.8)
    ap.add_argument("--retainer-height-mm", type=float, default=1.5)
    args = ap.parse_args()
    print(json.dumps(create_display_delivery(
        model_path=args.model,
        output_dir=args.output_dir,
        margin_mm=args.margin_mm,
        base_thickness_mm=args.base_thickness_mm,
        clearance_mm=args.clearance_mm,
        retainer_height_mm=args.retainer_height_mm,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
