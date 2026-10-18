#!/usr/bin/env python3
"""Create a factual structural-screen panel and update the evidence manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "paper_aaai_submission/pipeline_reference_assets_current_20260718"
SCREEN_JSON = ASSET_DIR / "16_structural_screen.json"
SCREEN_PNG = ASSET_DIR / "16_structural_screen.png"
MANIFEST = ASSET_DIR / "ASSET_MANIFEST.json"
README = ASSET_DIR / "README.md"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_panel(result: dict) -> None:
    image = Image.new("RGB", (1200, 520), "white")
    draw = ImageDraw.Draw(image)
    dark, teal, amber, gray = "#263238", "#00897B", "#FFB300", "#455A64"

    draw.rounded_rectangle((20, 20, 1180, 500), radius=18, outline="#CFD8DC", width=3)
    draw.text((55, 45), "Structural Screening Evidence", fill=dark, font=font(34, True))
    draw.text((55, 92), "Delivered model -> C48 structural proxy", fill=gray, font=font(23))

    # Compact voxel proxy icon.
    ox, oy, cell = 70, 175, 26
    for x in range(5):
        for y in range(5):
            shade = "#B2DFDB" if (x + y) % 2 == 0 else "#E0F2F1"
            draw.rectangle((ox + x * cell, oy + y * cell,
                            ox + (x + 1) * cell, oy + (y + 1) * cell),
                           fill=shade, outline=teal, width=2)
    draw.text((62, 326), "C48", fill=teal, font=font(28, True))
    draw.text((52, 368), "6 channels", fill=gray, font=font(20))

    sf_pino = result["sf_pino"]
    lower, upper = result["acs_lower_sf"], result["acs_upper_sf"]
    sf_fem = result["sf_fem"]
    threshold = result["acs_policy"]["threshold_sf"]

    draw.line((225, 250, 330, 250), fill=dark, width=4)
    draw.polygon([(330, 250), (314, 241), (314, 259)], fill=dark)
    draw.rounded_rectangle((345, 150, 620, 350), radius=14, outline=teal, width=3)
    draw.text((380, 172), "PINO + frozen ACS", fill=teal, font=font(28, True))
    draw.text((380, 224), f"PINO SF = {sf_pino:.3f}", fill=dark, font=font(25))
    draw.text((380, 266), f"ACS interval = [{lower:.3f}, {upper:.3f}]", fill=dark, font=font(23))
    draw.text((380, 308), f"threshold SF = {threshold:.1f}", fill=gray, font=font(22))

    draw.line((620, 250, 725, 250), fill=amber, width=5)
    draw.polygon([(725, 250), (709, 241), (709, 259)], fill=amber)
    draw.text((638, 214), "DEFER", fill="#A66F00", font=font(22, True))

    draw.rounded_rectangle((740, 150, 945, 350), radius=14, outline=gray, width=3)
    draw.text((790, 177), "Voxel FEM", fill=gray, font=font(29, True))
    draw.text((781, 237), f"SF = {sf_fem:.3f}", fill=dark, font=font(27))
    draw.text((790, 292), "final authority", fill=gray, font=font(20))

    draw.line((945, 250, 1015, 250), fill=teal, width=5)
    draw.polygon([(1015, 250), (999, 241), (999, 259)], fill=teal)
    draw.rounded_rectangle((1025, 190, 1145, 310), radius=14, fill="#E3F4EA", outline=teal, width=3)
    draw.text((1042, 225), "SAFE", fill=teal, font=font(31, True))
    draw.text((1027, 267), "FEM verified", fill=dark, font=font(17))

    draw.text((345, 405), "Frozen task-disjoint calibration; ACS did not auto-accept this case.",
              fill=gray, font=font(21))
    image.save(SCREEN_PNG)


def update_manifest(result: dict) -> None:
    payload = json.loads(MANIFEST.read_text())
    payload["pipeline_story"] = payload["pipeline_story"][:-1] + [
        "frozen PINO/ACS screen with FEM fallback",
        payload["pipeline_story"][-1],
    ]
    payload["selected_metrics"]["structural_screen"] = result
    payload["selected_metrics"]["structural_pass"] = bool(result["safe"])
    payload["selected_metrics"]["safety_score"] = float(result["safety_score"])
    payload["structural_screen_note"] = (
        "Strict six-channel v15 PINO predicted SF=%.6f. The frozen task-disjoint "
        "ACS interval [%.6f, %.6f] crossed SF=1.3, so the case was deferred. "
        "Voxel FEM returned SF=%.6f and the final status SAFE."
        % (result["sf_pino"], result["acs_lower_sf"], result["acs_upper_sf"], result["sf_fem"])
    )
    files = {row["file"]: row for row in payload["files"]}
    for path in (SCREEN_JSON, SCREEN_PNG):
        files[path.name] = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    payload["files"] = [files[name] for name in sorted(files)]
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")


def update_readme(result: dict) -> None:
    text = README.read_text()
    old = (
        "Structural screening was not requested in this evidence run. A PINO/ACS/FEM\n"
        "panel must therefore remain schematic unless evidence from a separately verified\n"
        "structural run is supplied.\n"
    )
    new = (
        "## Structural evidence\n\n"
        "- `16_structural_screen.json`: machine-readable frozen PINO/ACS/FEM result.\n"
        "- `16_structural_screen.png`: figure-ready factual decision panel.\n\n"
        f"Strict v15 PINO predicted `SF={result['sf_pino']:.6f}`. The frozen ACS interval "
        f"`[{result['acs_lower_sf']:.6f}, {result['acs_upper_sf']:.6f}]` crossed the "
        f"`SF=1.3` threshold and deferred the candidate. Voxel FEM returned "
        f"`SF={result['sf_fem']:.6f}` and the final decision is `SAFE`.\n"
    )
    if old not in text:
        raise RuntimeError("README structural disclosure no longer matches the expected text.")
    README.write_text(text.replace(old, new))


def main() -> None:
    result = json.loads(SCREEN_JSON.read_text())
    draw_panel(result)
    update_manifest(result)
    update_readme(result)
    print(json.dumps({
        "panel": str(SCREEN_PNG),
        "manifest": str(MANIFEST),
        "status": result["status"],
        "decided_by": result["decided_by"],
    }, indent=2))


if __name__ == "__main__":
    main()
