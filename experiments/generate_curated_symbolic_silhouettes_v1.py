#!/usr/bin/env python3
"""Generate representative symbolic silhouettes: Chinese characters + AAAI Fig.1 targets."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "data/silhouettes/curated_symbolic_v1"
N = 512


def font_cjk(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    raise RuntimeError("No CJK font found")


def font_latin(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def normalize(mask: np.ndarray, margin: int = 46, fill_holes: bool = True) -> Image.Image:
    if fill_holes:
        mask = binary_fill_holes(mask)
    mask = binary_closing(mask, iterations=1)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return Image.new("L", (N, N), 0)
    crop = mask[max(0, ys.min() - 8): min(mask.shape[0], ys.max() + 9), max(0, xs.min() - 8): min(mask.shape[1], xs.max() + 9)]
    img = Image.fromarray((crop.astype(np.uint8) * 255), "L")
    img.thumbnail((N - 2 * margin, N - 2 * margin), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (N, N), 0)
    canvas.paste(img, ((N - img.width) // 2, (N - img.height) // 2))
    return canvas.point(lambda p: 255 if p > 96 else 0)


def text_mask(name: str, text: str, cjk: bool = True, fill_holes: bool = True) -> None:
    img = Image.new("L", (N, N), 0)
    draw = ImageDraw.Draw(img)
    fnt = font_cjk(390) if cjk else font_latin(390)
    bbox = draw.textbbox((0, 0), text, font=fnt)
    scale = min((N * 0.78) / max(bbox[2] - bbox[0], 1), (N * 0.78) / max(bbox[3] - bbox[1], 1))
    fnt = font_cjk(max(40, int(390 * scale))) if cjk else font_latin(max(40, int(390 * scale)))
    bbox = draw.textbbox((0, 0), text, font=fnt)
    x = (N - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (N - (bbox[3] - bbox[1])) / 2 - bbox[1]
    draw.text((x, y), text, fill=255, font=fnt)
    out = normalize(np.array(img) > 64, fill_holes=fill_holes)
    out.save(OUT / f"{name}.png")


def polygon(points: list[tuple[float, float]]) -> Image.Image:
    img = Image.new("L", (N, N), 0)
    ImageDraw.Draw(img).polygon([(int(x * N), int(y * N)) for x, y in points], fill=255)
    return img


def star() -> Image.Image:
    pts = []
    for i in range(10):
        r = 0.78 if i % 2 == 0 else 0.34
        a = -math.pi / 2 + i * math.pi / 5
        pts.append((0.5 + 0.5 * r * math.cos(a), 0.5 + 0.5 * r * math.sin(a)))
    return normalize(np.array(polygon(pts)) > 0)


def heart() -> Image.Image:
    yy, xx = np.mgrid[0:N, 0:N]
    x = (xx + 0.5) / N * 2 - 1
    y = (yy + 0.5) / N * 2 - 1
    xs = x * 1.18
    ys = -y * 1.18 + 0.12
    m = (xs * xs + ys * ys - 0.36) ** 3 - xs * xs * ys**3 <= 0
    return normalize(m, fill_holes=False)


def gear() -> Image.Image:
    yy, xx = np.mgrid[0:N, 0:N]
    x = (xx + 0.5) / N * 2 - 1
    y = (yy + 0.5) / N * 2 - 1
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    outer = 0.58 + 0.10 * (np.cos(10 * theta) > 0)
    m = ((r < outer) & (r > 0.19)) | ((np.abs(x) < 0.14) | (np.abs(y) < 0.14))
    m &= r < 0.78
    return normalize(m)


def tree() -> Image.Image:
    img = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(img)
    d.polygon([(256, 48), (82, 250), (430, 250)], fill=255)
    d.polygon([(256, 135), (50, 365), (462, 365)], fill=255)
    d.rectangle((205, 340, 307, 465), fill=255)
    return normalize(np.array(img) > 0)


def make_sheet() -> None:
    files = sorted(p for p in OUT.glob("*.png") if p.name != "CONTACT_SHEET.png")
    tiles = []
    for p in files:
        img = Image.open(p).convert("RGB").resize((140, 140))
        tile = Image.new("RGB", (180, 176), "white")
        tile.paste(img, (20, 0))
        ImageDraw.Draw(tile).text((4, 146), p.stem, fill=(0, 0, 0))
        tiles.append(tile)
    cols = 4
    rows = int(math.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * 180, rows * 176), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 180, (i // cols) * 176))
    sheet.save(OUT / "CONTACT_SHEET.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Chinese characters chosen for recognizable, stroke-rich silhouettes.
    for name, text in [
        ("han_ying_shadow", "影"),
        ("han_long_dragon", "龙"),
        ("han_shan_mountain", "山"),
        ("han_guang_light", "光"),
        ("han_zhi_intelligence", "智"),
        ("han_meng_dream", "梦"),
    ]:
        text_mask(name, text, cjk=True)
    for name, text, fill_holes in [
        ("aaai_A", "A", False),
        ("aaai_I", "I", True),
        ("aaai_2", "2", True),
        ("aaai_0", "0", False),
        ("aaai_7", "7", True),
    ]:
        text_mask(name, text, cjk=False, fill_holes=fill_holes)
    for name, fn in [("aaai_star", star), ("aaai_heart", heart), ("aaai_gear", gear), ("aaai_tree", tree)]:
        fn().save(OUT / f"{name}.png")
    make_sheet()
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
