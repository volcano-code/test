#!/usr/bin/env python3
"""Generate public-safe anime-character archetype silhouettes.

These are original archetypes, not copies of copyrighted characters.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "data/silhouettes/anime_archetype_v1"
N = 512


def norm(mask: np.ndarray, margin: int = 34, dilate: int = 1) -> Image.Image:
    mask = binary_fill_holes(mask)
    mask = binary_closing(mask, iterations=1)
    if dilate:
        mask = binary_dilation(mask, iterations=dilate)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return Image.new("L", (N, N), 0)
    crop = mask[max(0, ys.min() - 8): min(mask.shape[0], ys.max() + 9),
                max(0, xs.min() - 8): min(mask.shape[1], xs.max() + 9)]
    img = Image.fromarray((crop.astype(np.uint8) * 255), "L")
    img.thumbnail((N - 2 * margin, N - 2 * margin), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (N, N), 0)
    canvas.paste(img, ((N - img.width) // 2, (N - img.height) // 2))
    return canvas.point(lambda p: 255 if p > 96 else 0)


def draw_base() -> ImageDraw.ImageDraw:
    img = Image.new("L", (N, N), 0)
    return img, ImageDraw.Draw(img)


def limb(d: ImageDraw.ImageDraw, xy, w=34):
    d.line(xy, fill=255, width=w, joint="curve")


def spiky_swordsman():
    img, d = draw_base()
    d.ellipse((205, 86, 307, 188), fill=255)
    for p in [(256, 30, 282, 106), (206, 58, 238, 122), (306, 62, 272, 122),
              (178, 96, 232, 136), (334, 100, 280, 138)]:
        d.polygon([(p[0], p[1]), (p[2], p[3]), ((p[0]+p[2])//2, p[1]+72)], fill=255)
    d.polygon([(222, 190), (290, 190), (326, 336), (190, 336)], fill=255)
    limb(d, [(198, 226), (132, 304), (92, 392)], 32)
    limb(d, [(314, 226), (382, 292), (422, 370)], 32)
    limb(d, [(228, 332), (200, 448)], 34)
    limb(d, [(286, 332), (330, 448)], 34)
    limb(d, [(350, 96), (150, 438)], 18)
    d.polygon([(142, 440), (172, 416), (164, 466)], fill=255)
    return norm(np.array(img) > 0)


def twin_tail_magical():
    img, d = draw_base()
    d.ellipse((208, 82, 304, 178), fill=255)
    d.ellipse((120, 120, 210, 284), fill=255)
    d.ellipse((302, 120, 392, 284), fill=255)
    d.polygon([(222, 180), (290, 180), (348, 362), (164, 362)], fill=255)
    limb(d, [(178, 228), (96, 170)], 28)
    limb(d, [(332, 226), (420, 158)], 28)
    limb(d, [(224, 360), (188, 454)], 32)
    limb(d, [(288, 360), (326, 454)], 32)
    limb(d, [(420, 160), (458, 78)], 12)
    d.polygon([(458, 58), (474, 92), (438, 82)], fill=255)
    return norm(np.array(img) > 0)


def ninja_runner():
    img, d = draw_base()
    d.ellipse((214, 94, 306, 184), fill=255)
    d.polygon([(208, 138), (324, 112), (300, 178)], fill=255)
    d.polygon([(230, 188), (318, 214), (286, 330), (176, 306)], fill=255)
    limb(d, [(186, 220), (100, 184)], 30)
    limb(d, [(292, 238), (414, 210)], 30)
    limb(d, [(210, 308), (118, 404)], 34)
    limb(d, [(278, 318), (386, 428)], 34)
    d.polygon([(400, 206), (468, 184), (422, 242)], fill=255)
    return norm(np.array(img) > 0)


def mecha_robot():
    img, d = draw_base()
    d.rectangle((202, 70, 310, 168), fill=255)
    d.polygon([(202, 70), (166, 24), (230, 70)], fill=255)
    d.polygon([(310, 70), (346, 24), (282, 70)], fill=255)
    d.rectangle((176, 184, 336, 342), fill=255)
    d.rectangle((92, 206, 176, 284), fill=255)
    d.rectangle((336, 206, 420, 284), fill=255)
    d.rectangle((192, 342, 242, 470), fill=255)
    d.rectangle((270, 342, 320, 470), fill=255)
    d.rectangle((152, 152, 360, 198), fill=255)
    return norm(np.array(img) > 0)


def caped_hero():
    img, d = draw_base()
    d.ellipse((210, 74, 302, 166), fill=255)
    d.polygon([(168, 174), (344, 174), (398, 450), (114, 450)], fill=255)
    d.polygon([(220, 178), (292, 178), (318, 352), (194, 352)], fill=0)
    d.polygon([(218, 184), (294, 184), (322, 330), (190, 330)], fill=255)
    limb(d, [(188, 226), (92, 300)], 32)
    limb(d, [(324, 226), (420, 300)], 32)
    limb(d, [(230, 344), (210, 466)], 32)
    limb(d, [(282, 344), (302, 466)], 32)
    return norm(np.array(img) > 0)


def sailor_warrior():
    img, d = draw_base()
    d.ellipse((210, 76, 302, 168), fill=255)
    d.pieslice((170, 50, 342, 220), 200, 340, fill=255)
    d.polygon([(212, 178), (300, 178), (340, 350), (172, 350)], fill=255)
    d.polygon([(184, 278), (328, 278), (390, 384), (122, 384)], fill=255)
    limb(d, [(188, 220), (118, 132)], 28)
    limb(d, [(324, 220), (394, 132)], 28)
    limb(d, [(220, 378), (192, 468)], 30)
    limb(d, [(292, 378), (322, 468)], 30)
    d.polygon([(256, 194), (230, 230), (282, 230)], fill=0)
    return norm(np.array(img) > 0)


def masked_thief():
    img, d = draw_base()
    d.ellipse((206, 78, 306, 178), fill=255)
    d.polygon([(198, 120), (314, 120), (330, 154), (182, 154)], fill=255)
    d.polygon([(236, 178), (306, 204), (292, 342), (188, 314)], fill=255)
    limb(d, [(200, 222), (96, 280)], 26)
    limb(d, [(298, 222), (426, 180)], 26)
    limb(d, [(214, 318), (148, 438)], 32)
    limb(d, [(276, 332), (386, 418)], 32)
    d.polygon([(426, 180), (466, 154), (450, 206)], fill=255)
    return norm(np.array(img) > 0)


def umbrella_girl():
    img, d = draw_base()
    d.pieslice((84, 20, 428, 252), 180, 360, fill=255)
    limb(d, [(256, 128), (256, 430)], 14)
    d.ellipse((212, 168, 300, 256), fill=255)
    d.polygon([(214, 258), (298, 258), (350, 408), (162, 408)], fill=255)
    limb(d, [(210, 304), (130, 356)], 26)
    limb(d, [(302, 304), (382, 356)], 26)
    limb(d, [(226, 406), (204, 476)], 28)
    limb(d, [(286, 406), (314, 476)], 28)
    return norm(np.array(img) > 0)


def samurai_mask():
    img, d = draw_base()
    d.polygon([(172, 92), (340, 92), (310, 210), (202, 210)], fill=255)
    d.polygon([(172, 92), (106, 54), (190, 152)], fill=255)
    d.polygon([(340, 92), (406, 54), (322, 152)], fill=255)
    d.polygon([(210, 214), (302, 214), (340, 360), (172, 360)], fill=255)
    limb(d, [(178, 240), (92, 336)], 30)
    limb(d, [(330, 240), (420, 336)], 30)
    limb(d, [(232, 360), (208, 468)], 32)
    limb(d, [(282, 360), (312, 468)], 32)
    limb(d, [(118, 88), (408, 432)], 18)
    return norm(np.array(img) > 0)


def chibi_mascot():
    img, d = draw_base()
    d.ellipse((154, 74, 358, 278), fill=255)
    d.polygon([(180, 92), (120, 30), (220, 82)], fill=255)
    d.polygon([(332, 92), (392, 30), (292, 82)], fill=255)
    d.ellipse((186, 280, 326, 420), fill=255)
    d.ellipse((102, 292, 190, 378), fill=255)
    d.ellipse((322, 292, 410, 378), fill=255)
    d.ellipse((176, 400, 238, 472), fill=255)
    d.ellipse((274, 400, 336, 472), fill=255)
    return norm(np.array(img) > 0)


def winged_idol():
    img, d = draw_base()
    d.ellipse((212, 76, 300, 164), fill=255)
    d.polygon([(218, 166), (294, 166), (342, 352), (170, 352)], fill=255)
    for side in [-1, 1]:
        pts = [(256, 210), (256 + side * 210, 120), (256 + side * 160, 252), (256 + side * 220, 336)]
        d.polygon(pts, fill=255)
    limb(d, [(220, 352), (198, 470)], 30)
    limb(d, [(292, 352), (318, 470)], 30)
    d.polygon([(256, 178), (224, 212), (288, 212)], fill=0)
    return norm(np.array(img) > 0)


def magic_beast():
    img, d = draw_base()
    d.ellipse((144, 170, 330, 334), fill=255)
    d.ellipse((286, 122, 398, 232), fill=255)
    d.polygon([(306, 132), (260, 42), (350, 118)], fill=255)
    d.polygon([(374, 132), (430, 48), (414, 154)], fill=255)
    d.polygon([(398, 172), (462, 188), (402, 218)], fill=255)
    for x in [178, 246, 314]:
        limb(d, [(x, 312), (x - 28, 450)], 30)
    d.polygon([(146, 218), (52, 150), (102, 258)], fill=255)
    return norm(np.array(img) > 0)


SHAPES = {
    "anime_spiky_swordsman": spiky_swordsman,
    "anime_twin_tail_magical": twin_tail_magical,
    "anime_ninja_runner": ninja_runner,
    "anime_mecha_robot": mecha_robot,
    "anime_caped_hero": caped_hero,
    "anime_sailor_warrior": sailor_warrior,
    "anime_masked_thief": masked_thief,
    "anime_umbrella_girl": umbrella_girl,
    "anime_samurai_mask": samurai_mask,
    "anime_chibi_mascot": chibi_mascot,
    "anime_winged_idol": winged_idol,
    "anime_magic_beast": magic_beast,
}


def make_sheet() -> None:
    files = sorted(p for p in OUT.glob("*.png") if p.name != "CONTACT_SHEET.png")
    cols = 4
    tile_w, tile_h = 190, 176
    rows = int(math.ceil(len(files) / cols))
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    for i, p in enumerate(files):
        img = Image.open(p).convert("RGB").resize((138, 138))
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        tile.paste(img, (26, 0))
        ImageDraw.Draw(tile).text((4, 146), p.stem, fill=(0, 0, 0))
        sheet.paste(tile, ((i % cols) * tile_w, (i // cols) * tile_h))
    sheet.save(OUT / "CONTACT_SHEET.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in SHAPES.items():
        fn().save(OUT / f"{name}.png")
    make_sheet()
    print(f"DONE -> {OUT}")
    print("n", len(SHAPES))


if __name__ == "__main__":
    main()
