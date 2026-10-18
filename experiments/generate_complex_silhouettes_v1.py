#!/usr/bin/env python3
"""Generate recognizable complex binary silhouettes for physical experiments."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path("/home/lin/xvxinyao/ShadowForge")
OUT = ROOT / "data/silhouettes/complex_v1"
SHEET = OUT / "CONTACT_SHEET.png"
N = 512


def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("L", (N, N), 0)
    return img, ImageDraw.Draw(img)


def save(name: str, img: Image.Image) -> None:
    img.save(OUT / f"{name}.png")


def poly(draw: ImageDraw.ImageDraw, pts, fill=255):
    draw.polygon([(int(x), int(y)) for x, y in pts], fill=fill)


def dragon() -> Image.Image:
    img, d = canvas()
    d.ellipse((130, 235, 370, 350), fill=255)
    d.polygon([(350, 250), (450, 205), (410, 270)], fill=255)
    d.ellipse((410, 190, 475, 250), fill=255)
    d.polygon([(455, 180), (488, 142), (474, 197)], fill=255)
    d.polygon([(430, 188), (418, 128), (454, 190)], fill=255)
    d.polygon([(360, 250), (500, 252), (395, 300)], fill=255)
    d.polygon([(170, 250), (85, 155), (250, 232)], fill=255)
    d.polygon([(230, 248), (305, 115), (355, 270)], fill=255)
    d.polygon([(140, 320), (48, 350), (150, 285)], fill=255)
    for x in [185, 245, 305]:
        d.polygon([(x, 338), (x + 22, 420), (x + 44, 338)], fill=255)
    for x in range(190, 350, 36):
        d.polygon([(x, 230), (x + 16, 185), (x + 36, 235)], fill=255)
    return img


def castle() -> Image.Image:
    img, d = canvas()
    d.rectangle((95, 245, 420, 410), fill=255)
    for x, w, h in [(105, 70, 210), (220, 82, 170), (345, 70, 225)]:
        d.rectangle((x, h, x + w, 410), fill=255)
        poly(d, [(x - 8, h), (x + w / 2, h - 55), (x + w + 8, h)])
        for k in range(3):
            d.rectangle((x + 8 + k * 20, h - 12, x + 20 + k * 20, h + 18), fill=0)
    for x in range(105, 400, 42):
        d.rectangle((x, 220, x + 20, 250), fill=255)
    d.rounded_rectangle((230, 320, 285, 410), radius=25, fill=0)
    d.ellipse((140, 280, 170, 318), fill=0)
    d.ellipse((345, 285, 375, 323), fill=0)
    return img


def ship() -> Image.Image:
    img, d = canvas()
    poly(d, [(70, 340), (430, 340), (370, 405), (130, 405)])
    d.rectangle((245, 110, 258, 340), fill=255)
    d.rectangle((165, 175, 176, 340), fill=255)
    d.rectangle((330, 205, 340, 340), fill=255)
    poly(d, [(180, 185), (245, 128), (245, 302), (180, 302)])
    poly(d, [(263, 140), (335, 215), (263, 300)])
    poly(d, [(85, 250), (162, 195), (162, 320)])
    poly(d, [(345, 220), (435, 275), (345, 325)])
    for x in range(150, 350, 42):
        d.ellipse((x, 358, x + 24, 382), fill=0)
    return img


def robot() -> Image.Image:
    img, d = canvas()
    d.rounded_rectangle((165, 110, 345, 235), radius=25, fill=255)
    d.rounded_rectangle((130, 240, 380, 390), radius=22, fill=255)
    d.rectangle((245, 65, 265, 110), fill=255)
    d.ellipse((235, 45, 275, 80), fill=255)
    d.ellipse((205, 155, 235, 185), fill=0)
    d.ellipse((275, 155, 305, 185), fill=0)
    d.rectangle((220, 205, 290, 218), fill=0)
    d.rectangle((75, 255, 130, 295), fill=255)
    d.rectangle((380, 255, 435, 295), fill=255)
    d.rectangle((155, 390, 205, 455), fill=255)
    d.rectangle((305, 390, 355, 455), fill=255)
    for x in [170, 225, 280, 335]:
        d.ellipse((x, 285, x + 30, 315), fill=0)
    return img


def bonsai() -> Image.Image:
    img, d = canvas()
    d.rectangle((190, 370, 330, 430), fill=255)
    d.polygon([(170, 350), (350, 350), (330, 385), (190, 385)], fill=255)
    d.line([(260, 350), (250, 290), (220, 250), (245, 200)], fill=255, width=35)
    d.line([(250, 300), (180, 275), (135, 230)], fill=255, width=24)
    d.line([(250, 285), (320, 245), (365, 205)], fill=255, width=24)
    for box in [(85, 180, 220, 285), (170, 130, 315, 245), (285, 155, 430, 270), (120, 245, 285, 335)]:
        d.ellipse(box, fill=255)
    return img


def phoenix() -> Image.Image:
    img, d = canvas()
    d.ellipse((220, 180, 305, 330), fill=255)
    d.ellipse((260, 115, 330, 180), fill=255)
    d.polygon([(310, 130), (382, 155), (320, 170)], fill=255)
    d.polygon([(230, 215), (60, 100), (178, 270)], fill=255)
    d.polygon([(292, 215), (455, 105), (340, 270)], fill=255)
    for pts in [
        [(235, 315), (165, 470), (260, 350)],
        [(260, 320), (252, 490), (292, 345)],
        [(285, 315), (360, 470), (300, 348)],
    ]:
        poly(d, pts)
    d.polygon([(285, 115), (300, 70), (315, 120)], fill=255)
    return img


def cathedral() -> Image.Image:
    img, d = canvas()
    d.rectangle((125, 235, 390, 420), fill=255)
    for x in [135, 220, 305]:
        d.rectangle((x, 165, x + 70, 420), fill=255)
        poly(d, [(x - 8, 165), (x + 35, 90), (x + 78, 165)])
    for x in [155, 245, 330]:
        d.rounded_rectangle((x, 260, x + 32, 345), radius=16, fill=0)
    d.rounded_rectangle((230, 325, 285, 420), radius=28, fill=0)
    for x in range(135, 370, 38):
        d.rectangle((x, 220, x + 18, 248), fill=0)
    return img


def bicycle() -> Image.Image:
    img, d = canvas()
    d.ellipse((80, 295, 205, 420), outline=255, width=28)
    d.ellipse((310, 295, 435, 420), outline=255, width=28)
    d.line([(145, 355), (245, 245), (370, 355), (220, 355), (145, 355), (245, 245), (290, 355)], fill=255, width=24)
    d.line([(245, 245), (235, 190), (290, 190)], fill=255, width=20)
    d.line([(370, 355), (345, 230), (405, 215)], fill=255, width=18)
    d.line([(220, 355), (250, 395)], fill=255, width=18)
    d.ellipse((230, 335, 270, 375), fill=255)
    return img


def owl_totem() -> Image.Image:
    img, d = canvas()
    d.ellipse((135, 90, 375, 405), fill=255)
    d.polygon([(150, 125), (105, 55), (210, 105)], fill=255)
    d.polygon([(360, 125), (405, 55), (300, 105)], fill=255)
    d.ellipse((180, 165, 240, 225), fill=0)
    d.ellipse((275, 165, 335, 225), fill=0)
    d.polygon([(250, 220), (225, 270), (285, 270)], fill=0)
    for y in [285, 330]:
        for x in [180, 225, 270, 315]:
            d.ellipse((x, y, x + 30, y + 30), fill=0)
    d.rectangle((180, 400, 330, 430), fill=255)
    return img


def stag() -> Image.Image:
    img, d = canvas()
    d.ellipse((135, 230, 380, 345), fill=255)
    d.ellipse((330, 170, 420, 250), fill=255)
    for x in [180, 245, 315]:
        d.rectangle((x, 330, x + 24, 430), fill=255)
    d.line([(365, 180), (320, 85), (280, 50)], fill=255, width=22)
    d.line([(385, 178), (455, 90), (485, 55)], fill=255, width=22)
    for pts in [[(318, 95), (290, 120)], [(338, 110), (312, 145)], [(452, 95), (430, 130)], [(438, 110), (470, 145)]]:
        d.line(pts, fill=255, width=18)
    d.polygon([(415, 205), (470, 220), (418, 235)], fill=255)
    return img


def violin() -> Image.Image:
    img, d = canvas()
    d.line((260, 45, 260, 455), fill=255, width=18)
    d.ellipse((185, 120, 335, 250), fill=255)
    d.ellipse((170, 255, 350, 430), fill=255)
    d.rectangle((230, 205, 290, 315), fill=255)
    d.ellipse((205, 170, 240, 215), fill=0)
    d.ellipse((285, 170, 320, 215), fill=0)
    d.ellipse((200, 315, 238, 365), fill=0)
    d.ellipse((288, 315, 326, 365), fill=0)
    d.rectangle((225, 55, 295, 85), fill=255)
    return img


def scifi_city() -> Image.Image:
    img, d = canvas()
    widths = [(55, 400, 55), (120, 300, 70), (205, 180, 55), (275, 250, 80), (370, 120, 65)]
    for x, top, w in widths:
        d.rectangle((x, top, x + w, 430), fill=255)
        if top < 220:
            poly(d, [(x, top), (x + w / 2, top - 60), (x + w, top)])
        for y in range(top + 25, 410, 38):
            d.rectangle((x + 12, y, x + 26, y + 18), fill=0)
            d.rectangle((x + w - 28, y, x + w - 14, y + 18), fill=0)
    d.rectangle((40, 430, 470, 455), fill=255)
    d.ellipse((245, 85, 295, 135), fill=255)
    return img


SHAPES = {
    "complex_dragon": dragon,
    "complex_castle": castle,
    "complex_ship": ship,
    "complex_robot": robot,
    "complex_bonsai": bonsai,
    "complex_phoenix": phoenix,
    "complex_cathedral": cathedral,
    "complex_bicycle": bicycle,
    "complex_owl_totem": owl_totem,
    "complex_stag": stag,
    "complex_violin": violin,
    "complex_scifi_city": scifi_city,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    for name, fn in SHAPES.items():
        img = fn()
        save(name, img)
        tile = Image.new("RGB", (180, 210), "white")
        tile.paste(img.convert("RGB").resize((180, 180)), (0, 0))
        ImageDraw.Draw(tile).text((4, 184), name, fill=(0, 0, 0))
        tiles.append(tile)
    sheet = Image.new("RGB", (4 * 180, 3 * 210), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 4) * 180, (i // 4) * 210))
    sheet.save(SHEET)
    print(f"DONE -> {OUT}")
    print(f"n={len(SHAPES)} contact_sheet={SHEET}")


if __name__ == "__main__":
    main()
