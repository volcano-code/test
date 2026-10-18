from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


FORMAL = Path("/home/lin/下载/P_CRAFT/overleaf_shadowforge_ideal_20260719_v2/figures")
WORK = Path("/home/lin/xvxinyao/ShadowForge/paper_aaai_submission/figures")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def centered(draw: ImageDraw.ImageDraw, xy, text, fnt, fill):
    box = draw.textbbox((0, 0), text, font=fnt)
    x = xy[0] - (box[2] - box[0]) / 2
    y = xy[1] - (box[3] - box[1]) / 2
    draw.text((x, y), text, font=fnt, fill=fill)


def arrow(draw: ImageDraw.ImageDraw, start, end, fill, width=3):
    draw.line([start, end], fill=fill, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex > sx else -1
        head = [(ex, ey), (ex - 10 * sign, ey - 6), (ex - 10 * sign, ey + 6)]
    else:
        sign = 1 if ey > sy else -1
        head = [(ex, ey), (ex - 6, ey - 10 * sign), (ex + 6, ey - 10 * sign)]
    draw.polygon(head, fill=fill)


def refine_fig2():
    source = FORMAL / "fig2_pcraft_pipeline_geometry_first_20260727.png"
    backup = FORMAL / "fig2_pcraft_pipeline_geometry_first_20260727.pre_compact.png"
    if not backup.exists():
        backup.write_bytes(source.read_bytes())

    # Always rebuild from the untouched full figure. Reading the previously
    # compacted output would compound its crop on repeated runs.
    original = Image.open(backup).convert("RGB")
    top = original.crop((0, 0, original.width, 650))
    canvas = Image.new("RGB", (original.width, 805), "white")
    canvas.paste(top, (0, 0))
    draw = ImageDraw.Draw(canvas)

    teal = "#00796B"
    violet = "#6A3DCE"
    charcoal = "#263238"
    pale = "#F8F6FF"
    gray = "#64748B"

    # Compact structural-query path. It is deliberately not presented as a
    # learned carving controller.
    y_mid = 720
    centered(draw, (595, 720), "Q48 = max-pool(Vdel,256)", font(23, True), charcoal)

    draw.rounded_rectangle((805, 676, 1120, 770), radius=14, fill=pale, outline=violet, width=3)
    centered(draw, (962, 704), "PINO prediction", font(22, True), violet)
    centered(draw, (962, 738), "û, σ̂, SF̂", font(24, True), charcoal)

    draw.rounded_rectangle((1165, 676, 1455, 770), radius=14, fill="#FAFAFA", outline=gray, width=3)
    centered(draw, (1310, 704), "Selective voxel FEM", font(21, True), charcoal)
    centered(draw, (1310, 738), "boundary verification", font(19, True), charcoal)

    arrow(draw, (735, y_mid), (805, y_mid), violet, 4)
    arrow(draw, (1120, y_mid), (1165, y_mid), violet, 4)
    # Handoff from guarded trajectories and return to selection.
    draw.line([(925, 650), (925, 664), (595, 664), (595, 700)], fill=violet, width=3)
    draw.line([(1310, 676), (1310, 655), (1265, 655), (1265, 640)], fill=teal, width=3)
    draw.polygon([(1265, 640), (1258, 651), (1272, 651)], fill=teal)

    canvas = canvas.resize((2400, 1156), Image.Resampling.LANCZOS)
    canvas = ImageEnhance.Contrast(canvas).enhance(1.06)
    canvas = canvas.filter(ImageFilter.UnsharpMask(radius=1.2, percent=115, threshold=2))

    for folder in (FORMAL, WORK):
        out = folder / "fig2_pcraft_pipeline_geometry_first_20260727.png"
        canvas.save(out, quality=95)


def refine_fig3():
    name = "pino_architecture_raster_simplified_with_derived_outputs_20260719.png"
    source = FORMAL / name
    backup = FORMAL / "pino_architecture_raster_simplified_with_derived_outputs_20260719.pre_readability.png"
    if not backup.exists():
        backup.write_bytes(source.read_bytes())

    # Always refine the untouched backup. Morphological dark dilation makes
    # compact math glyphs merge after single-column downsampling.
    image = Image.open(backup).convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(1.025)
    image = ImageEnhance.Sharpness(image).enhance(1.10)
    image = image.filter(ImageFilter.UnsharpMask(radius=0.45, percent=70, threshold=3))

    for folder in (FORMAL, WORK):
        image.save(folder / name, quality=95)


if __name__ == "__main__":
    refine_fig2()
    refine_fig3()
