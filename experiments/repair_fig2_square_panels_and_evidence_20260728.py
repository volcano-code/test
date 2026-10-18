from pathlib import Path

from PIL import Image, ImageDraw


PAPER = Path("/home/lin/下载/P_CRAFT/overleaf_shadowforge_ideal_20260719_v2")
FIGURE = PAPER / "figures/fig2_pcraft_pipeline_geometry_first_20260727.png"
TARGET_X = Path(
    "/home/lin/xvxinyao/ShadowForge/results/"
    "anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709/"
    "dark_knight_left__magic_scholar_right/inputs/left_wall_x_view1_target.png"
)
TARGET_Y = Path(
    "/home/lin/xvxinyao/ShadowForge/results/"
    "anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709/"
    "dark_knight_left__magic_scholar_right/inputs/right_wall_y_view2_target.png"
)
DELIVERY = Path(
    "/home/lin/xvxinyao/ShadowForge/paper_aaai_submission/"
    "pipeline_reference_assets_current_20260718/08_selected_delivery_3d.png"
)
PRINT_PHOTO = Path(
    "/home/lin/xvxinyao/ShadowForge/paper_aaai_submission/"
    "physical_experiment_figures_20260724/"
    "aaai_maple_clean_box_original_model_locked_v9.png"
)
OUTPUT = Path(
    "/home/lin/xvxinyao/ShadowForge/paper_aaai_submission/"
    "fig2_pipeline_candidates_20260728/"
    "pcraft_pipeline_square_panels_locked_evidence_3200x1800.png"
)


def fit_contain(image: Image.Image, size: tuple[int, int], background=(255, 255, 255)):
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def paste_target(canvas: Image.Image, source: Path, box: tuple[int, int, int, int]):
    x0, y0, x1, y1 = box
    size = x1 - x0
    target = Image.open(source).convert("RGB")
    target = fit_contain(target, (size - 24, size - 24), background=(0, 0, 0))
    panel = Image.new("RGB", (size, size), (0, 0, 0))
    panel.paste(target, (12, 12))
    canvas.paste(panel, (x0, y0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(112, 112, 108), width=6)


def arrow(draw: ImageDraw.ImageDraw, start, end, color, width=12):
    draw.line((start, end), fill=color, width=width)
    ex, ey = end
    sx, sy = start
    dx, dy = ex - sx, ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = 22
    wing = 11
    p1 = (ex, ey)
    p2 = (ex - head * ux + wing * px, ey - head * uy + wing * py)
    p3 = (ex - head * ux - wing * px, ey - head * uy - wing * py)
    draw.polygon((p1, p2, p3), fill=color)


def delivery_cutout(source: Path):
    image = Image.open(source).convert("RGBA")
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            r, g, b, _ = pixels[x, y]
            alpha = 0 if max(r, g, b) < 18 else min(255, max(r, g, b) * 4)
            pixels[x, y] = (r, g, b, alpha)
    bbox = image.getbbox()
    return image.crop(bbox) if bbox else image


def paste_rgba_contain(
    canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]
):
    x0, y0, x1, y1 = box
    image = image.copy()
    image.thumbnail((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    x = x0 + (x1 - x0 - image.width) // 2
    y = y0 + (y1 - y0 - image.height) // 2
    canvas.paste(image, (x, y), image)


def main():
    image = Image.open(FIGURE).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Rebuild only the two-view projection panel; preserve the surrounding formula.
    draw.rectangle((420, 300, 1168, 1222), fill=(255, 255, 255))
    left = (438, 430, 786, 778)
    right = (796, 430, 1144, 778)
    paste_target(image, TARGET_X, left)
    paste_target(image, TARGET_Y, right)
    draw = ImageDraw.Draw(image)

    # A shared vertical hinge conveys the orthogonal fixed-wall relation.
    draw.line((791, 410, 791, 798), fill=(105, 105, 100), width=7)
    draw.line((438, 788, 1144, 788), fill=(150, 150, 145), width=4)

    orange = (235, 112, 0)
    blue = (34, 104, 218)
    for offset in (-66, -22, 22, 66):
        arrow(draw, (610, 604 + offset), (900, 604 + offset), orange)
        arrow(draw, (970, 626 + offset), (680, 626 + offset), blue)

    # Restore the two stage-transition arrows without retaining old arrow fragments.
    draw.rectangle((382, 752, 437, 826), fill=(255, 255, 255))
    draw.rectangle((1144, 752, 1215, 826), fill=(255, 255, 255))
    arrow(draw, (386, 784), (430, 784), (0, 125, 145), width=14)
    arrow(draw, (1148, 784), (1210, 784), (0, 125, 145), width=14)

    # Replace generated evidence with pixel-locked local evidence.
    draw.rectangle((2640, 660, 3165, 1122), fill=(255, 255, 255))
    draw.line((2898, 720, 2898, 1070), fill=(224, 230, 232), width=3)

    delivery = delivery_cutout(DELIVERY)
    paste_rgba_contain(image, delivery, (2662, 720, 2880, 1065))

    photo = Image.open(PRINT_PHOTO).convert("RGB")
    photo = fit_contain(photo, (240, 345), background=(255, 255, 255))
    image.paste(photo, (2910, 720))
    draw = ImageDraw.Draw(image)
    arrow(draw, (2868, 895), (2910, 895), (0, 125, 145), width=12)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, quality=100)
    image.save(FIGURE, quality=100)
    print(OUTPUT)
    print(FIGURE)


if __name__ == "__main__":
    main()
