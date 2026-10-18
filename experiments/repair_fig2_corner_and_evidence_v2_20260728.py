from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
PAPER = Path("/home/lin/下载/P_CRAFT/overleaf_shadowforge_ideal_20260719_v2")
SOURCE = (
    PAPER
    / "figures/fig2_pcraft_pipeline_before_square_and_evidence_fix_20260728.png"
)
TARGET_X = (
    ROOT
    / "results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709"
    / "dark_knight_left__magic_scholar_right/inputs/left_wall_x_view1_target.png"
)
TARGET_Y = (
    ROOT
    / "results/anime_two_wall_fixed_no_floor_wall_aligned_v2_upright_base_10cm_256_20260709"
    / "dark_knight_left__magic_scholar_right/inputs/right_wall_y_view2_target.png"
)
DELIVERY = (
    ROOT
    / "paper_aaai_submission/pipeline_reference_assets_current_20260718"
    / "08_selected_delivery_3d.png"
)
PRINT_PHOTO = (
    ROOT
    / "paper_aaai_submission/physical_experiment_figures_20260724"
    / "aaai_maple_clean_box_original_model_locked_v9.png"
)
OUTPUT = (
    ROOT
    / "paper_aaai_submission/fig2_pipeline_candidates_20260728"
    / "pcraft_pipeline_square_corner_clean_output_v3_3200x1800.png"
)


def fit_contain(
    image: Image.Image,
    size: tuple[int, int],
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    canvas.paste(
        image,
        ((size[0] - image.width) // 2, (size[1] - image.height) // 2),
    )
    return canvas


def arrow(draw: ImageDraw.ImageDraw, start, end, color, width=11, head=22):
    draw.line((start, end), fill=color, width=width)
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    wing = head * 0.5
    draw.polygon(
        (
            (ex, ey),
            (ex - head * ux + wing * px, ey - head * uy + wing * py),
            (ex - head * ux - wing * px, ey - head * uy - wing * py),
        ),
        fill=color,
    )


def square_target(path: Path, size=640) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = fit_contain(image, (size - 36, size - 36), background=(0, 0, 0))
    panel = Image.new("RGB", (size, size), (0, 0, 0))
    panel.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return np.asarray(panel)


def warp_panel(
    canvas: np.ndarray,
    source: np.ndarray,
    quad: list[tuple[int, int]],
) -> None:
    height, width = source.shape[:2]
    src = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    dst = np.float32(quad)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        source,
        matrix,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    mask_source = np.full((height, width), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        mask_source,
        matrix,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    canvas[mask > 0] = warped[mask > 0]


def delivery_cutout(path: Path) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    data = np.asarray(image).copy()
    rgb = data[:, :, :3]
    brightness = rgb.max(axis=2)
    data[:, :, 3] = np.where(brightness < 18, 0, np.minimum(255, brightness * 4))
    result = Image.fromarray(data)
    bbox = result.getbbox()
    return result.crop(bbox) if bbox else result


def paste_rgba_contain(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    image = image.copy()
    image.thumbnail((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    x = x0 + (x1 - x0 - image.width) // 2
    y = y0 + (y1 - y0 - image.height) // 2
    canvas.paste(image, (x, y), image)


def rebuild_corner(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)
    draw.rectangle((410, 285, 1180, 1225), fill=(255, 255, 255))

    # Equal physical square walls represented by symmetric perspective quads.
    seam_top = (795, 555)
    seam_bottom = (795, 915)
    left_quad = [(420, 535), seam_top, seam_bottom, (420, 940)]
    right_quad = [seam_top, (1170, 535), (1170, 940), seam_bottom]

    canvas = np.asarray(image).copy()
    warp_panel(canvas, square_target(TARGET_X), left_quad)
    warp_panel(canvas, square_target(TARGET_Y), right_quad)
    image.paste(Image.fromarray(canvas))
    draw = ImageDraw.Draw(image)

    outer = (151, 148, 141)
    inner = (205, 202, 196)
    for quad in (left_quad, right_quad):
        draw.line(quad + [quad[0]], fill=outer, width=9, joint="curve")
        inset = [
            (
                int(0.96 * x + 0.04 * 795),
                int(0.96 * y + 0.04 * 752),
            )
            for x, y in quad
        ]
        draw.line(inset + [inset[0]], fill=inner, width=4, joint="curve")
    draw.line((seam_top, seam_bottom), fill=(116, 112, 106), width=10)

    # Rays originate on each target plane and follow its perspective direction.
    orange = (237, 112, 0)
    blue = (31, 103, 214)
    for y in (635, 685, 735, 785, 835):
        arrow(draw, (590, y), (840, y + 24), orange, width=11, head=22)
    for y in (635, 685, 735, 785, 835):
        arrow(draw, (1000, y), (750, y + 24), blue, width=11, head=22)

    # Restore clean stage-transition arrows.
    draw.rectangle((365, 725, 419, 845), fill=(255, 255, 255))
    draw.rectangle((1171, 725, 1220, 845), fill=(255, 255, 255))
    teal = (0, 126, 145)
    arrow(draw, (370, 785), (420, 785), teal, width=14, head=25)
    arrow(draw, (1170, 785), (1218, 785), teal, width=14, head=25)


def rebuild_output_evidence(image: Image.Image) -> None:
    draw = ImageDraw.Draw(image)

    # Remove the complete old output panel, including every generated-image trace.
    draw.rectangle((2570, 475, 3195, 1210), fill=(255, 255, 255))

    frame = (2592, 500, 3180, 1180)
    teal = (0, 112, 132)
    draw.rounded_rectangle(frame, radius=28, fill=(255, 255, 255), outline=teal, width=6)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 39)
    title = "Output and Print Result"
    title_box = draw.textbbox((0, 0), title, font=font)
    title_width = title_box[2] - title_box[0]
    draw.text(
        ((frame[0] + frame[2] - title_width) // 2, 545),
        title,
        fill=teal,
        font=font,
    )

    delivery = delivery_cutout(DELIVERY)
    paste_rgba_contain(image, delivery, (2630, 685, 2845, 1090))

    photo = Image.open(PRINT_PHOTO).convert("RGB")
    photo = fit_contain(photo, (235, 405), background=(255, 255, 255))
    image.paste(photo, (2915, 685))

    draw = ImageDraw.Draw(image)
    green = (35, 150, 92)
    arrow(draw, (2845, 888), (2912, 888), green, width=15, head=28)

    # Restore the incoming stage arrow after clearing the old panel.
    draw.rectangle((2520, 805, 2591, 875), fill=(255, 255, 255))
    arrow(draw, (2522, 840), (2592, 840), (0, 126, 145), width=15, head=28)


def main():
    image = Image.open(SOURCE).convert("RGB")
    rebuild_corner(image)
    rebuild_output_evidence(image)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, quality=100)
    print(OUTPUT)


if __name__ == "__main__":
    main()
