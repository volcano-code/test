#!/usr/bin/env python3
"""Render the evidence-grounded ShadowForge overview figure."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
EVIDENCE = ROOT / "results/dark_knight_fdm_7cm_evidence_20260713"
OUT_DIR = ROOT / "paper_aaai_submission/generated_pipeline_assets"
OUT = OUT_DIR / "shadowforge_final_evidence_pipeline_20260714.png"

W, H = 2600, 1420
WHITE = "#FFFFFF"
SLATE = "#263238"
TEAL = "#00897B"
AMBER = "#FFB300"
GREY = "#90A4AE"
PALE = "#F4F8F8"
PANEL = "#F7FAFA"
LAV = "#F8F6FC"


def font(size, bold=False):
    # This overview is designed for a two-column paper width; compact labels
    # must not collide when the full figure is rasterized.
    size = max(10, int(size * 0.78))
    name = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def fit(img, box, pad=8, bg=WHITE, remove_black=False):
    x, y, w, h = box
    canvas = Image.new("RGB", (w, h), bg)
    src = Image.open(img).convert("RGB")
    if remove_black:
        pixels = src.load()
        for yy in range(src.height):
            for xx in range(src.width):
                r, g, b = pixels[xx, yy]
                if r < 25 and g < 25 and b < 25:
                    pixels[xx, yy] = (255, 255, 255)
    src.thumbnail((max(1, w - pad * 2), max(1, h - pad * 2)), Image.Resampling.LANCZOS)
    canvas.paste(src, ((w - src.width) // 2, (h - src.height) // 2))
    return canvas


def rounded(draw, box, fill=WHITE, outline=GREY, width=2, radius=16):
    draw.rounded_rectangle(box, radius, fill=fill, outline=outline, width=width)


def label(draw, xy, text, size=24, color=SLATE, bold=False, anchor="la"):
    draw.text(xy, text, font=font(size, bold), fill=color, anchor=anchor)


def arrow(draw, start, end, color=TEAL, width=5, dash=False):
    if dash:
        x1, y1 = start; x2, y2 = end
        steps = 14
        for i in range(0, steps, 2):
            a = i / steps; b = min(1, (i + 1) / steps)
            draw.line((x1 + (x2-x1)*a, y1 + (y2-y1)*a, x1 + (x2-x1)*b, y1 + (y2-y1)*b), fill=color, width=width)
    else:
        draw.line((*start, *end), fill=color, width=width)
    x1, y1 = start; x2, y2 = end
    import math
    a = math.atan2(y2-y1, x2-x1)
    s = 15
    p1 = (x2 - s*math.cos(a-0.45), y2 - s*math.sin(a-0.45))
    p2 = (x2 - s*math.cos(a+0.45), y2 - s*math.sin(a+0.45))
    draw.polygon([end, p1, p2], fill=color)


def image_card(canvas, draw, box, image_path, title, outline=TEAL):
    x, y, w, h = box
    rounded(draw, (x, y, x+w, y+h), WHITE, outline, 2, 12)
    label(draw, (x+w//2, y+16), title, 17, SLATE, True, "ma")
    canvas.paste(fit(image_path, (0, 0, w-16, h-38), 4), (x+8, y+30))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(canvas)

    # Title and three restrained panels.
    label(draw, (W//2, 42), "ShadowForge: Fabrication-Aware Two-View Shadow-Art Generation", 43, SLATE, True, "ma")
    panels = [(30, 95, 825, 1315, "I. DIFFERENTIABLE SOFT OCCUPANCY FITTING", PANEL),
              (860, 95, 1715, 1315, "II. SOFT-GUIDED CANDIDATE TRAJECTORY", "#FCFAF6"),
              (1750, 95, 2570, 1315, "III. FABRICATION-AWARE DELIVERY", LAV)]
    for x1, y1, x2, y2, title, fill in panels:
        rounded(draw, (x1, y1, x2, y2), fill, "#B0BEC5", 2, 18)
        label(draw, ((x1+x2)//2, y1+28), title, 25, TEAL, True, "ma")
        draw.line((x1+18, y1+55, x2-18, y1+55), fill="#CFD8DC", width=2)

    inputs = EVIDENCE / "inputs"
    soft = EVIDENCE / "soft_training"
    renders = ROOT / "paper_aaai_submission/figure_sources_pipeline_20260713/real_mesh_renders_20260714"
    candidates = EVIDENCE / "candidate_evidence/delivery_candidates"
    scaffold = candidates / "soft_guided_full_hull_scaffold_core48_delivery256"
    carved = candidates / "soft_guided_carve_phase_1_removed_2544_core48_delivery256"

    # Panel I: exact target and soft render evidence.
    image_card(canvas, draw, (60, 165, 175, 215), inputs / "left_wall_x_target.png", "Target T_X")
    image_card(canvas, draw, (60, 395, 175, 215), inputs / "right_wall_y_target.png", "Target T_Y")
    rounded(draw, (265, 165, 470, 610), WHITE, TEAL, 2, 14)
    label(draw, (367, 198), "Fixed two-view", 25, SLATE, True, "ma")
    label(draw, (367, 230), "projection protocol", 25, SLATE, True, "ma")
    label(draw, (367, 302), "P = {(R_X,t_X), (R_Y,t_Y)}", 18, SLATE, False, "ma")
    label(draw, (367, 355), "View X  [camera] -> wall", 18, SLATE, False, "ma")
    label(draw, (367, 412), "View Y  [camera] -> wall", 18, SLATE, False, "ma")
    label(draw, (367, 496), "fixed calibrated views", 17, GREY, False, "ma")
    arrow(draw, (175, 273), (265, 273), TEAL)
    arrow(draw, (175, 503), (265, 503), TEAL)
    rounded(draw, (515, 165, 665, 610), WHITE, TEAL, 2, 14)
    label(draw, (590, 199), "Learnable occupancy", 18, SLATE, True, "ma")
    label(draw, (590, 226), "S_theta in [0,1]^Omega", 17, SLATE, True, "ma")
    # A compact vector proxy for the measured continuous density field.
    for z in range(5):
        for x in range(5):
            px = 545 + x * 19 + z * 7; py = 275 + z * 13
            shade = (85 + x * 17 + z * 9, 180 + x * 6, 176 + z * 4)
            draw.rectangle((px, py, px+15, py+15), fill=shade, outline="#5D8D8B")
    label(draw, (590, 465), "MLP occupancy", 15, SLATE, False, "ma")
    label(draw, (590, 492), "(x,y,z) -> f_theta -> rho", 14, SLATE, False, "ma")
    arrow(draw, (470, 387), (515, 387), TEAL)
    rounded(draw, (680, 165, 805, 610), WHITE, TEAL, 2, 14)
    label(draw, (742, 196), "Differentiable", 16, SLATE, True, "ma")
    label(draw, (742, 220), "renderer", 16, SLATE, True, "ma")
    canvas.paste(fit(soft / "pred_view_0.png", (0, 0, 102, 135), 3), (691, 244))
    canvas.paste(fit(soft / "pred_view_1.png", (0, 0, 102, 135), 3), (691, 407))
    label(draw, (742, 385), "T_hat_X", 13, GREY, False, "ma")
    label(draw, (742, 548), "T_hat_Y", 13, GREY, False, "ma")
    arrow(draw, (665, 387), (680, 387), TEAL)

    rounded(draw, (60, 665, 400, 840), WHITE, AMBER, 3, 14)
    label(draw, (230, 700), "Two-view fidelity loss L_img", 23, SLATE, True, "ma")
    label(draw, (230, 748), "L2 + L1 + Dice + SSIM + FP", 20, SLATE, False, "ma")
    label(draw, (230, 790), "d(T_X, T_hat_X) + d(T_Y, T_hat_Y)", 18, GREY, False, "ma")
    rounded(draw, (430, 665, 800, 840), WHITE, AMBER, 3, 14)
    label(draw, (615, 700), "Soft regularization L_struct", 23, SLATE, True, "ma")
    for i, txt in enumerate(["edge", "smooth", "deviation", "connectivity", "wall", "reachability"]):
        xx = 458 + (i % 3) * 112; yy = 738 + (i // 3) * 46
        rounded(draw, (xx, yy, xx+98, yy+32), WHITE, AMBER, 1, 8)
        label(draw, (xx+49, yy+16), txt, 14, SLATE, False, "mm")
    rounded(draw, (250, 885, 615, 945), WHITE, AMBER, 3, 12)
    label(draw, (432, 915), "L_soft = L_img + alpha_t L_struct", 20, SLATE, False, "mm")
    arrow(draw, (230, 840), (330, 885), AMBER, 4)
    arrow(draw, (615, 840), (530, 885), AMBER, 4)
    arrow(draw, (250, 915), (515, 570), AMBER, 4, True)
    label(draw, (340, 845), "backpropagate ∇theta L_soft", 16, AMBER, False, "ma")

    # Panel II: local source data only, using actual local target-vs-projection strips.
    arrow(draw, (805, 387), (900, 387), GREY, 3, True)
    label(draw, (805, 355), "freeze S_theta; deterministic handoff", 15, GREY, False, "ma")
    rounded(draw, (895, 165, 1125, 495), WHITE, TEAL, 2, 14)
    label(draw, (1010, 197), "Full-hull scaffold", 25, SLATE, True, "ma")
    canvas.paste(fit(renders / "selected_delivery.png", (0, 0, 190, 175), 0, remove_black=True), (915, 220))
    label(draw, (1010, 418), "soft-guided scaffold", 17, SLATE, False, "ma")
    label(draw, (1010, 447), "same target-ray envelope", 16, GREY, False, "ma")
    canvas.paste(fit(scaffold / "target_vs_projection.png", (0, 0, 200, 48), 0), (910, 465))
    rounded(draw, (1160, 165, 1415, 495), WHITE, AMBER, 2, 14)
    label(draw, (1287, 197), "Guarded material removal", 23, SLATE, True, "ma")
    for i, line in enumerate(["rank boundary voxels", "remove redundant material", "retain rays + 6-connectivity"]):
        label(draw, (1190, 265+i*58), f"{i+1}. {line}", 18, SLATE, False, "la")
    label(draw, (1287, 452), "V_(j+1) = V_j \\ Delta V_j", 17, AMBER, False, "ma")
    rounded(draw, (1450, 165, 1690, 495), WHITE, AMBER, 2, 14)
    label(draw, (1570, 197), "Carve phase 1", 25, SLATE, True, "ma")
    canvas.paste(fit(renders / "selected_delivery.png", (0, 0, 195, 175), 0, remove_black=True), (1472, 220))
    label(draw, (1570, 417), "soft-guided candidate", 17, SLATE, False, "ma")
    label(draw, (1570, 446), "fewer redundant voxels", 16, GREY, False, "ma")
    canvas.paste(fit(carved / "target_vs_projection.png", (0, 0, 210, 48), 0), (1465, 465))
    arrow(draw, (1125, 330), (1160, 330), AMBER)
    arrow(draw, (1415, 330), (1450, 330), AMBER)
    rounded(draw, (990, 565, 1595, 765), WHITE, TEAL, 2, 14)
    label(draw, (1292, 598), "Shared feasibility gates", 25, SLATE, True, "ma")
    for i, txt in enumerate(["two-view fidelity", "single connected body", "wall / erosion survival", "watertight mesh"]):
        xx = 1035 + (i % 2) * 285; yy = 650 + (i // 2) * 48
        label(draw, (xx, yy), "✓", 22, TEAL, True, "la")
        label(draw, (xx+28, yy+2), txt, 18, SLATE, False, "la")
    arrow(draw, (1570, 495), (1570, 565), TEAL)
    rounded(draw, (1080, 805, 1498, 875), WHITE, TEAL, 3, 14)
    label(draw, (1289, 835), "Fidelity-band selector", 24, SLATE, True, "ma")
    label(draw, (1289, 862), "select min volume within frozen IoU band", 16, GREY, False, "ma")
    arrow(draw, (1290, 765), (1290, 805), TEAL)

    # Panel III: actual delivered mesh render and evidence card.
    arrow(draw, (1498, 840), (1795, 840), TEAL)
    rounded(draw, (1785, 165, 2205, 760), WHITE, TEAL, 3, 16)
    label(draw, (1995, 197), "Selected delivered ShadowForge object", 25, SLATE, True, "ma")
    canvas.paste(fit(renders / "selected_delivery.png", (0, 0, 355, 375), 0, remove_black=True), (1818, 235))
    label(draw, (1995, 630), "C_48 -> V_del,256", 22, SLATE, True, "ma")
    label(draw, (1995, 665), "core-shell delivery", 18, GREY, False, "ma")
    image_card(canvas, draw, (2235, 165, 300, 330), EVIDENCE / "input_vs_projection.png", "Two-view input vs. delivery")
    arrow(draw, (2205, 360), (2235, 315), GREY, 3, True)
    label(draw, (2212, 288), "View X / View Y", 15, GREY, False, "ma")
    badges = [(1790, 795, 2130, 852, "Mean IoU = 0.9894", TEAL), (2145, 795, 2520, 852, "Material saving = 24.68%", AMBER),
              (1790, 870, 1980, 927, "Single-body", TEAL), (1995, 870, 2185, 927, "Watertight", TEAL), (2200, 870, 2520, 927, "FEM-safe, SF_FEM = 5.35", TEAL)]
    for x1, y1, x2, y2, txt, color in badges:
        rounded(draw, (x1, y1, x2, y2), WHITE, color, 3, 12)
        label(draw, ((x1+x2)//2, (y1+y2)//2), txt, 18, SLATE, True, "mm")
    rounded(draw, (1815, 1000, 2525, 1190), WHITE, "#7E57C2", 2, 14)
    label(draw, (2170, 1032), "Load-bearing mode: fail-closed structural triage", 22, "#5E35B1", True, "ma")
    blocks = [(1850, 1080, 2020, 1150, "Physics-residual\nPINO"), (2080, 1080, 2250, 1150, "ACS\nscreen"), (2310, 1080, 2485, 1150, "Voxel FEM\nfallback")]
    for x1, y1, x2, y2, txt in blocks:
        rounded(draw, (x1, y1, x2, y2), WHITE, "#7E57C2", 2, 10)
        a, b = txt.split("\n")
        label(draw, ((x1+x2)//2, y1+25), a, 18, SLATE, True, "ma")
        label(draw, ((x1+x2)//2, y1+50), b, 18, SLATE, False, "ma")
    arrow(draw, (2020, 1115), (2080, 1115), "#7E57C2", 4)
    arrow(draw, (2250, 1115), (2310, 1115), "#7E57C2", 4)
    label(draw, (2375, 1173), "FEM is final authority on defer", 15, GREY, False, "ma")

    # Legend.
    label(draw, (55, 1348), "Legend:", 16, SLATE, True, "la")
    arrow(draw, (145, 1352), (195, 1352), TEAL, 3); label(draw, (205, 1352), "forward / accepted flow", 16, GREY, False, "lm")
    arrow(draw, (410, 1352), (460, 1352), AMBER, 3); label(draw, (470, 1352), "optimization / guarded carving", 16, GREY, False, "lm")
    arrow(draw, (790, 1352), (840, 1352), GREY, 3, True); label(draw, (850, 1352), "frozen handoff / verification", 16, GREY, False, "lm")
    canvas.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
