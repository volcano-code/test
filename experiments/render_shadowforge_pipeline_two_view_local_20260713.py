#!/usr/bin/env python3
"""Render a two-view ShadowForge pipeline using local experimental masks verbatim."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
import numpy as np
from PIL import Image


ROOT = Path('/home/lin/xvxinyao/ShadowForge')
OUT = ROOT / 'paper_aaai_submission'
LEFT = ROOT / 'results/dark_knight_soft_seeded_hybrid_20260712/inputs/left_wall_x_target.png'
RIGHT = ROOT / 'results/dark_knight_soft_seeded_hybrid_20260712/inputs/right_wall_y_target.png'

TEAL = '#007D78'
TEAL_LIGHT = '#E8F5F2'
AMBER = '#DE7C00'
AMBER_LIGHT = '#FFF4E5'
BLUE = '#497A9C'
BLUE_LIGHT = '#EEF4F8'
INK = '#1D2730'
MUTED = '#60717E'


def local_mask(path):
    image = np.asarray(Image.open(path).convert('L'))
    # The inputs are white foreground on black. Display as black foreground on white.
    return np.where(image > 127, 0, 255).astype(np.uint8)


def box(ax, x, y, w, h, title, color=BLUE, fill='white', dashed=False, radius=0.015):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=f'round,pad=0.008,rounding_size={radius}',
        linewidth=1.25, edgecolor=color, facecolor=fill,
        linestyle='--' if dashed else '-', transform=ax.transAxes, zorder=1,
    )
    ax.add_patch(patch)
    ax.text(x + 0.014, y + h - 0.026, title, transform=ax.transAxes, va='top',
            fontsize=6.8, fontweight='bold', color=color, zorder=4)


def arrow(ax, a, b, color=TEAL, lw=1.8, style='-|>', rad=0.0, dashed=False):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle=style, mutation_scale=12, linewidth=lw,
        color=color, connectionstyle=f'arc3,rad={rad}', linestyle='--' if dashed else '-',
        transform=ax.transAxes, zorder=5,
    ))


def add_mask(ax, mask, extent, label=None):
    ax.imshow(mask, cmap='gray', vmin=0, vmax=255, extent=extent, zorder=3, interpolation='nearest')
    x0, x1, y0, y1 = extent
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, transform=ax.transAxes,
                           fill=False, linewidth=0.7, edgecolor='#87939B', zorder=4))
    if label:
        ax.text((x0+x1)/2, y0-0.014, label, transform=ax.transAxes, ha='center', va='top',
                fontsize=7, color=MUTED)


def voxel(ax, cx, cy, scale=0.038, color='#B8C5CD', accent=None):
    # compact isometric voxel block
    for row in range(3):
        for col in range(3):
            x = cx + (col-row) * scale * .34
            y = cy + (row+col) * scale * .17
            face = Rectangle((x, y), scale*.31, scale*.31, transform=ax.transAxes,
                             facecolor=accent if accent and (row == 1 and col == 1) else color,
                             edgecolor='#6D7D88', linewidth=.35, zorder=3)
            ax.add_patch(face)


def main():
    left, right = local_mask(LEFT), local_mask(RIGHT)
    fig = plt.figure(figsize=(18.5, 9.3), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(.5, .973, 'ShadowForge: Fabrication-Aware Shadow-Art Generation',
            ha='center', va='top', transform=ax.transAxes, fontsize=18, fontweight='bold', color=INK)
    ax.text(.5, .943, 'Fixed two-view differentiable proposal, deterministic fabrication repair, and auditable selection',
            ha='center', va='top', transform=ax.transAxes, fontsize=9.6, color=MUTED)

    # Upper training band
    box(ax, .018, .545, .964, .345, 'CONTINUOUS PROPOSAL OPTIMIZATION', TEAL, TEAL_LIGHT, radius=.02)

    # Fixed local inputs
    box(ax, .038, .594, .158, .222, 'FIXED TWO-VIEW\nSPECIFICATION', TEAL, 'white')
    add_mask(ax, left, (.053, .107, .636, .755), r'$T_1$')
    add_mask(ax, right, (.125, .179, .636, .755), r'$T_2$')
    ax.text(.117, .602, r'fixed calibrated cameras  $P=\{(R_i,t_i)\}_{i=1}^{2}$',
            transform=ax.transAxes, ha='center', fontsize=5.5, color=INK)
    ax.add_patch(Rectangle((.054,.612),.055,.009, transform=ax.transAxes, color='#677782', zorder=3))
    ax.add_patch(Circle((.067,.607),.009, transform=ax.transAxes, color='#677782', zorder=3))
    ax.add_patch(Circle((.096,.607),.009, transform=ax.transAxes, color='#677782', zorder=3))

    # Occupancy field and renderer
    box(ax, .235, .594, .214, .222, 'LEARNABLE OCCUPANCY\nREPRESENTATION', TEAL, 'white')
    voxel(ax, .282, .677, .075, '#D5E2E5', accent='#698B91')
    ax.text(.345, .733, r'$S_\theta\in[0,1]^{\Omega_{256}}$', transform=ax.transAxes, fontsize=10, color=INK)
    ax.text(.345, .702, 'continuous density field', transform=ax.transAxes, fontsize=7.4, color=MUTED)
    ax.text(.345, .671, 'learnable parameters  $\theta$', transform=ax.transAxes, fontsize=7.4, color=MUTED)
    ax.text(.253, .615, r'$\mathcal{L}_{soft}=\mathcal{L}_{img}+\alpha_t\mathcal{L}_{proposal}$',
            transform=ax.transAxes, fontsize=8.1, color=INK)

    box(ax, .492, .594, .175, .222, 'DIFFERENTIABLE\nSILHOUETTE RENDERER', TEAL, 'white')
    voxel(ax, .52, .67, .052, '#C8D5DB', accent='#93AEB6')
    ax.text(.583, .723, r'$\hat T_i=R_{\mathrm{silh}}(S_\theta,P_i)$', transform=ax.transAxes, fontsize=8.1, color=INK)
    ax.text(.583, .691, 'orthographic ray marching', transform=ax.transAxes, fontsize=7.1, color=MUTED)
    add_mask(ax, left, (.594, .622, .629, .668), None)
    add_mask(ax, right, (.625, .653, .629, .668), None)

    # loss
    box(ax, .710, .594, .245, .222, 'FIDELITY + PROPOSAL\nREGULARIZATION', AMBER, 'white')
    add_mask(ax, left, (.730, .761, .681, .730), r'$\hat T_1\;\leftrightarrow\;T_1$')
    add_mask(ax, right, (.770, .801, .681, .730), r'$\hat T_2\;\leftrightarrow\;T_2$')
    ax.text(.823, .704, r'$\mathcal{L}_{img}:\;L_2,L_1,\mathrm{Dice},\mathrm{SSIM},L_{FP}$', transform=ax.transAxes, fontsize=7.1, color=INK)
    ax.text(.823, .675, r'$\mathcal{L}_{proposal}:\;L_{edge},L_{smooth},L_{conn},L_{wall},\ldots$', transform=ax.transAxes, fontsize=6.2, color=INK)
    ax.text(.823, .642, 'curriculum activates integrity terms', transform=ax.transAxes, fontsize=6.3, color=MUTED)

    arrow(ax, (.196,.705), (.235,.705))
    arrow(ax, (.449,.705), (.492,.705))
    arrow(ax, (.667,.705), (.710,.705), color=AMBER)
    arrow(ax, (.827,.594), (.827,.557), color=AMBER)
    arrow(ax, (.827,.557), (.342,.557), color=AMBER)
    arrow(ax, (.342,.557), (.342,.594), color=AMBER)
    ax.text(.51, .562, r'backpropagate  $\nabla_\theta\mathcal{L}_{total}$', transform=ax.transAxes, fontsize=7.2, color=AMBER)

    # lower handoff/fabrication band
    box(ax, .018, .105, .964, .375, 'DISCRETE FABRICATION AND SELECTION', BLUE, BLUE_LIGHT, radius=.02)
    arrow(ax, (.342,.545), (.342,.473), color=BLUE)

    box(ax, .042, .165, .200, .245, 'THRESHOLD ENSEMBLE +\nSOFT-GUIDED STATES', BLUE, 'white')
    voxel(ax,.08,.285,.045,'#C7D6DE',accent='#5E9BA2')
    voxel(ax,.126,.285,.045,'#C7D6DE',accent='#5E9BA2')
    voxel(ax,.172,.285,.045,'#C7D6DE',accent='#5E9BA2')
    ax.text(.141,.350, r'$B_\tau=\mathbf{1}[G_{48}(S_\theta)\geq\tau]\cap H_{48}$', transform=ax.transAxes, ha='center', fontsize=7.3, color=INK)
    ax.text(.141,.225, 'fixed thresholds; density-guided carve states', transform=ax.transAxes, ha='center', fontsize=5.8, color=MUTED)
    ax.text(.141,.198, r'$C_{\mathrm{raw}}$', transform=ax.transAxes, ha='center', fontsize=10, color=BLUE)

    box(ax, .282, .165, .220, .245, 'PROJECTION-PRESERVING\nFABRICATION REPAIR', BLUE, 'white')
    voxel(ax,.316,.285,.048,'#D5E0E6',accent='#D48D4D')
    ax.text(.362,.314, 'RestoreRays', transform=ax.transAxes, fontsize=8.2, color=INK)
    ax.text(.362,.292, 'recover missing target-ray support', transform=ax.transAxes, fontsize=6.6, color=MUTED)
    ax.text(.362,.255, r'$\rightarrow\;\mathrm{Bridge}_6$', transform=ax.transAxes, fontsize=8.2, color=INK)
    ax.text(.362,.233, 'face-connect material inside $H_{48}$', transform=ax.transAxes, fontsize=5.8, color=MUTED)
    ax.text(.392,.195, r'$C_{\mathrm{fab}}$', transform=ax.transAxes, ha='center', fontsize=10, color=BLUE)

    box(ax, .540, .165, .188, .245, 'SHARED ELIGIBILITY GATES', TEAL, 'white')
    checks = ['per-view projection fidelity', 'single 6-connected body', 'scale-aware wall survival', 'watertight mesh export']
    for i, label in enumerate(checks):
        yy=.334-i*.035
        ax.text(.565, yy, '✓', transform=ax.transAxes, fontsize=11, color=TEAL, fontweight='bold')
        ax.text(.586, yy+.003, label, transform=ax.transAxes, fontsize=7.1, color=INK)
    ax.text(.634,.195, r'$C_{\mathrm{elig}}$', transform=ax.transAxes, ha='center', fontsize=10, color=TEAL)

    box(ax, .765, .165, .192, .245, 'FROZEN-BAND SELECTOR\nAND DELIVERY', TEAL, 'white')
    ax.text(.781,.344, r'$C_\delta=\{V\in C_{\mathrm{elig}}:s(V)\geq s_{\max}-\delta\}$', transform=ax.transAxes, fontsize=6.8, color=INK)
    ax.text(.781,.313, r'$V^*=\arg\min_{V\in C_\delta}|V|$', transform=ax.transAxes, fontsize=9.0, color=INK)
    voxel(ax,.797,.239,.040,'#B7C5CB',accent='#48A9A2')
    ax.text(.835,.257, '48³ core  +  256³ shell', transform=ax.transAxes, fontsize=7.2, color=INK)
    add_mask(ax,left,(.887,.914,.202,.245),None)
    add_mask(ax,right,(.918,.945,.202,.245),None)
    ax.text(.861,.183, 'faithful   reduced   watertight', transform=ax.transAxes, ha='center', fontsize=5.8, color=TEAL)

    arrow(ax, (.242,.287), (.282,.287), color=BLUE)
    arrow(ax, (.502,.287), (.540,.287), color=TEAL)
    arrow(ax, (.728,.287), (.765,.287), color=TEAL)

    # Optional structural branch: explicitly subordinate
    box(ax, .426, .112, .365, .055, '', MUTED, 'white', dashed=True, radius=.012)
    ax.text(.609,.148, 'OPTIONAL STRUCTURAL TRIAGE', transform=ax.transAxes,
            ha='center', fontsize=5.7, fontweight='bold', color=MUTED)
    ax.text(.609,.132, 'physics-residual PINO  →  ACS  →  voxel FEM fallback',
            transform=ax.transAxes, ha='center', fontsize=5.7, color=INK)
    ax.text(.609,.118, 'calibrated decisions only; deferred candidates retain FEM authority.',
            transform=ax.transAxes, ha='center', fontsize=5.0, color=MUTED)
    arrow(ax, (.861,.165), (.776,.167), color=MUTED, dashed=True)

    # minimal legend
    ax.plot([.025,.054],[.035,.035], transform=ax.transAxes, color=TEAL, lw=2)
    ax.text(.058,.035,'differentiable forward path', transform=ax.transAxes, va='center', fontsize=6.4, color=MUTED)
    ax.plot([.025,.054],[.018,.018], transform=ax.transAxes, color=AMBER, lw=2)
    ax.text(.058,.018,'optimization feedback', transform=ax.transAxes, va='center', fontsize=6.4, color=MUTED)

    # imshow uses data coordinates: lock them after adding all local raster masks.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    out_png = OUT / 'fig2_shadowforge_pipeline_two_view_local_20260713.png'
    out_pdf = OUT / 'fig2_shadowforge_pipeline_two_view_local_20260713.pdf'
    fig.savefig(out_png, dpi=260, bbox_inches='tight', pad_inches=.03)
    fig.savefig(out_pdf, bbox_inches='tight', pad_inches=.03)
    print(out_png)
    print(out_pdf)


if __name__ == '__main__':
    main()
