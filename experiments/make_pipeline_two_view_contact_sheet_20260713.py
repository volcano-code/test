#!/usr/bin/env python3
"""Pack four local two-view masks into one exact-pixel reference board for image editing."""

from pathlib import Path
from PIL import Image

ROOT = Path('/home/lin/xvxinyao/ShadowForge')
files = [
    ROOT / 'generation/results/soft_full_loss_ablation_smoke_20260704/full/free/gt_view_0.png',
    ROOT / 'generation/results/soft_full_loss_ablation_smoke_20260704/full/free/gt_view_1.png',
    ROOT / 'generation/results/soft_full_loss_ablation_smoke_20260704/full/free/sample_0_renderer_view_0.png',
    ROOT / 'generation/results/soft_full_loss_ablation_smoke_20260704/full/free/sample_0_renderer_view_1.png',
]
images = [Image.open(path).convert('RGBA') for path in files]
w = max(image.width for image in images)
h = max(image.height for image in images)
gap = 18
board = Image.new('RGBA', (w * 2 + gap * 3, h * 2 + gap * 3), 'white')
for idx, image in enumerate(images):
    x = gap + (idx % 2) * (w + gap)
    y = gap + (idx // 2) * (h + gap)
    board.alpha_composite(image, (x + (w - image.width)//2, y + (h - image.height)//2))
out = ROOT / 'paper_aaai_submission/generated_pipeline_assets/two_view_masks_exact_contact_sheet.png'
board.convert('RGB').save(out)
print(out)
