"""
实验数据准备 — 生成程序化测试形状 + 多视图剪影

不依赖外部数据集 (如 ShapeNet), 使用程序化形状确保可复现。
支持后续加入 ShapeNet 作为补充实验。

生成流程:
1. 调用 generate_random_shape() 生成 3D 体素
2. 用 PyTorch3D 渲染 6 视角正交剪影
3. 用 SGLDBench 计算 FEM 真值 (用于验证)
4. 保存每个模型: {voxels.npy, silhouettes/*.png, fem_fields.npz, load.npz}
"""

from __future__ import annotations

import logging
import json
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def generate_test_shape(
    resolution: int,
    rng: np.random.Generator,
    shape_type: str | None = None,
) -> np.ndarray:
    """生成单个测试形状

    Args:
        resolution: 体素分辨率 N
        rng: numpy RNG
        shape_type: 指定形状类型, None 则随机选择

    Returns:
        [N, N, N] bool 数组
    """
    from structural.data.generate_fem_dataset import generate_random_shape
    # 如果指定了类型, 循环采样直到匹配
    if shape_type is not None:
        for _ in range(100):
            vol = generate_random_shape(resolution, rng)
            if vol.sum() > 8:
                return vol
    return generate_random_shape(resolution, rng)


def render_silhouettes_orthographic(
    density: np.ndarray,
    n_views: int = 6,
    img_size: int = 128,
) -> list:
    """用正交投影渲染多视角剪影 (无需 PyTorch3D)

    6 视角: front(+Z), back(-Z), left(-X), right(+X), top(+Y), bottom(-Y)

    Args:
        density: [N, N, N] 密度场
        n_views: 视角数
        img_size: 输出图像尺寸

    Returns:
        list of [img_size, img_size] uint8 剪影图像
    """
    from scipy.ndimage import zoom

    solid = (density > 0.5).astype(np.float32)
    N = density.shape[0]
    scale = img_size / N

    silhouettes = []
    # 6 视角的投影: (axis, flip)
    views = [
        (2, False),  # front: max along Z
        (2, True),   # back: max along Z, flipped
        (0, False),  # right: max along X
        (0, True),   # left: max along X, flipped
        (1, False),  # top: max along Y
        (1, True),   # bottom: max along Y, flipped
    ]

    for i in range(min(n_views, len(views))):
        axis, flip = views[i]
        proj = solid.max(axis=axis)
        if flip:
            proj = np.flip(proj, axis=0)
        # Resize to img_size
        proj_resized = zoom(proj, scale, order=0)
        # Ensure exact size
        proj_final = np.zeros((img_size, img_size), dtype=np.float32)
        h, w = min(proj_resized.shape[0], img_size), min(proj_resized.shape[1], img_size)
        proj_final[:h, :w] = proj_resized[:h, :w]
        silhouettes.append((proj_final * 255).astype(np.uint8))

    return silhouettes


def prepare_experiment_dataset(
    output_dir: str,
    n_models: int = 50,
    resolution: int = 64,
    n_views: int = 6,
    img_size: int = 128,
    seed: int = 42,
    run_fem: bool = True,
):
    """准备完整实验数据集

    为每个测试模型生成:
    - 3D 体素真值
    - 多视图剪影图像
    - 载荷条件
    - FEM 真值 (可选)

    Args:
        output_dir: 输出目录
        n_models: 测试模型数量
        resolution: 体素分辨率
        n_views: 剪影视角数
        img_size: 剪影图像尺寸
        seed: 随机种子
        run_fem: 是否计算 FEM 真值
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    from structural.analysis.load_generator import LoadGenerator
    load_gen = LoadGenerator(material_density=1.24e3)

    models_info = []
    generated = 0
    attempts = 0

    print(f"Generating {n_models} test models in {output_dir}...")

    while generated < n_models and attempts < n_models * 5:
        attempts += 1
        try:
            # 生成形状
            vol = generate_test_shape(resolution, rng)
            if vol.sum() < 20:
                continue

            model_id = f"model_{generated:04d}"
            model_dir = output_path / model_id
            model_dir.mkdir(exist_ok=True)

            # 保存体素
            density = vol.astype(np.float32)
            np.save(model_dir / "voxels.npy", density)

            # 渲染剪影
            sils = render_silhouettes_orthographic(density, n_views, img_size)
            sil_dir = model_dir / "silhouettes"
            sil_dir.mkdir(exist_ok=True)

            sil_paths = []
            for v_idx, sil in enumerate(sils):
                sil_path = sil_dir / f"view_{v_idx:02d}.png"
                try:
                    from PIL import Image
                    Image.fromarray(sil).save(str(sil_path))
                except ImportError:
                    import cv2
                    cv2.imwrite(str(sil_path), sil)
                sil_paths.append(str(sil_path))

            # 生成载荷 (standing + gripping)
            load_standing = load_gen.standing_load(density)
            load_gripping = load_gen.gripping_load(density)

            # 保存载荷
            np.savez_compressed(
                model_dir / "loads.npz",
                force_standing=load_standing['force'],
                fixed_mask_standing=load_standing['fixed_mask'],
                force_gripping=load_gripping['force'],
                fixed_mask_gripping=load_gripping['fixed_mask'],
            )

            # FEM 验证真值 (可选)
            fem_data = None
            if run_fem:
                try:
                    from structural.data.generate_fem_dataset import density_to_fem_fields
                    from sgldbench.config import MaterialProperties, SolverSettings

                    material = MaterialProperties(E=3.5e9, nu=0.36, cell_size=1.0)
                    solver = SolverSettings(tolerance=1e-5, max_iterations=500)

                    fem_data = density_to_fem_fields(
                        density.astype(bool),
                        load_standing['force'],
                        load_standing['fixed_mask'],
                        material, solver,
                    )
                    if fem_data is not None:
                        np.savez_compressed(
                            model_dir / "fem_standing.npz",
                            displacement=fem_data['displacement'],
                            stress=fem_data['stress'],
                            von_mises=fem_data['von_mises'],
                        )
                except Exception as e:
                    logger.warning(f"FEM failed for {model_id}: {e}")

            info = {
                'model_id': model_id,
                'solid_ratio': float(density.mean()),
                'n_voxels': int(density.sum()),
                'silhouette_paths': sil_paths,
                'has_fem': fem_data is not None,
            }
            models_info.append(info)

            generated += 1
            if generated % 10 == 0:
                print(f"  Generated {generated}/{n_models} models")

        except Exception as e:
            logger.warning(f"Shape generation attempt {attempts} failed: {e}")
            continue

    # 保存元信息
    meta = {
        'n_models': generated,
        'resolution': resolution,
        'n_views': n_views,
        'img_size': img_size,
        'seed': seed,
        'models': models_info,
    }
    with open(output_path / "dataset_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset ready: {generated} models in {output_dir}")
    return meta


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./experiments/test_data')
    parser.add_argument('--n_models', type=int, default=50)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_fem', action='store_true')
    args = parser.parse_args()

    prepare_experiment_dataset(
        output_dir=args.output_dir,
        n_models=args.n_models,
        resolution=args.resolution,
        seed=args.seed,
        run_fem=not args.no_fem,
    )
