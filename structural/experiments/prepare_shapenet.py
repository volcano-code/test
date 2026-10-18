"""
ShapeNet 真实模型评估数据集准备

从 ShapeNet Core v2 加载真实 3D 模型, 体素化后生成与程序化数据集
格式完全一致的测试集, 用于验证方法在真实几何上的泛化能力。

支持的类别 (synsetId → 语义):
  02691156 → airplane
  02828884 → bench
  02933112 → cabinet
  03001627 → chair
  03211117 → display
  03636649 → lamp
  03691459 → loudspeaker
  04256520 → sofa
  04379243 → table
  04530566 → vessel

输出格式与 prepare_data.py 完全一致:
  {output_dir}/
    dataset_meta.json
    model_XXXX/
      voxels.npy           # [N, N, N] float32
      silhouettes/*.png    # 6 视角正交投影
      loads.npz            # force_standing, fixed_mask_standing, ...
      fem_standing.npz     # (可选) displacement, stress, von_mises
      source.json          # ShapeNet 来源信息

用法:
    # 方式1: 从 ShapeNet 目录加载
    python prepare_shapenet.py \
        --shapenet_dir /path/to/ShapeNetCore.v2 \
        --output_dir ./shapenet_test_data \
        --categories chair,table,lamp \
        --per_category 5

    # 方式2: 从已有 .obj/.off/.stl 文件列表加载
    python prepare_shapenet.py \
        --mesh_list meshes.txt \
        --output_dir ./shapenet_test_data
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ShapeNet synsetId → 类别名称
SHAPENET_CATEGORIES = {
    '02691156': 'airplane',
    '02828884': 'bench',
    '02933112': 'cabinet',
    '03001627': 'chair',
    '03211117': 'display',
    '03636649': 'lamp',
    '03691459': 'loudspeaker',
    '04256520': 'sofa',
    '04379243': 'table',
    '04530566': 'vessel',
}

# 反向映射
CATEGORY_TO_SYNSET = {v: k for k, v in SHAPENET_CATEGORIES.items()}

# 推荐实验类别 (结构多样性好, 可打印性挑战大)
RECOMMENDED_CATEGORIES = ['chair', 'table', 'lamp', 'airplane', 'vessel']


def voxelize_mesh(
    mesh_path: str,
    resolution: int = 64,
    padding: int = 2,
) -> np.ndarray | None:
    """将 mesh 体素化为 [N, N, N] 占据网格

    使用 trimesh 库做 ray-based 体素化。
    若 trimesh 不可用, 退化为基于采样点的方法。

    Args:
        mesh_path: .obj / .off / .stl / .ply 文件路径
        resolution: 目标分辨率 N
        padding: 边界留空体素数

    Returns:
        [N, N, N] float32 占据网格 (0 or 1), or None if failed
    """
    try:
        import trimesh

        mesh = trimesh.load(mesh_path, force='mesh')

        if isinstance(mesh, trimesh.Scene):
            # 合并场景中的所有 mesh
            meshes = [g for g in mesh.geometry.values()
                      if isinstance(g, trimesh.Trimesh)]
            if not meshes:
                logger.warning(f"No valid mesh in scene: {mesh_path}")
                return None
            mesh = trimesh.util.concatenate(meshes)

        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            logger.warning(f"Invalid mesh: {mesh_path}")
            return None

        # 归一化到 unit cube
        mesh = _normalize_mesh(mesh)

        # 体素化
        inner_res = resolution - 2 * padding
        pitch = 1.0 / inner_res

        voxel_grid = mesh.voxelized(pitch=pitch)
        filled = voxel_grid.fill()  # 填充内部

        # 转换为 numpy 数组
        matrix = filled.matrix.astype(np.float32)

        # 放入目标分辨率 (居中 + padding)
        result = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        sx, sy, sz = matrix.shape
        # 裁剪到 inner_res
        sx = min(sx, inner_res)
        sy = min(sy, inner_res)
        sz = min(sz, inner_res)

        ox = padding + (inner_res - sx) // 2
        oy = padding + (inner_res - sy) // 2
        oz = padding + (inner_res - sz) // 2

        result[ox:ox+sx, oy:oy+sy, oz:oz+sz] = matrix[:sx, :sy, :sz]

        # 确保底面有实体 (可站立)
        result = _ensure_standing_base(result)

        if result.sum() < 20:
            logger.warning(f"Too few solid voxels ({result.sum():.0f}): {mesh_path}")
            return None

        return result

    except ImportError:
        logger.warning("trimesh not installed. pip install trimesh")
        return _voxelize_fallback(mesh_path, resolution, padding)
    except Exception as e:
        logger.warning(f"Voxelization failed for {mesh_path}: {e}")
        return None


def _normalize_mesh(mesh):
    """归一化 mesh 到 [0, 1]³ 中心"""
    import trimesh
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = (bounds[1] - bounds[0]).max()

    if scale < 1e-8:
        return mesh

    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / scale)
    mesh.apply_translation([0.5, 0.5, 0.5])

    return mesh


def _ensure_standing_base(voxels: np.ndarray) -> np.ndarray:
    """确保模型有站立底面 (底部至少有 1 层实体)

    如果模型浮空, 将其下移到底部 padding 区域。
    """
    solid = voxels > 0.5
    if not solid.any():
        return voxels

    # 找最低实体层 (Y 轴 = axis 1)
    for y in range(voxels.shape[1]):
        if solid[:, y, :].any():
            if y > 2:
                # 模型浮空, 下移
                shift = y - 2
                result = np.zeros_like(voxels)
                result[:, :voxels.shape[1]-shift, :] = voxels[:, shift:, :]
                return result
            return voxels

    return voxels


def _voxelize_fallback(
    mesh_path: str,
    resolution: int,
    padding: int,
) -> np.ndarray | None:
    """不依赖 trimesh 的简单体素化 (基于点云采样)"""
    try:
        # 尝试用 numpy 直接解析简单 OBJ
        vertices = _parse_obj_vertices(mesh_path)
        if vertices is None or len(vertices) < 10:
            return None

        # 归一化到 [padding, resolution-padding]
        vmin = vertices.min(axis=0)
        vmax = vertices.max(axis=0)
        scale = (vmax - vmin).max()
        if scale < 1e-8:
            return None

        inner = resolution - 2 * padding
        normed = (vertices - vmin) / scale * (inner - 1) + padding

        # 点云占据
        result = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        coords = np.round(normed).astype(int)
        coords = np.clip(coords, 0, resolution - 1)
        result[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0

        # 形态学闭运算填充间隙
        from scipy.ndimage import binary_closing, binary_fill_holes
        result = binary_closing(result > 0.5, iterations=2).astype(np.float32)

        # 沿每个轴做 fill_holes
        for axis in range(3):
            for i in range(resolution):
                slc = [slice(None)] * 3
                slc[axis] = i
                layer = result[tuple(slc)]
                if layer.any():
                    result[tuple(slc)] = binary_fill_holes(layer > 0.5).astype(np.float32)

        result = _ensure_standing_base(result)

        if result.sum() < 20:
            return None

        return result

    except Exception as e:
        logger.warning(f"Fallback voxelization failed: {e}")
        return None


def _parse_obj_vertices(path: str) -> np.ndarray | None:
    """从 OBJ 文件中解析顶点坐标"""
    vertices = []
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if vertices:
            return np.array(vertices, dtype=np.float32)
    except Exception:
        pass
    return None


def discover_shapenet_models(
    shapenet_dir: str,
    categories: list[str] | None = None,
    per_category: int = 5,
    seed: int = 42,
) -> List[Tuple[str, str, str]]:
    """从 ShapeNet 目录中发现模型

    ShapeNet Core v2 目录结构:
      {shapenet_dir}/{synsetId}/{modelId}/models/model_normalized.obj

    Args:
        shapenet_dir: ShapeNet 根目录
        categories: 要加载的类别名 (e.g. ['chair', 'table'])
        per_category: 每类加载的模型数
        seed: 随机种子 (用于选子集)

    Returns:
        list of (mesh_path, category_name, model_id)
    """
    shapenet_path = Path(shapenet_dir)
    if not shapenet_path.exists():
        logger.error(f"ShapeNet directory not found: {shapenet_dir}")
        return []

    if categories is None:
        categories = RECOMMENDED_CATEGORIES

    rng = np.random.default_rng(seed)
    models = []

    for cat_name in categories:
        synset = CATEGORY_TO_SYNSET.get(cat_name)
        if synset is None:
            logger.warning(f"Unknown category: {cat_name}")
            continue

        cat_dir = shapenet_path / synset
        if not cat_dir.exists():
            logger.warning(f"Category dir not found: {cat_dir}")
            continue

        # 列出所有 model_normalized.obj
        model_dirs = sorted([d for d in cat_dir.iterdir() if d.is_dir()])
        if not model_dirs:
            continue

        # 随机选取子集
        indices = rng.choice(len(model_dirs), size=min(per_category, len(model_dirs)), replace=False)
        for idx in sorted(indices):
            model_dir = model_dirs[idx]
            obj_path = model_dir / "models" / "model_normalized.obj"

            if not obj_path.exists():
                # 尝试其他格式
                for ext in ['.off', '.stl', '.ply']:
                    alt = model_dir / "models" / f"model_normalized{ext}"
                    if alt.exists():
                        obj_path = alt
                        break
                else:
                    continue

            models.append((str(obj_path), cat_name, model_dir.name))

    logger.info(f"Discovered {len(models)} ShapeNet models across {len(categories)} categories")
    return models


def discover_mesh_list(mesh_list_file: str) -> List[Tuple[str, str, str]]:
    """从文件列表加载 mesh 路径

    文件格式 (每行一个, 可选带类别):
      /path/to/model.obj
      /path/to/model.obj,chair
      /path/to/model.obj,chair,shapenet_id_123

    Returns:
        list of (mesh_path, category, model_id)
    """
    models = []
    with open(mesh_list_file) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            mesh_path = parts[0].strip()
            category = parts[1].strip() if len(parts) > 1 else 'unknown'
            model_id = parts[2].strip() if len(parts) > 2 else f"mesh_{i:04d}"
            models.append((mesh_path, category, model_id))
    return models


def prepare_shapenet_dataset(
    output_dir: str,
    shapenet_dir: str | None = None,
    mesh_list: str | None = None,
    categories: list[str] | None = None,
    per_category: int = 5,
    resolution: int = 64,
    n_views: int = 6,
    img_size: int = 128,
    seed: int = 42,
    run_fem: bool = True,
) -> dict:
    """准备 ShapeNet 测试数据集

    输出格式与 prepare_experiment_dataset() 完全一致。

    Args:
        output_dir: 输出目录
        shapenet_dir: ShapeNet Core v2 根目录
        mesh_list: 或者, mesh 路径列表文件
        categories: ShapeNet 类别 (默认: chair, table, lamp, airplane, vessel)
        per_category: 每类模型数
        resolution: 体素分辨率 (64)
        n_views: 剪影视角数 (6)
        img_size: 剪影尺寸 (128)
        seed: 随机种子
        run_fem: 是否计算 FEM 真值

    Returns:
        dataset_meta dict
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 发现模型
    if mesh_list:
        mesh_models = discover_mesh_list(mesh_list)
    elif shapenet_dir:
        mesh_models = discover_shapenet_models(
            shapenet_dir, categories, per_category, seed
        )
    else:
        raise ValueError("Must provide either shapenet_dir or mesh_list")

    if not mesh_models:
        logger.error("No models found!")
        return {'n_models': 0, 'models': []}

    from structural.experiments.prepare_data import render_silhouettes_orthographic
    from structural.analysis.load_generator import LoadGenerator

    load_gen = LoadGenerator(material_density=1.24e3)
    models_info = []
    generated = 0

    print(f"Processing {len(mesh_models)} ShapeNet models → {output_dir}")

    for mesh_path, category, source_id in mesh_models:
        # 体素化
        voxels = voxelize_mesh(mesh_path, resolution)
        if voxels is None:
            logger.warning(f"Skip: voxelization failed for {source_id}")
            continue

        model_id = f"model_{generated:04d}"
        model_dir = output_path / model_id
        model_dir.mkdir(exist_ok=True)

        # 保存体素
        np.save(model_dir / "voxels.npy", voxels)

        # 保存来源信息
        source_info = {
            'source': 'shapenet',
            'category': category,
            'source_id': source_id,
            'mesh_path': mesh_path,
        }
        with open(model_dir / "source.json", 'w') as f:
            json.dump(source_info, f, indent=2)

        # 渲染剪影
        sils = render_silhouettes_orthographic(voxels, n_views, img_size)
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

        # 生成载荷
        load_standing = load_gen.standing_load(voxels)
        load_gripping = load_gen.gripping_load(voxels)

        np.savez_compressed(
            model_dir / "loads.npz",
            force_standing=load_standing['force'],
            fixed_mask_standing=load_standing['fixed_mask'],
            force_gripping=load_gripping['force'],
            fixed_mask_gripping=load_gripping['fixed_mask'],
        )

        # FEM 真值 (可选)
        has_fem = False
        if run_fem:
            try:
                from structural.data.generate_fem_dataset import density_to_fem_fields
                from sgldbench.config import MaterialProperties, SolverSettings

                material = MaterialProperties(E=3.5e9, nu=0.36, cell_size=1.0)
                solver = SolverSettings(tolerance=1e-5, max_iterations=500)

                fem_data = density_to_fem_fields(
                    voxels.astype(bool),
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
                    has_fem = True
            except Exception as e:
                logger.warning(f"FEM failed for {model_id}: {e}")

        info = {
            'model_id': model_id,
            'category': category,
            'source_id': source_id,
            'solid_ratio': float(voxels.mean()),
            'n_voxels': int((voxels > 0.5).sum()),
            'silhouette_paths': sil_paths,
            'has_fem': has_fem,
        }
        models_info.append(info)
        generated += 1

        if generated % 5 == 0:
            print(f"  Processed {generated}/{len(mesh_models)} models")

    # 保存元信息
    meta = {
        'n_models': generated,
        'resolution': resolution,
        'n_views': n_views,
        'img_size': img_size,
        'seed': seed,
        'source': 'shapenet',
        'categories': list(set(m['category'] for m in models_info)),
        'category_counts': {},
        'models': models_info,
    }

    # 统计每类数量
    for m in models_info:
        cat = m['category']
        meta['category_counts'][cat] = meta['category_counts'].get(cat, 0) + 1

    with open(output_path / "dataset_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nShapeNet dataset ready: {generated} models in {output_dir}")
    print(f"  Categories: {meta['category_counts']}")

    return meta


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    parser = argparse.ArgumentParser(
        description='Prepare ShapeNet test dataset for StructShadow experiments'
    )
    parser.add_argument('--shapenet_dir', type=str, default=None,
                        help='Path to ShapeNetCore.v2 directory')
    parser.add_argument('--mesh_list', type=str, default=None,
                        help='Text file with mesh paths (one per line)')
    parser.add_argument('--output_dir', type=str, default='./shapenet_test_data')
    parser.add_argument('--categories', type=str, default='chair,table,lamp,airplane,vessel',
                        help='Comma-separated ShapeNet category names')
    parser.add_argument('--per_category', type=int, default=5)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_fem', action='store_true')

    args = parser.parse_args()

    cats = [c.strip() for c in args.categories.split(',')]

    prepare_shapenet_dataset(
        output_dir=args.output_dir,
        shapenet_dir=args.shapenet_dir,
        mesh_list=args.mesh_list,
        categories=cats,
        per_category=args.per_category,
        resolution=args.resolution,
        seed=args.seed,
        run_fem=not args.no_fem,
    )
