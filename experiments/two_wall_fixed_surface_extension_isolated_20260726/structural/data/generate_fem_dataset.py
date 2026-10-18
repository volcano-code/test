"""
FEM数据集生成脚本

使用SGLDBench的FEM求解器生成PINO训练数据:
- 随机/参数化体素模型
- 多种载荷工况 (standing/gripping)
- 输出: {ρ, F, u, σ, σ_vm} 以.npz格式存储

用法:
    python generate_fem_dataset.py --num_samples 2000 --resolution 64 --output_dir ./fem_dataset
"""
import sys
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 路径设置
PROJECT_ROOT = Path(__file__).parent.parent.parent
SGLDBENCH_DIR = PROJECT_ROOT / "SGLDBench" / "python" / "src"
sys.path.insert(0, str(SGLDBENCH_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "structural"))

from sgldbench.config import MaterialProperties, SolverSettings
from sgldbench.models import VoxelModel, BoundaryConditions
from sgldbench.discretization import discretize_voxel_model, apply_boundary_conditions
from sgldbench.fea import solve_linear_elasticity, run_stress_analysis
from sgldbench.geometry import make_cuboid_model, make_cylinder_model, make_lshape_model

from analysis.load_generator import LoadGenerator


def generate_random_shape(resolution: int, rng: np.random.Generator, _depth: int = 0) -> np.ndarray:
    """生成随机体素形状

    随机选择:
    1. 基本几何体 (长方体, 圆柱, L形)
    2. 随机blob (高斯核卷积)
    3. 组合形状 (递归深度≤2)
    """
    N = resolution
    if _depth >= 2:
        shape_type = rng.choice(['cuboid', 'cylinder', 'lshape', 'blob'])
    else:
        shape_type = rng.choice(['cuboid', 'cylinder', 'lshape', 'blob', 'composite'])

    if shape_type == 'cuboid':
        # 随机尺寸长方体
        sx = rng.integers(N // 4, N * 3 // 4)
        sy = rng.integers(N // 4, N * 3 // 4)
        sz = rng.integers(N // 4, N * 3 // 4)
        vol = np.zeros((N, N, N), dtype=bool)
        ox = (N - sx) // 2
        oy = (N - sy) // 2
        oz = (N - sz) // 2
        vol[ox:ox+sx, oy:oy+sy, oz:oz+sz] = True
        return vol

    elif shape_type == 'cylinder':
        vol = np.zeros((N, N, N), dtype=bool)
        radius = rng.integers(N // 6, N // 3)
        height = rng.integers(N // 3, N * 2 // 3)
        cx, cz = N // 2, N // 2
        oy = (N - height) // 2
        for y in range(oy, oy + height):
            for x in range(N):
                for z in range(N):
                    if (x - cx)**2 + (z - cz)**2 <= radius**2:
                        vol[x, y, z] = True
        return vol

    elif shape_type == 'lshape':
        vol = np.zeros((N, N, N), dtype=bool)
        w = rng.integers(N // 4, N // 2)
        h = rng.integers(N // 2, N * 3 // 4)
        d = rng.integers(N // 4, N // 2)
        # 主体
        vol[:w, :h, :d] = True
        # L的臂
        arm_len = rng.integers(N // 4, N // 2)
        vol[:w, :w, :d + arm_len] = True
        return vol

    elif shape_type == 'blob':
        from scipy.ndimage import gaussian_filter
        noise = rng.standard_normal((N, N, N)).astype(np.float32)
        sigma = rng.uniform(3, 8)
        smooth = gaussian_filter(noise, sigma=sigma)
        threshold = np.percentile(smooth, rng.uniform(55, 75))
        vol = smooth > threshold
        # 保留最大连通分量
        from scipy.ndimage import label
        labeled, n_cc = label(vol)
        if n_cc > 0:
            sizes = [(labeled == i).sum() for i in range(1, n_cc + 1)]
            largest = np.argmax(sizes) + 1
            vol = labeled == largest
        return vol

    else:  # composite
        vol = np.zeros((N, N, N), dtype=bool)
        n_parts = rng.integers(2, 4)
        for _ in range(n_parts):
            sub = generate_random_shape(N, rng, _depth=_depth + 1)
            # 随机平移
            shift = rng.integers(-N // 4, N // 4, size=3)
            sub = np.roll(sub, shift[0], axis=0)
            sub = np.roll(sub, shift[1], axis=1)
            sub = np.roll(sub, shift[2], axis=2)
            vol |= sub
        return vol


def density_to_fem_fields(
    density: np.ndarray,
    force: np.ndarray,
    fixed_mask: np.ndarray,
    material: MaterialProperties,
    solver: SolverSettings,
) -> dict | None:
    """对单个样本运行FEM分析，返回全场结果

    Args:
        density: [N, N, N] bool
        force: [3, N, N, N] 力场
        fixed_mask: [1, N, N, N] 固定掩码
    Returns:
        dict with fields or None if FEM fails
    """
    N = density.shape[0]

    if density.sum() < 8:
        return None

    # 创建VoxelModel
    volume = density.astype(bool)
    voxel_model = VoxelModel(
        resolution=(N, N, N),
        volume=volume,
    )

    # 网格离散化
    try:
        mesh = discretize_voxel_model(voxel_model)
    except ValueError:
        return None

    # 构建边界条件
    n_nodes = mesh.num_nodes
    load_vector = np.zeros((mesh.num_dofs,), dtype=np.float64)
    fixed_dof_mask = np.zeros((mesh.num_dofs,), dtype=bool)

    # 将体素级力场映射到节点
    node_map_back = mesh.node_map_back
    node_shape = (N + 1, N + 1, N + 1)

    # 将力场从体素映射到节点(简单平均)
    for elem_idx in range(mesh.num_elements):
        grid_coord = mesh.element_grid_coords[elem_idx]
        y, x, z = grid_coord
        fx, fy, fz = force[0, x, y, z], force[1, x, y, z], force[2, x, y, z]

        # 均分到8个节点
        node_ids = mesh.element_nodes[elem_idx]
        for nid in node_ids:
            base = nid * 3
            load_vector[base] += fx / 8.0
            load_vector[base + 1] += fy / 8.0
            load_vector[base + 2] += fz / 8.0

    # 固定节点
    for elem_idx in range(mesh.num_elements):
        grid_coord = mesh.element_grid_coords[elem_idx]
        y, x, z = grid_coord
        if fixed_mask[0, x, y, z] > 0.5:
            node_ids = mesh.element_nodes[elem_idx]
            for nid in node_ids:
                base = nid * 3
                fixed_dof_mask[base:base + 3] = True

    if not fixed_dof_mask.any():
        return None
    if not (np.abs(load_vector) > 1e-20).any():
        return None

    from sgldbench.models import AppliedBoundaryConditions
    applied_bc = AppliedBoundaryConditions(
        load_vector=load_vector,
        fixed_dofs=np.flatnonzero(fixed_dof_mask),
        free_dofs=np.flatnonzero(~fixed_dof_mask),
        mapped_fixations=np.zeros((0, 4)),
        mapped_loads=np.zeros((0, 4)),
    )

    # FEM求解
    try:
        fea_result = solve_linear_elasticity(mesh, applied_bc, material, solver)
        stress_result = run_stress_analysis(mesh, fea_result, material)
    except Exception as e:
        logger.debug(f"FEM solve failed: {e}")
        return None

    # 映射回全场
    displacement_field = np.zeros((3, N, N, N), dtype=np.float32)
    stress_field = np.zeros((6, N, N, N), dtype=np.float32)
    von_mises_field = np.zeros((N, N, N), dtype=np.float32)

    for elem_idx in range(mesh.num_elements):
        grid_coord = mesh.element_grid_coords[elem_idx]
        y, x, z = grid_coord

        # 位移: 取元素节点平均
        node_ids = mesh.element_nodes[elem_idx]
        u_elem = np.zeros(3)
        for nid in node_ids:
            base = nid * 3
            u_elem += fea_result.displacements[base:base + 3]
        u_elem /= 8.0
        displacement_field[:, x, y, z] = u_elem.astype(np.float32)

        # 应力
        stress_field[:, x, y, z] = stress_result.element_stress[elem_idx].astype(np.float32)
        von_mises_field[x, y, z] = stress_result.element_von_mises[elem_idx].astype(np.float32)

    return {
        'density': density.astype(np.float32),
        'force': force.astype(np.float32),
        'fixed_mask': fixed_mask.astype(np.float32),
        'displacement': displacement_field,
        'stress': stress_field,
        'von_mises': von_mises_field,
    }


def generate_dataset(
    num_samples: int = 2000,
    resolution: int = 64,
    output_dir: str = './fem_dataset',
    seed: int = 42,
):
    """生成完整FEM数据集"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    load_gen = LoadGenerator()
    material = MaterialProperties(
        youngs_modulus=3.5e9,
        poisson_ratio=0.36,
        cell_size=1.0,
    )
    solver = SolverSettings(tolerance=1e-5, max_iterations=500)

    generated = 0
    attempted = 0
    pbar = tqdm(total=num_samples, desc="Generating FEM dataset")

    while generated < num_samples:
        attempted += 1

        # 生成随机形状
        volume = generate_random_shape(resolution, rng)
        density = volume.astype(np.float32)

        # 生成随机载荷
        load_dict = load_gen.generate_random_load(density, rng)

        # FEM求解
        result = density_to_fem_fields(
            density, load_dict['force'], load_dict['fixed_mask'],
            material, solver,
        )

        if result is None:
            continue

        # 检查结果有效性
        if not np.all(np.isfinite(result['displacement'])):
            continue
        if not np.all(np.isfinite(result['stress'])):
            continue

        # 保存
        np.savez_compressed(
            output_path / f"sample_{generated:05d}.npz",
            **result,
        )

        generated += 1
        pbar.update(1)
        pbar.set_postfix(success_rate=f"{generated}/{attempted}")

    pbar.close()
    print(f"\nDataset generated: {generated} samples in {output_dir}")
    print(f"Success rate: {generated}/{attempted} ({100*generated/attempted:.1f}%)")

    # 计算归一化统计量 (遍历所有样本)
    print("Computing normalization statistics...")
    stats = {
        'force_absmax': 0.0,
        'displacement_absmax': 0.0,
        'stress_absmax': 0.0,
    }
    for i in range(generated):
        data = np.load(output_path / f"sample_{i:05d}.npz")
        stats['force_absmax'] = max(stats['force_absmax'],
                                     float(np.abs(data['force']).max()))
        stats['displacement_absmax'] = max(stats['displacement_absmax'],
                                            float(np.abs(data['displacement']).max()))
        stats['stress_absmax'] = max(stats['stress_absmax'],
                                      float(np.abs(data['stress']).max()))

    # 避免除以0
    for k in stats:
        if stats[k] < 1e-20:
            stats[k] = 1.0
    print(f"  Force absmax:        {stats['force_absmax']:.4e}")
    print(f"  Displacement absmax: {stats['displacement_absmax']:.4e}")
    print(f"  Stress absmax:       {stats['stress_absmax']:.4e}")

    # 保存元信息 (含归一化统计)
    meta = {
        'num_samples': generated,
        'resolution': resolution,
        'material': {
            'youngs_modulus': material.youngs_modulus,
            'poisson_ratio': material.poisson_ratio,
        },
        'seed': seed,
        'normalization': stats,
    }
    import json
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate FEM dataset for PINO training')
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./fem_dataset')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        num_samples=args.num_samples,
        resolution=args.resolution,
        output_dir=args.output_dir,
        seed=args.seed,
    )
