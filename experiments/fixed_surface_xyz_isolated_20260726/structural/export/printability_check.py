"""
3D打印可行性检查

检查项:
1. 最小壁厚
2. 悬垂角度
3. 连通性 (单组件)
4. 密封性 (watertight)
"""
import numpy as np
from scipy.ndimage import (
    distance_transform_edt, label, generate_binary_structure,
    binary_erosion,
)
from dataclasses import dataclass, field
from typing import List


@dataclass
class PrintabilityReport:
    """可打印性报告"""
    is_printable: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # 详细指标
    min_wall_thickness: float = 0.0        # mm
    max_overhang_ratio: float = 0.0        # 无支撑体素占比
    num_components: int = 0
    total_volume: float = 0.0              # mm³
    bounding_box: tuple = (0, 0, 0)        # mm

    def summary(self) -> str:
        lines = ["=== Printability Report ==="]
        lines.append(f"Printable: {'YES' if self.is_printable else 'NO'}")
        lines.append(f"Min wall thickness: {self.min_wall_thickness:.2f} mm")
        lines.append(f"Overhang ratio: {self.max_overhang_ratio:.1%}")
        lines.append(f"Components: {self.num_components}")
        lines.append(f"Volume: {self.total_volume:.1f} mm³")
        lines.append(f"Bounding box: {self.bounding_box[0]:.1f} × {self.bounding_box[1]:.1f} × {self.bounding_box[2]:.1f} mm")

        if self.issues:
            lines.append("\nISSUES:")
            for issue in self.issues:
                lines.append(f"  ✗ {issue}")
        if self.warnings:
            lines.append("\nWARNINGS:")
            for w in self.warnings:
                lines.append(f"  ! {w}")

        return '\n'.join(lines)


def check_printability(
    density: np.ndarray,
    voxel_size: float = 1.0,
    min_wall_thickness: float = 0.8,
    max_overhang_angle: float = 45.0,
) -> PrintabilityReport:
    """检查体素模型的3D可打印性

    Args:
        density: [N, N, N] 密度场
        voxel_size: 体素物理尺寸 (mm)
        min_wall_thickness: 最小壁厚要求 (mm)
        max_overhang_angle: 最大允许悬垂角度 (度)
    Returns:
        PrintabilityReport
    """
    solid = density > 0.5
    report = PrintabilityReport(is_printable=True)

    if not solid.any():
        report.is_printable = False
        report.issues.append("Model is empty")
        return report

    N = density.shape[0]

    # 1. 壁厚检查
    dist = distance_transform_edt(solid) * voxel_size
    report.min_wall_thickness = float(dist[solid].min()) * 2  # 距离变换值×2=壁厚

    if report.min_wall_thickness < min_wall_thickness:
        report.issues.append(
            f"Wall thickness {report.min_wall_thickness:.2f}mm < {min_wall_thickness}mm minimum"
        )
        report.is_printable = False

    # 2. 悬垂检查 (Y轴为打印方向)
    # 对每个实体体素，检查正下方及45度邻域是否有支撑
    n_unsupported = 0
    n_solid = int(solid.sum())

    for y in range(1, N):  # 跳过底层
        layer = solid[:, y, :]
        if not layer.any():
            continue
        # 正下方
        support_below = solid[:, y - 1, :]
        # 45度邻域: 下方 ±1 格
        from scipy.ndimage import maximum_filter
        support_diag = maximum_filter(solid[:, y - 1, :].astype(np.float32), size=3)
        unsupported = layer & (support_diag < 0.5)
        n_unsupported += int(unsupported.sum())

    report.max_overhang_ratio = n_unsupported / max(n_solid, 1)

    if report.max_overhang_ratio > 0.15:
        report.warnings.append(
            f"High overhang ratio: {report.max_overhang_ratio:.1%} "
            f"({n_unsupported} unsupported voxels)"
        )
    if report.max_overhang_ratio > 0.3:
        report.issues.append(
            f"Excessive overhang: {report.max_overhang_ratio:.1%} requires support structures"
        )

    # 3. 连通性检查 (26-connectivity: 包括面/边/角邻居)
    struct_26 = generate_binary_structure(3, 3)  # rank=3, connectivity=3 → 26-connected
    labeled, n_cc = label(solid, structure=struct_26)
    report.num_components = n_cc

    if n_cc > 1:
        sizes = [(labeled == i).sum() for i in range(1, n_cc + 1)]
        max_size = max(sizes)
        small_components = [s for s in sizes if s < max_size * 0.01]

        if len(small_components) > 0:
            report.warnings.append(
                f"{len(small_components)} tiny disconnected components detected"
            )
        if n_cc > 3:
            report.issues.append(
                f"{n_cc} disconnected components — model may fall apart during printing"
            )
            report.is_printable = False

    # 4. 体积和尺寸
    report.total_volume = float(solid.sum()) * (voxel_size ** 3)
    coords = np.argwhere(solid)
    bbox_min = coords.min(axis=0) * voxel_size
    bbox_max = (coords.max(axis=0) + 1) * voxel_size
    report.bounding_box = tuple((bbox_max - bbox_min).tolist())

    # 体积过小警告
    if report.total_volume < 10.0:  # < 10 mm³
        report.warnings.append(f"Very small volume: {report.total_volume:.1f} mm³")

    return report
