"""泛化 TSF(k 正交轴, k=2/3)严格评估 —— 新模块, 不改 voxel_ops.py.
fixed/free = 视图->轴指派(identity / 全置换), mr=0/1 = 是否加镜像(fliplr)视图.
build/project 的逆-d4 约定与 voxel_ops 完全一致 => k=3,mr=1 复刻官方 0.977/0.994."""
import itertools
import numpy as np
from voxel_ops import (apply_d4_transform, resize_binary_mask, compute_iou_np,
                       _get_inverse_d4_map)
INV = _get_inverse_d4_map()
ROT = [0, 1, 2, 3]; ALLD4 = list(range(8))

def _base_proj(vol):
    return {0: vol.max(axis=0), 1: vol.max(axis=1), 2: vol.max(axis=2)}  # [Y,Z],[X,Z],[X,Y]
def _extrude(plane, axis, R):
    if axis == 0: return np.broadcast_to(plane[None, :, :], (R, R, R))
    if axis == 1: return np.broadcast_to(plane[:, None, :], (R, R, R))
    return np.broadcast_to(plane[:, :, None], (R, R, R))

def build_hull_k(sils, perm, ops, R):
    """sils[i] 指派到轴 perm[i], 朝向 op ops[i]. 缺失轴自由."""
    occ = np.ones((R, R, R), bool)
    for i, (ax, op) in enumerate(zip(perm, ops)):
        plane = apply_d4_transform(resize_binary_mask(sils[i], R), INV[op]).astype(bool)
        occ = occ & _extrude(plane, ax, R)
    return occ

def eval_k(vol, sils, perm, ops, mr, imsz, msils=None):
    """严格: 沿指派轴投影, 与基础剪影比; mr=1 时 fliplr(投影) 与【真实镜像目标 msils】比. bool-safe."""
    bp = _base_proj(vol); ious = []
    for i, (ax, op) in enumerate(zip(perm, ops)):
        pred = apply_d4_transform(bp[ax], op).astype(bool)
        predr = resize_binary_mask(pred, imsz)                # bool 直传
        ious.append(compute_iou_np(predr, resize_binary_mask(sils[i], imsz)))
        if mr and msils is not None:                          # 真实镜像目标
            ious.append(compute_iou_np(resize_binary_mask(np.fliplr(pred), imsz),
                                       resize_binary_mask(msils[i], imsz)))
    return float(np.mean(ious))

# 官方固定约定: axis_for_view = (z, x, y) = 索引 (2,0,1), 取前 k 个
OFFICIAL_PERM = (2, 0, 1)
def search_calib(sils, k, fixed, mr, Rs=64, msils=None, identity_assign=False):
    """fixed: 无搜索(固定轴指派 + 朝向 ops=0). free: 全 k-轴置换 x 全 d4 朝向.
    fixed 默认用官方约定 (z,x,y); identity_assign=True 时降级为 (0,1,2)."""
    if fixed:
        perm = tuple(range(k)) if identity_assign else OFFICIAL_PERM[:k]
        return perm, tuple([0] * k)                            # 完全不搜索
    axperm = list(itertools.permutations(range(3), k))
    best = None
    for perm in axperm:
        for ops in itertools.product(ALLD4, repeat=k):
            h = build_hull_k(sils, perm, ops, Rs)
            v = eval_k(h, sils, perm, ops, mr, Rs, msils)
            if best is None or v > best[0]: best = (v, perm, ops)
    return best[1], best[2]
