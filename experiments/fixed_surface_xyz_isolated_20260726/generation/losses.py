"""
losses.py
轮廓损失函数 + 体素 3D 正则化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


# ==================== MS-SSIM ====================
class MultiScaleSSIM(nn.Module):
    """多尺度结构相似性"""
    def __init__(self, window_size=11, size_average=True, device='cuda'):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.device = device
        self.weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
        self.window = self._create_window()
    
    def _create_window(self):
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - self.window_size//2)**2 / (2*sigma**2))
            for x in range(self.window_size)
        ])
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32, device=self.device)
    
    def _ssim(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def forward(self, img1, img2):
        if img1.dim() == 2:
            img1 = img1.unsqueeze(0).unsqueeze(0)
        if img2.dim() == 2:
            img2 = img2.unsqueeze(0).unsqueeze(0)
        
        img1 = img1.to(dtype=torch.float32, device=self.device)
        img2 = img2.to(dtype=torch.float32, device=self.device)
        
        if self.window.device != self.device:
            self.window = self.window.to(self.device)
        
        levels = self.weights.size(0)
        mssim = []
        
        for i in range(levels):
            ssim_val = self._ssim(img1, img2)
            mssim.append(ssim_val)
            
            if i < levels - 1:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
        
        mssim = torch.stack(mssim)
        value = (mssim ** self.weights).prod()
        return 1.0 - value


class FastSSIM(nn.Module):
    """快速单尺度SSIM"""
    def __init__(self, window_size=11, device='cuda'):
        super().__init__()
        self.window_size = window_size
        self.device = device
        window = self._create_window()
        self.register_buffer('window', window)
    
    def _create_window(self):
        sigma = 1.5
        gauss = torch.Tensor([
            np.exp(-(x - self.window_size//2)**2 / (2*sigma**2))
            for x in range(self.window_size)
        ])
        gauss = gauss / gauss.sum()
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32, device=self.device)
    
    def forward(self, img1, img2):
        if img1.dim() == 2:
            img1 = img1.unsqueeze(0).unsqueeze(0)
        if img2.dim() == 2:
            img2 = img2.unsqueeze(0).unsqueeze(0)
        
        img1 = img1.to(dtype=torch.float32, device=self.device)
        img2 = img2.to(dtype=torch.float32, device=self.device)
        
        if self.window.device != self.device:
            self.window = self.window.to(self.device)
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2) - mu12
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2*mu12 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1.0 - ssim_map.mean()


MS_SSIM = FastSSIM


# ==================== 基础损失（来自 mesh8.1.7）====================
def silh_loss(silh_pred, silh_gt):
    """轮廓L2损失"""
    silh_pred = silh_pred.to(dtype=torch.float32).clamp(0, 1)
    silh_gt = silh_gt.to(dtype=torch.float32).clamp(0, 1)
    return ((silh_pred - silh_gt) ** 2).mean()


def l1_loss(silh_pred, silh_gt):
    """L1损失"""
    silh_pred = silh_pred.to(dtype=torch.float32).clamp(0, 1)
    silh_gt = silh_gt.to(dtype=torch.float32).clamp(0, 1)
    return torch.abs(silh_pred - silh_gt).mean()


# ==================== 新增：可微分 IOU/Dice Loss ====================
def differentiable_iou_loss(pred, gt, smooth=1e-6):
    """
    可微分 IOU 损失
    使用软阈值保持可微性
    """
    pred = pred.to(dtype=torch.float32).clamp(0, 1)
    gt = gt.to(dtype=torch.float32).clamp(0, 1)
    
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return 1.0 - iou


def differentiable_dice_loss(pred, gt, smooth=1e-6):
    """
    可微分 Dice 损失
    """
    pred = pred.to(dtype=torch.float32).clamp(0, 1)
    gt = gt.to(dtype=torch.float32).clamp(0, 1)
    
    intersection = (pred * gt).sum()
    dice = (2 * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    return 1.0 - dice


# ==================== 新增：不对称轮廓损失（消除飞边的核心）====================
def asymmetric_silhouette_loss(silh_pred, silh_gt, fp_weight=3.0):
    """
    不对称轮廓损失 — 飞边消除的关键
    
    飞边产生的像素 = 假阳性（pred=白, gt=黑）
    这个损失对假阳性施加 fp_weight 倍的惩罚
    
    Args:
        silh_pred: 预测剪影 [H, W]
        silh_gt: GT剪影 [H, W]
        fp_weight: 假阳性（多余覆盖/飞边）的惩罚倍数
    """
    pred = silh_pred.to(dtype=torch.float32).clamp(0, 1)
    gt = silh_gt.to(dtype=torch.float32).clamp(0, 1)
    
    # 假阳性：pred有值但gt没有 → 飞边区域
    false_positive = pred * (1.0 - gt)      # pred白色 & gt黑色
    
    # 假阴性：gt有值但pred没有 → 缺失区域
    false_negative = (1.0 - pred) * gt      # pred黑色 & gt白色
    
    # 不对称惩罚：飞边区域惩罚 fp_weight 倍
    loss = (fp_weight * (false_positive ** 2) + (false_negative ** 2)).mean()
    
    return loss


def false_negative_only_loss(silh_pred, silh_gt):
    """
    只惩罚假阴性（缺失覆盖）— 不产生假阳性梯度
    
    用途：Phase 2 中替代 silh_loss，避免与不对称损失冲突
    silh_loss 的 L2 会推动 pred→gt 的双向梯度（包括拉回飞边处的像素）
    这个损失只推动 pred 去覆盖 gt=1 的区域，不惩罚 pred>gt 的区域
    """
    pred = silh_pred.to(dtype=torch.float32).clamp(0, 1)
    gt = silh_gt.to(dtype=torch.float32).clamp(0, 1)
    
    # 只惩罚 gt=1 但 pred=0 的区域
    fn_mask = gt * (1.0 - pred)  # gt白色 & pred黑色
    return (fn_mask ** 2).mean()


# ==================== 新增：Focal Tversky Loss（边界聚焦）====================
def focal_tversky_loss(pred, gt, alpha=0.3, beta=0.7, gamma=2.0, smooth=1e-6):
    """
    Focal Tversky Loss — 在高IOU阶段聚焦边界难样本

    alpha < beta → 更重惩罚假阳性（多余覆盖/飞边）
    gamma > 1 → 聚焦难分类的边界像素（类似 Focal Loss）

    当 IOU 已经很高时，Tversky Index 接近1，(1-TI)^gamma 让容易区域
    梯度几乎为0，只在边界困难区域产生有效梯度
    """
    pred = pred.to(dtype=torch.float32).clamp(0, 1)
    gt = gt.to(dtype=torch.float32).clamp(0, 1)

    tp = (pred * gt).sum()
    fp = (pred * (1.0 - gt)).sum()
    fn = ((1.0 - pred) * gt).sum()

    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    focal = (1.0 - tversky_index) ** gamma

    return focal


# ==================== 新增：Boundary Loss（距离变换加权）====================
def boundary_loss(pred, gt, device='cuda'):
    """
    Boundary Loss — 将梯度集中在轮廓边界附近

    原理：计算 GT 轮廓的距离变换，边界像素权重最高，
    远离边界的内部/外部像素权重低。这样优化器只需要
    精细调整边界位置，而不会被大面积正确区域的梯度稀释。

    对 IOU 从 0.97→0.98 特别有效，因为误差几乎全在边界。
    """
    pred = pred.to(dtype=torch.float32).clamp(0, 1)
    gt = gt.to(dtype=torch.float32).clamp(0, 1)

    # 计算 GT 边界的距离变换（作为权重图）
    with torch.no_grad():
        gt_np = (gt.detach().cpu().numpy() * 255).astype(np.uint8)

        # 内部距离：GT内部像素到边界的距离
        dist_inside = cv2.distanceTransform(gt_np, cv2.DIST_L2, 5)
        # 外部距离：GT外部像素到边界的距离
        dist_outside = cv2.distanceTransform(255 - gt_np, cv2.DIST_L2, 5)

        # 合并：所有像素到最近边界的距离
        dist_to_boundary = dist_inside + dist_outside

        # 转为权重：边界处权重最高，衰减系数控制影响范围
        # sigma=8 意味着距离边界8像素以内权重较高
        sigma = 8.0
        boundary_weight = np.exp(-(dist_to_boundary ** 2) / (2 * sigma ** 2))

        # 归一化使权重和约等于像素数（保持损失量级稳定）
        boundary_weight = boundary_weight / (boundary_weight.mean() + 1e-8)

        weight_tensor = torch.from_numpy(boundary_weight).to(dtype=torch.float32, device=device)

    # 加权 per-pixel L2 损失
    pixel_loss = (pred - gt) ** 2
    weighted_loss = (pixel_loss * weight_tensor).mean()

    return weighted_loss


# ==================== 体素 3D 正则化 ====================
def tv_regularization_3d(densities):
    """3D 全变分正则化：鼓励体素密度的空间平滑性（替代 mesh 的 laplacian/edge）"""
    dx = torch.abs(densities[:, 1:, :, :] - densities[:, :-1, :, :])
    dy = torch.abs(densities[:, :, 1:, :] - densities[:, :, :-1, :])
    dz = torch.abs(densities[:, :, :, 1:] - densities[:, :, :, :-1])
    return (dx.mean() + dy.mean() + dz.mean()) / 3.0


def sparsity_regularization(densities):
    """稀疏性正则化：鼓励密度接近 0 或 1（减少中间模糊值）"""
    eps = 1e-6
    p = densities.clamp(eps, 1 - eps)
    entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
    return entropy.mean()


def volume_minimization_loss(densities, hull_mask=None):
    """体积最小化损失：直接惩罚占据材料的总量（mean density）。

    与 sparsity(熵, 逼 0/1) 不同——它逼总体积↓。配合剪影损失(保覆盖)形成
    平衡: 只保留作为某剪影像素唯一支撑的体素 = 近最小剪影一致实体。
    hull_mask 给定时只罚 hull 内冗余(hull 外本就被禁止),梯度更聚焦。
    """
    if hull_mask is not None:
        m = hull_mask.to(densities.dtype)
        denom = m.sum().clamp_min(1.0)
        return (densities * m).sum() / denom
    return densities.mean()


def topo_connectivity_loss(densities, down=3):
    """可微拓扑(0维持续同调)连通损失:惩罚所有"会死亡的"连通分量的持续性,
    使其合并为单一分量。其梯度抬高分隔分量的鞍点密度(长桥连接)而非删除体素
    ——配合剪影损失(撑住峰)→ 净效果=建桥。修复了 reachability 损失只激励删除的根因。
    用 gudhi cubical persistence(detached 计算临界点,torch 索引保可微)。"""
    import gudhi
    f = densities
    if f.dim() == 4:
        f = f[0]
    if down > 1:
        f = F.avg_pool3d(f[None, None], down, down)[0, 0]
    g = 1.0 - f
    gnp = g.detach().cpu().numpy().astype(np.float64)
    cc = gudhi.CubicalComplex(top_dimensional_cells=gnp)
    cc.compute_persistence()
    pairs = cc.cofaces_of_persistence_pairs()
    if len(pairs[0]) == 0 or len(pairs[0][0]) == 0:
        return (f * 0.0).sum()
    p0 = np.asarray(pairs[0][0])  # (N,2) 0维有限对的顶维cell flat索引
    gflat = g.reshape(-1)
    b = torch.as_tensor(p0[:, 0], device=f.device, dtype=torch.long)
    d = torch.as_tensor(p0[:, 1], device=f.device, dtype=torch.long)
    return torch.abs(gflat[d] - gflat[b]).sum()   # abs=正确持续度(非负);梯度抬鞍点(建桥)


def connectivity_reachability_loss(densities, steps=48, down=4):
    """可微 6 连通损失：从最密种子做面相邻可达性扩散，惩罚"占据但从种子经
    高密度面路径不可达"的体素。直接优化 6 连通(非厚度代理)。

    R ← min(density, max(R, faceNeighborMax(R))) 迭代 steps 步;损失=mean(d·(1-R))。
    在 down 倍降采样网格上传播(连通是粗尺度属性,省算力)。
    """
    d = densities
    if d.dim() == 4:
        d = d.unsqueeze(0)
    x = d[:, :1]                                   # [1,1,D,H,W]
    if down > 1:
        x = F.avg_pool3d(x, kernel_size=down, stride=down)
    dd = x
    with torch.no_grad():
        flat = dd.reshape(-1)
        seed = torch.zeros_like(flat)
        seed[torch.argmax(flat)] = 1.0
        seed = seed.reshape(dd.shape)
    R = seed * dd
    def shift_max(R):
        m = R
        z = torch.zeros_like(R)
        for dim, sl_src, sl_dst in [
            (2, slice(0, -1), slice(1, None)), (2, slice(1, None), slice(0, -1)),
            (3, slice(0, -1), slice(1, None)), (3, slice(1, None), slice(0, -1)),
            (4, slice(0, -1), slice(1, None)), (4, slice(1, None), slice(0, -1)),
        ]:
            s = z.clone()
            idx_dst = [slice(None)] * 5; idx_dst[dim] = sl_dst
            idx_src = [slice(None)] * 5; idx_src[dim] = sl_src
            s[tuple(idx_dst)] = R[tuple(idx_src)]
            m = torch.maximum(m, s)
        return m
    for _ in range(steps):
        R = torch.minimum(dd, torch.maximum(R, shift_max(R)))
    return (dd * (1.0 - R)).mean()


def isolation_penalty(densities):
    """孤立体素惩罚：惩罚邻域为空但自身密度高的体素（消除飞边/游离体素）

    对于主体内部体素：邻域密度高 → 惩罚≈0
    对于表面体素：邻域密度中等 → 惩罚小
    对于孤立飞散体素：邻域密度低 → 惩罚大
    """
    kernel = torch.zeros(1, 1, 3, 3, 3, device=densities.device)
    kernel[0, 0, 0, 1, 1] = 1  # -x
    kernel[0, 0, 2, 1, 1] = 1  # +x
    kernel[0, 0, 1, 0, 1] = 1  # -y
    kernel[0, 0, 1, 2, 1] = 1  # +y
    kernel[0, 0, 1, 1, 0] = 1  # -z
    kernel[0, 0, 1, 1, 2] = 1  # +z
    kernel /= 6.0

    d = densities.unsqueeze(0)  # [1, 1, D, H, W]
    neighbor_avg = F.conv3d(d, kernel, padding=1).squeeze(0)  # [1, D, H, W]

    # density * (1 - neighbor_avg)^3
    # 孤立体素: ~density * 1.0, 薄条纹: ~density * 0.30, 表面: ~density * 0.125, 内部: ~0
    penalty = densities * (1 - neighbor_avg) ** 3
    return penalty.mean()


def support_regularization_3d(densities, min_support=0.30, wide_support=0.18):
    """局部支撑正则：同时约束近邻和稍大范围支撑，减少细颈/细丝连接。"""
    near_kernel = torch.ones(1, 1, 3, 3, 3, device=densities.device)
    near_kernel[0, 0, 1, 1, 1] = 0.0
    near_kernel /= near_kernel.sum()

    wide_kernel = torch.ones(1, 1, 5, 5, 5, device=densities.device)
    wide_kernel[0, 0, 2, 2, 2] = 0.0
    wide_kernel /= wide_kernel.sum()

    d = densities.unsqueeze(0)  # [1, 1, D, H, W]
    near_avg = F.conv3d(d, near_kernel, padding=1).squeeze(0)
    wide_avg = F.conv3d(d, wide_kernel, padding=2).squeeze(0)
    combined_support = 0.72 * near_avg + 0.28 * wide_avg
    support_deficit = F.relu(float(min_support) - combined_support)
    wide_deficit = F.relu(float(wide_support) - wide_avg)
    penalty = densities * (support_deficit ** 2) * (1.0 + 1.5 * wide_deficit)
    return penalty.mean()


def export_consistency_regularization_3d(
    densities,
    target_volume,
    inside_weight=1.2,
    outside_weight=3.5,
    dice_weight=0.35,
    smooth=1e-6,
):
    """
    将“最终导出体素”先验直接拉入训练。

    - target_volume=1 的区域，惩罚缺失体素
    - target_volume=0 的区域，强惩罚外溢/飞边
    - 附加一个 soft Dice，避免只靠逐体素 L2 导致边界碎裂
    """
    pred = densities.to(dtype=torch.float32).clamp(0, 1)
    target = target_volume.to(dtype=torch.float32, device=pred.device).clamp(0, 1)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    missing = (1.0 - pred) * target
    overflow = pred * (1.0 - target)
    pointwise = float(inside_weight) * (missing ** 2) + float(outside_weight) * (overflow ** 2)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return pointwise.mean() + float(dice_weight) * (1.0 - dice)


def manufacturability_regularization_3d(
    densities,
    min_neighbor=0.30,
    min_axis_support=0.22,
    wide_support=0.12,
):
    """
    比简单 connected cleanup 更强的局部制造性先验。

    惩罚三类结构：
    - 邻域几乎为空的孤立块
    - 只有一像素厚、缺少轴向支撑的薄片/飞边
    - 大范围支撑不足的细颈连接
    """
    d = densities.to(dtype=torch.float32)
    if d.dim() == 4:
        d5 = d.unsqueeze(0)
    elif d.dim() == 5:
        d5 = d
        d = d.squeeze(0)
    else:
        raise ValueError("densities must have shape [C,D,H,W] or [N,C,D,H,W]")

    near_kernel = torch.ones(1, 1, 3, 3, 3, device=d.device, dtype=d.dtype)
    near_kernel[0, 0, 1, 1, 1] = 0.0
    near_kernel /= near_kernel.sum()

    wide_kernel = torch.ones(1, 1, 5, 5, 5, device=d.device, dtype=d.dtype)
    wide_kernel[0, 0, 2, 2, 2] = 0.0
    wide_kernel /= wide_kernel.sum()

    line_x = torch.ones(1, 1, 5, 1, 1, device=d.device, dtype=d.dtype) / 5.0
    line_y = torch.ones(1, 1, 1, 5, 1, device=d.device, dtype=d.dtype) / 5.0
    line_z = torch.ones(1, 1, 1, 1, 5, device=d.device, dtype=d.dtype) / 5.0

    near_avg = F.conv3d(d5, near_kernel, padding=1).squeeze(0)
    wide_avg = F.conv3d(d5, wide_kernel, padding=2).squeeze(0)
    axis_x = F.conv3d(d5, line_x, padding=(2, 0, 0)).squeeze(0)
    axis_y = F.conv3d(d5, line_y, padding=(0, 2, 0)).squeeze(0)
    axis_z = F.conv3d(d5, line_z, padding=(0, 0, 2)).squeeze(0)
    axial_support = torch.maximum(torch.maximum(axis_x, axis_y), axis_z)

    iso_deficit = F.relu(float(min_neighbor) - near_avg)
    axis_deficit = F.relu(float(min_axis_support) - axial_support)
    wide_deficit = F.relu(float(wide_support) - wide_avg)

    penalty = d * (
        1.35 * (iso_deficit ** 2) +
        1.10 * (axis_deficit ** 2) +
        0.65 * (wide_deficit ** 2)
    )
    return penalty.mean()


def main_structure_regularization_3d(
    densities,
    core_support=0.16,
    axis_thickness=0.12,
):
    """
    主结构优先先验。

    用大尺度局部平均近似“是否属于主体”，惩罚离开主体核心的游离体素；
    同时约束三个轴向上的最小厚度，减少细颈、尖刺和单层薄片。
    """
    d = densities.to(dtype=torch.float32)
    if d.dim() == 4:
        d5 = d.unsqueeze(0)
    elif d.dim() == 5:
        d5 = d
        d = d.squeeze(0)
    else:
        raise ValueError("densities must have shape [C,D,H,W] or [N,C,D,H,W]")

    core_kernel = torch.ones(1, 1, 7, 7, 7, device=d.device, dtype=d.dtype)
    core_kernel[0, 0, 3, 3, 3] = 0.0
    core_kernel /= core_kernel.sum()

    line_x = torch.ones(1, 1, 7, 1, 1, device=d.device, dtype=d.dtype) / 7.0
    line_y = torch.ones(1, 1, 1, 7, 1, device=d.device, dtype=d.dtype) / 7.0
    line_z = torch.ones(1, 1, 1, 1, 7, device=d.device, dtype=d.dtype) / 7.0

    core_avg = F.conv3d(d5, core_kernel, padding=3).squeeze(0)
    thick_x = F.conv3d(d5, line_x, padding=(3, 0, 0)).squeeze(0)
    thick_y = F.conv3d(d5, line_y, padding=(0, 3, 0)).squeeze(0)
    thick_z = F.conv3d(d5, line_z, padding=(0, 0, 3)).squeeze(0)
    min_thickness = torch.minimum(torch.minimum(thick_x, thick_y), thick_z)

    core_deficit = F.relu(float(core_support) - core_avg)
    thickness_deficit = F.relu(float(axis_thickness) - min_thickness)
    penalty = d * (1.45 * (core_deficit ** 2) + 1.10 * (thickness_deficit ** 2))
    return penalty.mean()
