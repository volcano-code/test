"""
train.py — 体素 Shadow Art 三阶段训练
"""
import sys
import os
import random
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm

if os.environ.get("SHADOWART_DISABLE_CUDNN", "0") == "1":
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

from models import VolumeModel
from losses import (
    silh_loss, l1_loss, MS_SSIM,
    tv_regularization_3d, isolation_penalty, sparsity_regularization,
    support_regularization_3d,
    export_consistency_regularization_3d,
    main_structure_regularization_3d,
    manufacturability_regularization_3d,
    asymmetric_silhouette_loss, false_negative_only_loss,
    differentiable_iou_loss, differentiable_dice_loss, focal_tversky_loss,
    boundary_loss, volume_minimization_loss, connectivity_reachability_loss, topo_connectivity_loss,
)
from utils import (
    get_voxel_renderer, create_cameras, create_cameras_TFS_mode,
    create_cameras_4VTFS_mode,
)
from datasets import load_data_from_list
from voxel_ops import (
    analyze_connected_components,
    build_multiview_visual_hull,
    build_tfs_visual_hull,
    calibrate_tfs_projection,
    choose_best_multiview_candidate,
    constrain_printable_candidate_pool,
    connect_binary_volume,
    choose_best_tfs_candidate,
    clean_binary_volume,
    estimate_voxel_normal_consistency,
    evaluate_multiview_volume,
    evaluate_tfs_volume,
    minimize_material_printable,
    make_watertight_printable,
    project_tfs_density_torch,
    repair_multiview_projection_gaps,
    repair_tfs_projection_gaps,
    resize_binary_mask,
    resize_binary_volume,
    resize_float_volume,
    save_voxel_mesh_as_obj,
)

from pytorch3d.renderer import FoVOrthographicCameras
from export_high_quality_mesh import export_reconstructed_mesh

try:
    from post_export import generate_post_export_bundle
    POST_EXPORT_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - defensive import guard
    generate_post_export_bundle = None
    POST_EXPORT_IMPORT_ERROR = exc


def compute_visual_hull(silhs_tensors, Rs, Ts, volume_size, device):
    """Visual hull 初始化：多视角反向投影求交集"""
    print("  Computing visual hull ...")
    coords = torch.linspace(-0.85, 0.85, volume_size, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(coords, coords, coords, indexing='ij')
    points_3d = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)

    view_votes = torch.zeros(volume_size ** 3, device=device)

    for view_idx, silh_gt in enumerate(silhs_tensors):
        R = Rs[view_idx].to(device)
        T = Ts[view_idx].to(device)
        points_cam = points_3d @ R.T + T

        img_size = silh_gt.shape[0]
        proj_x = (points_cam[:, 0] + 0.85) / 1.7 * (img_size - 1)
        proj_y = (points_cam[:, 1] + 0.85) / 1.7 * (img_size - 1)

        x0 = torch.floor(proj_x).long()
        y0 = torch.floor(proj_y).long()
        x1 = x0 + 1
        y1 = y0 + 1

        valid = (x0 >= 0) & (x1 < img_size) & (y0 >= 0) & (y1 < img_size) & (points_cam[:, 2] > 0)
        valid_idx = valid.nonzero(as_tuple=True)[0]

        if len(valid_idx) > 0:
            x0_v, y0_v = x0[valid_idx], y0[valid_idx]
            x1_v, y1_v = x1[valid_idx], y1[valid_idx]
            wx = proj_x[valid_idx] - x0_v.float()
            wy = proj_y[valid_idx] - y0_v.float()

            v00 = silh_gt[y0_v, x0_v]
            v01 = silh_gt[y1_v, x0_v]
            v10 = silh_gt[y0_v, x1_v]
            v11 = silh_gt[y1_v, x1_v]

            silh_vals = (1 - wx) * (1 - wy) * v00 + (1 - wx) * wy * v01 + wx * (1 - wy) * v10 + wx * wy * v11
            view_votes[valid_idx] += silh_vals

    num_views = len(silhs_tensors)
    occupancy = (view_votes / num_views) > 0.6
    occupancy_3d = occupancy.float().reshape(volume_size, volume_size, volume_size)

    from scipy.ndimage import binary_closing, binary_opening, binary_fill_holes, label
    occ_np = occupancy_3d.cpu().numpy()
    occ_np = binary_fill_holes(occ_np).astype(np.float32)
    occ_np = binary_closing(occ_np, structure=np.ones((5, 5, 5)), iterations=2)
    occ_np = binary_opening(occ_np, structure=np.ones((3, 3, 3)), iterations=1)

    labeled, num_cc = label(occ_np)
    if num_cc > 0:
        sizes = [(labeled == i).sum() for i in range(1, num_cc + 1)]
        largest = np.argmax(sizes) + 1
        occ_np = (labeled == largest).astype(np.float32)
        if num_cc > 1:
            print(f"  Removed {num_cc - 1} small components")

    occupancy_3d = torch.from_numpy(occ_np).float().to(device)
    total = occupancy_3d.sum().item()
    pct = total / volume_size ** 3 * 100
    print(f"  Visual hull: {total:.0f} voxels ({pct:.1f}%)")
    return occupancy_3d


def iou_in_memory(pred_np, gt_np, threshold=127):
    b1 = (pred_np > threshold).astype(np.uint8)
    b2 = (gt_np > threshold).astype(np.uint8)
    intersection = int((b1 & b2).sum())
    union = int((b1 | b2).sum())
    if union == 0:
        return 1.0
    return intersection / union


def dice_in_memory(pred_np, gt_np, threshold=127):
    b1 = (pred_np > threshold).astype(np.uint8)
    b2 = (gt_np > threshold).astype(np.uint8)
    intersection = int((b1 & b2).sum())
    total = int(b1.sum()) + int(b2.sum())
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def compute_boundary_weight(mask_np, size, device):
    """预计算边界权重图，把梯度集中在轮廓附近。"""
    mask = resize_binary_mask(mask_np, size).astype(np.uint8) * 255
    dist_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_outside = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
    dist_to_boundary = dist_inside + dist_outside
    sigma = 6.0
    weight = np.exp(-(dist_to_boundary ** 2) / (2 * sigma ** 2))
    weight = weight / (weight.mean() + 1e-8)
    return torch.from_numpy(weight).float().to(device)


def _build_printable_minimal(hard_hull):
    """vol_min export: connectivity-preserving minimal solid, thickened to a
    guaranteed watertight-manifold printable body. Falls back to the (non-
    watertight) minimal solid if trimesh/skimage are unavailable."""
    minimal, carve_info = minimize_material_printable(hard_hull, bridge_radius=1)
    try:
        solid, watertight_info = make_watertight_printable(minimal, hard_hull)
        return solid, {**carve_info, **watertight_info}
    except Exception as exc:
        print(f"  [printable] watertight thickening skipped ({exc}); exporting minimal solid")
        return minimal, {
            **carve_info,
            "watertight": False,
            "fallback": "minimal",
            "error": str(exc),
        }


def train(device, shadow_files, output_dir, Niter, lr, exp_sample_id,
          num_views, mirror_mode, TSF_mode, TSF4v_mode,
          zdist, camera_mode, img_size, debug_mode,
          voxel_res=128, use_visual_hull=True, export_res=0,
          post_export_bundle=True, history_roots=None,
          rotation_frames=72, rotation_fps=18, keyview_pause_seconds=10.0,
          recon_mesh_method="mcubes", recon_mesh_smooth_sigma=0.35,
          recon_mesh_taubin_iterations=2, recon_mesh_upsample_factor=2,
          fixed_calib=False, vol_min=0.0, conn_w=1.0,
          loss_export_consistency_scale=1.0,
          loss_support_scale=1.0,
          loss_main_structure_scale=1.0,
          loss_manufacturability_scale=1.0,
          loss_export_distill_scale=1.0,
          loss_tfs_projection_scale=1.0,
          loss_volume_scale=1.0,
          loss_topo_connectivity_scale=1.0,
          candidate_min_iou=0.95,
          candidate_min_saving=0.40,
          candidate_max_saving=0.85,
          save_export_candidates=False,
          renderer_phase4_iters=None):
    train_start_time = time.time()
    requested_voxel_res = int(voxel_res)
    print("\n" + "=" * 80)
    print("SHADOW ART - VOXEL VERSION")
    print(f"  Resolution: {requested_voxel_res}^3 | Visual Hull: {use_visual_hull}")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Iterations: {Niter}")
    print(f"Learning Rate: {lr}")
    print(f"Views: {num_views} (Mirror: {mirror_mode})")
    print("=" * 80 + "\n")

    # ========== 相机 ==========
    if TSF_mode:
        cameras, Rs, Ts = create_cameras_TFS_mode(
            device,
            zdist,
            mirror_mode,
            camera_mode,
            num_input_views=num_views,
        )
    elif TSF4v_mode:
        cameras, Rs, Ts = create_cameras_4VTFS_mode(device, zdist, mirror_mode, camera_mode)
    else:
        cameras, Rs, Ts = create_cameras(num_views, device, zdist, camera_mode, mirror_mode)

    volume_extent_world = 1.7

    # ========== 数据 ==========
    silhs, silhs_tensors = load_data_from_list(
        shadow_files, mirror_mode, (img_size, img_size), device, num_views, debug_mode
    )
    silhs_tensors = [s.to(dtype=torch.float32, device=device) for s in silhs_tensors]
    actual_num_views = len(silhs_tensors)
    print(f"Loaded {len(silhs)} base images, {actual_num_views} total views\n")

    os.makedirs(output_dir, exist_ok=True)
    gt_arrays = [img.copy() for img in silhs]
    for idx, gt_img in enumerate(gt_arrays):
        cv2.imwrite(os.path.join(output_dir, f"gt_view_{idx}.png"), gt_img)
    use_tfs_hard_constraints = TSF_mode and actual_num_views == 6 and camera_mode != "perp"
    use_multiview_export_constraints = (
        camera_mode != "perp"
        and actual_num_views >= 2
        and (
            TSF_mode
            or TSF4v_mode
            or (mirror_mode and actual_num_views in (2, 4, 6, 8))
        )
    )
    proxy_res = requested_voxel_res
    if use_multiview_export_constraints:
        proxy_res = max(proxy_res, 192)
    final_export_res = export_res if export_res > 0 else (max(img_size, proxy_res) if use_multiview_export_constraints else proxy_res)
    if proxy_res != requested_voxel_res:
        print(f"  Internal Proxy Resolution upgraded to {proxy_res}^3 for renderer training")

    # ========== 渲染器 ==========
    renderer = get_voxel_renderer(device, cameras, img_size, volume_extent_world, proxy_res)
    raymarcher = renderer.renderer.raymarcher

    tfs_calibration = None
    tfs_target_masks = None
    tfs_boundary_weights = None
    export_target_volume_np = None
    export_target_volume_t = None
    export_target_metrics = None
    export_target_masks_t = None
    export_target_source = None

    if use_tfs_hard_constraints:
        print("=" * 60)
        print("TFS HARD PROJECTION CALIBRATION")
        print("=" * 60)
        tfs_calibration = calibrate_tfs_projection(gt_arrays, search_res=min(96, proxy_res), fix_axis=fixed_calib)
        if fixed_calib:
            print("  [FIXED-CALIB] axis permutation locked to identity (no axis search)")
        print(f"  Calibration mean IOU@{tfs_calibration['search_res']}: {tfs_calibration['mean_iou']:.4f}")
        print(f"  Axis mapping: {tfs_calibration['axis_for_view']}")
        print(f"  View transforms: {tfs_calibration['ops']}\n")

        tfs_target_masks = [
            torch.from_numpy(resize_binary_mask(mask, proxy_res).astype(np.float32)).to(device)
            for mask in gt_arrays[:6]
        ]
        tfs_boundary_weights = [
            compute_boundary_weight(mask, proxy_res, device)
            for mask in gt_arrays[:6]
        ]

    def build_training_export_prior(target_res):
        from scipy.ndimage import binary_dilation

        if use_tfs_hard_constraints:
            base_hull, _ = build_tfs_visual_hull(
                gt_arrays,
                target_res,
                calibration=tfs_calibration,
                apply_cleanup=False,
            )
            support = binary_dilation(base_hull, structure=np.ones((3, 3, 3), dtype=bool), iterations=1)
            gapfill, _ = repair_tfs_projection_gaps(
                gt_arrays,
                target_res,
                calibration=tfs_calibration,
                guidance_density=support.astype(np.float32),
            )
            connected, _ = connect_binary_volume(gapfill, bridge_radius=0)
            target = clean_binary_volume(
                connected,
                fill_holes=True,
                closing_iters=1,
                opening_iters=0,
                keep_largest=True,
            )
            metrics = evaluate_tfs_volume(target, gt_arrays, tfs_calibration, image_size=img_size)
            return target.astype(bool), metrics

        if use_multiview_export_constraints:
            balanced_vote = 0.58 if actual_num_views <= 4 else 0.52
            loose_vote = max(0.40, balanced_vote - 0.10)
            base_hull = build_multiview_visual_hull(
                gt_arrays,
                Rs,
                Ts,
                target_res,
                image_size=img_size,
                vote_threshold=balanced_vote,
                apply_cleanup=False,
            )
            loose_support = build_multiview_visual_hull(
                gt_arrays,
                Rs,
                Ts,
                target_res,
                image_size=img_size,
                vote_threshold=loose_vote,
                apply_cleanup=False,
            )
            support = binary_dilation(
                np.logical_or(base_hull, loose_support),
                structure=np.ones((3, 3, 3), dtype=bool),
                iterations=1,
            )
            gapfill, _ = repair_multiview_projection_gaps(
                base_hull,
                gt_arrays,
                Rs,
                Ts,
                target_res,
                image_size=img_size,
                support_volume=support,
            )
            connected, _ = connect_binary_volume(gapfill, bridge_radius=0)
            target = clean_binary_volume(
                connected,
                fill_holes=True,
                closing_iters=2,
                opening_iters=0,
                keep_largest=True,
            )
            metrics = evaluate_multiview_volume(target, gt_arrays, Rs, Ts, image_size=img_size)
            return target.astype(bool), metrics

        return None, None

    # ========== 模型 ==========
    volume_model = VolumeModel(
        renderer,
        volume_size=proxy_res,
        voxel_size=volume_extent_world / proxy_res,
    ).to(device)

    # ========== Visual Hull ==========
    if use_visual_hull:
        print("=" * 60)
        print("VISUAL HULL INITIALIZATION")
        print("=" * 60)
        if use_tfs_hard_constraints:
            visual_hull_np, _ = build_tfs_visual_hull(gt_arrays, proxy_res, calibration=tfs_calibration, apply_cleanup=False)
            visual_hull = torch.from_numpy(visual_hull_np.astype(np.float32)).to(device)
            print(f"  TFS hard visual hull: {visual_hull_np.sum()} voxels")
        else:
            visual_hull = compute_visual_hull(silhs_tensors, Rs, Ts, proxy_res, device)
        with torch.no_grad():
            volume_model.log_densities.data[0] = torch.where(
                visual_hull > 0.5,
                torch.full_like(visual_hull, 2.0),
                torch.full_like(visual_hull, -8.0),
            )
        print("Model initialized from visual hull\n")

    ms_ssim_loss = MS_SSIM(device=device)

    # ========== 训练状态 ==========
    view_weights = torch.ones(actual_num_views, device=device, dtype=torch.float32)
    best_iou = 0.0
    best_state = None
    best_temp = 1.0
    best_tau = 10.0
    best_joint_score = -1e9
    best_teacher_agreement = 0.0
    best_renderer_min_iou = 0.0
    metrics_history = {'history': [], 'final_metrics': {}}
    teacher_source_history = []

    def compute_export_structure_losses(densities):
        zero = torch.tensor(0.0, device=device)
        export_consistency = zero
        if export_target_volume_t is not None:
            export_consistency = export_consistency_regularization_3d(
                densities,
                export_target_volume_t,
                inside_weight=1.15,
                outside_weight=3.8,
                dice_weight=0.35,
            )

        support = support_regularization_3d(densities, min_support=0.30, wide_support=0.16)
        main_struct = main_structure_regularization_3d(
            densities,
            core_support=0.16,
            axis_thickness=0.12,
        )
        manuf = manufacturability_regularization_3d(
            densities,
            min_neighbor=0.30,
            min_axis_support=0.24,
            wide_support=0.14,
        )
        return export_consistency, support, main_struct, manuf

    def set_training_export_target(target_volume_np, target_metrics, source="static_prior"):
        nonlocal export_target_volume_np, export_target_volume_t
        nonlocal export_target_metrics, export_target_masks_t, export_target_source
        if target_volume_np is None or target_metrics is None:
            return
        export_target_volume_np = target_volume_np.astype(bool)
        export_target_volume_t = torch.from_numpy(export_target_volume_np.astype(np.float32)).to(device)
        export_target_metrics = target_metrics
        export_target_source = str(source)
        teacher_source_history.append({
            'source': str(source),
            'mean_iou': float(target_metrics.get('mean_iou', 0.0)),
            'component_count': int(target_metrics.get('component_count', 0)),
            'occupied_voxels': int(target_metrics.get('occupied_voxels', int(export_target_volume_np.sum()))),
        })
        pred_masks = list(target_metrics.get('pred_masks', []))
        export_target_masks_t = [
            torch.from_numpy(resize_binary_mask(mask.astype(np.uint8) * 255, img_size).astype(np.float32)).to(device)
            for mask in pred_masks[:actual_num_views]
        ]

    def maybe_refresh_export_teacher(current_binary, current_density, phase_name, iter_idx, refresh_every, min_teacher_iou_gain=5e-4):
        if not use_multiview_export_constraints or refresh_every <= 0:
            return False
        if iter_idx < 0 or (iter_idx % int(refresh_every)) != 0:
            return False

        export_bundle = (
            build_tfs_export_bundle(current_binary, current_density, proxy_res)
            if use_tfs_hard_constraints
            else build_multiview_export_bundle(current_binary, current_density, proxy_res)
        )
        cand_volume = export_bundle['export_volume']
        cand_metrics = export_bundle['export_metrics']
        cand_iou = float(cand_metrics.get('mean_iou', 0.0))
        prev_iou = float(export_target_metrics.get('mean_iou', 0.0)) if export_target_metrics is not None else -1.0
        prev_min_iou = float(export_target_metrics.get('min_iou', -1.0)) if export_target_metrics is not None else -1.0
        cand_components = int(cand_metrics.get('component_count', 999))
        prev_components = int(export_target_metrics.get('component_count', 999)) if export_target_metrics is not None else 999
        cand_min_iou = float(cand_metrics.get('min_iou', 0.0))
        prev_voxels = int(export_target_metrics.get('occupied_voxels', 10**18)) if export_target_metrics is not None else 10**18
        cand_voxels = int(cand_metrics.get('occupied_voxels', int(cand_volume.sum())))

        should_accept = export_target_metrics is None
        should_accept = should_accept or (cand_components < prev_components)
        should_accept = should_accept or (
            cand_components == prev_components and cand_min_iou > prev_min_iou + float(min_teacher_iou_gain)
        )
        should_accept = should_accept or (
            cand_components == prev_components
            and abs(cand_min_iou - prev_min_iou) <= float(min_teacher_iou_gain)
            and cand_iou >= prev_iou + float(min_teacher_iou_gain)
        )
        should_accept = should_accept or (
            cand_components == prev_components
            and abs(cand_min_iou - prev_min_iou) <= float(min_teacher_iou_gain)
            and abs(cand_iou - prev_iou) <= float(min_teacher_iou_gain)
            and cand_voxels < prev_voxels
        )

        if should_accept:
            set_training_export_target(cand_volume, cand_metrics, source=f"{phase_name}@{iter_idx}")
            print(
                f"  Export teacher refreshed ({phase_name} {iter_idx}): "
                f"IOU={cand_iou:.4f}, components={cand_components}, strategy={export_bundle['export_strategy']}"
            )
            return True
        return False

    def compute_teacher_agreement(current_binary):
        if export_target_volume_np is None:
            return 0.0
        target = export_target_volume_np.astype(bool)
        pred = current_binary.astype(bool)
        union = np.logical_or(pred, target).sum()
        if union == 0:
            return 1.0
        return float(np.logical_and(pred, target).sum() / union)

    if use_multiview_export_constraints:
        init_export_target_np, init_export_target_metrics = build_training_export_prior(proxy_res)
        if init_export_target_np is not None:
            set_training_export_target(init_export_target_np, init_export_target_metrics, source="gt_prior")
            print("=" * 60)
            print("TRAINING EXPORT PRIOR")
            print("=" * 60)
            print(
                f"  Prior target IOU={float(export_target_metrics.get('mean_iou', 0.0)):.4f}, "
                f"components={int(export_target_metrics.get('component_count', 0))}, "
                f"voxels={int(export_target_metrics.get('occupied_voxels', int(export_target_volume_np.sum())))}\n"
            )

    def compute_tfs_projection_losses(densities, fp_weight=4.0):
        if not use_tfs_hard_constraints:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero, zero

        preds = project_tfs_density_torch(densities, tfs_calibration)
        proj_l1 = torch.tensor(0.0, device=device)
        proj_iou = torch.tensor(0.0, device=device)
        proj_tversky = torch.tensor(0.0, device=device)
        proj_boundary = torch.tensor(0.0, device=device)

        for idx, pred in enumerate(preds):
            gt = tfs_target_masks[idx]
            boundary_w = tfs_boundary_weights[idx]
            proj_l1 = proj_l1 + asymmetric_silhouette_loss(pred, gt, fp_weight=fp_weight)
            proj_iou = proj_iou + differentiable_iou_loss(pred, gt)
            proj_tversky = proj_tversky + focal_tversky_loss(pred, gt, alpha=0.25, beta=0.75, gamma=1.5)
            proj_boundary = proj_boundary + (((pred - gt) ** 2) * boundary_w).mean()

        n_views = float(len(preds))
        return proj_l1 / n_views, proj_iou / n_views, proj_tversky / n_views, proj_boundary / n_views

    def render_view(n):
        R = Rs[n:n + 1].to(device=device, dtype=torch.float32).contiguous()
        T = Ts[n:n + 1].to(device=device, dtype=torch.float32).contiguous()
        bc = FoVOrthographicCameras(R=R, T=T, device=device)
        rendered_images, _ = volume_model(bc).split([3, 1], dim=-1)
        return rendered_images[0][..., 0]

    def evaluate(phase_name, global_iter, total_loss=None, normalized_total_loss=None, raw_total_loss=None, loss_components=None):
        nonlocal best_iou, best_state, best_temp, best_tau
        nonlocal best_joint_score, best_teacher_agreement, best_renderer_min_iou
        view_ious = []
        view_dices = []
        current_normal_consistency = None
        volume_model.eval()
        with torch.no_grad():
            for n in range(actual_num_views):
                silh_pred = render_view(n).cpu().numpy()
                pred_img = (silh_pred * 255).astype(np.uint8)
                _, pred_img = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                view_ious.append(iou_in_memory(pred_img, gt_arrays[n]))
                view_dices.append(dice_in_memory(pred_img, gt_arrays[n]))
            current_binary = volume_model.get_binary_voxels(threshold=0.5)[0].cpu().numpy() > 0.5
        volume_model.train()
        current_normal_consistency = estimate_voxel_normal_consistency(
            current_binary,
            device=device,
            max_res=min(128, proxy_res),
        )
        if current_normal_consistency is None and proxy_res > 96:
            current_normal_consistency = estimate_voxel_normal_consistency(
                current_binary,
                device=device,
                max_res=min(160, proxy_res),
            )

        mean_iou = np.mean(view_ious)
        mean_dice = np.mean(view_dices)
        min_iou = float(np.min(view_ious)) if view_ious else 0.0
        teacher_agreement = compute_teacher_agreement(current_binary)
        # Keep checkpoint selection closer to the voxel_test55 behavior:
        # renderer mean IoU remains primary, while min-IoU / teacher-agreement
        # only provide moderate tie-breaking pressure.
        joint_checkpoint_score = float(mean_iou + 0.22 * min_iou + 0.12 * teacher_agreement)
        history_entry = {
            'phase': phase_name, 'global_iter': global_iter,
            'mean_iou': float(mean_iou),
            'mean_dice': float(mean_dice),
            'min_iou': float(min_iou),
            'teacher_agreement': float(teacher_agreement),
            'joint_checkpoint_score': float(joint_checkpoint_score),
        }
        if total_loss is not None:
            history_entry['total_loss'] = float(total_loss)
        if normalized_total_loss is not None:
            history_entry['normalized_total_loss'] = float(normalized_total_loss)
        if raw_total_loss is not None:
            history_entry['raw_total_loss'] = float(raw_total_loss)
        if current_normal_consistency is not None:
            history_entry['normal_consistency'] = float(current_normal_consistency)
        if loss_components:
            history_entry['loss_components'] = {
                str(k): float(v) for k, v in loss_components.items()
            }
        metrics_history['history'].append(history_entry)

        if mean_iou > best_iou:
            best_iou = mean_iou
        if (
            joint_checkpoint_score > best_joint_score + 1e-8
            or (
                abs(joint_checkpoint_score - best_joint_score) <= 1e-8
                and min_iou > best_renderer_min_iou + 1e-8
            )
        ):
            best_state = volume_model.log_densities.clone().detach()
            best_temp = volume_model.temperature
            best_tau = raymarcher.tau
            best_joint_score = float(joint_checkpoint_score)
            best_teacher_agreement = float(teacher_agreement)
            best_renderer_min_iou = float(min_iou)

        return mean_iou, mean_dice, view_ious, view_dices

    def update_view_weights(view_ious):
        """温和的视图权重更新（防止训练不稳定）"""
        nonlocal view_weights
        for vn in range(actual_num_views):
            # 使用 1/iou（非 1/iou^3），且下限 0.7（防止极端权重）
            view_weights[vn] = 1.0 / max(view_ious[vn], 0.7)
        view_weights = view_weights / view_weights.mean()

    def restore_best():
        if best_state is not None:
            volume_model.log_densities.data = best_state.clone()
            volume_model.temperature = best_temp
            raymarcher.tau = best_tau

    def build_tfs_export_bundle(learned_binary, learned_density, export_resolution):
        hard_hull, _ = build_tfs_visual_hull(
            gt_arrays,
            export_resolution,
            calibration=tfs_calibration,
            apply_cleanup=False,
        )
        learned_up = resize_binary_volume(learned_binary, export_resolution)
        # Match the high-IoU voxel_test32 behavior: keep bridges as single-voxel links
        # instead of thick dilated tubes that distort silhouettes and carve visible gaps.
        hard_hull_connected, hard_hull_connected_info = connect_binary_volume(hard_hull, bridge_radius=0)
        gap_filled, gap_filled_info = repair_tfs_projection_gaps(
            gt_arrays,
            export_resolution,
            calibration=tfs_calibration,
            guidance_density=learned_density,
        )
        gap_filled_connected, gap_filled_connected_info = connect_binary_volume(gap_filled, bridge_radius=0)
        learned_connected, _ = connect_binary_volume(learned_up & hard_hull, bridge_radius=0)
        try:
            learned_printable, learned_printable_info = make_watertight_printable(
                learned_connected,
                hard_hull,
            )
        except Exception as exc:
            print(f"  WARNING: learned printable post-process failed: {exc}")
            learned_printable = learned_connected
            learned_printable_info = {"watertight": False, "fallback": "learned_connected"}
        if vol_min > 0.0:
            printable_minimal, printable_minimal_info = _build_printable_minimal(hard_hull)
        else:
            printable_minimal = learned_up & hard_hull
            printable_minimal_info = {"watertight": False, "disabled": True}
        candidates = {
            'tfs_hard': hard_hull,
            'tfs_hard_connected': hard_hull_connected,
            'tfs_gapfill': gap_filled,
            'tfs_gapfill_connected': gap_filled_connected,
            'tfs_hard_clean': clean_binary_volume(hard_hull, fill_holes=True, closing_iters=1, opening_iters=0, keep_largest=True),
            'learned_up': learned_up,
            'learned_clip': learned_up & hard_hull,
            'learned_clip_clean': clean_binary_volume(
                learned_up & hard_hull,
                fill_holes=True,
                closing_iters=1,
                opening_iters=0,
                keep_largest=True,
            ),
            # vol_min branch: pure learned (soft-carved) voxels clipped to hull,
            # only re-connected into a single body — NO target-rebuild. Silhouette
            # fidelity here comes solely from the soft losses (tests if they bind).
            'learned_connected': learned_connected,
            # Fair soft-branch export: apply the same print-oriented thickening
            # and watertightness repair used by the geometric minimal candidate.
            'learned_printable': learned_printable,
            # printable minimal solid: connectivity-preserving carve + thick bridges,
            # then thicken junctions to a WATERTIGHT MANIFOLD (guaranteed printable):
            # 6-connected, silhouette-faithful, ~52-76% material saved, >=0.8mm walls.
            'printable_minimal': printable_minimal,
        }

        candidate_metrics_log = {}
        for cand_name, cand_volume in candidates.items():
            cand_metrics = choose_best_tfs_candidate(
                {cand_name: cand_volume},
                gt_arrays,
                tfs_calibration,
                image_size=img_size,
            )[2]
            candidate_metrics_log[cand_name] = {
                'mean_iou': float(cand_metrics['mean_iou']),
                'mean_dice': float(cand_metrics['mean_dice']),
                'mean_recall': float(cand_metrics.get('mean_recall', 0.0)),
                'min_iou': float(cand_metrics.get('min_iou', 0.0)),
                'occupied_voxels': int(cand_metrics.get('occupied_voxels', int(cand_volume.sum()))),
                'component_count': int(cand_metrics.get('component_count', 0)),
                'largest_component_ratio': float(cand_metrics.get('largest_component_ratio', 0.0)),
                'small_component_voxels': int(cand_metrics.get('small_component_voxels', 0)),
                'repair_voxels_added': int(
                    gap_filled_info.get('repair_voxels_added', 0)
                    if cand_name in {'tfs_gapfill', 'tfs_gapfill_connected'}
                    else cand_metrics.get('repair_voxels_added', 0)
                ),
                'bridge_voxels_added': int(
                    hard_hull_connected_info.get('bridge_voxels_added', 0)
                    if cand_name == 'tfs_hard_connected'
                    else gap_filled_connected_info.get('bridge_voxels_added', 0)
                    if cand_name == 'tfs_gapfill_connected'
                    else cand_metrics.get('bridge_voxels_added', 0)
                ),
            }

        selection_gate = None
        if vol_min > 0.0:
            # Material-minimization mode keeps soft and geometric candidates in
            # one pool. Prefer candidates satisfying the shared fidelity,
            # connectivity, and material-saving gates; record any relaxation.
            printable_candidates = {
                name: candidates[name]
                for name in ('learned_printable', 'printable_minimal')
            }
            candidate_metrics_log['learned_printable']['printability_ok'] = bool(
                learned_printable_info.get('watertight', False)
            )
            candidate_metrics_log['printable_minimal']['printability_ok'] = bool(
                printable_minimal_info.get('watertight', False)
            )
            constrained_pool, selection_gate = constrain_printable_candidate_pool(
                printable_candidates,
                candidate_metrics_log,
                reference_voxels=int(hard_hull.sum()),
                min_mean_iou=float(candidate_min_iou),
                min_material_saving=float(candidate_min_saving),
                max_material_saving=float(candidate_max_saving),
            )
            for cand_name, gate_record in selection_gate['candidates'].items():
                candidate_metrics_log[cand_name].update(gate_record)
            candidate_metrics_log['learned_printable']['watertight_postprocess'] = {
                key: (bool(value) if isinstance(value, np.bool_) else value)
                for key, value in learned_printable_info.items()
            }
            candidate_metrics_log['printable_minimal']['watertight_postprocess'] = {
                key: (bool(value) if isinstance(value, np.bool_) else value)
                for key, value in printable_minimal_info.items()
            }
            export_strategy, export_volume, export_metrics = choose_best_tfs_candidate(
                constrained_pool,
                gt_arrays, tfs_calibration, image_size=img_size,
            )
        else:
            export_strategy, export_volume, export_metrics = choose_best_tfs_candidate(
                candidates,
                gt_arrays,
                tfs_calibration,
                image_size=img_size,
            )
        if vol_min <= 0.0 and int(export_metrics.get('component_count', 1)) > 1:
            connected_fallback = {
                'tfs_hard': 'tfs_hard_connected',
                'tfs_gapfill': 'tfs_gapfill_connected',
            }.get(export_strategy)
            if connected_fallback in candidates:
                print(
                    "  Connectivity safeguard: "
                    f"{export_strategy} -> {connected_fallback}"
                )
                export_strategy, export_volume, export_metrics = choose_best_tfs_candidate(
                    {connected_fallback: candidates[connected_fallback]},
                    gt_arrays,
                    tfs_calibration,
                    image_size=img_size,
                )

        if export_strategy == 'tfs_hard_connected':
            export_metrics['bridge_voxels_added'] = int(hard_hull_connected_info.get('bridge_voxels_added', 0))
            export_metrics['initial_component_count'] = int(
                hard_hull_connected_info.get('initial_component_count', export_metrics.get('component_count', 1))
            )
        if export_strategy in {'tfs_gapfill', 'tfs_gapfill_connected'}:
            export_metrics['repair_voxels_added'] = int(gap_filled_info.get('repair_voxels_added', 0))
            export_metrics['repair_pixels_by_axis'] = {
                str(k): int(v) for k, v in gap_filled_info.get('repair_pixels_by_axis', {}).items()
            }
        if export_strategy == 'tfs_gapfill_connected':
            export_metrics['bridge_voxels_added'] = int(gap_filled_connected_info.get('bridge_voxels_added', 0))
            export_metrics['initial_component_count'] = int(
                gap_filled_connected_info.get('initial_component_count', export_metrics.get('component_count', 1))
            )

        return {
            'candidates': candidates,
            'candidate_metrics_log': candidate_metrics_log,
            'export_strategy': export_strategy,
            'export_volume': export_volume,
            'export_metrics': export_metrics,
            'selection_gate': selection_gate,
        }

    def build_multiview_export_bundle(learned_binary, learned_density, export_resolution):
        learned_up = resize_binary_volume(learned_binary, export_resolution)
        learned_density_up = resize_float_volume(learned_density, export_resolution)
        learned_density_soft = learned_density_up >= (0.40 if actual_num_views >= 6 else 0.42)
        learned_native_clean = clean_binary_volume(
            learned_binary,
            fill_holes=True,
            closing_iters=1,
            opening_iters=0,
            keep_largest=True,
        )
        strict_vote = 0.60 if actual_num_views <= 4 else 0.58
        balanced_vote = 0.56 if actual_num_views <= 4 else 0.52
        loose_vote = max(0.40, balanced_vote - 0.10)

        hard_hull = build_multiview_visual_hull(
            gt_arrays,
            Rs,
            Ts,
            export_resolution,
            image_size=img_size,
            vote_threshold=strict_vote,
            apply_cleanup=False,
        )
        balanced_hull = build_multiview_visual_hull(
            gt_arrays,
            Rs,
            Ts,
            export_resolution,
            image_size=img_size,
            vote_threshold=balanced_vote,
            apply_cleanup=False,
        )
        loose_hull = build_multiview_visual_hull(
            gt_arrays,
            Rs,
            Ts,
            export_resolution,
            image_size=img_size,
            vote_threshold=loose_vote,
            apply_cleanup=False,
        )
        hard_hull_connected, hard_hull_connected_info = connect_binary_volume(hard_hull, bridge_radius=0)
        gap_filled, gap_filled_info = repair_multiview_projection_gaps(
            balanced_hull,
            gt_arrays,
            Rs,
            Ts,
            export_resolution,
            image_size=img_size,
            guidance_density=learned_density,
            support_volume=np.logical_or(loose_hull, learned_density_soft | learned_up),
        )
        gap_filled_connected, gap_filled_connected_info = connect_binary_volume(gap_filled, bridge_radius=0)
        learned_density_clean = clean_binary_volume(
            learned_density_soft,
            fill_holes=True,
            closing_iters=1,
            opening_iters=0,
            keep_largest=True,
        )
        balanced_hull_clean = clean_binary_volume(
            balanced_hull,
            fill_holes=True,
            closing_iters=2,
            opening_iters=0,
            keep_largest=True,
        )
        loose_hull_clean = clean_binary_volume(
            loose_hull,
            fill_holes=True,
            closing_iters=2,
            opening_iters=0,
            keep_largest=True,
        )

        candidates = {
            'learned': learned_binary,
            'learned_clean': learned_native_clean,
            'hard_hull': hard_hull,
            'hard_hull_connected': hard_hull_connected,
            'balanced_hull': balanced_hull,
            'balanced_hull_clean': balanced_hull_clean,
            'loose_hull_clean': loose_hull_clean,
            'gapfill': gap_filled,
            'gapfill_connected': gap_filled_connected,
            'hard_hull_clean': clean_binary_volume(hard_hull, fill_holes=True, closing_iters=1, opening_iters=0, keep_largest=True),
            'learned_up': learned_up,
            'learned_density_soft': learned_density_soft,
            'learned_density_clean': learned_density_clean,
            'learned_clip': learned_up & hard_hull,
            'learned_clip_clean': clean_binary_volume(
                learned_up & hard_hull,
                fill_holes=True,
                closing_iters=1,
                opening_iters=0,
                keep_largest=True,
            ),
            'learned_density_clip_balanced': clean_binary_volume(
                learned_density_soft & balanced_hull,
                fill_holes=True,
                closing_iters=1,
                opening_iters=0,
                keep_largest=True,
            ),
            'learned_density_clip_loose': clean_binary_volume(
                learned_density_soft & loose_hull,
                fill_holes=True,
                closing_iters=1,
                opening_iters=0,
                keep_largest=True,
            ),
        }

        candidate_metrics_log = {}
        for cand_name, cand_volume in candidates.items():
            cand_metrics = choose_best_multiview_candidate(
                {cand_name: cand_volume},
                gt_arrays,
                Rs,
                Ts,
                image_size=img_size,
            )[2]
            candidate_metrics_log[cand_name] = {
                'mean_iou': float(cand_metrics['mean_iou']),
                'mean_dice': float(cand_metrics['mean_dice']),
                'mean_recall': float(cand_metrics.get('mean_recall', 0.0)),
                'min_iou': float(cand_metrics.get('min_iou', 0.0)),
                'occupied_voxels': int(cand_metrics.get('occupied_voxels', int(cand_volume.sum()))),
                'component_count': int(cand_metrics.get('component_count', 0)),
                'largest_component_ratio': float(cand_metrics.get('largest_component_ratio', 0.0)),
                'small_component_voxels': int(cand_metrics.get('small_component_voxels', 0)),
                'repair_voxels_added': int(
                    gap_filled_info.get('repair_voxels_added', 0)
                    if cand_name in {'gapfill', 'gapfill_connected'}
                    else cand_metrics.get('repair_voxels_added', 0)
                ),
                'bridge_voxels_added': int(
                    hard_hull_connected_info.get('bridge_voxels_added', 0)
                    if cand_name == 'hard_hull_connected'
                    else gap_filled_connected_info.get('bridge_voxels_added', 0)
                    if cand_name == 'gapfill_connected'
                    else cand_metrics.get('bridge_voxels_added', 0)
                ),
            }

        export_strategy, export_volume, export_metrics = choose_best_multiview_candidate(
            candidates,
            gt_arrays,
            Rs,
            Ts,
            image_size=img_size,
        )
        if int(export_metrics.get('component_count', 1)) > 1:
            connected_fallback = {
                'hard_hull': 'hard_hull_connected',
                'gapfill': 'gapfill_connected',
            }.get(export_strategy)
            if connected_fallback in candidates:
                print(
                    "  Connectivity safeguard: "
                    f"{export_strategy} -> {connected_fallback}"
                )
                export_strategy, export_volume, export_metrics = choose_best_multiview_candidate(
                    {connected_fallback: candidates[connected_fallback]},
                    gt_arrays,
                    Rs,
                    Ts,
                    image_size=img_size,
                )

        if export_strategy == 'hard_hull_connected':
            export_metrics['bridge_voxels_added'] = int(hard_hull_connected_info.get('bridge_voxels_added', 0))
            export_metrics['initial_component_count'] = int(
                hard_hull_connected_info.get('initial_component_count', export_metrics.get('component_count', 1))
            )
        if export_strategy in {'gapfill', 'gapfill_connected'}:
            export_metrics['repair_voxels_added'] = int(gap_filled_info.get('repair_voxels_added', 0))
            export_metrics['repair_pixels_by_view'] = {
                str(k): int(v) for k, v in gap_filled_info.get('repair_pixels_by_view', {}).items()
            }
        if export_strategy == 'gapfill_connected':
            export_metrics['bridge_voxels_added'] = int(gap_filled_connected_info.get('bridge_voxels_added', 0))
            export_metrics['initial_component_count'] = int(
                gap_filled_connected_info.get('initial_component_count', export_metrics.get('component_count', 1))
            )

        return {
            'candidates': candidates,
            'candidate_metrics_log': candidate_metrics_log,
            'export_strategy': export_strategy,
            'export_volume': export_volume,
            'export_metrics': export_metrics,
        }

    # ==========================================================
    # PHASE 1: 形状形成
    # ==========================================================
    phase1_iters = Niter
    phase1_lr = lr

    optimizer = torch.optim.Adam(volume_model.parameters(), lr=phase1_lr,
                                  betas=(0.9, 0.999), eps=1e-8)
    # CosineAnnealingLR（无 warm restart）→ 稳定下降
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase1_iters, eta_min=phase1_lr * 0.01
    )

    p1_temp_start, p1_temp_end = 1.0, 5.0

    print("=" * 60)
    print(f"PHASE 1: SHAPE FORMATION ({phase1_iters} iters, lr={phase1_lr:.5f})")
    print(f"  Temp: {p1_temp_start} -> {p1_temp_end}")
    print("=" * 60 + "\n")

    mean_iou = 0.0
    loop = tqdm(range(phase1_iters), desc="Phase 1")

    for i in loop:
        progress = i / max(phase1_iters - 1, 1)
        volume_model.temperature = p1_temp_start + (p1_temp_end - p1_temp_start) * progress
        # tau: 10→25（温和递增，前期保持平滑梯度）
        raymarcher.tau = 10.0 + 15.0 * progress

        optimizer.zero_grad(set_to_none=True)

        n_silh, n_l1, n_ssim = 0.0, 0.0, 0.0
        n_iou, n_dice = 0.0, 0.0
        n_export_distill = 0.0

        for n, silh_gt in enumerate(silhs_tensors):
            silh_pred = render_view(n)
            w = view_weights[n]

            n_silh += silh_loss(silh_pred, silh_gt) * w
            n_l1 += l1_loss(silh_pred, silh_gt) * w
            n_ssim += ms_ssim_loss(
                silh_pred.view(1, 1, img_size, img_size),
                silh_gt.view(1, 1, img_size, img_size)
            ) * w

            # 后 40% 加入 IOU/Dice 精细化
            if i >= int(phase1_iters * 0.6):
                n_iou += differentiable_iou_loss(silh_pred, silh_gt) * w
                n_dice += differentiable_dice_loss(silh_pred, silh_gt) * w
            if export_target_masks_t is not None and n < len(export_target_masks_t) and i >= int(phase1_iters * 0.5):
                teacher_mask = export_target_masks_t[n]
                n_export_distill += (
                    asymmetric_silhouette_loss(silh_pred, teacher_mask, fp_weight=4.0) +
                    differentiable_iou_loss(silh_pred, teacher_mask)
                ) * w

        total_w = view_weights.sum().item()
        densities = torch.sigmoid(volume_model.temperature * volume_model.log_densities)
        tv = tv_regularization_3d(densities)
        sp = sparsity_regularization(densities)
        export_consistency, support_reg, main_struct_reg, manuf_reg = compute_export_structure_losses(densities)
        proj_l1, proj_iou, proj_tversky, _ = compute_tfs_projection_losses(densities[0], fp_weight=3.0)

        loss = ((n_silh / total_w) * 7.0 +
                (n_l1 / total_w) * 6.0 +
                (n_ssim / total_w) * 2.5 +
                tv * 0.3 +
                sp * 0.02)

        if i >= int(phase1_iters * 0.6):
            loss = loss + (n_iou / total_w) * 3.0 + (n_dice / total_w) * 2.0
            if use_tfs_hard_constraints:
                loss = loss + proj_l1 * 2.0 + proj_iou * 1.5 + proj_tversky * 1.0
        if use_multiview_export_constraints and i >= int(phase1_iters * 0.35):
            ramp = (i - int(phase1_iters * 0.35)) / max(int(phase1_iters * 0.65), 1)
            ramp = float(np.clip(ramp, 0.0, 1.0))
            loss = loss + export_consistency * (0.35 + 0.45 * ramp) * float(loss_export_consistency_scale)
            loss = loss + support_reg * (0.04 + 0.04 * ramp) * float(loss_support_scale)
            loss = loss + main_struct_reg * (0.10 + 0.12 * ramp) * float(loss_main_structure_scale)
            loss = loss + manuf_reg * (0.06 + 0.06 * ramp) * float(loss_manufacturability_scale)
            loss = loss + (n_export_distill / total_w) * (0.18 + 0.22 * ramp) * float(loss_export_distill_scale)
        if vol_min > 0.0:
            vol_ramp = float(np.clip((i - int(phase1_iters * 0.5)) / max(int(phase1_iters * 0.5), 1), 0.0, 1.0))
            loss = loss + volume_minimization_loss(densities) * (vol_min * vol_ramp) * float(loss_volume_scale)
            # differentiable 6-connectivity: penalize occupied-but-unreachable voxels so the
            # soft-carve stays a single face-connected printable body (discrete carve -> dust).
            loss = loss + topo_connectivity_loss(densities, down=4) * (conn_w * (3.0 + 5.0 * vol_ramp)) * float(loss_topo_connectivity_scale)
        raw_loss = loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(volume_model.parameters(), max_norm=1.0)

        if any(torch.isnan(p.grad).any() for p in volume_model.parameters() if p.grad is not None):
            optimizer.zero_grad()
            continue

        optimizer.step()
        scheduler.step()

        # 评估（不要太早更新 view_weights，避免不稳定）
        if (i % 100 == 0 and i > 0) or i == phase1_iters - 1:
            with torch.no_grad():
                phase1_components = {
                    'silhouette': float((n_silh / total_w).item()),
                    'l1': float((n_l1 / total_w).item()),
                    'ms_ssim': float((n_ssim / total_w).item()),
                    'iou': float((n_iou / total_w).item()) if i >= int(phase1_iters * 0.6) else 0.0,
                    'dice': float((n_dice / total_w).item()) if i >= int(phase1_iters * 0.6) else 0.0,
                    'export_render_distill': float((n_export_distill / total_w).item()) if export_target_masks_t is not None and i >= int(phase1_iters * 0.5) else 0.0,
                    'reg_tv': float(tv.item()),
                    'reg_sparsity': float(sp.item()),
                    'export_consistency': float(export_consistency.item()),
                    'support': float(support_reg.item()),
                    'main_structure': float(main_struct_reg.item()),
                    'manufacturability': float(manuf_reg.item()),
                    'projection_l1': float(proj_l1.item()),
                    'projection_iou': float(proj_iou.item()),
                    'projection_tversky': float(proj_tversky.item()),
                }
                mean_iou, _, view_ious, _ = evaluate(
                    'Phase 1',
                    i,
                    total_loss=loss.item(),
                    normalized_total_loss=loss.item(),
                    raw_total_loss=loss.item(),
                    loss_components=phase1_components,
                )
                if i > int(phase1_iters * 0.3):
                    update_view_weights(view_ious)
                if i >= int(phase1_iters * 0.55):
                    current_binary = volume_model.get_binary_voxels(threshold=0.5)[0].detach().cpu().numpy() > 0.5
                    current_density = torch.sigmoid(volume_model.temperature * volume_model.log_densities)[0].detach().cpu().numpy()
                    maybe_refresh_export_teacher(current_binary, current_density, "Phase1", i, refresh_every=500)

            loop.set_postfix({'IOU': f'{mean_iou:.4f}', 'best': f'{best_iou:.4f}',
                              'temp': f'{volume_model.temperature:.2f}'})

            if i % 500 == 0:
                print(f"\n  [Phase1 {i}/{phase1_iters}] IOU={mean_iou:.4f} Best={best_iou:.4f}")

    restore_best()
    phase1_best = best_iou
    print(f"\n  Phase 1 best IOU: {phase1_best:.4f}")

    # ==========================================================
    # PHASE 2: 飞散体素消除
    # ==========================================================
    phase2_iters = int(Niter * 0.8)
    phase2_lr = lr * 0.05
    p2_temp_start = volume_model.temperature
    p2_temp_end = 15.0

    print("\n" + "=" * 60)
    print(f"PHASE 2: FLYING VOXEL REMOVAL ({phase2_iters} iters, lr={phase2_lr:.6f})")
    print(f"  Temp: {p2_temp_start:.1f} -> {p2_temp_end}")
    print("=" * 60 + "\n")

    optimizer = torch.optim.Adam(volume_model.parameters(), lr=phase2_lr,
                                  betas=(0.9, 0.999), eps=1e-8)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase2_iters, eta_min=phase2_lr * 0.02
    )

    phase2_best = best_iou
    global_offset = phase1_iters

    phase2_loop = tqdm(range(phase2_iters), desc="Phase 2")
    for ri in phase2_loop:
        progress = ri / max(phase2_iters - 1, 1)
        volume_model.temperature = p2_temp_start + (p2_temp_end - p2_temp_start) * progress
        raymarcher.tau = 25.0 + 15.0 * progress  # 25→40

        optimizer.zero_grad(set_to_none=True)

        p2_asym, p2_l1, p2_ssim = 0.0, 0.0, 0.0
        p2_iou, p2_dice = 0.0, 0.0
        p2_tversky, p2_boundary = 0.0, 0.0
        p2_export_distill = 0.0

        for n, silh_gt in enumerate(silhs_tensors):
            silh_pred = render_view(n)
            w = view_weights[n]

            p2_asym += asymmetric_silhouette_loss(silh_pred, silh_gt, fp_weight=5.0) * w
            p2_l1 += l1_loss(silh_pred, silh_gt) * w
            p2_ssim += ms_ssim_loss(
                silh_pred.view(1, 1, img_size, img_size),
                silh_gt.view(1, 1, img_size, img_size)
            ) * w
            p2_iou += differentiable_iou_loss(silh_pred, silh_gt) * w
            p2_dice += differentiable_dice_loss(silh_pred, silh_gt) * w
            if use_multiview_export_constraints and not use_tfs_hard_constraints:
                p2_tversky += focal_tversky_loss(silh_pred, silh_gt, alpha=0.28, beta=0.72, gamma=1.5) * w
                p2_boundary += boundary_loss(silh_pred, silh_gt, device=device) * w
            if export_target_masks_t is not None and n < len(export_target_masks_t):
                teacher_mask = export_target_masks_t[n]
                p2_export_distill += (
                    F.binary_cross_entropy(silh_pred.clamp(1e-4, 1 - 1e-4), teacher_mask) +
                    differentiable_iou_loss(silh_pred, teacher_mask) +
                    asymmetric_silhouette_loss(silh_pred, teacher_mask, fp_weight=4.5)
                ) * w

        total_w = view_weights.sum().item()
        densities = torch.sigmoid(volume_model.temperature * volume_model.log_densities)
        tv = tv_regularization_3d(densities)
        iso = isolation_penalty(densities)
        sp = sparsity_regularization(densities)
        export_consistency, support_reg, main_struct_reg, manuf_reg = compute_export_structure_losses(densities)
        proj_l1, proj_iou, proj_tversky, proj_boundary = compute_tfs_projection_losses(densities[0], fp_weight=5.0)

        iso_w = 0.02 + 0.06 * progress   # 0.02→0.08

        loss = ((p2_asym / total_w) * 8.0 +
                (p2_l1 / total_w) * 6.0 +
                (p2_ssim / total_w) * 3.0 +
                (p2_iou / total_w) * 5.0 +
                (p2_dice / total_w) * 3.0 +
                tv * 0.15 +
                iso * iso_w +
                sp * 0.03)

        if use_tfs_hard_constraints:
            loss = loss + proj_l1 * 3.0 + proj_iou * 2.0 + proj_tversky * 1.5 + proj_boundary * 0.8
        if use_multiview_export_constraints:
            loss = loss + export_consistency * 1.15 * float(loss_export_consistency_scale)
            loss = loss + support_reg * 0.16 * float(loss_support_scale)
            loss = loss + main_struct_reg * 0.32 * float(loss_main_structure_scale)
            loss = loss + manuf_reg * 0.26 * float(loss_manufacturability_scale)
            loss = loss + (p2_export_distill / total_w) * 0.90 * float(loss_export_distill_scale)
            if not use_tfs_hard_constraints:
                loss = loss + ((p2_tversky / total_w) * 1.35 + (p2_boundary / total_w) * 0.70) * float(loss_tfs_projection_scale)
        if vol_min > 0.0:
            loss = loss + volume_minimization_loss(densities) * (vol_min * (0.5 + 0.5 * progress)) * float(loss_volume_scale)
            loss = loss + topo_connectivity_loss(densities, down=4) * (conn_w * (4.0 + 6.0 * progress)) * float(loss_topo_connectivity_scale) \
                        + iso * (0.3 + 0.3 * progress)
        raw_loss = loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(volume_model.parameters(), max_norm=0.8)

        if any(torch.isnan(p.grad).any() for p in volume_model.parameters() if p.grad is not None):
            optimizer.zero_grad()
            continue

        optimizer.step()
        scheduler2.step()

        if (ri % 200 == 0 and ri > 0) or ri == phase2_iters - 1:
            with torch.no_grad():
                phase2_components = {
                    'asymmetric': float((p2_asym / total_w).item()),
                    'l1': float((p2_l1 / total_w).item()),
                    'ms_ssim': float((p2_ssim / total_w).item()),
                    'iou': float((p2_iou / total_w).item()),
                    'dice': float((p2_dice / total_w).item()),
                    'focal_tversky': float((p2_tversky / total_w).item()) if use_multiview_export_constraints and not use_tfs_hard_constraints else 0.0,
                    'boundary': float((p2_boundary / total_w).item()) if use_multiview_export_constraints and not use_tfs_hard_constraints else 0.0,
                    'export_render_distill': float((p2_export_distill / total_w).item()) if export_target_masks_t is not None else 0.0,
                    'reg_tv': float(tv.item()),
                    'reg_sparsity': float(sp.item()),
                    'isolation': float(iso.item()),
                    'export_consistency': float(export_consistency.item()),
                    'support': float(support_reg.item()),
                    'main_structure': float(main_struct_reg.item()),
                    'manufacturability': float(manuf_reg.item()),
                    'projection_l1': float(proj_l1.item()),
                    'projection_iou': float(proj_iou.item()),
                    'projection_tversky': float(proj_tversky.item()),
                    'projection_boundary': float(proj_boundary.item()),
                }
                mean_iou, _, view_ious, _ = evaluate(
                    'Phase 2',
                    global_offset + ri,
                    total_loss=loss.item(),
                    normalized_total_loss=loss.item(),
                    raw_total_loss=loss.item(),
                    loss_components=phase2_components,
                )
                update_view_weights(view_ious)
                current_binary = volume_model.get_binary_voxels(threshold=0.5)[0].detach().cpu().numpy() > 0.5
                current_density = torch.sigmoid(volume_model.temperature * volume_model.log_densities)[0].detach().cpu().numpy()
                maybe_refresh_export_teacher(current_binary, current_density, "Phase2", ri, refresh_every=400)
                if mean_iou > phase2_best:
                    phase2_best = mean_iou

            phase2_loop.set_postfix({'IOU': f'{mean_iou:.4f}', 'best': f'{best_iou:.4f}'})

    restore_best()
    print(f"\n  Phase 2 best IOU: {best_iou:.4f} (P1={phase1_best:.4f})")

    # ==========================================================
    # PHASE 3: 缺口填补 + 精修
    # ==========================================================
    phase3_iters = int(Niter * 0.8)
    phase3_lr = lr * 0.02
    p3_temp_start = volume_model.temperature
    p3_temp_end = 25.0

    print("\n" + "=" * 60)
    print(f"PHASE 3: GAP FILLING + POLISH ({phase3_iters} iters, lr={phase3_lr:.6f})")
    print(f"  Temp: {p3_temp_start:.1f} -> {p3_temp_end}")
    print("=" * 60 + "\n")

    optimizer = torch.optim.Adam(volume_model.parameters(), lr=phase3_lr,
                                  betas=(0.9, 0.999), eps=1e-8)
    scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=phase3_iters, eta_min=phase3_lr * 0.05
    )

    phase3_best = best_iou
    global_offset = phase1_iters + phase2_iters

    phase3_loop = tqdm(range(phase3_iters), desc="Phase 3")
    for ri in phase3_loop:
        progress = ri / max(phase3_iters - 1, 1)
        volume_model.temperature = p3_temp_start + (p3_temp_end - p3_temp_start) * progress
        raymarcher.tau = 35.0 + 10.0 * progress  # 35→45

        optimizer.zero_grad(set_to_none=True)

        p3_fn, p3_silh, p3_l1 = 0.0, 0.0, 0.0
        p3_iou, p3_dice = 0.0, 0.0
        p3_tversky, p3_boundary = 0.0, 0.0
        p3_export_distill = 0.0

        for n, silh_gt in enumerate(silhs_tensors):
            silh_pred = render_view(n)
            w = view_weights[n]

            p3_fn += false_negative_only_loss(silh_pred, silh_gt) * w
            p3_silh += silh_loss(silh_pred, silh_gt) * w
            p3_l1 += l1_loss(silh_pred, silh_gt) * w
            p3_iou += differentiable_iou_loss(silh_pred, silh_gt) * w
            p3_dice += differentiable_dice_loss(silh_pred, silh_gt) * w
            if use_multiview_export_constraints and not use_tfs_hard_constraints:
                p3_tversky += focal_tversky_loss(silh_pred, silh_gt, alpha=0.26, beta=0.74, gamma=1.6) * w
                p3_boundary += boundary_loss(silh_pred, silh_gt, device=device) * w
            if export_target_masks_t is not None and n < len(export_target_masks_t):
                teacher_mask = export_target_masks_t[n]
                p3_export_distill += (
                    F.binary_cross_entropy(silh_pred.clamp(1e-4, 1 - 1e-4), teacher_mask) +
                    differentiable_iou_loss(silh_pred, teacher_mask) +
                    differentiable_dice_loss(silh_pred, teacher_mask) +
                    asymmetric_silhouette_loss(silh_pred, teacher_mask, fp_weight=5.0)
                ) * w

        total_w = view_weights.sum().item()
        densities = torch.sigmoid(volume_model.temperature * volume_model.log_densities)
        tv = tv_regularization_3d(densities)
        sp = sparsity_regularization(densities)
        export_consistency, support_reg, main_struct_reg, manuf_reg = compute_export_structure_losses(densities)
        proj_l1, proj_iou, proj_tversky, proj_boundary = compute_tfs_projection_losses(densities[0], fp_weight=6.0)

        loss = ((p3_fn / total_w) * 7.0 +
                (p3_silh / total_w) * 3.0 +
                (p3_l1 / total_w) * 5.0 +
                (p3_iou / total_w) * 7.0 +
                (p3_dice / total_w) * 3.0 +
                tv * 0.1 +
                sp * 0.04)

        if use_tfs_hard_constraints:
            loss = loss + proj_l1 * 3.5 + proj_iou * 2.5 + proj_tversky * 2.0 + proj_boundary * 1.2
        if use_multiview_export_constraints:
            loss = loss + export_consistency * 1.65 * float(loss_export_consistency_scale)
            loss = loss + support_reg * 0.22 * float(loss_support_scale)
            loss = loss + main_struct_reg * 0.44 * float(loss_main_structure_scale)
            loss = loss + manuf_reg * 0.34 * float(loss_manufacturability_scale)
            loss = loss + (p3_export_distill / total_w) * 1.35 * float(loss_export_distill_scale)
            if not use_tfs_hard_constraints:
                loss = loss + ((p3_tversky / total_w) * 1.80 + (p3_boundary / total_w) * 0.90) * float(loss_tfs_projection_scale)
        raw_loss = loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(volume_model.parameters(), max_norm=0.3)

        if any(torch.isnan(p.grad).any() for p in volume_model.parameters() if p.grad is not None):
            optimizer.zero_grad()
            continue

        optimizer.step()
        scheduler3.step()

        if (ri % 200 == 0 and ri > 0) or ri == phase3_iters - 1:
            with torch.no_grad():
                phase3_components = {
                    'false_negative': float((p3_fn / total_w).item()),
                    'silhouette': float((p3_silh / total_w).item()),
                    'l1': float((p3_l1 / total_w).item()),
                    'iou': float((p3_iou / total_w).item()),
                    'dice': float((p3_dice / total_w).item()),
                    'focal_tversky': float((p3_tversky / total_w).item()) if use_multiview_export_constraints and not use_tfs_hard_constraints else 0.0,
                    'boundary': float((p3_boundary / total_w).item()) if use_multiview_export_constraints and not use_tfs_hard_constraints else 0.0,
                    'export_render_distill': float((p3_export_distill / total_w).item()) if export_target_masks_t is not None else 0.0,
                    'reg_tv': float(tv.item()),
                    'reg_sparsity': float(sp.item()),
                    'export_consistency': float(export_consistency.item()),
                    'support': float(support_reg.item()),
                    'main_structure': float(main_struct_reg.item()),
                    'manufacturability': float(manuf_reg.item()),
                    'projection_l1': float(proj_l1.item()),
                    'projection_iou': float(proj_iou.item()),
                    'projection_tversky': float(proj_tversky.item()),
                    'projection_boundary': float(proj_boundary.item()),
                }
                mean_iou, _, view_ious, _ = evaluate(
                    'Phase 3',
                    global_offset + ri,
                    total_loss=loss.item(),
                    normalized_total_loss=loss.item(),
                    raw_total_loss=loss.item(),
                    loss_components=phase3_components,
                )
                update_view_weights(view_ious)
                current_binary = volume_model.get_binary_voxels(threshold=0.5)[0].detach().cpu().numpy() > 0.5
                current_density = torch.sigmoid(volume_model.temperature * volume_model.log_densities)[0].detach().cpu().numpy()
                maybe_refresh_export_teacher(current_binary, current_density, "Phase3", ri, refresh_every=250)
                if mean_iou > phase3_best:
                    phase3_best = mean_iou

            phase3_loop.set_postfix({'IOU': f'{mean_iou:.4f}', 'best': f'{best_iou:.4f}'})

    restore_best()
    phase4_best = best_iou
    renderer_phase4_enabled = use_multiview_export_constraints

    if renderer_phase4_enabled:
        print("\n" + "=" * 60)
        print("PHASE 4: RENDERER POLISH")
        print("=" * 60)
        print(f"  Starting from renderer IOU={best_iou:.4f}")

        phase4_iters = int(renderer_phase4_iters) if renderer_phase4_iters is not None else max(800, int(Niter * 0.25))
        phase4_lr = max(lr * 0.004, 2e-5)
        p4_temp_start = volume_model.temperature
        p4_temp_end = max(float(p4_temp_start), 28.0)

        optimizer = torch.optim.Adam(volume_model.parameters(), lr=phase4_lr,
                                      betas=(0.9, 0.999), eps=1e-8)
        scheduler4 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=phase4_iters, eta_min=phase4_lr * 0.2
        )

        global_offset = phase1_iters + phase2_iters + phase3_iters
        phase4_loop = tqdm(range(phase4_iters), desc="Phase 4")
        for ri in phase4_loop:
            progress = ri / max(phase4_iters - 1, 1)
            volume_model.temperature = p4_temp_start + (p4_temp_end - p4_temp_start) * progress
            raymarcher.tau = 40.0 + 8.0 * progress

            optimizer.zero_grad(set_to_none=True)

            p4_fn, p4_bce = 0.0, 0.0
            p4_asym, p4_iou, p4_dice = 0.0, 0.0, 0.0
            p4_tversky, p4_boundary = 0.0, 0.0
            p4_export_distill = 0.0

            for n, silh_gt in enumerate(silhs_tensors):
                silh_pred = render_view(n)
                w = view_weights[n]
                silh_pred_clamped = silh_pred.clamp(1e-4, 1.0 - 1e-4)

                p4_fn += false_negative_only_loss(silh_pred, silh_gt) * w
                p4_bce += F.binary_cross_entropy(silh_pred_clamped, silh_gt) * w
                p4_asym += asymmetric_silhouette_loss(silh_pred, silh_gt, fp_weight=4.0) * w
                p4_iou += differentiable_iou_loss(silh_pred, silh_gt) * w
                p4_dice += differentiable_dice_loss(silh_pred, silh_gt) * w
                if use_multiview_export_constraints and not use_tfs_hard_constraints:
                    p4_tversky += focal_tversky_loss(silh_pred, silh_gt, alpha=0.25, beta=0.75, gamma=1.8) * w
                    p4_boundary += boundary_loss(silh_pred, silh_gt, device=device) * w
                if export_target_masks_t is not None and n < len(export_target_masks_t):
                    teacher_mask = export_target_masks_t[n]
                    p4_export_distill += (
                        F.binary_cross_entropy(silh_pred_clamped, teacher_mask) +
                        differentiable_iou_loss(silh_pred, teacher_mask) +
                        differentiable_dice_loss(silh_pred, teacher_mask) +
                        asymmetric_silhouette_loss(silh_pred, teacher_mask, fp_weight=5.0)
                    ) * w

            total_w = view_weights.sum().item()
            densities = torch.sigmoid(volume_model.temperature * volume_model.log_densities)
            tv = tv_regularization_3d(densities)
            sp = sparsity_regularization(densities)
            export_consistency, support_reg, main_struct_reg, manuf_reg = compute_export_structure_losses(densities)
            proj_l1, proj_iou, proj_tversky, proj_boundary = compute_tfs_projection_losses(
                densities[0], fp_weight=4.0
            )

            loss = (
                (p4_fn / total_w) * 2.0 +
                (p4_bce / total_w) * 6.0 +
                (p4_asym / total_w) * 4.0 +
                (p4_iou / total_w) * 7.0 +
                (p4_dice / total_w) * 3.0 +
                tv * 0.05 +
                sp * 0.02
            )
            loss = loss + (proj_l1 * 1.5 + proj_iou * 1.0 + proj_tversky * 0.7 + proj_boundary * 0.3) * float(loss_tfs_projection_scale)
            if use_multiview_export_constraints:
                loss = loss + export_consistency * 1.10 * float(loss_export_consistency_scale)
                loss = loss + support_reg * 0.15 * float(loss_support_scale)
                loss = loss + main_struct_reg * 0.28 * float(loss_main_structure_scale)
                loss = loss + manuf_reg * 0.22 * float(loss_manufacturability_scale)
                loss = loss + (p4_export_distill / total_w) * 1.45 * float(loss_export_distill_scale)
                if not use_tfs_hard_constraints:
                    loss = loss + ((p4_tversky / total_w) * 2.00 + (p4_boundary / total_w) * 1.10) * float(loss_tfs_projection_scale)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(volume_model.parameters(), max_norm=0.25)

            if any(torch.isnan(p.grad).any() for p in volume_model.parameters() if p.grad is not None):
                optimizer.zero_grad()
                continue

            optimizer.step()
            scheduler4.step()

            if ri % 150 == 0 or ri == phase4_iters - 1:
                with torch.no_grad():
                    phase4_components = {
                        'false_negative': float((p4_fn / total_w).item()),
                        'render_bce': float((p4_bce / total_w).item()),
                        'render_asymmetric': float((p4_asym / total_w).item()),
                        'render_iou': float((p4_iou / total_w).item()),
                        'render_dice': float((p4_dice / total_w).item()),
                        'focal_tversky': float((p4_tversky / total_w).item()) if use_multiview_export_constraints and not use_tfs_hard_constraints else 0.0,
                        'boundary': float((p4_boundary / total_w).item()) if use_multiview_export_constraints and not use_tfs_hard_constraints else 0.0,
                        'export_render_distill': float((p4_export_distill / total_w).item()) if export_target_masks_t is not None else 0.0,
                        'reg_tv': float(tv.item()),
                        'reg_sparsity': float(sp.item()),
                        'export_consistency': float(export_consistency.item()),
                        'support': float(support_reg.item()),
                        'main_structure': float(main_struct_reg.item()),
                        'manufacturability': float(manuf_reg.item()),
                        'projection_l1': float(proj_l1.item()),
                        'projection_iou': float(proj_iou.item()),
                        'projection_tversky': float(proj_tversky.item()),
                        'projection_boundary': float(proj_boundary.item()),
                    }
                    mean_iou, _, view_ious, _ = evaluate(
                        'Phase 4',
                        global_offset + ri,
                        total_loss=loss.item(),
                        normalized_total_loss=loss.item(),
                        raw_total_loss=loss.item(),
                        loss_components=phase4_components,
                    )
                    update_view_weights(view_ious)
                    current_binary = volume_model.get_binary_voxels(threshold=0.5)[0].detach().cpu().numpy() > 0.5
                    current_density = torch.sigmoid(volume_model.temperature * volume_model.log_densities)[0].detach().cpu().numpy()
                    maybe_refresh_export_teacher(current_binary, current_density, "Phase4", ri, refresh_every=150)
                    if mean_iou > phase4_best:
                        phase4_best = mean_iou

                phase4_loop.set_postfix({'IOU': f'{mean_iou:.4f}', 'best': f'{best_iou:.4f}'})

        restore_best()
    elif use_multiview_export_constraints:
        print("\n  Phase 4 skipped unexpectedly.")

    print(f"\n  Final best IOU: {best_iou:.4f} "
          f"(P1={phase1_best:.4f} -> P2={phase2_best:.4f} -> P3={phase3_best:.4f}"
          + (f" -> P4={phase4_best:.4f}" if renderer_phase4_enabled else "") + ")")

    print("\n" + "=" * 60)
    print("PHASE 5: JOINT CHECKPOINT SELECTION")
    print("=" * 60)
    restore_best()
    print(
        f"  Selected checkpoint by joint score={best_joint_score:.6f} | "
        f"min_iou={best_renderer_min_iou:.6f} | teacher_agreement={best_teacher_agreement:.6f}"
    )

    # ==========================================================
    # 最终评估
    # ==========================================================
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    renderer_ious, renderer_dices = [], []
    with torch.no_grad():
        volume_model.eval()
        for n in range(actual_num_views):
            silh_pred = render_view(n).cpu().numpy()
            pred_img = (silh_pred * 255).astype(np.uint8)
            _, pred_bin = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imwrite(os.path.join(output_dir, f"sample_{exp_sample_id}_renderer_view_{n}.png"), pred_bin)
            cv2.imwrite(os.path.join(output_dir, f"sample_{exp_sample_id}_pred_raw_view_{n}.png"), pred_bin)
            renderer_ious.append(iou_in_memory(pred_bin, gt_arrays[n]))
            renderer_dices.append(dice_in_memory(pred_bin, gt_arrays[n]))

    renderer_mean_iou = np.mean(renderer_ious)
    renderer_mean_dice = np.mean(renderer_dices)
    renderer_min_iou = float(np.min(renderer_ious)) if renderer_ious else 0.0

    print(f"\nRenderer Model: IOU={renderer_mean_iou:.6f}, DICE={renderer_mean_dice:.6f}")
    for idx, (iou_v, dice_v) in enumerate(zip(renderer_ious, renderer_dices)):
        s = '***' if iou_v > 0.97 else '**' if iou_v > 0.95 else '*' if iou_v > 0.93 else ''
        print(f"  Renderer View {idx}: IOU={iou_v:.4f}, DICE={dice_v:.4f} {s}")
    if use_multiview_export_constraints:
        print("  Note: Renderer Model is the learned proxy volume. The final target metric is Final Export Volume below.")

    learned_binary = volume_model.get_binary_voxels(threshold=0.5)[0].cpu().numpy() > 0.5
    export_strategy = "learned"
    export_volume = learned_binary
    selection_gate = None
    candidates = {}
    candidate_metrics_log = {
        'learned': {
            'mean_iou': float(renderer_mean_iou),
            'mean_dice': float(renderer_mean_dice),
            'occupied_voxels': int(learned_binary.sum()),
            **analyze_connected_components(learned_binary),
        }
    }
    export_metrics = {
        'view_ious': [float(v) for v in renderer_ious],
        'view_dices': [float(v) for v in renderer_dices],
        'mean_iou': float(renderer_mean_iou),
        'mean_dice': float(renderer_mean_dice),
        'occupied_voxels': int(learned_binary.sum()),
        **analyze_connected_components(learned_binary),
    }

    if use_multiview_export_constraints:
        learned_density = torch.sigmoid(volume_model.temperature * volume_model.log_densities)[0].detach().cpu().numpy()
        export_bundle = (
            build_tfs_export_bundle(learned_binary, learned_density, final_export_res)
            if use_tfs_hard_constraints
            else build_multiview_export_bundle(learned_binary, learned_density, final_export_res)
        )
        candidates = export_bundle['candidates']
        candidate_metrics_log.update(export_bundle['candidate_metrics_log'])
        export_strategy = export_bundle['export_strategy']
        export_volume = export_bundle['export_volume']
        export_metrics = export_bundle['export_metrics']
        selection_gate = export_bundle.get('selection_gate')
        export_min_iou = float(export_metrics.get('min_iou', 0.0))
        if (
            not use_tfs_hard_constraints and (
                float(export_metrics.get('mean_iou', 0.0)) + 1e-6 < float(renderer_mean_iou)
                or export_min_iou + 0.01 < float(renderer_min_iou)
            )
        ):
            print(
                "  Export safeguard: generic hard export underperformed the learned renderer "
                "on mean IOU or worst-view IOU; "
                "reverting to learned binary export."
            )
            export_strategy = "learned"
            export_volume = learned_binary
            export_metrics = {
                'view_ious': [float(v) for v in renderer_ious],
                'view_dices': [float(v) for v in renderer_dices],
                'mean_iou': float(renderer_mean_iou),
                'mean_dice': float(renderer_mean_dice),
                'occupied_voxels': int(learned_binary.sum()),
                **analyze_connected_components(learned_binary),
                'pred_masks': [resize_binary_mask(mask, img_size).astype(bool) for mask in gt_arrays[:actual_num_views]],
            }

        print(f"\nSelected export strategy: {export_strategy}")
        print(f"Final Export Volume: IOU={export_metrics['mean_iou']:.6f}, DICE={export_metrics['mean_dice']:.6f}")
        print(
            "  Connectivity: "
            f"components={int(export_metrics.get('component_count', 0))}, "
            f"largest_ratio={float(export_metrics.get('largest_component_ratio', 0.0)):.6f}, "
            f"bridge_added={int(export_metrics.get('bridge_voxels_added', 0))}"
        )
        if export_metrics.get('repair_voxels_added', 0):
            print(
                "  Silhouette repair: "
                f"voxels_added={int(export_metrics.get('repair_voxels_added', 0))}, "
                f"per_axis={export_metrics.get('repair_pixels_by_axis', export_metrics.get('repair_pixels_by_view', {}))}"
            )
        for idx, (iou_v, dice_v) in enumerate(zip(export_metrics['view_ious'], export_metrics['view_dices'])):
            s = '***' if iou_v > 0.97 else '**' if iou_v > 0.95 else '*' if iou_v > 0.93 else ''
            print(f"  Export View {idx}: IOU={iou_v:.4f}, DICE={dice_v:.4f} {s}")

        for idx, pred_mask in enumerate(export_metrics['pred_masks'][:actual_num_views]):
            pred_path = os.path.join(output_dir, f"sample_{exp_sample_id}_view_{idx}.png")
            cv2.imwrite(pred_path, pred_mask.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(output_dir, f"sample_{exp_sample_id}_pred_cleaned_view_{idx}.png"), pred_mask.astype(np.uint8) * 255)
    else:
        for idx in range(actual_num_views):
            src = os.path.join(output_dir, f"sample_{exp_sample_id}_renderer_view_{idx}.png")
            dst = os.path.join(output_dir, f"sample_{exp_sample_id}_view_{idx}.png")
            if os.path.exists(src):
                img = cv2.imread(src, 0)
                cv2.imwrite(dst, img)
                cv2.imwrite(os.path.join(output_dir, f"sample_{exp_sample_id}_pred_cleaned_view_{idx}.png"), img)

    voxels_path = os.path.join(output_dir, f"sample_{exp_sample_id}_final_voxels.npy")
    np.save(voxels_path, export_volume[None].astype(np.float32))
    if save_export_candidates and candidates:
        candidate_dir = os.path.join(output_dir, "export_candidates")
        os.makedirs(candidate_dir, exist_ok=True)
        for candidate_name, candidate_volume in candidates.items():
            safe_name = "".join(
                ch if ch.isalnum() or ch in {"-", "_"} else "_"
                for ch in str(candidate_name)
            )
            np.save(
                os.path.join(candidate_dir, f"{safe_name}.npy"),
                np.asarray(candidate_volume, dtype=np.uint8),
            )

    obj_path = os.path.join(output_dir, f"sample_{exp_sample_id}_output.obj")
    if not save_voxel_mesh_as_obj(export_volume, obj_path, extent_world=volume_extent_world):
        try:
            volume_model.save_obj(obj_path, threshold=0.5)
        except Exception as e:
            print(f"WARNING: save_obj failed: {e}")

    reconstructed_obj_path = os.path.join(output_dir, f"sample_{exp_sample_id}_reconstructed_mesh.obj")
    try:
        export_reconstructed_mesh(
            export_volume.astype(np.float32),
            reconstructed_obj_path,
            method=str(recon_mesh_method),
            extent_world=volume_extent_world,
            smooth_sigma=float(recon_mesh_smooth_sigma),
            taubin_iterations=int(recon_mesh_taubin_iterations),
            upsample_factor=int(recon_mesh_upsample_factor),
        )
    except Exception as e:
        print(f"WARNING: reconstructed mesh export failed: {e}")

    train_runtime_seconds = time.time() - train_start_time
    final_normal_consistency = estimate_voxel_normal_consistency(
        export_volume,
        device=device,
        max_res=min(128, final_export_res),
    )
    teacher_agreement_final = compute_teacher_agreement(learned_binary)
    renderer_export_gap = float(export_metrics['mean_iou'] - renderer_mean_iou)

    phase_best_ious = {
        'Phase 1': float(phase1_best),
        'Phase 2': float(phase2_best),
        'Phase 3': float(phase3_best),
    }
    if renderer_phase4_enabled:
        phase_best_ious['Phase 4'] = float(phase4_best)
    phase_best_ious['Global Best'] = float(best_iou)

    metrics_history['final_metrics'] = {
        'renderer_metrics': {
            'view_ious': [float(v) for v in renderer_ious],
            'view_dices': [float(v) for v in renderer_dices],
            'mean_iou': float(renderer_mean_iou),
            'mean_dice': float(renderer_mean_dice),
            'min_iou': float(renderer_min_iou),
        },
        'export_metrics': {
            'view_ious': [float(v) for v in export_metrics['view_ious']],
            'view_dices': [float(v) for v in export_metrics['view_dices']],
            'mean_iou': float(export_metrics['mean_iou']),
            'mean_dice': float(export_metrics['mean_dice']),
            'mean_recall': float(export_metrics.get('mean_recall', 0.0)),
            'min_iou': float(export_metrics.get('min_iou', 0.0)),
            'occupied_voxels': int(export_metrics.get('occupied_voxels', int(export_volume.sum()))),
            'component_count': int(export_metrics.get('component_count', 0)),
            'largest_component_ratio': float(export_metrics.get('largest_component_ratio', 0.0)),
            'small_component_voxels': int(export_metrics.get('small_component_voxels', 0)),
            'bridge_voxels_added': int(export_metrics.get('bridge_voxels_added', 0)),
            'repair_voxels_added': int(export_metrics.get('repair_voxels_added', 0)),
            'repair_pixels_by_axis': export_metrics.get('repair_pixels_by_axis', {}),
            'repair_pixels_by_view': export_metrics.get('repair_pixels_by_view', {}),
        },
        'export_strategy': export_strategy,
        'selection_gate': selection_gate,
        'renderer_export_gap': float(renderer_export_gap),
        'teacher_agreement': float(teacher_agreement_final),
        'joint_checkpoint_score': float(best_joint_score),
        'teacher_source_history': teacher_source_history,
        'normal_consistency': None if final_normal_consistency is None else float(final_normal_consistency),
        'export_resolution': int(final_export_res),
        'export_candidates': candidate_metrics_log,
        'phase_best_ious': phase_best_ious,
        'training_export_prior': (
            {
                'source': str(export_target_source),
                'mean_iou': float(export_target_metrics.get('mean_iou', 0.0)),
                'mean_dice': float(export_target_metrics.get('mean_dice', 0.0)),
                'occupied_voxels': int(export_target_metrics.get('occupied_voxels', int(export_target_volume_np.sum()))),
                'component_count': int(export_target_metrics.get('component_count', 0)),
                'largest_component_ratio': float(export_target_metrics.get('largest_component_ratio', 0.0)),
            }
            if export_target_metrics is not None and export_target_volume_np is not None
            else None
        ),
        'config': {
            'device': str(device),
            'shadow_files': [os.path.basename(path) for path in shadow_files],
            'num_iters': int(Niter),
            'base_lr': float(lr),
            'num_input_views': int(len(shadow_files)),
            'num_total_views': int(actual_num_views),
            'mirror_mode': bool(mirror_mode),
            'tsf_mode': bool(TSF_mode),
            'tsf4v_mode': bool(TSF4v_mode),
            'projection_mode': 'tsf' if TSF_mode else ('tsf4v' if TSF4v_mode else 'ring'),
            'camera_mode': str(camera_mode),
            'img_size': int(img_size),
            'z_distance': float(zdist),
            'voxel_res': int(requested_voxel_res),
            'proxy_res': int(proxy_res),
            'export_res': int(final_export_res),
            'recon_mesh_method': str(recon_mesh_method),
            'recon_mesh_smooth_sigma': float(recon_mesh_smooth_sigma),
            'recon_mesh_taubin_iterations': int(recon_mesh_taubin_iterations),
            'recon_mesh_upsample_factor': int(recon_mesh_upsample_factor),
            'use_visual_hull': bool(use_visual_hull),
            'keyview_pause_seconds': float(keyview_pause_seconds),
            'renderer_refine_phase': bool(renderer_phase4_enabled),
            'training_export_prior_enabled': bool(export_target_volume_t is not None),
            'candidate_min_iou': float(candidate_min_iou),
            'candidate_min_saving': float(candidate_min_saving),
            'candidate_max_saving': float(candidate_max_saving),
        },
        'runtime': {
            'train_seconds': float(train_runtime_seconds),
        },
    }
    if tfs_calibration is not None:
        metrics_history['final_metrics']['tfs_calibration'] = {
            'axis_for_view': {int(k): str(v) for k, v in tfs_calibration.get('axis_for_view', {}).items()},
            'ops': {int(k): int(v) for k, v in tfs_calibration.get('ops', {}).items()},
            'mean_iou': float(tfs_calibration.get('mean_iou', 0.0)),
            'view_ious': [float(v) for v in tfs_calibration.get('view_ious', [])],
            'search_res': int(tfs_calibration.get('search_res', proxy_res)),
            'occupied_voxels': int(tfs_calibration.get('occupied_voxels', 0)),
        }

    with open(os.path.join(output_dir, 'metrics_history.json'), 'w') as f:
        json.dump(metrics_history, f, indent=2, ensure_ascii=False)

    metrics_txt_path = os.path.join(output_dir, f"sample_{exp_sample_id}_metrics.txt")
    with open(metrics_txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("VOXEL RECONSTRUCTION METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write("Renderer Metrics\n")
        if use_tfs_hard_constraints:
            f.write("Note: Renderer Model is the learned proxy volume; target quality should be read from Final Export Metrics.\n")
        f.write(f"Mean IOU:  {renderer_mean_iou:.6f}\n")
        f.write(f"Mean DICE: {renderer_mean_dice:.6f}\n")
        f.write(f"Min IOU:   {renderer_min_iou:.6f}\n")
        f.write(f"Per-View IOUs:  {', '.join([f'{v:.6f}' for v in renderer_ious])}\n")
        f.write(f"Per-View DICEs: {', '.join([f'{v:.6f}' for v in renderer_dices])}\n\n")
        f.write("Final Export Metrics\n")
        f.write(f"Export Strategy: {export_strategy}\n")
        f.write(f"Export Resolution: {final_export_res}\n")
        f.write(f"Reconstructed Mesh Method: {recon_mesh_method}\n")
        f.write(f"Reconstructed Mesh Sigma: {float(recon_mesh_smooth_sigma):.3f}\n")
        f.write(f"Reconstructed Mesh Taubin Iterations: {int(recon_mesh_taubin_iterations)}\n")
        f.write(f"Reconstructed Mesh Upsample Factor: {int(recon_mesh_upsample_factor)}\n")
        f.write(f"Mean IOU:  {export_metrics['mean_iou']:.6f}\n")
        f.write(f"Mean DICE: {export_metrics['mean_dice']:.6f}\n")
        f.write(f"Per-View IOUs:  {', '.join([f'{v:.6f}' for v in export_metrics['view_ious']])}\n")
        f.write(f"Per-View DICEs: {', '.join([f'{v:.6f}' for v in export_metrics['view_dices']])}\n")
        f.write(f"Occupied Voxels: {int(export_metrics.get('occupied_voxels', int(export_volume.sum())))}\n\n")
        f.write(f"Renderer-Export Gap: {renderer_export_gap:.6f}\n")
        f.write(f"Teacher Agreement: {teacher_agreement_final:.6f}\n")
        f.write(f"Joint Checkpoint Score: {best_joint_score:.6f}\n\n")
        if export_target_metrics is not None and export_target_volume_np is not None:
            f.write("Training Export Prior\n")
            f.write(f"Prior Source: {export_target_source}\n")
            f.write(f"Prior Mean IOU:  {float(export_target_metrics.get('mean_iou', 0.0)):.6f}\n")
            f.write(f"Prior Mean DICE: {float(export_target_metrics.get('mean_dice', 0.0)):.6f}\n")
            f.write(f"Prior Components: {int(export_target_metrics.get('component_count', 0))}\n")
            f.write(f"Prior Occupied Voxels: {int(export_target_metrics.get('occupied_voxels', int(export_target_volume_np.sum())))}\n\n")
        f.write(f"Normal Consistency: {'N/A' if final_normal_consistency is None else f'{float(final_normal_consistency):.6f}'}\n")
        if export_metrics.get('repair_voxels_added', 0):
            f.write(f"Repair Voxels Added: {int(export_metrics.get('repair_voxels_added', 0))}\n")
            if export_metrics.get('repair_pixels_by_axis'):
                f.write(f"Repair Pixels by Axis: {export_metrics.get('repair_pixels_by_axis', {})}\n")
            if export_metrics.get('repair_pixels_by_view'):
                f.write(f"Repair Pixels by View: {export_metrics.get('repair_pixels_by_view', {})}\n")
        f.write("\n")
        f.write(f"Phase 1 Best IOU: {phase1_best:.6f}\n")
        f.write(f"Phase 2 Best IOU: {phase2_best:.6f}\n")
        f.write(f"Phase 3 Best IOU: {phase3_best:.6f}\n")
        if renderer_phase4_enabled:
            f.write(f"Phase 4 Best IOU: {phase4_best:.6f}\n")
        f.write(f"Global Best IOU:  {best_iou:.6f}\n")

    if post_export_bundle:
        print("\n" + "=" * 60)
        print("POST EXPORT BUNDLE")
        print("=" * 60)
        if generate_post_export_bundle is None:
            print(f"WARNING: post_export import failed: {POST_EXPORT_IMPORT_ERROR}")
        else:
            try:
                post_export_context = {
                    'output_dir': output_dir,
                    'sample_id': int(exp_sample_id),
                    'device': str(device),
                    'renderer_metrics': metrics_history['final_metrics']['renderer_metrics'],
                    'export_metrics': metrics_history['final_metrics']['export_metrics'],
                    'obj_path': obj_path,
                    'voxels_path': voxels_path,
                    'runtime_seconds': float(train_runtime_seconds),
                    'rotation_frames': int(rotation_frames),
                    'rotation_fps': int(rotation_fps),
                    'keyview_pause_seconds': float(keyview_pause_seconds),
                    'history_roots': history_roots,
                    'candidate_metrics': candidate_metrics_log,
                    'candidate_volumes': {
                        'learned': learned_binary,
                        **candidates,
                    },
                    'post_export_start_time': time.time(),
                }
                generate_post_export_bundle(post_export_context)
            except Exception as e:
                print(f"WARNING: post export bundle failed: {e}")
                import traceback
                traceback.print_exc()

    print(f"\nSaved to {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Shadow Art - Voxel Version")
    parser.add_argument("device", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("num_iters", type=int)
    parser.add_argument("lr", type=float)
    parser.add_argument("exp_id", type=int)
    parser.add_argument("-sdlist", "--shadow_files", nargs="+", required=True)
    parser.add_argument("-mr", "--mirror_mode", type=int, default=0)
    parser.add_argument("-tsf", "--tsf_mode", type=int, default=0)
    parser.add_argument("-tsf4", "--tsf4v_mode", type=int, default=0)
    parser.add_argument("-cam", "--camera_mode", type=str, default="ortho")
    parser.add_argument("-imsz", "--img_size", type=int, default=256)
    parser.add_argument("-zdist", "--z_distance", type=float, default=2.5)
    parser.add_argument("-debug", "--debug_mode", type=int, default=0)
    parser.add_argument("-res", "--voxel_res", type=int, default=128,
                        help="Voxel grid resolution (default 128)")
    parser.add_argument("-eres", "--export_res", type=int, default=0,
                        help="Final exported voxel resolution (0=auto, TFS defaults to img_size)")
    parser.add_argument("-vh", "--visual_hull", type=int, default=1,
                        help="Use visual hull init (0=off, 1=on)")
    parser.add_argument("--post_export_bundle", type=int, default=1,
                        help="Generate extra reports, charts, GIFs, and paper figures after training")
    parser.add_argument("--history_roots", nargs="*", default=None,
                        help="Optional history roots for paper-style figure aggregation")
    parser.add_argument("--rotation_frames", type=int, default=72,
                        help="Frame count for the exported turntable GIF")
    parser.add_argument("--rotation_fps", type=int, default=18,
                        help="FPS for the exported turntable GIF")
    parser.add_argument("--keyview_pause_seconds", type=float, default=10.0,
                        help="Pause duration in seconds at each silhouette key view")
    parser.add_argument("--recon_mesh_method", type=str, default="mcubes",
                        choices=["mcubes", "poisson"],
                        help="Method for the extra reconstructed mesh OBJ exported after training")
    parser.add_argument("--recon_mesh_smooth_sigma", type=float, default=0.35,
                        help="Smoothing sigma for reconstructed mesh extraction")
    parser.add_argument("--recon_mesh_taubin_iterations", type=int, default=2,
                        help="Taubin smoothing iterations for reconstructed marching-cubes mesh")
    parser.add_argument("--recon_mesh_upsample_factor", type=int, default=2,
                        help="Nearest-neighbor voxel upsample factor before marching cubes")
    parser.add_argument("-fixedcalib", "--fixed_calib", type=int, default=0,
                        help="1 = fixed-camera branch: lock axis permutation (no axis search)")
    parser.add_argument("-volmin", "--vol_min", type=float, default=0.0,
                        help=">0 = enable differentiable material loss and constrained soft/geometry candidate selection")
    parser.add_argument("-connw", "--conn_w", type=float, default=1.0,
                        help="multiplier on the differentiable reachability connectivity loss (0=ablate off)")
    parser.add_argument("--candidate_min_iou", type=float, default=0.95,
                        help="preferred minimum mean IoU for final candidate selection")
    parser.add_argument("--candidate_min_saving", type=float, default=0.40,
                        help="preferred minimum material saving relative to the visual hull")
    parser.add_argument("--candidate_max_saving", type=float, default=0.85,
                        help="preferred maximum material saving relative to the visual hull")
    parser.add_argument("--save_export_candidates", type=int, default=0,
                        help="save every export candidate as output_dir/export_candidates/*.npy")
    parser.add_argument("--loss_export_consistency_scale", type=float, default=1.0)
    parser.add_argument("--loss_support_scale", type=float, default=1.0)
    parser.add_argument("--loss_main_structure_scale", type=float, default=1.0)
    parser.add_argument("--loss_manufacturability_scale", type=float, default=1.0)
    parser.add_argument("--loss_export_distill_scale", type=float, default=1.0)
    parser.add_argument("--loss_tfs_projection_scale", type=float, default=1.0)
    parser.add_argument("--loss_volume_scale", type=float, default=1.0)
    parser.add_argument("--loss_topo_connectivity_scale", type=float, default=1.0)
    parser.add_argument("--renderer_phase4_iters", type=int, default=None)
    args = parser.parse_args()

    if torch.cuda.is_available() and "cuda" in args.device.lower():
        device = torch.device(args.device)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    train(
        device=device,
        shadow_files=args.shadow_files,
        output_dir=args.output_dir,
        Niter=args.num_iters,
        lr=args.lr,
        exp_sample_id=args.exp_id,
        num_views=len(args.shadow_files),
        mirror_mode=bool(args.mirror_mode),
        TSF_mode=bool(args.tsf_mode),
        TSF4v_mode=bool(args.tsf4v_mode),
        zdist=args.z_distance,
        camera_mode=args.camera_mode,
        img_size=args.img_size,
        debug_mode=bool(args.debug_mode),
        voxel_res=args.voxel_res,
        use_visual_hull=bool(args.visual_hull),
        export_res=args.export_res,
        post_export_bundle=bool(args.post_export_bundle),
        history_roots=args.history_roots,
        rotation_frames=args.rotation_frames,
        rotation_fps=args.rotation_fps,
        keyview_pause_seconds=args.keyview_pause_seconds,
        recon_mesh_method=args.recon_mesh_method,
        recon_mesh_smooth_sigma=args.recon_mesh_smooth_sigma,
        recon_mesh_taubin_iterations=args.recon_mesh_taubin_iterations,
        recon_mesh_upsample_factor=args.recon_mesh_upsample_factor,
        fixed_calib=bool(args.fixed_calib),
        vol_min=float(args.vol_min),
        conn_w=float(args.conn_w),
        loss_export_consistency_scale=float(args.loss_export_consistency_scale),
        loss_support_scale=float(args.loss_support_scale),
        loss_main_structure_scale=float(args.loss_main_structure_scale),
        loss_manufacturability_scale=float(args.loss_manufacturability_scale),
        loss_export_distill_scale=float(args.loss_export_distill_scale),
        loss_tfs_projection_scale=float(args.loss_tfs_projection_scale),
        loss_volume_scale=float(args.loss_volume_scale),
        loss_topo_connectivity_scale=float(args.loss_topo_connectivity_scale),
        candidate_min_iou=float(args.candidate_min_iou),
        candidate_min_saving=float(args.candidate_min_saving),
        candidate_max_saving=float(args.candidate_max_saving),
        save_export_candidates=bool(args.save_export_candidates),
        renderer_phase4_iters=args.renderer_phase4_iters,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
