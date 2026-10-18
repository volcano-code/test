"""Paired CPU warm-start mechanism probe; NOT a standard soft artifact producer.

Original training and delivery modules are read-only dependencies. New fields
cannot be relabelled as a full from-scratch run or a passed printable result.
"""
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from scipy import ndimage as ndi

from .artifacts import load_soft
from .contracts import CameraContract, TargetSet, file_hash, mask_hash
from .entity_projection import render_probability_max
from .layout import iou_report
from .losses import compute_loss, _balanced_bce_per_view
from .metrics import soft_projection_metrics
from .rendering import render_binary


FROZEN_CAMERA = '01bf5e162d7e1868fa44bfe49ac70a0fbca5642f1011062933af92b829e9f2f8'
FROZEN_FIELDS = {
    'guidance.npy': 'c7b6be119e52cc5217c991d6b4c43e7e1653c09fa8e2d88c0dd8997bb27730fd',
    'density.npy': '803e523e9d8542f5cd03c83dd979182d792644ff8b74e6484f2f38bd9c4eb6e1',
    'occupancy.npy': 'd3453299a5869d42b3fcf3bc0bceec712db88c7c7a26b505a0f75527161c448e',
}


def load_frozen(source, silhouettes):
    source = Path(source)
    contract = CameraContract.from_dict(json.loads((source / 'CAMERA_CONTRACT.json').read_text()))
    if contract.digest != FROZEN_CAMERA or contract.volume_resolution != 128:
        raise ValueError('FAILED_FROZEN_CAMERA')
    for name, digest in FROZEN_FIELDS.items():
        if not (source / name).is_file() or file_hash(source / name) != digest:
            raise ValueError('FAILED_FROZEN_FIELD: ' + name)
    targets = TargetSet.from_paths(silhouettes)
    if (targets.input_hashes != contract.input_hashes or
            tuple(mask_hash(m) for m in targets.render(contract.cameras, contract.image_size,
                                                       contract.layout)) != contract.target_hashes):
        raise ValueError('FAILED_TARGET_HASH')
    return contract, targets, load_soft(source, contract)


def _initial_logits(initial):
    p = np.array(initial, copy=True)
    if (p.ndim != 3 or len(set(p.shape)) != 1 or not p.size or
            p.dtype not in (np.dtype('float32'), np.dtype('float64')) or
            not np.isfinite(p).all() or np.any(p <= 0) or np.any(p >= 1)):
        raise ValueError('initial probability must be a finite strictly interior floating cube')
    return torch.logit(torch.from_numpy(p)).requires_grad_(True)


def _nonnegative(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(name + ' must be finite and ' + ('positive' if positive else 'nonnegative'))


def _base_loss(probability, targets, cameras, extent):
    predictions = render_probability_max(probability, cameras, targets.shape[-1], extent)
    return compute_loss(predictions, targets)[0]


def _aux_loss(probability, targets, cameras, width, root, backend, extent):
    from .rooted_core import rooted_core
    core = rooted_core(probability, width, root, backend)
    predictions = render_probability_max(core, cameras, targets.shape[-1], extent)
    return _balanced_bce_per_view(predictions, targets).mean()


def _truth(targets, cameras, dtype):
    truth = torch.as_tensor(np.array(targets, copy=True), dtype=dtype)
    if (truth.ndim != 3 or truth.shape[0] != len(cameras) or not len(cameras) or
            truth.shape[1] != truth.shape[2] or truth.shape[1] < 2 or
            not torch.isfinite(truth).all() or ((truth != 0) & (truth != 1)).any()):
        raise ValueError('one finite binary square target per camera required')
    return truth


def calibrate(initial, targets, cameras, *, width, root, backend,
              gradient_ratio=.1, extent=1.7):
    """Freeze ONE auxiliary coefficient on the common initial logit gradients."""
    _nonnegative(gradient_ratio, 'gradient_ratio', positive=True)
    cameras = tuple(cameras)
    logits = _initial_logits(initial)
    truth = _truth(targets, cameras, logits.dtype)
    base = _base_loss(logits.sigmoid(), truth, cameras, extent)
    base_gradient, = torch.autograd.grad(base, logits)
    aux = _aux_loss(logits.sigmoid(), truth, cameras, width, root, backend, extent)
    aux_gradient, = torch.autograd.grad(aux, logits)
    bn, an = float(base_gradient.norm()), float(aux_gradient.norm())
    if not math.isfinite(bn + an) or min(bn, an) <= 0:
        raise ValueError('cannot calibrate a zero or nonfinite logit gradient')
    coefficient = gradient_ratio * bn / an
    return {'rule': 'initial_logit_gradient_norm_ratio_once_no_recalibration',
            'gradient_ratio_requested': gradient_ratio, 'coefficient': coefficient,
            'base_gradient_norm': bn, 'aux_gradient_norm': an,
            'weighted_gradient_ratio': coefficient * an / bn,
            'gradient_cosine': float((base_gradient * aux_gradient).sum()) / (bn * an),
            'base_nonzero_gradient_voxels': int(torch.count_nonzero(base_gradient)),
            'aux_nonzero_gradient_voxels': int(torch.count_nonzero(aux_gradient)),
            'base_loss': float(base.detach()), 'aux_loss': float(aux.detach())}


def run_arm(initial, targets, cameras, *, width, root, backend, coefficient,
            steps=20, lr=.007, extent=1.7, callback=None):
    """Same final-step reset-Adam trajectory; zero coefficient is exact baseline."""
    _nonnegative(coefficient, 'coefficient')
    _nonnegative(lr, 'lr', positive=True)
    if type(steps) is not int or steps < 1:
        raise ValueError('steps must be a positive integer')
    if type(width) is not int or width < 1 or width > np.shape(initial)[0]:
        raise ValueError('invalid block width')
    cameras = tuple(cameras)
    logits = _initial_logits(initial)
    truth = _truth(targets, cameras, logits.dtype)
    initial_hash = hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest()
    optimizer = torch.optim.Adam([logits], lr=lr)
    history = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        probability = logits.sigmoid()
        base = _base_loss(probability, truth, cameras, extent)
        aux = (_aux_loss(probability, truth, cameras, width, root, backend, extent)
               if coefficient else base.new_zeros(()))
        total = base + coefficient * aux
        total.backward()
        norm = float(logits.grad.norm())
        if not torch.isfinite(total) or not torch.isfinite(logits.grad).all():
            raise ValueError('nonfinite pilot loss/gradient')
        optimizer.step()
        row = {'step': step, 'loss_phase': 'pre_optimizer_update',
               'base_loss': float(base.detach()),
               'aux_loss': float(aux.detach()) if coefficient else None,
               'weighted_aux_loss': coefficient * float(aux.detach()),
               'total_loss': float(total.detach()), 'gradient_norm': norm,
               'elapsed_seconds': time.monotonic() - started}
        history.append(row)
        if callback is not None:
            callback(dict(row))
    return {'probability': logits.detach().sigmoid().numpy().copy(), 'history': history,
            'initial_logits_sha256': initial_hash, 'checkpoint_selection': 'final_step',
            'optimizer_reset': True, 'steps': steps, 'lr': lr, 'coefficient': coefficient}


def measure(probability, contract, target_set, *, width, root, backend,
            sizes=(256, 512, 1024)):
    """Stage diagnostics only: discrete/continuous IoUs never claim mesh QA."""
    from .rooted_core import rooted_core
    from .manufacturing import thickness_audit
    p = np.array(probability, copy=True)
    with torch.no_grad():
        tensor = torch.from_numpy(p)
        core_tensor = rooted_core(tensor, width, root, backend)
        core = core_tensor.numpy().copy()
        evaluations = []
        for size in sizes:
            target = target_set.render(contract.cameras, size, contract.layout)
            original_projection = render_probability_max(tensor, contract.cameras, size,
                                                          contract.extent).numpy()
            core_projection = render_probability_max(core_tensor, contract.cameras, size,
                                                      contract.extent).numpy()
            hard = render_binary(p >= .5, contract.cameras, size, contract.extent)
            core_hard = render_binary(core >= .5, contract.cameras, size, contract.extent)
            parity = (np.array_equal(original_projection >= .5, hard),
                      np.array_equal(core_projection >= .5, core_hard))
            if not all(parity):
                raise ValueError('FAILED_MAX_HARD_PARITY')
            original_soft = soft_projection_metrics(original_projection, target)
            core_soft = soft_projection_metrics(core_projection, target)
            for record in (original_soft, core_soft):
                record.update(projection_rule='complete_ray_max_probability_not_optical_alpha',
                              threshold_rule='max_probability_on_complete_ray>=0.5')
            evaluations.append({'image_size': size, 'original_soft': original_soft,
                                'core_soft': core_soft, 'hard_iou': iou_report(hard, target),
                                'core_hard_iou': iou_report(core_hard, target),
                                'original_threshold_hard_parity': bool(parity[0]),
                                'core_threshold_hard_parity': bool(parity[1])})
    occupied = p >= .5
    points = np.argwhere(occupied)
    pitch = 100. / int(np.ptp(points, axis=0).max() + 1) if len(points) else 100. / p.shape[0]
    audit = thickness_audit(occupied, pitch, min_wall_mm=.9, min_neck_mm=1.35)
    face = ndi.generate_binary_structure(3, 1)
    report = {'stage': 'warm_start_soft_mechanism_only', 'native_resolution': p.shape[0],
              'evaluations': evaluations, 'occupied_voxels': int(occupied.sum()),
              'core_occupied_voxels': int((core >= .5).sum()),
              'component_count_6': int(ndi.label(occupied, face)[1]),
              'core_component_count_6': int(ndi.label(core >= .5, face)[1]),
              'occupied_bbox_pitch_mm_at_100mm': pitch,
              'block_proxy_width_cells': width,
              'block_proxy_width_mm_at_current_bbox': width * pitch,
              'thickness_audit': audit, 'mesh_qa_status': 'NOT_RUN',
              'digital_manufacturing_passed': False,
              'digital_manufacturing_status': 'NOT_EVALUATED_COMPLETE_MESH_REQUIRED'}
    return report, core
