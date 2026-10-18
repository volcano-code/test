"""From-scratch continuation with a shared Adam prefix and early core auxiliary.

No historical probability/density field is consumed. A frozen target/camera
protocol supplies only layout and input identity. This is a development pair,
not an independent confirmation ablation or a manufacturing certificate.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .contracts import CameraContract, TargetSet, mask_hash
from .continuation_projection import render_probability_continuation, continuation_state
from .losses import compute_loss
from .rendering import visual_hull
from .rooted_core import choose_root
from .rooted_core_probe import FROZEN_CAMERA, _nonnegative, _truth, _aux_loss


def load_protocol(path, silhouettes):
    """Read ONLY camera/layout JSON and source images, never trained fields."""
    contract = CameraContract.from_dict(json.loads(Path(path).read_text()))
    if contract.digest != FROZEN_CAMERA or contract.volume_resolution != 128:
        raise ValueError('FAILED_FROZEN_CAMERA')
    targets = TargetSet.from_paths(silhouettes)
    if (targets.input_hashes != contract.input_hashes or
            tuple(mask_hash(m) for m in targets.render(contract.cameras, contract.image_size,
                                                       contract.layout)) != contract.target_hashes):
        raise ValueError('FAILED_TARGET_HASH')
    return contract, targets


def _state_hash(logits, optimizer):
    """Bind parameters, Adam step/moments AND hyperparameters across the split."""
    digest = hashlib.sha256(logits.detach().numpy().tobytes())
    state = optimizer.state_dict()
    digest.update(json.dumps(state['param_groups'], sort_keys=True).encode())
    for key, value in sorted(state['state'].items()):
        digest.update(str(key).encode())
        for name, item in sorted(value.items()):
            digest.update(name.encode())
            if isinstance(item, torch.Tensor):
                digest.update(str((tuple(item.shape), item.dtype)).encode())
                digest.update(item.detach().numpy().tobytes())
            else:
                digest.update(repr(item).encode())
    return digest.hexdigest()


def _base_backward(logits, truth, cameras, step, steps, extent):
    # Match existing trainer's view_chunk_size=1 and accumulation order exactly.
    total = 0.
    for index, camera in enumerate(cameras):
        prediction = render_probability_continuation(logits.sigmoid(), [camera], truth.shape[-1],
                                                     step=step, steps=steps, extent=extent)
        loss, _ = compute_loss(prediction, truth[index:index + 1])
        (loss * (1. / len(cameras))).backward()
        total += float(loss.detach()) / len(cameras)
    return total


def train_pair(targets, cameras, *, resolution=128, steps=2000, warmup=300, ramp=300,
               lr=.007, gradient_ratio=.1, width=2, extent=1.7, backend,
               callback=None, split_callback=None, arm_callback=None):
    """Train common prefix once, fork parameters AND Adam state, finish both arms.

    Auxiliary coefficient is calibrated once at the shared post-warmup state
    using the base projector scheduled for the next step. Multiplication by a
    fixed linear ramp begins next step; no coefficient recalibration follows.
    The fixed root is chosen from target visual-hull anchors before any update.
    """
    for name, value in (('resolution', resolution), ('steps', steps), ('warmup', warmup),
                        ('ramp', ramp), ('width', width)):
        if type(value) is not int or value < 1:
            raise ValueError(name + ' must be a positive integer')
    if warmup >= steps or warmup + ramp > steps or width > resolution:
        raise ValueError('invalid warmup/ramp/block bounds')
    _nonnegative(lr, 'lr', positive=True)
    _nonnegative(extent, 'extent', positive=True)
    _nonnegative(gradient_ratio, 'gradient_ratio')
    cameras = tuple(cameras)
    truth = _truth(targets, cameras, torch.float32)
    hull = visual_hull(truth.numpy().astype(bool), cameras, resolution, extent=extent)
    root = choose_root(hull.astype(np.float32), width)
    logits = torch.full((resolution,) * 3, -4., dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=lr)
    initial_hash = hashlib.sha256(logits.detach().numpy().tobytes()).hexdigest()
    started = time.monotonic()

    def update(current, opt, step, coefficient, arm):
        opt.zero_grad(set_to_none=True)
        base_value = _base_backward(current, truth, cameras, step, steps, extent)
        factor = min(1., max(0., (step - warmup) / ramp)) if coefficient else 0.
        aux_value = None
        if factor:
            aux = _aux_loss(current.sigmoid(), truth, cameras, width, root, backend, extent)
            (coefficient * factor * aux).backward()
            aux_value = float(aux.detach())
        if current.grad is None or not torch.isfinite(current.grad).all():
            raise ValueError('nonfinite or missing training gradient')
        norm = float(current.grad.norm())
        opt.step()
        if not torch.isfinite(current).all():
            raise ValueError('nonfinite training logits')
        state = continuation_state(step, steps)
        row = {'step': step, 'base_loss': base_value, 'aux_loss': aux_value,
               'auxiliary_ramp': factor, 'weighted_aux_loss': coefficient * factor * (aux_value or 0.),
               'gradient_norm': norm, 'projection_phase': state['phase'],
               'projection_exponent': state['exponent'], 'loss_phase': 'pre_optimizer_update'}
        if callback is not None:
            callback(arm, {**row, 'elapsed_seconds': time.monotonic() - started})
        return row

    prefix = [update(logits, optimizer, step, 0., 'shared') for step in range(1, warmup + 1)]
    common_logits = logits.detach().clone()
    common_optimizer = copy.deepcopy(optimizer.state_dict())
    common_hash = _state_hash(logits, optimizer)
    calibration = {'coefficient': 0., 'gradient_ratio_requested': gradient_ratio,
                   'next_training_step': warmup + 1, 'state_sha256': common_hash,
                   'rule': 'post_warmup_next_step_base_gradient_ratio_once',
                   'phase': continuation_state(warmup + 1, steps),
                   'evaluated': bool(gradient_ratio)}
    if gradient_ratio:
        optimizer.zero_grad(set_to_none=True)
        base_value = _base_backward(logits, truth, cameras, warmup + 1, steps, extent)
        base_gradient = logits.grad.detach().clone()
        aux = _aux_loss(logits.sigmoid(), truth, cameras, width, root, backend, extent)
        auxiliary_gradient, = torch.autograd.grad(aux, logits)
        bn, an = float(base_gradient.norm()), float(auxiliary_gradient.norm())
        if not math.isfinite(bn + an) or min(bn, an) <= 0:
            raise ValueError('cannot calibrate zero or nonfinite gradients')
        coefficient = gradient_ratio * bn / an
        calibration.update(coefficient=coefficient, base_gradient_norm=bn, aux_gradient_norm=an,
                           weighted_gradient_ratio=coefficient * an / bn,
                           base_loss=base_value, aux_loss=float(aux.detach()),
                           gradient_cosine=float((base_gradient * auxiliary_gradient).sum()) / (bn * an))
    if _state_hash(logits, optimizer) != common_hash:
        raise RuntimeError('calibration changed the common optimizer state')
    common = {'initial_logits_sha256': initial_hash, 'initial_logit_value': -4.,
              'checkpoint_step': warmup, 'state_sha256': common_hash,
              'root_anchor_xyz': list(root), 'root_source': 'target_visual_hull_only_no_learned_field',
              'target_hull_sha256': mask_hash(hull), 'optimizer_state_preserved': True}
    if split_callback is not None:
        split_callback(common_logits.numpy().copy(), copy.deepcopy(common), copy.deepcopy(calibration))
    arms = {}
    for name, coefficient in (('baseline', 0.), ('rooted_core', calibration['coefficient'])):
        current = common_logits.clone().requires_grad_(True)
        opt = torch.optim.Adam([current], lr=lr)
        opt.load_state_dict(copy.deepcopy(common_optimizer))
        start_hash = _state_hash(current, opt)
        if start_hash != common_hash:
            raise RuntimeError('FAILED_SHARED_ADAM_STATE')
        history = copy.deepcopy(prefix)
        history.extend(update(current, opt, step, coefficient, name)
                       for step in range(warmup + 1, steps + 1))
        arms[name] = {'probability': current.detach().sigmoid().numpy().copy(),
                      'history': history, 'start_state_sha256': start_hash,
                      'initial_logits_sha256': initial_hash,
                      'optimizer_step': int(opt.state[current]['step'].item()),
                      'coefficient': coefficient, 'checkpoint_selection': 'final_step'}
        if arm_callback is not None:
            arm_callback(name, arms[name]['probability'].copy(),
                         copy.deepcopy({key: value for key, value in arms[name].items()
                                        if key != 'probability'}))
    return {'common': common, 'calibration': calibration, 'arms': arms}
