"""Unified model loader — supports PINO3d and UFNO variants from checkpoint."""
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _load_state_dict_compat(model: torch.nn.Module, state: dict, checkpoint_path: str):
    try:
        return model.load_state_dict(state)
    except RuntimeError as exc:
        if "scale_head" not in str(exc) and "scale_" not in str(exc):
            raise
        filtered_state = dict(state)
        model_state = model.state_dict()
        for key in list(filtered_state.keys()):
            if key.startswith('scale_') or key.startswith('scale_head.'):
                if key not in model_state or filtered_state[key].shape != model_state[key].shape:
                    filtered_state.pop(key, None)
        logger.warning(
            "Checkpoint %s predates the current scale_head layout; loading with "
            "strict=False and keeping a freshly initialized scale head.",
            checkpoint_path,
        )
        return model.load_state_dict(filtered_state, strict=False)


def load_surrogate(checkpoint_path: str, device: torch.device):
    """Load PINO3d or UFNO3d from checkpoint based on 'arch' field.

    Returns (model, ckpt_dict) with model in eval mode.
    """
    from structural.models.pino import PINO3d, UFNO3d, UFNODualBranch3d

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    arch = ckpt.get('arch', 'pino')

    in_ch = cfg.get('in_channels', 4)
    out_ch = cfg.get('out_channels', 9)
    modes = cfg.get('modes', 16)
    width = cfg.get('width', 64)
    n_layers = cfg.get('n_layers', 6)
    padding = cfg.get('padding', 8 if arch == 'pino' else 0)

    if arch == 'ufno':
        model = UFNO3d(
            in_channels=in_ch, out_channels=out_ch,
            modes=modes, width=width, n_layers=n_layers, padding=padding,
        ).to(device)
    elif arch == 'ufno_dual':
        model = UFNODualBranch3d(
            in_channels=in_ch, out_channels=out_ch,
            modes=modes, width=width, n_layers=n_layers, padding=padding,
        ).to(device)
    else:
        model = PINO3d(
            in_channels=in_ch, out_channels=out_ch,
            modes=modes, width=width, n_layers=n_layers, padding=padding,
        ).to(device)

    state = ckpt.get('model_state_dict') or ckpt.get('ema_state_dict') or ckpt
    incompatible = _load_state_dict_compat(model, state, str(checkpoint_path))
    model.eval()

    # Attach normalization metadata
    norm = ckpt.get('normalization', {})
    model._pino_norm = {
        'force_scale': norm.get('force_scale', 1.0),
        'disp_scale': norm.get('disp_scale', 1.0),
        'stress_scale': norm.get('stress_scale', 1.0),
    }
    # Per-sample normalized models output stress/vm_max (relative, not physical)
    model._per_sample_normalized = (norm.get('stress_mode') == 'log10'
                                     and ckpt.get('normalization_mode') == 'log_stress')
    model._is_v3 = (ckpt.get('normalization_mode') == 'log_stress'
                     or 'v3' in str(ckpt.get('version', ''))
                     or 'v1' in str(ckpt.get('version', ''))  # v16+ are all log-stress
                     )
    model._has_scale_head = bool(
        hasattr(model, 'forward_with_scale')
        and any(k.startswith('scale_head.') for k in state.keys())
    )
    model._missing_checkpoint_keys = list(getattr(incompatible, 'missing_keys', []))
    model._unexpected_checkpoint_keys = list(getattr(incompatible, 'unexpected_keys', []))
    model._arch = arch
    model._checkpoint_path = str(checkpoint_path)

    logger.info(f"Loaded {arch.upper()} from {checkpoint_path} "
                f"(epoch={ckpt.get('epoch', '?')}, version={ckpt.get('version', '?')}, "
                f"v3={model._is_v3})")
    return model, ckpt


def build_pino_input(pino_model, density_tensor, force_tensor):
    """Build PINO input with optional SDF channel for UFNO/per-sample-normalized models.

    Args:
        pino_model: loaded surrogate model
        density_tensor: [B, 1, N, N, N]
        force_tensor: [B, 3, N, N, N]
    Returns:
        [B, C, N, N, N] where C=4 (PINO) or C=5 (UFNO with SDF)
    """
    parts = [density_tensor, force_tensor]
    if getattr(pino_model, '_per_sample_normalized', False):
        from scipy.ndimage import distance_transform_edt
        d_np = density_tensor[0, 0].detach().cpu().numpy()
        solid = d_np > 0.5
        if solid.any() and (~solid).any():
            dist_in = distance_transform_edt(solid)
            dist_out = distance_transform_edt(~solid)
            sdf_np = (dist_in - dist_out).astype(np.float32) / (d_np.shape[0] / 2)
        else:
            sdf_np = np.zeros_like(d_np)
        sdf = torch.from_numpy(sdf_np).float().unsqueeze(0).unsqueeze(0).to(density_tensor.device)
        parts.append(sdf)
    return torch.cat(parts, dim=1)


def forward_surrogate(pino_model, density_tensor, force_tensor):
    """Forward surrogate with optional scale-head support.

    Returns:
        pino_input: [B, C, N, N, N]
        pino_out: [B, 9, N, N, N]
        log_vm_max_pred: [B, 1] or None
    """
    pino_input = build_pino_input(pino_model, density_tensor, force_tensor)
    if getattr(pino_model, '_has_scale_head', False) and hasattr(pino_model, 'forward_with_scale'):
        pino_out, log_vm_max_pred = pino_model.forward_with_scale(pino_input)
    else:
        pino_out = pino_model(pino_input)
        log_vm_max_pred = None
    pino_model._last_log_vm_max_pred = log_vm_max_pred
    return pino_input, pino_out, log_vm_max_pred


def forward_surrogate_physical(pino_model, density_tensor, force_tensor):
    """Forward surrogate and recover physical displacement/stress/von Mises."""
    from structural.analysis.stress_eval import (
        compute_von_mises_torch,
        get_yield_stress_for_model,
        pino_displacement_to_physical,
        pino_stress_to_physical,
    )

    _, pino_out, log_vm_max_pred = forward_surrogate(
        pino_model, density_tensor, force_tensor
    )
    is_v3 = getattr(pino_model, '_is_v3', False)
    stress_scale = getattr(pino_model, '_pino_norm', {}).get('stress_scale', 1.0)
    disp_scale = getattr(pino_model, '_pino_norm', {}).get('disp_scale', 1.0)
    displacement = pino_displacement_to_physical(
        pino_out[:, :3], is_v3, disp_scale=disp_scale
    )
    stress = pino_stress_to_physical(
        pino_out[:, 3:],
        is_v3,
        log_vm_max_pred=log_vm_max_pred,
        stress_scale=stress_scale,
    )
    von_mises = compute_von_mises_torch(stress)
    return {
        'output': pino_out,
        'log_vm_max_pred': log_vm_max_pred,
        'displacement': displacement,
        'stress': stress,
        'von_mises': von_mises,
        'effective_yield_stress': get_yield_stress_for_model(pino_model),
    }
