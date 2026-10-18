"""Isolated final-step soft voxel training; no file writes or external trainer."""
import hashlib
import math
import copy
from pathlib import Path
import platform

import numpy as np
import torch
import torch.nn.functional as F

from .losses import LossConfig, compute_loss
from .contracts import mask_hash


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def logits_to_density(logits, extent=1.7):
    """sigma=softplus(logits)/(extent/256); reference opacity=sigmoid(logits).

    The fixed physical reference length never changes with voxel resolution.
    Occupancy uses reference opacity >= .5, equivalently sigma >= log(2)/length.
    """
    extent = _positive("extent", extent)
    return F.softplus(logits) / (extent / 256.), torch.sigmoid(logits)


def _numpy(value, dtype):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=dtype, copy=True, order="C")


def render_training_projection(density, cameras, image_size, *, extent=1.7, pixel_supersampling=2):
    """Area-average physical alpha from full rays at the supersampled pixels."""
    if type(pixel_supersampling) is not int or pixel_supersampling <= 0:
        raise ValueError("pixel_supersampling must be a positive integer")
    from . import rendering
    high = rendering.render_optical(density, cameras, image_size * pixel_supersampling, extent=extent)
    return F.avg_pool2d(high[:, None], pixel_supersampling, pixel_supersampling)[:, 0]


def render_hard_training_projection(logits, cameras, image_size, *, extent=1.7):
    """Exact exported-cell forward, explicitly biased optical surrogate backward.

    This is not the derivative of binary geometry. Use the existing complete
    binary ray implementation, including half-open ownership, for forward;
    neither accumulated opacity nor an image threshold defines that forward.
    """
    from . import rendering
    cameras = tuple(cameras)
    density, probability = logits_to_density(logits, extent)
    surrogate = rendering.render_optical(density, cameras, image_size, extent=extent)
    occupied = _numpy(probability.detach() >= .5, bool)
    hard = torch.as_tensor(rendering.render_binary(occupied, cameras, image_size, extent=extent),
                           dtype=logits.dtype, device=logits.device)
    return hard + (surrogate - surrogate.detach())


def train_soft(targets, cameras, *, extent=1.7, resolution=256, image_size=256,
               steps=2000, lr=.007, seed=42, device="cpu", loss_config=LossConfig(),
               callback=None, view_chunk_size=1, pixel_supersampling=2,
               supersampled_targets=None, calibration_steps=(), projection_mode="area_soft",
               initialization_mode="visual_hull"):
    """Fit all supplied views jointly, with chunked backward and one Adam step.

    Geometry initialization defaults to the canonical visual hull (+2/-8
    logits) with seed-controlled Gaussian logit noise std .01. The controlled
    full_minus4 option uses exact -4 float32 logits without noise. A chunk never
    triggers its own optimizer update. Returned arrays are owned NumPy copies;
    intermediate checkpoints are never selected for export.
    """
    extent, lr = _positive("extent", extent), _positive("lr", lr)
    for name, value in (("resolution", resolution), ("image_size", image_size),
                        ("steps", steps), ("view_chunk_size", view_chunk_size),
                        ("pixel_supersampling", pixel_supersampling)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2**63)")
    if not isinstance(loss_config, LossConfig) or (callback is not None and not callable(callback)):
        raise ValueError("invalid loss configuration or callback")
    if projection_mode not in ("area_soft", "dense_soft", "dense_hard_st", "dense_entity_max"):
        raise ValueError("invalid projection_mode")
    if initialization_mode not in ("visual_hull", "full_minus4"):
        raise ValueError("invalid initialization_mode")
    if projection_mode != "area_soft" and supersampled_targets is None:
        raise ValueError("dense projection modes require source-derived targets")
    if projection_mode != "area_soft" and calibration_steps:
        raise ValueError("dense projection probe cannot use historical optical calibration")
    try:
        calibration_steps = tuple(calibration_steps)
    except TypeError as error:
        raise ValueError("calibration_steps must be an explicit sequence") from error
    if any(type(step) is not int or not 1 <= step <= steps for step in calibration_steps) or len(set(calibration_steps)) != len(calibration_steps):
        raise ValueError("calibration steps must be unique integers within the training budget")
    if calibration_steps and (loss_config.control or loss_config.auxiliary != "none" or loss_config.core == "none"):
        raise ValueError("weight calibration requires a pure nonempty core trajectory")
    cameras = tuple(cameras)
    target_array = _numpy(targets, np.float32)
    if not cameras or target_array.shape != (len(cameras), image_size, image_size):
        raise ValueError("targets must contain exactly one square image per camera")
    if not np.isfinite(target_array).all() or not np.isin(target_array, [0., 1.]).all():
        raise ValueError("canonical training targets must be finite binary masks")
    target_array = target_array.astype(bool)
    hull_image_size = image_size * pixel_supersampling
    if supersampled_targets is None:
        hull_targets = np.repeat(np.repeat(target_array, pixel_supersampling, axis=1), pixel_supersampling, axis=2)
        hull_source = "nearest_nominal_binary"
    else:
        hull_targets = _numpy(supersampled_targets, np.float32)
        if hull_targets.shape != (len(cameras), hull_image_size, hull_image_size) or not np.isin(hull_targets, [0., 1.]).all():
            raise ValueError("supersampled targets must be binary with the supersampled image shape")
        hull_targets = hull_targets.astype(bool)
        hull_source = "caller_source_derived_binary"
    try:
        torch_device = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError("invalid torch device") from error
    if torch_device.type not in {"cpu", "cuda"}:
        raise ValueError("supported training devices are CPU and CUDA")
    from . import rendering
    source_paths = {"training.py":Path(__file__), "losses.py":Path(__file__).with_name("losses.py"),
                    "rendering.py":Path(rendering.__file__)}
    if projection_mode == 'dense_entity_max':
        from . import entity_projection
        source_paths['entity_projection.py'] = Path(entity_projection.__file__)
    source_hashes = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in source_paths.items()}
    environment = {"torch_version":str(torch.__version__), "cuda_build_version":torch.version.cuda,
        "device_type":torch_device.type, "device":str(torch_device),
        "device_name":torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else platform.machine(),
        "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic":torch.backends.cudnn.deterministic,
        "cuda_matmul_fp32_precision":getattr(torch.backends.cuda.matmul,"fp32_precision","unavailable")}
    if torch_device.type == "cuda":
        properties = torch.cuda.get_device_properties(torch_device)
        environment.update(device_index=torch_device.index if torch_device.index is not None else torch.cuda.current_device(),
            device_total_memory=properties.total_memory,device_capability=[properties.major,properties.minor])

    hull = _numpy(rendering.visual_hull(hull_targets, cameras, resolution, extent=extent), bool)
    if hull.shape != (resolution,) * 3:
        raise ValueError("visual hull returned the wrong shape")
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(seed)
    hull_tensor = torch.as_tensor(hull, device=torch_device)
    if initialization_mode == "visual_hull":
        initial = torch.where(hull_tensor, torch.tensor(2.,device=torch_device,dtype=torch.float32),
                              torch.tensor(-8.,device=torch_device,dtype=torch.float32)) + .01 * torch.randn(
            hull_tensor.shape, generator=generator, device=torch_device, dtype=torch.float32)
    else:
        initial = torch.full(hull_tensor.shape, -4., device=torch_device, dtype=torch.float32)
    logits = initial.detach().requires_grad_(True)
    initial_state = np.ascontiguousarray(initial.detach().cpu().numpy())
    initial_hash = hashlib.sha256(str((initial_state.shape, str(initial_state.dtype))).encode()
                                  + initial_state.tobytes()).hexdigest()
    del initial_state
    training_targets = target_array if projection_mode == "area_soft" else hull_targets
    training_size = image_size if projection_mode == "area_soft" else hull_image_size
    truth = torch.tensor(training_targets, dtype=torch.float32, device=torch_device)
    optimizer = torch.optim.Adam([logits], lr=lr)
    history = []
    calibration_records = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        hard_weight = min(1., max(0., (step / steps - .25) / .25)) if projection_mode == "dense_hard_st" else 0.
        row = {"step": step, "hard_forward_weight": hard_weight}
        for start in range(0, len(cameras), view_chunk_size):
            stop = min(start + view_chunk_size, len(cameras))
            if projection_mode == 'dense_entity_max':
                projections = entity_projection.render_probability_max(
                    torch.sigmoid(logits), cameras[start:stop], training_size, extent=extent)
            elif hard_weight:
                hard = render_hard_training_projection(logits, cameras[start:stop], training_size, extent=extent)
                if hard_weight == 1.:
                    projections = hard
                else:
                    density, _ = logits_to_density(logits, extent)
                    optical = rendering.render_optical(density, cameras[start:stop], training_size, extent=extent)
                    projections = hard_weight * hard + (1. - hard_weight) * optical
            else:
                density, _ = logits_to_density(logits, extent)
                if projection_mode == "area_soft":
                    projections = render_training_projection(density, cameras[start:stop], image_size,
                        extent=extent, pixel_supersampling=pixel_supersampling)
                else:
                    projections = rendering.render_optical(density, cameras[start:stop], training_size, extent=extent)
            loss, diagnostics = compute_loss(projections, truth[start:stop], loss_config, step / steps)
            fraction = (stop - start) / len(cameras)
            (loss * fraction).backward()
            for name, value in diagnostics.items():
                if isinstance(value, str):
                    row[name] = value
                else:
                    row[name] = row.get(name, 0.) + fraction * value
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise ValueError(f"nonfinite or absent joint gradient at step {step}")
        row["gradient_norm"] = torch.linalg.vector_norm(logits.grad).item()
        if step in calibration_steps:
            # autograd.grad does not modify logits.grad; probes cause no Adam step.
            state = np.ascontiguousarray(logits.detach().cpu().numpy())
            state_hash = hashlib.sha256(str((state.shape,str(state.dtype))).encode()+state.tobytes()).hexdigest()
            step_records = []
            for auxiliary in ("lovasz","ray_bce"):
                auxiliary_gradient = torch.zeros_like(logits)
                probe_config = LossConfig(core="none",auxiliary=auxiliary,aux_weight=1.)
                for start in range(0,len(cameras),view_chunk_size):
                    stop = min(start+view_chunk_size,len(cameras))
                    probe_density,_ = logits_to_density(logits,extent)
                    prediction = render_training_projection(probe_density,cameras[start:stop],image_size,
                        extent=extent,pixel_supersampling=pixel_supersampling)
                    probe_loss,_ = compute_loss(prediction,truth[start:stop],probe_config,1.)
                    gradient, = torch.autograd.grad(probe_loss,logits)
                    auxiliary_gradient.add_(gradient,alpha=(stop-start)/len(cameras))
                norm = torch.linalg.vector_norm(auxiliary_gradient).item()
                if not math.isfinite(norm):
                    raise ValueError(f"nonfinite {auxiliary} calibration gradient at step {step}")
                step_records.append({"step":step,"auxiliary":auxiliary,"core_norm":row["gradient_norm"],
                    "aux_norm":norm,"logits_sha256":state_hash,"phase":"pre_optimizer_update",
                    "raw_auxiliary_weight":1.,"raw_auxiliary_ramp":1.})
            calibration_records.extend(step_records)
            row["calibration"] = step_records
        optimizer.step()
        if not torch.isfinite(logits).all():
            raise ValueError(f"nonfinite model state at step {step}")
        history.append(copy.deepcopy(row))
        if callback:
            callback(copy.deepcopy(row))
    with torch.no_grad():
        density, probabilities = logits_to_density(logits, extent)
        optical = []
        for start in range(0, len(cameras), view_chunk_size):
            optical.append(_numpy(render_training_projection(density, cameras[start:start + view_chunk_size], image_size,
                extent=extent, pixel_supersampling=pixel_supersampling), np.float32))
        density_array = _numpy(density, np.float32)
        probability_array = _numpy(probabilities, np.float32)
    occupancy = probability_array >= .5
    hard = _numpy(rendering.render_binary(occupancy, cameras, image_size, extent=extent), bool)
    hard_512 = _numpy(rendering.render_binary(occupancy, cameras, 512, extent=extent), bool)
    target_512 = F.interpolate(torch.tensor(target_array[:, None], dtype=torch.float32),
                               size=(512, 512), mode="nearest")[:, 0].numpy().astype(bool)
    intersection = (hard_512 & target_512).sum((1, 2))
    union = (hard_512 | target_512).sum((1, 2))
    ious_512 = np.divide(intersection, union, out=np.ones_like(union, dtype=float), where=union > 0)
    metadata = {
        "version": "shadowart_vnext_soft_v1", "seed": seed, "device": str(torch_device),
        "source_hashes":source_hashes,"environment":environment,
        "calibration_steps":list(calibration_steps),"calibration_records":calibration_records,
        "resolution": resolution, "image_size": image_size, "extent": extent,
        "steps": steps, "lr": lr, "optimizer": "Adam_default_betas_eps",
        "view_count": len(cameras), "view_chunk_size": view_chunk_size,
        "pixel_supersampling": pixel_supersampling,
        "initial_logits_sha256": initial_hash,
        "training_projection": {"mode": projection_mode, "image_size": training_size,
            "target_rule": "nominal_binary" if projection_mode == "area_soft" else "source_derived_binary",
            "target_hashes": [mask_hash(mask) for mask in training_targets],
            "hard_start_fraction": .25 if projection_mode == "dense_hard_st" else None,
            "hard_full_fraction": .5 if projection_mode == "dense_hard_st" else None,
            "backward_rule": ("ray_max_subgradient" if projection_mode == "dense_entity_max" else
                              "optical_straight_through_biased" if projection_mode == "dense_hard_st" else "optical_autograd"),
            "soft_projections_are": "area_optical_diagnostic_not_training_forward"},
        "hard_diagnostics_512": {"target_rule": "nearest_nominal_binary", "target_fidelity": "approximate_stage_diagnostic_not_final_QA", "per_view_iou": ious_512.tolist(),
                                 "mean_iou": float(ious_512.mean()), "worst_iou": float(ious_512.min())},
        "view_update_mode": "joint_view_equal", "export_step": steps,
        "checkpoint_selection": "final_step", "loss_config": loss_config.to_dict(),
        "reference_length": extent / 256., "density_units": "inverse_canonical_length",
        "density_rule": "softplus(logits)/(extent/256)",
        "occupancy_rule": "sigmoid(logits)>=0.5", "occupancy_reference_opacity": .5,
        "initialization": ({"method": "canonical_visual_hull", "inside_logit": 2., "outside_logit": -8., "noise_std": .01,
                            "hull_image_size": hull_image_size, "target_source": hull_source, "post_init_hull_clipping": False,
                            "target_hashes": [mask_hash(mask) for mask in hull_targets]}
                           if initialization_mode == "visual_hull" else
                           {"method": "full_minus4", "logit": -4., "noise_std": 0.,
                            "hull_image_size": hull_image_size, "target_source": hull_source, "post_init_hull_clipping": False,
                            "target_hashes": [mask_hash(mask) for mask in hull_targets]}),
        "target_hashes": [mask_hash(mask) for mask in target_array],
    }
    return {"occupancy": occupancy, "density": density_array,
            "probabilities": probability_array, "guidance": probability_array.copy(),
            "soft_projections": np.concatenate(optical), "hard_projections": hard,
            "hard_projections_512": hard_512,
            "metadata": metadata, "history": history}
