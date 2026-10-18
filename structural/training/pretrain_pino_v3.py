"""
PINO预训练脚本 v3 — Log-scale Stress + Asymmetric Safety Loss

相比v2的关键改进:
1. Log-scale stress 输出空间: 预测 log10(|σ|+1) 而非线性σ
   - 训练数据应力跨5个量级(1e4~1e9)，线性归一化失败
   - 对数空间让所有量级均等分布
2. 非对称安全损失: 低估应力(高估SF)惩罚2x，高估应力(低估SF)惩罚0.5x
   - 安全关键应用必须偏保守: SF_PINO ≤ SF_FEM
3. 混合数据集: 原训练数据(2000) + 薄壳数据(~300) 按1:1过采样
4. Per-sample adaptive normalization for force/displacement

用法:
    python pretrain_pino_v3.py \
        --data_dirs ./fem_dataset,./fem_dataset_v2 \
        --output_dir ./checkpoints/pino_v3 \
        --resume_from_v1 ./checkpoints/pino/best_model.pt \
        --epochs 500
"""
import os
import random
import sys
import argparse
import copy
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from structural.models.pino import PINO3d, UFNO3d, UFNODualBranch3d
from structural.models.official_operator_baselines import (
    OfficialGINO3d,
    OfficialTransolver3d,
)
from structural.configs import StructShadowConfig


def _load_state_dict_compat(model: torch.nn.Module, state: dict, checkpoint_path: str):
    # A strict run adds fixed_mask as a sixth physical input channel.  Preserve
    # a compatible five-channel checkpoint by zero-initialising only that new
    # channel, so the original mapping is unchanged at the first update.
    model_state = model.state_dict()
    migrated = dict(state)
    # Some serialized OrderedDict checkpoints expose module metadata as a
    # mapping entry. It is not a learnable tensor and must not reach strict
    # state-dict loading.
    migrated.pop('_metadata', None)
    for key in ('lifting.weight', 'scale_lifting.weight'):
        if key not in migrated or key not in model_state:
            continue
        old, new = migrated[key], model_state[key]
        if (
            old.ndim == 5 and new.ndim == 5
            and old.shape[0] == new.shape[0]
            and old.shape[1] + 1 == new.shape[1]
            and old.shape[2:] == new.shape[2:]
        ):
            expanded = torch.zeros_like(new)
            expanded[:, :old.shape[1]] = old
            migrated[key] = expanded
            print(f"Migrated {key}: added zero-initialized fixed-mask channel.")
        elif (
            old.ndim == 5 and new.ndim == 5
            and old.shape[0] == new.shape[0]
            and old.shape[1] > new.shape[1]
            and old.shape[2:] == new.shape[2:]
        ):
            # Structural-context ablations retain the leading rho+force channels
            # and deliberately remove trailing SDF/support channels.
            migrated[key] = old[:, :new.shape[1]].contiguous()
            print(f"Migrated {key}: retained the first {new.shape[1]} input channels.")
    state = migrated
    try:
        result = model.load_state_dict(state)
        model._loaded_missing_scale_head = False
        model._missing_checkpoint_keys = list(getattr(result, 'missing_keys', []))
        model._unexpected_checkpoint_keys = list(getattr(result, 'unexpected_keys', []))
        return result
    except RuntimeError as exc:
        if "scale_head" not in str(exc) and "scale_" not in str(exc):
            raise
        filtered_state = dict(state)
        dropped = []
        model_state = model.state_dict()
        for key in list(filtered_state.keys()):
            if key.startswith('scale_') or key.startswith('scale_head.'):
                if key not in model_state or filtered_state[key].shape != model_state[key].shape:
                    filtered_state.pop(key, None)
                    dropped.append(key)
        print(
            f"WARNING: {checkpoint_path} predates the current scale_head layout; "
            "loading backbone with strict=False and initializing scale head from scratch."
        )
        if dropped:
            print(f"  Dropped incompatible scale keys: {len(dropped)}")
        result = model.load_state_dict(filtered_state, strict=False)
        model._loaded_missing_scale_head = True
        model._missing_checkpoint_keys = list(getattr(result, 'missing_keys', []))
        model._unexpected_checkpoint_keys = list(getattr(result, 'unexpected_keys', []))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Log-scale utilities
# ─────────────────────────────────────────────────────────────────────────────

def stress_to_log(stress: torch.Tensor) -> torch.Tensor:
    """Convert stress to log-scale: sign(σ) * log10(|σ| + 1).

    Maps any stress magnitude to a bounded range:
      1e4 Pa  → ±4.0
      1e6 Pa  → ±6.0
      1e9 Pa  → ±9.0
    """
    return torch.sign(stress) * torch.log10(torch.abs(stress) + 1.0)


def log_to_stress(log_stress: torch.Tensor, max_abs_log: float = 8.0) -> torch.Tensor:
    """Convert log-scale back to physical stress: sign(x) * (10^|x| - 1).

    The PINO head is unconstrained; once predictions drift to very large log values,
    the inverse transform can explode and poison the VM loss with inf/NaN. We clamp
    the inverse transform to a conservative but finite stress range before exponentiation.
    """
    log_stress = torch.clamp(log_stress, min=-max_abs_log, max=max_abs_log)
    return torch.sign(log_stress) * (10.0 ** torch.abs(log_stress) - 1.0)


def compute_von_mises(stress: torch.Tensor) -> torch.Tensor:
    """Von Mises stress from 6-channel Voigt tensor.

    Args:
        stress: [B, 6, N, N, N] (physical units)
    Returns:
        [B, 1, N, N, N]
    """
    sxx, syy, szz = stress[:, 0:1], stress[:, 1:2], stress[:, 2:3]
    syz, szx, sxy = stress[:, 3:4], stress[:, 4:5], stress[:, 5:6]
    vm_sq = (0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
             + 3.0 * (sxy**2 + syz**2 + szx**2))
    return torch.sqrt(vm_sq.clamp(min=1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# Loss Function v3: Log-space + Asymmetric Safety
# ─────────────────────────────────────────────────────────────────────────────

class PINOLossV3(nn.Module):
    """PINO v3 loss: data supervision + PDE physics residual + asymmetric safety.

    This is a proper PINN loss combining:
      - Data-driven: displacement/stress/VM supervision from FEM
      - Physics-informed: PDE residual (equilibrium ∇·σ+f=0, constitutive σ=C:ε)
      - Safety-biased: asymmetric penalty for stress underestimation

    The model predicts:
      - Channels 0-2: displacement (linear, normalized by disp_scale)
      - Channels 3-8: stress in log-space: sign(σ) * log10(|σ| + 1)

    Loss components:
      1. Displacement MSE (linear space, masked)
      2. Stress log-space L1 (smooth L1, all magnitudes equal)
      3. Von Mises asymmetric loss (penalize underestimation 2x)
      4. Von Mises peak loss (per-sample max stress accuracy)
      5. Dirichlet BC loss
      6. Equilibrium PDE residual: ∇·σ + f = 0
      7. Constitutive PDE residual: σ = C:ε(u)
    """

    def __init__(
        self,
        lambda_disp: float = 1.0,
        lambda_stress_log: float = 3.0,
        lambda_vm_asym: float = 2.0,
        lambda_vm_peak: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_pde: float = 0.1,
        lambda_constitutive: float = 0.1,
        lambda_scale: float = 5.0,
        lambda_decision: float = 0.0,
        lambda_decision_margin: float = 0.0,
        decision_temperature: float = 0.12,
        decision_margin_log: float = 0.06,
        safety_threshold: float = 1.3,
        underestimate_weight: float = 2.0,
        overestimate_weight: float = 0.5,
        disp_scale: float = 1.0,
        max_log_stress: float = 8.0,
        resolution: int = 64,
        youngs_modulus: float = 3.5e9,
        poisson_ratio: float = 0.36,
        simp_penalty: float = 3.0,
    ):
        super().__init__()
        self.lambda_disp = lambda_disp
        self.lambda_stress_log = lambda_stress_log
        self.lambda_vm_asym = lambda_vm_asym
        self.lambda_vm_peak = lambda_vm_peak
        self.lambda_bc = lambda_bc
        self.lambda_pde = lambda_pde
        self.lambda_constitutive = lambda_constitutive
        self.lambda_scale = lambda_scale
        self.lambda_decision = lambda_decision
        self.lambda_decision_margin = lambda_decision_margin
        self.decision_temperature = decision_temperature
        self.decision_margin_log = decision_margin_log
        self.safety_threshold = safety_threshold
        self.log_safety_threshold = math.log10(safety_threshold)
        self.log_yield_stress = math.log10(50.0e6)
        self.w_under = underestimate_weight
        self.w_over = overestimate_weight
        self.disp_scale = disp_scale
        self.max_log_stress = max_log_stress
        self.h = 1.0 / resolution
        self.simp_penalty = simp_penalty

        # Precompute Hooke stiffness matrix C (6×6 Voigt notation)
        E, nu = youngs_modulus, poisson_ratio
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        C = torch.zeros(6, 6)
        C[0, 0] = C[1, 1] = C[2, 2] = lam + 2 * mu
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = lam
        C[3, 3] = C[4, 4] = C[5, 5] = mu
        self.register_buffer('C_matrix', C)

    # ── PDE residual methods (PINN physics constraints) ──

    def _grad_3d(self, f: torch.Tensor, dim: int) -> torch.Tensor:
        """Non-periodic central difference with one-sided boundary correction."""
        spatial_dim = dim + 2
        N = f.shape[spatial_dim]
        h = self.h
        f_plus = torch.roll(f, -1, spatial_dim)
        f_minus = torch.roll(f, 1, spatial_dim)
        grad = (f_plus - f_minus) / (2 * h)
        # Boundary correction
        idx_f = [slice(None)] * 5
        idx_f[spatial_dim] = 0
        idx_f1 = [slice(None)] * 5
        idx_f1[spatial_dim] = 1
        grad[tuple(idx_f)] = (f[tuple(idx_f1)] - f[tuple(idx_f)]) / h
        idx_b = [slice(None)] * 5
        idx_b[spatial_dim] = N - 1
        idx_b1 = [slice(None)] * 5
        idx_b1[spatial_dim] = N - 2
        grad[tuple(idx_b)] = (f[tuple(idx_b)] - f[tuple(idx_b1)]) / h
        return grad

    def equilibrium_residual(self, stress, force, density):
        """Normalized ∇·σ + f = 0 residual.

        Normalization: divide by characteristic scale (force_rms²) to prevent
        overflow when stress is in physical units (~1e6-1e8 Pa).
        """
        sxx, syy, szz = stress[:, 0:1], stress[:, 1:2], stress[:, 2:3]
        syz, szx, sxy = stress[:, 3:4], stress[:, 4:5], stress[:, 5:6]
        res_x = self._grad_3d(sxx, 0) + self._grad_3d(sxy, 1) + self._grad_3d(szx, 2) + force[:, 0:1]
        res_y = self._grad_3d(sxy, 0) + self._grad_3d(syy, 1) + self._grad_3d(syz, 2) + force[:, 1:2]
        res_z = self._grad_3d(szx, 0) + self._grad_3d(syz, 1) + self._grad_3d(szz, 2) + force[:, 2:3]
        residual = torch.cat([res_x, res_y, res_z], dim=1)
        mask = (density > 0.1).float()
        # Normalize by characteristic scale to keep loss ~O(1)
        char_scale = (force ** 2).mean().clamp(min=1e-8)
        return ((residual * mask) ** 2).mean() / char_scale

    def constitutive_residual(self, stress, displacement, density):
        """Normalized σ = C:ε(u) residual."""
        ux, uy, uz = displacement[:, 0:1], displacement[:, 1:2], displacement[:, 2:3]
        exx = self._grad_3d(ux, 0)
        eyy = self._grad_3d(uy, 1)
        ezz = self._grad_3d(uz, 2)
        eyz = self._grad_3d(uy, 2) + self._grad_3d(uz, 1)
        ezx = self._grad_3d(uz, 0) + self._grad_3d(ux, 2)
        exy = self._grad_3d(ux, 1) + self._grad_3d(uy, 0)
        strain = torch.cat([exx, eyy, ezz, eyz, ezx, exy], dim=1)
        B, _, Nx, Ny, Nz = stress.shape
        strain_flat = strain.reshape(B, 6, -1)
        sigma_from_u = torch.einsum('ij,bjm->bim', self.C_matrix, strain_flat)
        sigma_from_u = sigma_from_u.reshape(B, 6, Nx, Ny, Nz)
        sigma_from_u = sigma_from_u * (density ** self.simp_penalty)
        residual = stress - sigma_from_u
        mask = (density > 0.1).float()
        # Normalize by characteristic stress scale
        char_scale = (stress ** 2).mean().clamp(min=1e-8)
        return ((residual * mask) ** 2).mean() / char_scale

    def forward(
        self,
        pred: torch.Tensor,
        target_disp_norm: torch.Tensor,
        target_stress_log: torch.Tensor,
        target_vm: torch.Tensor,
        density: torch.Tensor,
        fixed_mask: torch.Tensor,
        force: torch.Tensor | None = None,
        epoch: int = 0,
        pde_warmup_epochs: int = 20,
        has_disp: torch.Tensor | None = None,
        disp_scale: torch.Tensor | None = None,
        log_vm_max_pred: torch.Tensor | None = None,
        log_vm_max_target: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            pred:               [B, 9, N, N, N] model output
                                  pred[:,:3] = displacement (normalized)
                                  pred[:,3:] = log-stress
            target_disp_norm:   [B, 3, N, N, N] displacement / per-sample scale
            target_stress_log:  [B, 6, N, N, N] sign(σ)*log10(|σ|+1)
            target_vm:          [B, 1, N, N, N] Von Mises in physical units
            density:            [B, 1, N, N, N]
            fixed_mask:         [B, 1, N, N, N]
            force:              [B, 3, N, N, N] physical force (for PDE residual)
            has_disp:           [B] 1.0 if sample has displacement data, 0.0 otherwise
            disp_scale:         [B] per-sample displacement scale for de-normalization
            log_vm_max_pred:    [B, 1] predicted log10(vm_max) from scale head
            log_vm_max_target:  [B] ground-truth log10(vm_max) for scale loss
        """
        mask = (density > 0.1).float()
        B = pred.shape[0]

        pred_disp = pred[:, :3]
        pred_stress_log = pred[:, 3:]

        # ── 1. Displacement loss (per-sample normalized, skip zero-disp samples) ──
        if has_disp is not None:
            # Weight each sample's disp loss by has_disp flag
            disp_weight = has_disp.view(B, 1, 1, 1, 1)  # [B,1,1,1,1]
            l_disp = ((TF.smooth_l1_loss(pred_disp * mask, target_disp_norm * mask,
                                          reduction='none') * disp_weight).sum()
                      / mask.sum().clamp(min=1) / max(has_disp.sum().item(), 1.0) * B)
        else:
            l_disp = (TF.smooth_l1_loss(pred_disp * mask, target_disp_norm * mask,
                                          reduction='sum')
                      / mask.sum().clamp(min=1))

        # ── 2. Stress loss (asymmetric: penalize magnitude underestimation heavier) ──
        # In per-sample normalized space, pred_stress_log and target_stress_log
        # are raw stress / vm_max. Symmetric smooth_L1 pulls peaks toward mean,
        # causing VM underestimation. Apply asymmetric weighting based on whether
        # prediction reduces magnitude relative to target.
        mask6 = mask.expand_as(pred_stress_log)
        stress_err = pred_stress_log - target_stress_log
        # Magnitude underestimation: |pred| < |target| (dangerous, unconservative)
        mag_under = (pred_stress_log.abs() < target_stress_log.abs()).float()
        stress_weight = mag_under * self.w_under + (1.0 - mag_under) * self.w_over
        l_stress_log = ((stress_weight * stress_err**2 * mask6).sum()
                        / mask6.sum().clamp(min=1))

        # ── 3. Von Mises asymmetric loss (normalized space, no log transform) ──
        # After per-sample normalization, both pred_stress and target are ~O(1).
        # VM is computed directly, target_vm_max = 1.0 for all samples.
        pred_vm = compute_von_mises(pred_stress_log)  # pred_stress_log is normalized stress

        # Asymmetric: heavier penalty when pred_vm < target_vm (underestimation)
        vm_error = pred_vm - target_vm  # negative = underestimate (dangerous)
        asym_weight = torch.where(vm_error < 0,
                                   self.w_under * torch.ones_like(vm_error),
                                   self.w_over * torch.ones_like(vm_error))
        l_vm_asym = ((asym_weight * vm_error**2 * mask).sum()
                     / mask.sum().clamp(min=1))

        # ── 4. Von Mises peak loss (per-sample max; target max = 1.0 always) ──
        l_vm_peak = torch.tensor(0.0, device=pred.device)
        for i in range(B):
            m = mask[i, 0] > 0.5
            if m.any():
                pred_max = pred_vm[i, 0][m].max()
                target_max = target_vm[i, 0][m].max()
                error = pred_max - target_max
                if error < 0:  # underestimate peak → overestimate SF (dangerous)
                    l_vm_peak = l_vm_peak + self.w_under * error**2
                else:
                    l_vm_peak = l_vm_peak + self.w_over * error**2
        l_vm_peak = l_vm_peak / max(B, 1)

        # ── 5. BC loss + void zero loss ──
        l_bc = ((pred_disp * fixed_mask)**2).mean()
        # Push all outputs to zero in void region (density < 0.1)
        void_mask = (density < 0.1).float()
        l_void = (pred[:, :3] ** 2 * void_mask).mean() + (pred[:, 3:] ** 2 * void_mask).mean()
        l_bc = l_bc + l_void

        # ── 6 & 7. PDE residual (PINN physics constraints) ──
        # Delayed start: only enable after pde_warmup_epochs to let data loss stabilize first
        l_equilibrium = torch.tensor(0.0, device=pred.device)
        l_constitutive = torch.tensor(0.0, device=pred.device)
        pde_active = force is not None and self.lambda_pde > 0 and epoch >= pde_warmup_epochs
        if pde_active:
            # The six field channels are supervised as stress / vm_max, not as
            # log-physical stress. Recover physical stress with the predicted
            # sample scale. Treating these normalized channels as log stress
            # exponentiates O(1) values and destabilizes the PDE residual.
            if log_vm_max_pred is None:
                pde_stress = pred_stress_log
            else:
                pde_log_scale = torch.clamp(log_vm_max_pred, min=4.0, max=9.0)
                pde_scale = torch.pow(10.0, pde_log_scale).view(B, 1, 1, 1, 1)
                pde_stress = pred_stress_log * pde_scale
            # De-normalize displacement from log-scale to physical units
            # log_disp = sign(u)*log10(|u|*1e6+1) → u = sign(x)*(10^|x|-1)/1e6
            pred_disp_clamped = torch.clamp(pred_disp, -8.0, 8.0)
            pred_disp_phys = (torch.sign(pred_disp_clamped)
                              * (10.0 ** torch.abs(pred_disp_clamped) - 1.0) / 1e6)
            l_equilibrium = self.equilibrium_residual(pde_stress, force, density)
            l_constitutive = self.constitutive_residual(
                pde_stress, pred_disp_phys, density)

        l_pde = l_equilibrium + self.lambda_constitutive * l_constitutive

        # ── 8. Scale head loss: predict log10(vm_max) ──
        # Recovers absolute stress magnitude lost by per-sample normalization.
        # Asymmetric: underestimating vm_max → overestimating SF (dangerous).
        l_scale = torch.tensor(0.0, device=pred.device)
        if log_vm_max_pred is not None and log_vm_max_target is not None:
            target = log_vm_max_target.view(-1, 1).to(pred.device)
            scale_err = log_vm_max_pred - target  # positive = overestimate vm_max (conservative)
            scale_weight = torch.where(scale_err < 0,
                                       self.w_under * torch.ones_like(scale_err),
                                       self.w_over * torch.ones_like(scale_err))
            l_scale = (scale_weight * scale_err ** 2).mean()

        # ── 9. Decision-aware auxiliary loss on the absolute scale branch ──
        # The field/PDE losses retain physical fidelity.  This term only shapes
        # the scalar safety-factor ordering around the deployment threshold so
        # ACS does not need to absorb all boundary uncertainty as deferral.
        l_decision = torch.tensor(0.0, device=pred.device)
        l_decision_margin = torch.tensor(0.0, device=pred.device)
        if (
            log_vm_max_pred is not None
            and log_vm_max_target is not None
            and epoch >= pde_warmup_epochs
            and (self.lambda_decision > 0.0 or self.lambda_decision_margin > 0.0)
        ):
            target_log_sf = self.log_yield_stress - log_vm_max_target.view(-1, 1)
            pred_log_sf = self.log_yield_stress - log_vm_max_pred
            true_safe = (target_log_sf >= self.log_safety_threshold).float()
            # Balance the two classes per batch.  This avoids letting the large
            # high-SF majority turn the decision term into an always-safe bias.
            n_safe = true_safe.sum().clamp(min=1.0)
            n_unsafe = (1.0 - true_safe).sum().clamp(min=1.0)
            weights = torch.where(true_safe > 0.5,
                                  torch.full_like(true_safe, 0.5 * B) / n_safe,
                                  torch.full_like(true_safe, 0.5 * B) / n_unsafe)
            logits = (pred_log_sf - self.log_safety_threshold) / self.decision_temperature
            l_decision = (TF.binary_cross_entropy_with_logits(logits, true_safe, reduction='none') * weights).mean()
            safe_gap = TF.relu(self.log_safety_threshold + self.decision_margin_log - pred_log_sf)
            unsafe_gap = TF.relu(pred_log_sf - (self.log_safety_threshold - self.decision_margin_log))
            l_decision_margin = ((true_safe * safe_gap.square() + (1.0 - true_safe) * unsafe_gap.square())
                                 * weights).mean()

        total = (self.lambda_disp * l_disp
                 + self.lambda_stress_log * l_stress_log
                 + self.lambda_vm_asym * l_vm_asym
                 + self.lambda_vm_peak * l_vm_peak
                 + self.lambda_bc * l_bc
                 + self.lambda_pde * l_pde
                 + self.lambda_scale * l_scale
                 + self.lambda_decision * l_decision
                 + self.lambda_decision_margin * l_decision_margin)

        return {
            'total': total,
            'disp': l_disp,
            'stress_log': l_stress_log,
            'vm_asym': l_vm_asym,
            'vm_peak': l_vm_peak,
            'bc': l_bc,
            'pde': l_pde,
            'equilibrium': l_equilibrium,
            'constitutive': l_constitutive,
            'scale': l_scale,
            'decision': l_decision,
            'decision_margin': l_decision_margin,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset v3: Log-scale stress + multi-directory support
# ─────────────────────────────────────────────────────────────────────────────

class FEMDatasetV3(Dataset):
    """FEM dataset with log-scale stress targets.

    Target format:
      - displacement: normalized by per-dataset scale
      - stress: log-space: sign(σ) * log10(|σ| + 1)
      - von_mises: physical units (for loss computation)
    """

    # Global scales (from v1/v2 training data)
    FORCE_SCALE = 12164.43
    DISP_SCALE = 0.00997422
    SOLID_THRESHOLD = 0.3

    def __init__(self, data_dirs=None, split='train', train_ratio=0.85,
                 augment=False, seed=42, index_file=None, target_res=None,
                 include_fixed_mask=False, include_sdf=True):
        rng = np.random.RandomState(seed)
        self.target_res = target_res
        self.include_fixed_mask = include_fixed_mask
        self.include_sdf = include_sdf
        self.files = []
        dir_stats = []

        if index_file is not None:
            import json
            with open(index_file) as f:
                idx = json.load(f)
            root = Path(index_file).resolve().parent
            # Strict task-grouped format: rows are already split by the full
            # silhouette task.  calibration is validation; sealed_test is
            # deliberately never visible to this trainer.
            if "rows" in idx:
                source_split = "train" if split == "train" else "calibration"
                recs = [rec for rec in idx["rows"] if rec.get("split") == source_split]
                selected = [root / rec["path"] for rec in recs]
                self.files.extend(selected)
                by_dir = {}
                for fp in selected:
                    by_dir[fp.parent.name] = by_dir.get(fp.parent.name, 0) + 1
                dir_stats = [f"{k}={v}" for k, v in by_dir.items()]
            # New format: top-level "train" and "val" lists
            elif "train" in idx and "val" in idx:
                key = "train" if split == "train" else "val"
                recs = idx[key]
                selected = [root / rec["path"] for rec in recs]
                self.files.extend(selected)
                # Group by parent dir for stats
                by_dir = {}
                for fp in selected:
                    by_dir[fp.parent.name] = by_dir.get(fp.parent.name, 0) + 1
                dir_stats = [f"{k}={v}" for k, v in by_dir.items()]
            else:
                # Legacy format: single "samples" list, split randomly per-dir
                all_files = [root / rec["path"] for rec in idx["samples"]]
                by_dir = {}
                for fp in all_files:
                    by_dir.setdefault(fp.parent, []).append(fp)
                for d, files in by_dir.items():
                    files = sorted(files)
                    indices = rng.permutation(len(files))
                    n_train = int(len(files) * train_ratio)
                    selected = [files[i] for i in (indices[:n_train] if split == 'train' else indices[n_train:])]
                    self.files.extend(selected)
                    dir_stats.append(f"{d.name}={len(selected)}")
        else:
            if isinstance(data_dirs, str):
                data_dirs = [data_dirs]
            for d in data_dirs:
                files = sorted(Path(d).glob("sample_*.npz"))
                if not files:
                    continue
                indices = rng.permutation(len(files))
                n_train = int(len(files) * train_ratio)
                if split == 'train':
                    selected = [files[i] for i in indices[:n_train]]
                else:
                    selected = [files[i] for i in indices[n_train:]]
                self.files.extend(selected)
                dir_stats.append(f"{Path(d).name}={len(selected)}")

        self.augment = augment and (split == 'train')

        # Compute per-file solid fractions for weighted sampling
        self.solid_fractions = []
        for f in self.files:
            self.solid_fractions.append(None)  # computed on first access

        n_dirs = len(dir_stats)
        print(f"FEMDatasetV3 [{split}]: {len(self.files)} samples from "
              f"{n_dirs} dirs ({', '.join(dir_stats)}), augment={self.augment}")

    def __len__(self):
        return len(self.files)

    def _apply_flip(self, density, force, displacement, stress, von_mises,
                     fixed_mask, axis):
        """Physics-consistent flip (same as v2)."""
        spatial_dim = axis + 1
        density = density.flip(spatial_dim)
        fixed_mask = fixed_mask.flip(spatial_dim)
        von_mises = von_mises.flip(spatial_dim)
        force = force.flip(spatial_dim)
        displacement = displacement.flip(spatial_dim)
        stress = stress.flip(spatial_dim)

        force[axis] = -force[axis]
        displacement[axis] = -displacement[axis]

        # Voigt: [σxx(0), σyy(1), σzz(2), σyz(3), σzx(4), σxy(5)]
        if axis == 0:
            stress[4] = -stress[4]
            stress[5] = -stress[5]
        elif axis == 1:
            stress[3] = -stress[3]
            stress[5] = -stress[5]
        elif axis == 2:
            stress[3] = -stress[3]
            stress[4] = -stress[4]

        return density, force, displacement, stress, von_mises, fixed_mask

    def _apply_rot90(self, density, force, displacement, stress, von_mises,
                      fixed_mask, plane_axes, k=1):
        """Physics-consistent 90° rotation in a plane.

        Args:
            plane_axes: tuple (a, b) — spatial axes to rotate in (0=X, 1=Y, 2=Z)
            k: number of 90° rotations (1, 2, or 3)
        """
        a, b = plane_axes
        dim_a, dim_b = a + 1, b + 1  # +1 for channel dim

        density = torch.rot90(density, k, [dim_a, dim_b])
        fixed_mask = torch.rot90(fixed_mask, k, [dim_a, dim_b])
        von_mises = torch.rot90(von_mises, k, [dim_a, dim_b])
        force = torch.rot90(force, k, [dim_a, dim_b])
        displacement = torch.rot90(displacement, k, [dim_a, dim_b])
        stress = torch.rot90(stress, k, [dim_a, dim_b])

        # Rotate vector components (force, displacement)
        cos_vals = [1, 0, -1, 0]  # cos(k*90°) for k=0,1,2,3
        sin_vals = [0, 1, 0, -1]
        c, s = cos_vals[k % 4], sin_vals[k % 4]

        f_a, f_b = force[a].clone(), force[b].clone()
        force[a] = c * f_a - s * f_b
        force[b] = s * f_a + c * f_b

        d_a, d_b = displacement[a].clone(), displacement[b].clone()
        displacement[a] = c * d_a - s * d_b
        displacement[b] = s * d_a + c * d_b

        # Rotate stress tensor (Voigt notation)
        # σ = [σxx(0), σyy(1), σzz(2), σyz(3), σzx(4), σxy(5)]
        stress = self._rotate_stress_voigt(stress, a, b, c, s)

        return density, force, displacement, stress, von_mises, fixed_mask

    def _rotate_stress_voigt(self, stress, a, b, c, s):
        """Rotate 2D stress tensor in Voigt notation for rotation in plane (a, b)."""
        # Map axis pairs to Voigt shear indices
        # Normal: 0→xx, 1→yy, 2→zz; Shear: (1,2)→yz(3), (0,2)→zx(4), (0,1)→xy(5)
        shear_map = {(0, 1): 5, (0, 2): 4, (1, 2): 3}
        key = (min(a, b), max(a, b))
        shear_idx = shear_map[key]
        # Third axis (not in rotation plane)
        third = 3 - a - b

        sa, sb = stress[a].clone(), stress[b].clone()
        s_ab = stress[shear_idx].clone()

        # Normal stress rotation: σ_a' = c²σ_a + s²σ_b + 2cs·τ_ab
        stress[a] = c*c*sa + s*s*sb + 2*c*s*s_ab
        stress[b] = s*s*sa + c*c*sb - 2*c*s*s_ab
        # Shear in rotation plane: τ_ab' = -cs·σ_a + cs·σ_b + (c²-s²)·τ_ab
        stress[shear_idx] = -c*s*sa + c*s*sb + (c*c - s*s)*s_ab

        # Shear involving third axis also rotates
        shear_a_third = shear_map[(min(a, third), max(a, third))]
        shear_b_third = shear_map[(min(b, third), max(b, third))]
        s_at = stress[shear_a_third].clone()
        s_bt = stress[shear_b_third].clone()
        stress[shear_a_third] = c * s_at + s * s_bt if a < third else c * s_at - s * s_bt
        stress[shear_b_third] = -s * s_at + c * s_bt if b < third else s * s_at + c * s_bt

        return stress

    def _apply_force_scale(self, force, displacement, stress, von_mises, scale):
        """Scale force magnitude — displacement and stress scale linearly (linear elasticity)."""
        force = force * scale
        displacement = displacement * scale
        stress = stress * scale
        von_mises = von_mises * scale
        return force, displacement, stress, von_mises

    def _apply_density_noise(self, density):
        """Add small noise to density boundary voxels to improve generalization."""
        # Only perturb voxels near the surface (not deep interior or exterior)
        solid = (density > self.SOLID_THRESHOLD).float()
        # Detect surface: dilate - erode
        dilated = torch.nn.functional.max_pool3d(solid.unsqueeze(0), 3, 1, 1)[0]
        eroded = -torch.nn.functional.max_pool3d(-solid.unsqueeze(0), 3, 1, 1)[0]
        surface = (dilated - eroded).clamp(0, 1)
        noise = torch.randn_like(density) * 0.05  # small perturbation
        density = density + noise * surface
        return density.clamp(0, 1)

    def _apply_density_blur(self, density):
        """Gaussian blur on density to simulate continuous density from sigmoid annealing.

        Teaches the model that blurred boundaries should produce the same
        structural response as the original binary field.
        """
        sigma = torch.empty(1).uniform_(0.5, 2.0).item()
        ks = int(sigma * 4) | 1  # odd kernel size
        ks = max(ks, 3)
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        kernel_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        # Separable 3D Gaussian via three 1D convolutions
        d = density.unsqueeze(0)  # [1, 1, N, N, N]
        pad = ks // 2
        for dim in [2, 3, 4]:
            shape = [1, 1, 1, 1, 1]
            shape[dim] = ks
            w = kernel_1d.view(shape).to(d.device)
            d = torch.nn.functional.conv3d(d, w, padding=[pad if i == dim - 2 else 0 for i in range(3)])
        return d[0].clamp(0, 1)

    def _apply_density_simp(self, density):
        """SIMP-style penalization: ρ^p with random exponent.

        Pushes intermediate densities toward 0 or 1 (large p) or spreads them
        out (small p), mimicking the density distributions seen during topology
        optimization.
        """
        p = torch.empty(1).uniform_(0.5, 3.0).item()
        return density.clamp(0, 1).pow(p)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        density = torch.from_numpy(data['density']).float()
        if density.ndim == 3:
            density = density.unsqueeze(0)
        force = torch.from_numpy(data['force']).float()
        fixed_mask = torch.from_numpy(data['fixed_mask']).float()
        if fixed_mask.ndim == 3:
            fixed_mask = fixed_mask.unsqueeze(0)
        displacement = torch.from_numpy(data['displacement']).float()
        stress = torch.from_numpy(data['stress']).float()  # physical units

        if 'von_mises' in data:
            von_mises = torch.from_numpy(data['von_mises']).float()
            if von_mises.ndim == 3:
                von_mises = von_mises.unsqueeze(0)
        else:
            # Compute from stress
            s = stress.unsqueeze(0)
            von_mises = compute_von_mises(s)[0]

        # Unify mixed-source resolutions (dense=48, thin/paired=64) so batches stack.
        # FNO is resolution-agnostic; trilinear resize of dense replay data is acceptable.
        if self.target_res is not None and density.shape[-1] != self.target_res:
            r = self.target_res
            density = TF.interpolate(density.unsqueeze(0), size=(r, r, r), mode="trilinear", align_corners=False)[0]
            force = TF.interpolate(force.unsqueeze(0), size=(r, r, r), mode="trilinear", align_corners=False)[0]
            displacement = TF.interpolate(displacement.unsqueeze(0), size=(r, r, r), mode="trilinear", align_corners=False)[0]
            stress = TF.interpolate(stress.unsqueeze(0), size=(r, r, r), mode="trilinear", align_corners=False)[0]
            von_mises = TF.interpolate(von_mises.unsqueeze(0), size=(r, r, r), mode="trilinear", align_corners=False)[0]
            fixed_mask = (TF.interpolate(fixed_mask.unsqueeze(0), size=(r, r, r), mode="nearest")[0] > 0.5).float()

        # Data augmentation
        if self.augment:
            # 1. Random axis flips (p=0.5 each)
            for axis in range(3):
                if torch.rand(1).item() > 0.5:
                    density, force, displacement, stress, von_mises, fixed_mask = \
                        self._apply_flip(density, force, displacement, stress,
                                         von_mises, fixed_mask, axis)

            # 2. Random 90° rotations (p=0.3 per plane)
            for plane in [(0, 1), (0, 2), (1, 2)]:
                if torch.rand(1).item() < 0.3:
                    k = torch.randint(1, 4, (1,)).item()  # 90°, 180°, or 270°
                    density, force, displacement, stress, von_mises, fixed_mask = \
                        self._apply_rot90(density, force, displacement, stress,
                                          von_mises, fixed_mask, plane, k)

            # 3. Random force scaling (linear elasticity: u,σ scale linearly)
            if torch.rand(1).item() < 0.4:
                scale = torch.exp(torch.randn(1) * 0.3).item()  # ~0.5x to 2x
                scale = max(0.3, min(3.0, scale))
                force, displacement, stress, von_mises = \
                    self._apply_force_scale(force, displacement, stress, von_mises, scale)

            # 4. Density augmentation — teach model to handle continuous density
            r = torch.rand(1).item()
            if r < 0.25:
                # Gaussian blur: simulates sigmoid temperature annealing in Stage C
                density = self._apply_density_blur(density)
            elif r < 0.45:
                # SIMP penalization: simulates topology optimization density distributions
                density = self._apply_density_simp(density)
            elif r < 0.60:
                # Boundary noise: small perturbation on surface voxels
                density = self._apply_density_noise(density)

        # Normalize
        force_norm = force / self.FORCE_SCALE

        # Log-scale displacement: sign(u) * log10(|u| * 1e6 + 1)
        # Physical disp ranges from ~1e-5 to ~1e0 m.
        # Multiplying by 1e6 maps to ~10 to ~1e6, then log10 gives ~1 to ~6.
        # This preserves the input→output causal link (bigger force → bigger log-disp)
        # unlike per-sample normalization which destroys scale information.
        disp_log = torch.sign(displacement) * torch.log10(torch.abs(displacement) * 1e6 + 1.0)
        disp_abs_max = displacement.abs().max().clamp(min=1e-10)  # kept for PDE de-norm
        has_disp = float(displacement.abs().max() > 1e-12)

        # Per-sample stress normalization by von Mises max in solid region.
        # This makes target stress ~O(1) and target vm_max = 1.0 across all samples,
        # eliminating cross-dataset magnitude disparity that caused Rel L2 = 1.0.
        solid_3d = (density[0] > self.SOLID_THRESHOLD)
        if solid_3d.any():
            vm_max = float(von_mises[0][solid_3d].max().clamp(min=1e-6).item())
        else:
            vm_max = 1.0
        stress_log = stress / vm_max           # normalized stress [~ -5, 5]
        von_mises = von_mises / vm_max         # normalized vm, target max = 1.0

        # SDF (Signed Distance Field): geometry-aware input channel
        # Positive inside solid, negative outside. Normalized by resolution.
        density_np = density[0].numpy()
        solid_mask = density_np > self.SOLID_THRESHOLD
        if solid_mask.any() and (~solid_mask).any():
            dist_in = distance_transform_edt(solid_mask)
            dist_out = distance_transform_edt(~solid_mask)
            sdf_np = (dist_in - dist_out).astype(np.float32)
        else:
            sdf_np = np.zeros_like(density_np)
        # Normalize to ~[-1, 1] by dividing by half-resolution
        sdf = torch.from_numpy(sdf_np).unsqueeze(0).float() / (density_np.shape[0] / 2)

        # Input: [rho, Fx_norm, Fy_norm, Fz_norm, SDF, fixed_mask?].  The
        # support condition changes per v5 candidate, so it must be visible to
        # the operator instead of appearing only in the boundary loss.
        input_channels = [density, force_norm]
        if self.include_sdf:
            input_channels.append(sdf)
        x = torch.cat(input_channels, dim=0)
        if self.include_fixed_mask:
            x = torch.cat([x, fixed_mask], dim=0)

        return {
            'x': x,
            'disp_norm': disp_log,            # [3, N, N, N] log-scale displacement
            'disp_scale': disp_abs_max,        # scalar, for PDE de-normalization
            'has_disp': has_disp,              # 1.0 if displacement data exists, 0.0 otherwise
            'stress_log': stress_log,          # [6, N, N, N] stress / vm_max (normalized)
            'von_mises': von_mises,            # [1, N, N, N] vm / vm_max (target max = 1.0)
            'vm_scale': float(vm_max),         # per-sample scale (for physical recovery)
            'log_vm_max': float(math.log10(max(vm_max, 1e-6))),  # log10(vm_max) target for scale head
            'density': density,                # [1, N, N, N]
            'fixed_mask': fixed_mask,          # [1, N, N, N]
            'force': force,                    # [3, N, N, N] physical units (for PDE)
            'solid_fraction': float((density > self.SOLID_THRESHOLD).float().mean()),
        }


def collate_v3(batch):
    """Custom collate to handle dict-based dataset."""
    keys = batch[0].keys()
    result = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            result[k] = torch.stack([b[k] for b in batch])
        else:
            result[k] = [b[k] for b in batch]
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch_v3(model, loader, criterion, optimizer, scaler, device, epoch,
                   accum_steps=1):
    model.train()
    totals = {'total': 0, 'disp': 0, 'stress_log': 0,
              'vm_asym': 0, 'vm_peak': 0, 'bc': 0, 'pde': 0, 'scale': 0,
              'decision': 0, 'decision_margin': 0}
    n_batches = 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(pbar):
        x = batch['x'].to(device)
        disp_norm = batch['disp_norm'].to(device)
        stress_log = batch['stress_log'].to(device)
        vm = batch['von_mises'].to(device)
        density = batch['density'].to(device)
        fixed_mask = batch['fixed_mask'].to(device)
        force = batch['force'].to(device)
        has_disp = torch.as_tensor(batch['has_disp'], device=device).float() if 'has_disp' in batch else None
        disp_scale = torch.as_tensor(batch['disp_scale'], device=device).float() if 'disp_scale' in batch else None
        log_vm_max_target = torch.as_tensor(batch['log_vm_max'], device=device).float() if 'log_vm_max' in batch else None

        amp_enabled = scaler.is_enabled()
        with autocast('cuda', enabled=amp_enabled):
            # Use forward_with_scale if model has scale head
            has_scale_head = hasattr(model, 'forward_with_scale')
            if has_scale_head:
                pred, log_vm_max_pred = model.forward_with_scale(x)
            else:
                pred = model(x)
                log_vm_max_pred = None
            losses = criterion(pred, disp_norm, stress_log, vm, density, fixed_mask,
                               force=force, epoch=epoch, has_disp=has_disp, disp_scale=disp_scale,
                               log_vm_max_pred=log_vm_max_pred, log_vm_max_target=log_vm_max_target)
            loss = losses['total'] / accum_steps

        if not torch.isfinite(pred).all():
            print(f"[Epoch {epoch} step {step}] Non-finite model output detected; skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        if not torch.isfinite(loss):
            print(f"[Epoch {epoch} step {step}] Non-finite loss detected; skipping batch.")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                print(f"[Epoch {epoch} step {step}] Non-finite grad norm; skipping step.")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            optimizer.zero_grad()

        for k in totals:
            if k in losses:
                totals[k] += losses[k].item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{losses['total'].item():.4e}",
            slog=f"{losses['stress_log'].item():.4e}",
            vasym=f"{losses['vm_asym'].item():.4e}",
        )

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def validate_v3(model, loader, criterion, device, use_amp=True, epoch=0):
    model.eval()
    totals = {'total': 0, 'disp': 0, 'stress_log': 0,
              'vm_asym': 0, 'vm_peak': 0, 'bc': 0, 'pde': 0, 'scale': 0,
              'decision': 0, 'decision_margin': 0}
    n_batches = 0

    vm_rel_errors = []  # per-sample max VM relative errors
    sf_ratios = []      # SF_PINO / SF_FEM per sample
    vm_under_flags = []
    scale_errors = []   # log10(vm_max) prediction errors
    vm_phys_rel_errors = []
    sf_ratios_phys = []
    sf_phys_under_flags = []
    pred_sf_physical = []
    true_sf_physical = []

    # For relative L2 error (accumulated over all batches)
    disp_l2_num_sq = 0.0   # sum of ||pred - true||^2
    disp_l2_den_sq = 0.0   # sum of ||true||^2
    stress_l2_num_sq = 0.0
    stress_l2_den_sq = 0.0

    for batch in loader:
        x = batch['x'].to(device)
        disp_norm = batch['disp_norm'].to(device)
        stress_log = batch['stress_log'].to(device)
        vm = batch['von_mises'].to(device)
        density = batch['density'].to(device)
        fixed_mask = batch['fixed_mask'].to(device)
        force = batch['force'].to(device)
        has_disp = torch.as_tensor(batch['has_disp'], device=device).float() if 'has_disp' in batch else None
        disp_scale = torch.as_tensor(batch['disp_scale'], device=device).float() if 'disp_scale' in batch else None
        log_vm_max_target = torch.as_tensor(batch['log_vm_max'], device=device).float() if 'log_vm_max' in batch else None

        with autocast('cuda', enabled=use_amp):
            has_scale_head = hasattr(model, 'forward_with_scale')
            if has_scale_head:
                pred, log_vm_max_pred = model.forward_with_scale(x)
            else:
                pred = model(x)
                log_vm_max_pred = None

        if not torch.isfinite(pred).all():
            print("[validate_v3] Non-finite model output detected; skipping batch.")
            continue

        losses = criterion(pred.float(), disp_norm, stress_log, vm, density, fixed_mask,
                          force=force, epoch=epoch, has_disp=has_disp, disp_scale=disp_scale,
                          log_vm_max_pred=log_vm_max_pred.float() if log_vm_max_pred is not None else None,
                          log_vm_max_target=log_vm_max_target)
        if not torch.isfinite(losses['total']):
            print("[validate_v3] Non-finite loss detected; skipping batch.")
            continue

        for k in totals:
            if k in losses:
                totals[k] += losses[k].item()
        n_batches += 1

        # Compute actual metrics — normalized space + physical scale-aware space
        pred_stress_log = pred[:, 3:].float()  # now: normalized stress
        pred_vm = compute_von_mises(pred_stress_log)
        mask = (density > 0.1).float()

        # Relative L2 error — computed only in solid region (mask > 0.5)
        pred_disp = pred[:, :3].float()
        target_disp = disp_norm.float()
        target_stress_log_f = stress_log.float()
        solid_mask_3 = mask.expand_as(pred_disp)   # [B,3,N,N,N]
        solid_mask_6 = mask.expand_as(pred_stress_log)  # [B,6,N,N,N]
        if has_disp is not None:
            for i in range(pred_disp.shape[0]):
                if has_disp[i] > 0.5:
                    m3 = solid_mask_3[i] > 0.5
                    disp_l2_num_sq += ((pred_disp[i][m3] - target_disp[i][m3]) ** 2).sum().item()
                    disp_l2_den_sq += (target_disp[i][m3] ** 2).sum().item()
        else:
            m3 = solid_mask_3 > 0.5
            disp_l2_num_sq += ((pred_disp[m3] - target_disp[m3]) ** 2).sum().item()
            disp_l2_den_sq += (target_disp[m3] ** 2).sum().item()
        m6 = solid_mask_6 > 0.5
        stress_l2_num_sq += ((pred_stress_log[m6] - target_stress_log_f[m6]) ** 2).sum().item()
        stress_l2_den_sq += (target_stress_log_f[m6] ** 2).sum().item()

        B = pred.shape[0]
        for i in range(B):
            m = mask[i, 0] > 0.5
            if m.any():
                pred_max = pred_vm[i, 0][m].max().item()
                target_max = vm[i, 0][m].max().item()
                if target_max > 1e-6:
                    rel = abs(pred_max - target_max) / target_max
                    vm_rel_errors.append(rel)
                    vm_under_flags.append(float(pred_max < target_max))

                    # SF ratio (>1 means PINO overestimates SF = underestimates stress)
                    # target_max should be ~1.0 after per-sample normalization
                    sf_pred = 1.0 / max(pred_max, 1e-10)
                    sf_true = 1.0 / target_max
                    sf_ratios.append(sf_pred / sf_true)

                    if log_vm_max_pred is not None and log_vm_max_target is not None:
                        pred_log = log_vm_max_pred[i, 0].item()
                        true_log = (
                            log_vm_max_target[i].item()
                            if log_vm_max_target.dim() > 0
                            else log_vm_max_target.item()
                        )
                        pred_scale = 10.0 ** pred_log
                        true_scale = 10.0 ** true_log
                        pred_phys_max = pred_max * pred_scale
                        true_phys_max = target_max * true_scale
                        rel_phys = abs(pred_phys_max - true_phys_max) / max(true_phys_max, 1e-12)
                        vm_phys_rel_errors.append(rel_phys)
                        sf_pred_phys = 50.0e6 / max(pred_phys_max, 1e-12)
                        sf_true_phys = 50.0e6 / max(true_phys_max, 1e-12)
                        sf_ratios_phys.append(sf_pred_phys / max(sf_true_phys, 1e-12))
                        sf_phys_under_flags.append(float(sf_pred_phys <= sf_true_phys))
                        pred_sf_physical.append(sf_pred_phys)
                        true_sf_physical.append(sf_true_phys)

        # Scale head error tracking
        if log_vm_max_pred is not None and log_vm_max_target is not None:
            for i in range(pred.shape[0]):
                pred_log = log_vm_max_pred[i, 0].item()
                true_log = log_vm_max_target[i].item() if log_vm_max_target.dim() > 0 else log_vm_max_target.item()
                scale_errors.append(abs(pred_log - true_log))

    avg = {k: v / max(n_batches, 1) for k, v in totals.items()}

    # Relative L2 errors
    if disp_l2_den_sq > 0:
        avg['disp_rel_l2'] = float((disp_l2_num_sq / disp_l2_den_sq) ** 0.5)
    if stress_l2_den_sq > 0:
        avg['stress_rel_l2'] = float((stress_l2_num_sq / stress_l2_den_sq) ** 0.5)

    if vm_rel_errors:
        arr = np.array(vm_rel_errors)
        avg['vm_max_rel_mean'] = float(arr.mean())
        avg['vm_max_rel_median'] = float(np.median(arr))
        avg['vm_max_rel_p95'] = float(np.percentile(arr, 95))
        avg['vm_under_rate'] = float(np.mean(vm_under_flags)) if vm_under_flags else 0.0

    if sf_ratios:
        sr = np.array(sf_ratios)
        avg['sf_ratio_mean'] = float(sr.mean())
        avg['sf_ratio_median'] = float(np.median(sr))
        # % of samples where PINO is conservative (SF_PINO ≤ SF_FEM)
        avg['conservative_pct'] = float((sr <= 1.0).mean() * 100)
        avg['sf_over_rate'] = float((sr > 1.0).mean())

    if vm_phys_rel_errors:
        arr = np.array(vm_phys_rel_errors)
        avg['vm_phys_rel_mean'] = float(arr.mean())
        avg['vm_phys_rel_median'] = float(np.median(arr))
        avg['vm_phys_rel_p95'] = float(np.percentile(arr, 95))

    if sf_ratios_phys:
        sr = np.array(sf_ratios_phys)
        avg['sf_ratio_phys_mean'] = float(sr.mean())
        avg['sf_ratio_phys_median'] = float(np.median(sr))
        avg['conservative_pct_phys'] = float((sr <= 1.0).mean() * 100)
        avg['sf_over_rate_phys'] = float((sr > 1.0).mean())

    if pred_sf_physical:
        pred_sf = np.asarray(pred_sf_physical, dtype=np.float64)
        true_sf = np.asarray(true_sf_physical, dtype=np.float64)
        true_safe = true_sf >= criterion.safety_threshold
        pred_safe = pred_sf >= criterion.safety_threshold
        n_safe = int(true_safe.sum())
        n_unsafe = int((~true_safe).sum())
        true_positive = int(np.logical_and(pred_safe, true_safe).sum())
        true_negative = int(np.logical_and(~pred_safe, ~true_safe).sum())
        false_safe = int(np.logical_and(pred_safe, ~true_safe).sum())
        false_unsafe = int(np.logical_and(~pred_safe, true_safe).sum())
        safe_recall = true_positive / max(n_safe, 1)
        unsafe_recall = true_negative / max(n_unsafe, 1)
        avg['decision_accuracy'] = float((true_positive + true_negative) / max(len(true_safe), 1))
        avg['decision_balanced_accuracy'] = float(0.5 * (safe_recall + unsafe_recall))
        avg['decision_safe_recall'] = float(safe_recall)
        avg['decision_unsafe_recall'] = float(unsafe_recall)
        avg['decision_false_safe_count'] = false_safe
        avg['decision_false_safe_rate'] = float(false_safe / max(n_unsafe, 1))
        avg['decision_false_unsafe_count'] = false_unsafe

        # Rank-based AUROC without an sklearn dependency. Ties receive their
        # average rank. This is the threshold-independent quantity ACS needs.
        if n_safe and n_unsafe:
            order = np.argsort(pred_sf, kind='mergesort')
            ranks = np.empty(len(pred_sf), dtype=np.float64)
            start = 0
            while start < len(order):
                end = start + 1
                while end < len(order) and pred_sf[order[end]] == pred_sf[order[start]]:
                    end += 1
                ranks[order[start:end]] = 0.5 * (start + 1 + end)
                start = end
            rank_sum_safe = float(ranks[true_safe].sum())
            avg['decision_auroc'] = float(
                (rank_sum_safe - n_safe * (n_safe + 1) / 2.0) / (n_safe * n_unsafe)
            )

    if scale_errors:
        se = np.array(scale_errors)
        avg['scale_mae'] = float(se.mean())
        avg['scale_p95'] = float(np.percentile(se, 95))

    return avg


def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs,
                                 min_lr_ratio=0.01):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def train_pino_v3(
    data_dirs: list[str],
    output_dir: str = './checkpoints/pino_v3',
    epochs: int = 500,
    batch_size: int = 4,
    lr: float = 3e-4,
    warmup_epochs: int = 20,
    accum_steps: int = 2,
    hard_case_replay_ratio: float = 0.35,
    boundary_replay_ratio: float = 0.0,
    boundary_log_band: float = 0.12,
    resume: bool = False,
    resume_from: str | None = None,
    use_amp: bool = True,
    model_modes: int = 16,
    model_width: int = 64,
    model_n_layers: int = 6,
    arch: str = 'pino',
    seed: int = 42,
    checkpoint_selection: str = 'physics',
    scale_ramp_epochs: int = 12,
    scale_ramp_start_factor: float = 0.1,
    auto_freeze_scale_head_warmstart: bool = True,
    version_tag: str | None = None,
    index_file: str | None = None,
    target_res: int | None = None,
    train_augment: bool = True,
    include_fixed_mask: bool = False,
    include_sdf: bool = True,
    **loss_kwargs,
):
    if seed is not None:
        set_seed(seed)
        print(f"Random seed: {seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if version_tag is None:
        version_tag = output_path.name

    # Data
    train_set = FEMDatasetV3(data_dirs, split='train', augment=train_augment,
                             index_file=index_file, target_res=target_res,
                             include_fixed_mask=include_fixed_mask,
                             include_sdf=include_sdf)
    val_set = FEMDatasetV3(data_dirs, split='val', augment=False,
                           index_file=index_file, target_res=target_res,
                           include_fixed_mask=include_fixed_mask,
                           include_sdf=include_sdf)

    # Weighted sampler: oversample thin-shell, unsafe, and true threshold-near
    # cases.  The last term is explicitly based on FEM SF rather than a
    # stress-magnitude proxy so it targets the ACS deployment boundary.
    sample_weights = []
    for i in range(len(train_set)):
        d = np.load(train_set.files[i])
        sf = (d['density'] > FEMDatasetV3.SOLID_THRESHOLD).mean()
        vm_max = float(d['von_mises'].max()) if 'von_mises' in d else 0.0
        unsafe_ratio = vm_max / 50e6 if vm_max > 0 else 0.0
        if sf < 0.05:
            weight = 3.0
        elif sf < 0.10:
            weight = 2.0
        else:
            weight = 1.0
        if unsafe_ratio > 0.8:
            weight += 2.0 * hard_case_replay_ratio
        elif unsafe_ratio > 0.5:
            weight += 1.0 * hard_case_replay_ratio
        if "sf" in d and boundary_replay_ratio > 0.0:
            sf_gt = float(d["sf"])
            if sf_gt > 0.0 and abs(math.log10(sf_gt / 1.3)) <= boundary_log_band:
                weight += boundary_replay_ratio
        sample_weights.append(weight)

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_set), replacement=True
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler,
        num_workers=8, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=3,
        collate_fn=collate_v3,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=3,
        collate_fn=collate_v3,
    )

    # Model — load config from existing checkpoint if available
    config = StructShadowConfig.from_yaml()

    # Model config — configurable architecture size
    input_channels = 4 + int(include_sdf) + int(include_fixed_mask)
    model_kwargs = {
        'in_channels': input_channels,
        'out_channels': 9,  # 3 disp + 6 stress (now in log-space)
        'modes': model_modes,
        'width': model_width,
        'n_layers': model_n_layers,
        'padding': 0,
    }

    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        ckpt_cfg = ckpt.get('config', {})
        for k in model_kwargs:
            if k in ckpt_cfg:
                model_kwargs[k] = ckpt_cfg[k]
        # Do not let an older checkpoint silently remove the support input.
        model_kwargs['in_channels'] = input_channels
        # Force padding=0 for AMP compatibility (64+0=64, power of 2)
        model_kwargs['padding'] = 0
        print(f"Model config from checkpoint: {model_kwargs}")

    model_map = {
        'pino': PINO3d,
        'ufno': UFNO3d,
        'ufno_dual': UFNODualBranch3d,
        'gino_official': OfficialGINO3d,
        'transolver_official': OfficialTransolver3d,
    }
    ModelClass = model_map[arch]
    if arch in {'gino_official', 'transolver_official'}:
        model_kwargs['resolution'] = target_res or 48
    model = ModelClass(**model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{arch.upper()} parameters: {n_params:,}")

    # Initialize from previous checkpoint (fine-tune)
    if resume_from and not resume:
        if 'model_state_dict' in ckpt:
            _load_state_dict_compat(model, ckpt['model_state_dict'], resume_from)
        else:
            _load_state_dict_compat(model, ckpt, resume_from)
        print(f"Loaded weights from {resume_from}")
        print("NOTE: keeping all v1 weights including projection head")
        print("  The network will adapt from linear-stress to log-stress via fine-tuning")

    # Split train-control kwargs from pure loss kwargs before constructing criterion.
    freeze_backbone_epochs = loss_kwargs.pop('freeze_backbone_epochs', 0)

    # Loss
    criterion = PINOLossV3(
        disp_scale=FEMDatasetV3.DISP_SCALE,
        resolution=target_res or 48,
        **loss_kwargs,
    ).to(device)

    print(f"Loss config: {loss_kwargs}")

    # Freeze backbone for initial scale-head-only training (if requested).
    # This prevents the large scale loss gradient from damaging backbone weights.
    if (
        auto_freeze_scale_head_warmstart
        and freeze_backbone_epochs <= 0
        and resume_from
        and getattr(model, '_loaded_missing_scale_head', False)
        and hasattr(model, 'scale_head')
    ):
        freeze_backbone_epochs = max(8, min(max(epochs // 10, 1), 15))
        print(
            f"Auto-enabled scale-head warmstart: backbone frozen for "
            f"{freeze_backbone_epochs} epochs because the resume checkpoint has no scale_head weights."
        )
    scale_train_prefixes = tuple(getattr(model, '_scale_branch_prefixes', ('scale_head',)))
    if freeze_backbone_epochs > 0 and hasattr(model, 'scale_head'):
        for name, p in model.named_parameters():
            if not any(name.startswith(prefix) for prefix in scale_train_prefixes):
                p.requires_grad = False
        model._detach_scale_head_input = True
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Backbone FROZEN for first {freeze_backbone_epochs} epochs. "
              f"Trainable params: {n_trainable:,} (scale branch only)")
    else:
        model._detach_scale_head_input = False

    # Optimizer — simple AdamW, proven to work
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_epochs, epochs)
    scaler = GradScaler(enabled=(use_amp and device.type == 'cuda'))

    # Training state
    start_epoch = 1
    history = []
    best_vm_metric = float('inf')
    best_conservative = 0.0
    best_sf_ratio_phys = None
    best_norm_vm_metric = float('inf')
    best_norm_conservative = 0.0
    best_norm_score = float('inf')
    best_decision_auroc = float('-inf')
    no_improve = 0
    patience = 60

    if resume:
        checkpoint_paths = [
            path for path in (
                output_path / 'best_model.pt',
                output_path / 'last_model.pt',
            )
            if path.exists()
        ]
        if checkpoint_paths:
            loaded = [
                (path, torch.load(path, map_location=device, weights_only=False))
                for path in checkpoint_paths
            ]
            ckpt_path, ckpt = max(
                loaded,
                key=lambda item: int(item[1].get('epoch', 0)),
            )
            _load_state_dict_compat(model, ckpt['model_state_dict'], str(ckpt_path))
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            else:
                print(
                    f"Checkpoint {ckpt_path} has no optimizer state; "
                    "resuming its model weights with a fresh optimizer."
                )
            if 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            best_vm_metric = ckpt.get('vm_phys_rel_mean', ckpt.get('vm_max_rel_mean', float('inf')))
            best_conservative = ckpt.get('conservative_pct_phys', ckpt.get('conservative_pct', 0.0))
            best_sf_ratio_phys = ckpt.get('sf_ratio_phys_mean', None)
            best_norm_vm_metric = ckpt.get('vm_max_rel_mean', float('inf'))
            best_norm_conservative = ckpt.get('conservative_pct', 0.0)
            cons_penalty = max(0.0, 70.0 - best_norm_conservative) * 0.01 + max(0.0, best_norm_conservative - 90.0) * 0.015
            best_norm_score = best_norm_vm_metric + cons_penalty
            best_decision_auroc = ckpt.get('decision_auroc', float('-inf'))
            for _ in range(start_epoch - 1):
                scheduler.step()
            print(f"Resumed from epoch {start_epoch-1}, "
                  f"best VM_phys={best_vm_metric:.4f}, Cons_phys={best_conservative:.1f}%")

    # ── 2-phase curriculum ──
    # Phase 1 (epoch 1..warmup): disp + stress only — fast convergence
    # Phase 2 (warmup+1..): add VM asymmetric + PDE — safety tuning
    phase1_end = warmup_epochs

    # EMA model for stable validation — updated PER-EPOCH, so decay must be
    # tuned for epoch granularity. 0.999 freezes the model (99.5% old after 5 epochs).
    # Use 0.85 for responsive updates during fine-tune (~4-epoch half-life).
    ema_decay = 0.85
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    def update_ema(ema_m, m, decay):
        with torch.no_grad():
            for ema_p, p in zip(ema_m.parameters(), m.parameters()):
                ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)

    for epoch in range(start_epoch, epochs + 1):
        # Unfreeze backbone after scale-head warmup
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            for name, p in model.named_parameters():
                p.requires_grad = True
            model._detach_scale_head_input = False
            # Rebuild optimizer with all params and a lower LR for fine-tuning
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
            scheduler = get_cosine_warmup_scheduler(
                optimizer, 0, epochs - freeze_backbone_epochs
            )
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n*** Backbone UNFROZEN at epoch {epoch}. "
                  f"Trainable: {n_trainable:,}, LR: {lr*0.3:.2e} ***\n")

        # Curriculum phase switching
        if epoch <= phase1_end:
            criterion.lambda_disp = loss_kwargs.get('lambda_disp', 3.0)
            criterion.lambda_stress_log = loss_kwargs.get('lambda_stress_log', 5.0)
            criterion.lambda_vm_asym = 0.0
            criterion.lambda_vm_peak = 0.0
            criterion.lambda_pde = 0.0
            if epoch == start_epoch:
                print(f"\n*** Phase 1 (epoch 1-{phase1_end}): disp + stress ***\n")
        else:
            criterion.lambda_disp = loss_kwargs.get('lambda_disp', 3.0)
            criterion.lambda_stress_log = loss_kwargs.get('lambda_stress_log', 5.0)
            criterion.lambda_vm_asym = loss_kwargs.get('lambda_vm_asym', 2.0)
            criterion.lambda_vm_peak = loss_kwargs.get('lambda_vm_peak', 1.0)
            criterion.lambda_pde = loss_kwargs.get('lambda_pde', 0.1)
            if epoch == phase1_end + 1:
                print(f"\n*** Phase 2 (epoch {phase1_end+1}+): +VM asymmetric + PDE ***\n")

        base_lambda_scale = loss_kwargs.get('lambda_scale', criterion.lambda_scale)
        if scale_ramp_epochs > 0:
            ramp_progress = min(max(epoch - 1, 0) / float(scale_ramp_epochs), 1.0)
            scale_factor = scale_ramp_start_factor + (1.0 - scale_ramp_start_factor) * ramp_progress
        else:
            scale_factor = 1.0
        if freeze_backbone_epochs > 0 and epoch <= freeze_backbone_epochs:
            scale_factor = 1.0
        criterion.lambda_scale = base_lambda_scale * scale_factor

        train_losses = train_epoch_v3(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, accum_steps
        )
        # Update EMA model after each epoch
        update_ema(ema_model, model, ema_decay)
        # Validate with EMA model (more stable metrics)
        val_losses = validate_v3(ema_model, val_loader, criterion, device, use_amp=use_amp, epoch=epoch)
        # Validate raw model every 2 epochs (saves ~10% time)
        if epoch % 2 == 0:
            raw_val = validate_v3(model, val_loader, criterion, device, use_amp=use_amp, epoch=epoch)
            raw_cons = raw_val.get('conservative_pct', 0)
            raw_vm_med = raw_val.get('vm_max_rel_median', float('inf'))
            print(f"  [raw] Cons={raw_cons:.1f}% VM_med={raw_vm_med:.4f}")
        scheduler.step()

        lr_cur = optimizer.param_groups[0]['lr']
        vm_mean = val_losses.get('vm_max_rel_mean', float('inf'))
        vm_med = val_losses.get('vm_max_rel_median', float('inf'))
        sf_ratio = val_losses.get('sf_ratio_mean', float('nan'))
        cons_pct = val_losses.get('conservative_pct', 0)
        sf_over_rate = max(0.0, 100.0 - cons_pct)
        vm_phys_mean = val_losses.get('vm_phys_rel_mean', float('inf'))
        vm_phys_med = val_losses.get('vm_phys_rel_median', float('inf'))
        sf_ratio_phys = val_losses.get('sf_ratio_phys_mean', float('nan'))
        cons_pct_phys = val_losses.get('conservative_pct_phys', 0.0)
        sf_over_rate_phys = val_losses.get('sf_over_rate_phys', float('nan'))

        history.append({
            'epoch': epoch, 'lr': lr_cur,
            'train': train_losses, 'val': val_losses,
        })
        with open(output_path / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        print(f"  Train: {train_losses['total']:.4e} (pde={train_losses.get('pde', 0):.4e}) | "
              f"Val: {val_losses['total']:.4e} (pde={val_losses.get('pde', 0):.4e})")
        disp_rl2 = val_losses.get('disp_rel_l2', float('nan'))
        stress_rl2 = val_losses.get('stress_rel_l2', float('nan'))
        scale_mae = val_losses.get('scale_mae', float('nan'))
        scale_p95 = val_losses.get('scale_p95', float('nan'))
        print(f"  VM rel: mean={vm_mean:.4f} med={vm_med:.4f} | "
              f"SF ratio: {sf_ratio:.3f} | Conservative: {cons_pct:.1f}% | "
              f"LR: {lr_cur:.2e}")
        if np.isfinite(vm_phys_mean):
            print(f"  VM phys: mean={vm_phys_mean:.4f} med={vm_phys_med:.4f} | "
                  f"SF phys ratio: {sf_ratio_phys:.3f} | Conservative phys: {cons_pct_phys:.1f}%")
        print(f"  Rel L2: disp={disp_rl2:.4f}, stress={stress_rl2:.4f} | "
              f"Scale MAE: {scale_mae:.3f}, p95: {scale_p95:.3f} | "
              f"lambda_scale_eff: {criterion.lambda_scale:.3f}")

        # Save best — use physical metrics when scale head is reliable (MAE < 1.0),
        # otherwise fall back to normalized-space metrics to avoid garbage selection.
        scale_head_reliable = np.isfinite(scale_mae) and scale_mae < 1.0

        def sf_ratio_distance(value):
            if not np.isfinite(value):
                return float('inf')
            if 0.95 <= value <= 1.05:
                return 0.0
            return min(abs(value - 0.95), abs(value - 1.05))

        decision_auroc = val_losses.get('decision_auroc', float('nan'))
        if checkpoint_selection == 'decision' and np.isfinite(decision_auroc):
            if decision_auroc > best_decision_auroc + 1e-12:
                is_best = True
            elif decision_auroc < best_decision_auroc - 1e-12:
                is_best = False
            else:
                is_best = vm_phys_mean < best_vm_metric
        elif scale_head_reliable:
            # Physical-space selection
            cur_ratio_dist = sf_ratio_distance(sf_ratio_phys)
            best_ratio_dist = sf_ratio_distance(best_sf_ratio_phys) if best_sf_ratio_phys is not None else float('inf')
            if cur_ratio_dist < best_ratio_dist:
                is_best = True
            elif cur_ratio_dist > best_ratio_dist:
                is_best = False
            else:
                if vm_phys_mean < best_vm_metric - 1e-12:
                    is_best = True
                elif vm_phys_mean > best_vm_metric + 1e-12:
                    is_best = False
                else:
                    is_best = cons_pct_phys > best_conservative
        else:
            # Normalized-space fallback (scale head not yet reliable)
            def balance_score(cons, vm):
                cons_penalty = max(0.0, 70.0 - cons) * 0.01 + max(0.0, cons - 90.0) * 0.015
                return vm + cons_penalty
            cur_score = balance_score(cons_pct, vm_mean)
            prev_score = best_norm_score
            is_best = cur_score < prev_score
        if is_best:
            if scale_head_reliable:
                best_vm_metric = vm_phys_mean
                best_conservative = cons_pct_phys
                best_sf_ratio_phys = sf_ratio_phys
            if np.isfinite(decision_auroc):
                best_decision_auroc = max(best_decision_auroc, decision_auroc)
            else:
                best_norm_vm_metric = vm_mean
                best_norm_conservative = cons_pct
                best_norm_score = cur_score
            no_improve = 0

            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': ema_model.state_dict(),  # save EMA weights (more stable)
                'optimizer_state_dict': optimizer.state_dict(),
                'vm_max_rel_mean': vm_mean,
                'vm_phys_rel_mean': vm_phys_mean,
                'vm_phys_rel_p95': val_losses.get('vm_phys_rel_p95', vm_phys_mean),
                'sf_ratio_mean': sf_ratio,
                'sf_ratio_phys_mean': sf_ratio_phys,
                'val_sf_under_rate': cons_pct / 100.0,
                'val_sf_over_rate': sf_over_rate / 100.0,
                'val_vm_under_rate': 1.0 - cons_pct / 100.0,
                'val_sf_under_rate_phys': cons_pct_phys / 100.0,
                'val_sf_over_rate_phys': sf_over_rate_phys,
                'p95_vm_error': val_losses.get('vm_max_rel_p95', vm_mean),
                'p95_vm_error_phys': val_losses.get('vm_phys_rel_p95', vm_phys_mean),
                'conservative_pct': cons_pct,
                'conservative_pct_phys': cons_pct_phys,
                'best_selection_space': 'physical' if scale_head_reliable else 'normalized',
                'scale_head_reliable': bool(scale_head_reliable),
                'checkpoint_selection': checkpoint_selection,
                'decision_auroc': decision_auroc,
                'decision_balanced_accuracy': val_losses.get('decision_balanced_accuracy'),
                'decision_safe_recall': val_losses.get('decision_safe_recall'),
                'decision_unsafe_recall': val_losses.get('decision_unsafe_recall'),
                'decision_false_safe_count': val_losses.get('decision_false_safe_count'),
                'best_norm_vm_metric': best_norm_vm_metric,
                'best_norm_conservative': best_norm_conservative,
                'val_loss': val_losses['total'],
                'normalization_mode': 'log_stress',
                'normalization': {
                    'force_scale': FEMDatasetV3.FORCE_SCALE,
                    'disp_mode': 'log10_1e6',  # sign(u)*log10(|u|*1e6+1)
                    'stress_mode': 'log10',
                    'scale_mode': 'log10_vm_max',
                    'input_channels': 'density+force3+sdf',
                },
                'config': model_kwargs,
                'arch': arch,
                'version': version_tag,
                'resume_from': resume_from,
                'seed': seed,
                'loss_kwargs': loss_kwargs,
                'train_args': {
                    'data_dirs': data_dirs,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                    'warmup_epochs': warmup_epochs,
                    'accum_steps': accum_steps,
                },
                'scaler_state_dict': scaler.state_dict(),
            }
            torch.save(ckpt_data, output_path / 'best_model.pt')
            print(f"  -> Best (conservative_phys={cons_pct_phys:.1f}%, "
                  f"VM_phys={vm_phys_mean:.4f}, SF_ratio_phys={sf_ratio_phys:.3f})")
        else:
            no_improve += 1
        print(f"  [epoch {epoch}] VM_phys={vm_phys_mean:.4f} SF_ratio_phys={sf_ratio_phys:.3f} "
              f"cons_phys={cons_pct_phys:.1f}% reliable={scale_head_reliable}", flush=True)

        # Always keep the latest EMA snapshot so a crash never loses progress.
        last_data = {
            'epoch': epoch,
            'model_state_dict': ema_model.state_dict(),
            'vm_phys_rel_mean': vm_phys_mean,
            'sf_ratio_phys_mean': sf_ratio_phys,
            'val_loss': val_losses['total'],
            'normalization_mode': 'log_stress',
            'normalization': {
                'force_scale': FEMDatasetV3.FORCE_SCALE,
                'disp_mode': 'log10_1e6',
                'stress_mode': 'log10',
                'scale_mode': 'log10_vm_max',
                'input_channels': 'density+force3+sdf',
            },
            'config': model_kwargs,
            'arch': arch,
            'version': version_tag,
        }
        tmp_path = output_path / 'last_model.pt.tmp'
        torch.save(last_data, tmp_path)
        tmp_path.replace(output_path / 'last_model.pt')

        if epoch % 50 == 0:
            torch.save(ema_model.state_dict(), output_path / f'epoch_{epoch}.pt')

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nPINO v3 training complete.")
    print(f"  Best VM max rel error: {best_vm_metric:.4f}")
    print(f"  Conservative predictions: {best_conservative:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PINO v3: Log-stress + Asymmetric Safety')
    parser.add_argument('--index_file', type=str, default=None,
                        help='JSON index with train/val splits (overrides --data_dirs globbing)')
    parser.add_argument('--target_res', type=int, default=None,
                        help='Resize all samples to this cubic resolution (unify mixed sources)')
    parser.add_argument('--include_fixed_mask', action='store_true',
                        help='Append the per-candidate fixed support mask to PINO inputs.')
    parser.add_argument('--disable_sdf_input', action='store_true',
                        help='Remove the signed-distance channel for structural-context ablation.')
    parser.add_argument('--data_dirs', type=str, required=False, default='',
                        help='Comma-separated FEM dataset directories')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/pino_v3')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--accum_steps', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--hard_case_replay_ratio', type=float, default=0.35)
    parser.add_argument('--boundary_replay_ratio', type=float, default=0.0,
                        help='Extra sampler weight for FEM samples near SF=1.3.')
    parser.add_argument('--boundary_log_band', type=float, default=0.12,
                        help='Absolute log10(SF/1.3) band used by boundary replay.')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to v1/v2 checkpoint for fine-tuning')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--disable_train_augment', action='store_true',
                        help='Disable label-inconsistent geometry augmentation for PDE training.')
    # Model architecture
    parser.add_argument(
        '--arch',
        type=str,
        default='pino',
        choices=[
            'pino',
            'ufno',
            'ufno_dual',
            'gino_official',
            'transolver_official',
        ],
                        help='Model architecture: pino (F-FNO), ufno (U-shaped FNO), or ufno_dual (dual-branch scale-aware UFNO)')
    parser.add_argument('--modes', type=int, default=16)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=6)
    # Loss weights
    parser.add_argument('--lambda_disp', type=float, default=1.0)
    parser.add_argument('--lambda_stress_log', type=float, default=3.0)
    parser.add_argument('--lambda_vm_asym', type=float, default=2.0)
    parser.add_argument('--lambda_vm_peak', type=float, default=1.0)
    parser.add_argument('--lambda_bc', type=float, default=1.0)
    parser.add_argument('--underestimate_weight', type=float, default=6.0,
                        help='Penalty for VM underestimation; default tuned toward a balanced 5-8 regime')
    parser.add_argument('--overestimate_weight', type=float, default=0.3,
                        help='Penalty for VM overestimation; lower keeps the model mildly conservative')
    parser.add_argument('--max_log_stress', type=float, default=8.0)
    parser.add_argument('--lambda_pde', type=float, default=0.1,
                        help='PDE residual loss weight (equilibrium)')
    parser.add_argument('--lambda_scale', type=float, default=5.0,
                        help='Weight for log10(vm_max) scale prediction loss')
    parser.add_argument('--lambda_decision', type=float, default=0.0,
                        help='Auxiliary balanced safe/unsafe decision loss on the absolute scale branch.')
    parser.add_argument('--lambda_decision_margin', type=float, default=0.0,
                        help='Threshold-margin loss for safe/unsafe separation on the absolute scale branch.')
    parser.add_argument('--decision_temperature', type=float, default=0.12,
                        help='Log10(SF) temperature used by the auxiliary decision loss.')
    parser.add_argument('--decision_margin_log', type=float, default=0.06,
                        help='Required log10(SF) separation around the safety threshold.')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0,
                        help='Freeze backbone for N epochs, only train scale_head')
    parser.add_argument('--scale_ramp_epochs', type=int, default=12,
                        help='Ramp lambda_scale from a smaller value to full strength over N epochs')
    parser.add_argument('--scale_ramp_start_factor', type=float, default=0.1,
                        help='Initial fraction of lambda_scale used at epoch 1 before ramp-up')
    parser.add_argument('--disable_auto_scale_warmstart', action='store_true',
                        help='Disable automatic backbone-freeze warmstart for checkpoints missing scale_head')
    parser.add_argument('--version_tag', type=str, default=None,
                        help='Optional version string written into checkpoints (defaults to output_dir basename)')
    parser.add_argument('--lambda_constitutive', type=float, default=0.1,
                        help='Constitutive relation loss weight (σ=C:ε)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--checkpoint_selection', choices=['physics', 'decision'], default='physics',
                        help='Select checkpoints by physical error or calibration-split safety-factor AUROC.')

    args = parser.parse_args()

    loss_kwargs = {
        'lambda_disp': args.lambda_disp,
        'lambda_stress_log': args.lambda_stress_log,
        'lambda_vm_asym': args.lambda_vm_asym,
        'lambda_vm_peak': args.lambda_vm_peak,
        'lambda_bc': args.lambda_bc,
        'lambda_pde': args.lambda_pde,
        'lambda_constitutive': args.lambda_constitutive,
        'lambda_scale': args.lambda_scale,
        'lambda_decision': args.lambda_decision,
        'lambda_decision_margin': args.lambda_decision_margin,
        'decision_temperature': args.decision_temperature,
        'decision_margin_log': args.decision_margin_log,
        'freeze_backbone_epochs': args.freeze_backbone_epochs,
        'underestimate_weight': args.underestimate_weight,
        'overestimate_weight': args.overestimate_weight,
        'max_log_stress': args.max_log_stress,
    }

    train_pino_v3(
        data_dirs=([d for d in args.data_dirs.split(',') if d] or None),
        index_file=args.index_file,
        target_res=args.target_res,
        include_fixed_mask=args.include_fixed_mask,
        include_sdf=not args.disable_sdf_input,
        train_augment=not args.disable_train_augment,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        accum_steps=args.accum_steps,
        hard_case_replay_ratio=args.hard_case_replay_ratio,
        boundary_replay_ratio=args.boundary_replay_ratio,
        boundary_log_band=args.boundary_log_band,
        resume=args.resume,
        resume_from=args.resume_from,
        use_amp=not args.no_amp,
        model_modes=args.modes,
        model_width=args.width,
        model_n_layers=args.n_layers,
        arch=args.arch,
        seed=args.seed,
        checkpoint_selection=args.checkpoint_selection,
        scale_ramp_epochs=args.scale_ramp_epochs,
        scale_ramp_start_factor=args.scale_ramp_start_factor,
        auto_freeze_scale_head_warmstart=not args.disable_auto_scale_warmstart,
        version_tag=args.version_tag,
        **loss_kwargs,
    )
