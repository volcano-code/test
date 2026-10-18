#!/usr/bin/env python3
"""Fine-tune the candidate PINO stress-field tail using grouped FEM VM fields."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from train_candidate_pino_grouped import acs_metrics, metrics as scalar_metrics, split_groups, xg
from train_pino_aug_hardneg import load_geoattn_model
from structural.analysis.load_generator import LoadGenerator
from structural.analysis.stress_eval import compute_von_mises_torch

YIELD_STRESS = 50e6
TAU = 1.3
SEED = 20260703


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_field_map() -> dict[tuple[int, int, int], dict]:
    payload = np.load(ROOT / "data/candidate_vm_fields_grouped_20260703.npz")
    return {
        (int(branch), int(group), int(geometry)): {
            "field": field.astype(np.float32),
            "vm_base": float(vm_base),
            "base_multiplier": float(base_multiplier),
            "valid": bool(valid),
        }
        for branch, group, geometry, field, vm_base, base_multiplier, valid in zip(
            payload["branch"], payload["group"], payload["geometry"],
            payload["vm_normalized"], payload["vm_base"], payload["base_multiplier"], payload["valid"]
        )
    }


def load_rows(include_field: bool = True) -> list[dict]:
    field_map = load_field_map() if include_field else {}
    generator = LoadGenerator()
    rows = []
    for branch_id, path in enumerate((ROOT / "data/aug.npz", ROOT / "data/aug_fixed.npz")):
        payload = np.load(path)
        for index in range(len(payload["dens"])):
            density = payload["dens"][index].astype(np.float32)
            force = generator.standing_load(density)["force"] * float(payload["mult"][index])
            group = index // 16
            geometry = (index % 16) // 4
            field_record = field_map.get((branch_id, group, geometry))
            if field_record is None or not field_record["valid"]:
                continue
            vm_recomputed = field_record["vm_base"] * float(payload["mult"][index]) / field_record["base_multiplier"]
            rows.append({
                "density": density,
                "force": force.astype(np.float32),
                "log_vm": float(np.log10(max(vm_recomputed, 1e-12))),
                "sf": float(YIELD_STRESS / max(vm_recomputed, 1e-12)),
                "group": group,
                "vm_normalized": field_record["field"],
            })
    return rows


def load_scaleup_rows() -> list[dict]:
    payload = np.load(ROOT / "data/scaleup_test.npz")
    generator = LoadGenerator()
    rows = []
    for index in range(len(payload["dens"])):
        density = payload["dens"][index].astype(np.float32)
        force = generator.standing_load(density)["force"] * float(payload["mult"][index])
        rows.append({
            "density": density,
            "force": force.astype(np.float32),
            "log_vm": float(np.log10(max(float(payload["vm"][index]), 1e-12))),
            "sf": float(payload["sf"][index]),
        })
    return rows


def subset(rows: list[dict], groups: list[int]) -> list[dict]:
    allowed = set(groups)
    return [row for row in rows if row["group"] in allowed]


def forward_vm(model, row: dict, device: torch.device):
    inputs = xg(model, row["density"], row["force"], device)
    fields = model._field_forward(inputs)
    vm_normalized = compute_von_mises_torch(fields[:, 3:])[:, 0]
    log_scale = model._scale_forward(inputs).reshape(-1)
    solid = torch.as_tensor(row["density"] > 0.5, device=device).unsqueeze(0)
    vm_max_normalized = vm_normalized.masked_fill(~solid, 0.0).flatten(1).amax(1).clamp(min=1e-8)
    log_vm_physical = log_scale + torch.log10(vm_max_normalized)
    return vm_normalized, log_vm_physical, solid


@torch.no_grad()
def predict_log_vm(model, rows: list[dict], device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    for row in rows:
        _, log_vm, _ = forward_vm(model, row, device)
        output.append(float(log_vm.detach().cpu()))
    return np.asarray(output, dtype=float)


def cpu_state(module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--decision-weight", type=float, default=0.0)
    parser.add_argument("--unsafe-margin-log", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/pino_candidate_field_20260703")
    parser.add_argument("--result", type=Path, default=ROOT / "results/pino_candidate_field_20260703.json")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    rows = load_rows()
    splits = split_groups()
    train_rows = subset(rows, splits["train"])
    val_rows = subset(rows, splits["selection_validation"])
    cal_rows = subset(rows, splits["conformal_calibration"])
    test_rows = subset(rows, splits["heldout_test"])
    scaleup_rows = load_scaleup_rows()

    base = ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"
    scale_checkpoint = ROOT / "checkpoints/pino_aug_hardneg_20260624/scale_geoattn_e8.pt"
    model, _, _ = load_geoattn_model(device, base, scale_checkpoint)
    baseline_test_pred = predict_log_vm(model, test_rows, device)
    baseline_scaleup_pred = predict_log_vm(model, scaleup_rows, device)
    baseline_test = scalar_metrics(baseline_test_pred, test_rows)
    baseline_scaleup = scalar_metrics(baseline_scaleup_pred, scaleup_rows)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_modules = [model.dec1[-1], model.stress_head]
    parameters = []
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            parameters.append(parameter)
    initial_parameters = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    history = []
    best_key = None
    best_epoch = None
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train_rows))
        total = 0.0
        for row_index in order:
            row = train_rows[int(row_index)]
            vm_pred, log_vm_pred, solid = forward_vm(model, row, device)
            target_field = torch.as_tensor(row["vm_normalized"], dtype=torch.float32, device=device).unsqueeze(0)
            target_log_vm = torch.tensor([row["log_vm"]], dtype=torch.float32, device=device)
            values_pred = vm_pred[solid]
            values_target = target_field[solid]
            weights = 1.0 + 4.0 * values_target
            field_loss = torch.mean(F.smooth_l1_loss(values_pred, values_target, reduction="none") * weights)
            scalar_error = log_vm_pred - target_log_vm
            scalar_loss = F.smooth_l1_loss(log_vm_pred, target_log_vm)
            underprediction = F.relu(-scalar_error).mean()
            unsafe_hinge = torch.zeros((), device=device)
            if row["sf"] < TAU:
                unsafe_hinge = F.relu(target_log_vm + args.unsafe_margin_log - log_vm_pred).mean()
            parameter_anchor = sum(
                torch.mean(torch.abs(parameter - initial) ** 2)
                for parameter, initial in zip(parameters, initial_parameters)
            )
            loss = (
                field_loss + scalar_loss + 3.0 * underprediction
                + args.decision_weight * unsafe_hinge + 1e-4 * parameter_anchor
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.detach().cpu())

        val_pred = predict_log_vm(model, val_rows, device)
        val = scalar_metrics(val_pred, val_rows)
        key = (val["false_pass_count"], -val["raw_safe_coverage"], val["log_vm_mae"])
        history.append({"epoch": epoch, "train_loss": total / len(train_rows), "validation": val})
        print(json.dumps(history[-1]), flush=True)
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                "dec1_tail": cpu_state(model.dec1[-1]),
                "stress_head": cpu_state(model.stress_head),
            }

    model.dec1[-1].load_state_dict(best_state["dec1_tail"])
    model.stress_head.load_state_dict(best_state["stress_head"])
    cal_pred = predict_log_vm(model, cal_rows, device)
    test_pred = predict_log_vm(model, test_rows, device)
    scaleup_pred = predict_log_vm(model, scaleup_rows, device)
    candidate_test = scalar_metrics(test_pred, test_rows)
    candidate_scaleup = scalar_metrics(scaleup_pred, scaleup_rows)
    candidate_acs = acs_metrics(cal_pred, cal_rows, test_pred, test_rows, alpha=0.05)

    promotion = {
        "eligible": bool(
            candidate_scaleup["false_pass_count"] <= baseline_scaleup["false_pass_count"]
            and candidate_scaleup["raw_safe_coverage"] > baseline_scaleup["raw_safe_coverage"]
            and candidate_acs["false_safe"] == 0
            and candidate_acs["false_unsafe"] == 0
            and candidate_acs["autonomous_rate"] > 0
        ),
        "authoritative_pipeline_modified": False,
        "rule": "external false passes no worse, external coverage higher, and zero-error nonzero heldout ACS autonomy",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "field_tail_delta.pt"
    torch.save({
        **best_state,
        "base_checkpoint": str(base),
        "scale_checkpoint": str(scale_checkpoint),
        "best_epoch": best_epoch,
        "split_seed": SEED,
    }, checkpoint)
    result = {
        "schema": "candidate_pino_field_tail_v1",
        "field_labels": "data/candidate_vm_fields_grouped_20260703.npz",
        "source_hashes": {
            "field_labels": sha256(ROOT / "data/candidate_vm_fields_grouped_20260703.npz"),
            "base_checkpoint": sha256(base),
            "scale_checkpoint": sha256(scale_checkpoint),
        },
        "splits": splits,
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "decision_weight": args.decision_weight,
            "unsafe_margin_log": args.unsafe_margin_log,
            "best_epoch": best_epoch,
            "trainable": ["dec1[-1]", "stress_head"],
        },
        "baseline_heldout": baseline_test,
        "candidate_heldout": candidate_test,
        "baseline_scaleup_external": baseline_scaleup,
        "candidate_scaleup_external": candidate_scaleup,
        "candidate_acs_heldout": candidate_acs,
        "history": history,
        "checkpoint": str(checkpoint),
        "promotion": promotion,
        "limitations": [
            "Training supervises normalized von-Mises magnitude, not the six signed stress components.",
            "FEM labels are numerical references, not physical measurements.",
            "The output is an experimental delta and is not loaded by the authoritative pipeline.",
        ],
    }
    result_path = args.result
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(result_path), "checkpoint": str(checkpoint), "promotion": promotion}, indent=2))


if __name__ == "__main__":
    main()
