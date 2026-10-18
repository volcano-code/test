#!/usr/bin/env python3
"""Re-score saved 50-triple candidates with the frozen 85%-coverage PINO.

FEM labels and candidate metadata remain the original recorded values.  Only
PINO scores are recomputed from regenerated, checksum-audited candidate voxels.
The script refuses to emit paper-facing scores if candidate voxel counts differ
from the saved task artifacts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path("/data/xvxinyao/xvxinyao/StructShadow")
sys.path[:0] = [str(ROOT), str(ROOT / "generation"), str(ROOT / "experiments")]

from structural.analysis.load_generator import LoadGenerator
from train_pino_aug_hardneg import SCALE_BRANCH, load_geoattn_model
from train_pino_safe_side_boundary import predict


YIELD_STRESS = 50e6
TAU = 1.3
BASE = ROOT / "checkpoints/pino_v24_ufno_dual/best_model_e10_sf0991.pt"
DELTA = ROOT / "checkpoints/pino_safe_accept90_cal300_m08_20260704/safe_side_external_hardneg_scale.pt"
SOURCES = {"free": ROOT / "results/candidate_selection_batch_free50_20260624.json",
           "fixed": ROOT / "results/candidate_selection_batch_fixed50_20260624.json"}
OUT = ROOT / "results/acs_task_level_risk_coverage_20260711/rescored_best85"


def legacy_generator():
    """Load the exact generator revision that produced the saved 2026-06 tasks."""
    sys.path[:0] = [str(LEGACY_ROOT), str(LEGACY_ROOT / "generation"), str(LEGACY_ROOT / "experiments")]
    spec = importlib.util.spec_from_file_location(
        "shadowforge_legacy_candidate_selection", LEGACY_ROOT / "experiments/candidate_selection_batch.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.load_data_from_list, module.make_candidates


def load_model(device: torch.device):
    model, head, _ = load_geoattn_model(device, BASE, DELTA)
    payload = torch.load(DELTA, map_location=device, weights_only=False)
    for name, state in payload["scale_branch_state_dict"].items():
        getattr(model, name).load_state_dict(state)
    if "attn_scale_head_state_dict" in payload:
        head.load_state_dict(payload["attn_scale_head_state_dict"])
    model.eval()
    return model


def regenerate(case: dict, branch: str, load_data_from_list, make_candidates) -> tuple[dict[str, np.ndarray], list[dict]]:
    paths = [ROOT / "data/silhouettes/bench3" / f"{name}.png" for name in case["triple"]]
    if not all(p.exists() for p in paths):
        missing = [str(p) for p in paths if not p.exists()]
        raise FileNotFoundError(f"Missing task silhouettes: {missing}")
    sils, _ = load_data_from_list([str(p) for p in paths], True, (48, 48), torch.device("cpu"), 3, False)
    masks = [np.asarray(x).squeeze().astype(np.float32) for x in sils]
    volumes, _ = make_candidates(masks, 48, branch == "fixed")
    mismatches = []
    for old in case["candidates"]:
        name = old["candidate"]
        if name not in volumes:
            mismatches.append({"candidate": name, "reason": "missing"})
        elif int(volumes[name].sum()) != int(old["occupied_voxels"]):
            mismatches.append({"candidate": name, "expected_voxels": int(old["occupied_voxels"]),
                               "regenerated_voxels": int(volumes[name].sum())})
    return volumes, mismatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    load_data_from_list, make_candidates = legacy_generator()
    generator = LoadGenerator()
    audit, outputs = {"checkpoint": str(DELTA), "base": str(BASE), "branches": {}}, {}

    for branch, source in SOURCES.items():
        payload = json.loads(source.read_text())
        score_rows, links, mismatches = [], [], []
        for case in payload["cases"]:
            volumes, issue = regenerate(case, branch, load_data_from_list, make_candidates)
            if issue:
                mismatches.append({"case_id": case["case_id"], "triple": case["triple"], "issues": issue})
                continue
            for candidate in case["candidates"]:
                vol = volumes[candidate["candidate"]].astype(np.float32)
                score_rows.append({"density": vol, "force": generator.standing_load(vol)["force"].astype(np.float32)})
                links.append(candidate)
        audit["branches"][branch] = {"source": str(source), "candidate_rows": len(score_rows),
                                      "mismatch_cases": mismatches}
        if mismatches:
            continue
        pred_log_vm = predict(model, score_rows, device, args.batch_size)
        for candidate, log_vm in zip(links, pred_log_vm):
            sf = float(YIELD_STRESS / max(10.0 ** float(log_vm), 1e-12))
            candidate["sf_pino"] = sf
            candidate["raw_pino_safe"] = bool(sf >= TAU)
            # Old ACS fields belong to an earlier calibration and are invalid.
            for key in ("acs_lower_sf", "acs_upper_sf", "acs_certified_safe", "acs_certified_unsafe"):
                candidate.pop(key, None)
        outputs[branch] = payload

    (OUT / "regeneration_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    failures = {k: v["mismatch_cases"] for k, v in audit["branches"].items() if v["mismatch_cases"]}
    if failures:
        raise RuntimeError("Candidate reconstruction mismatch; no scores written: " + json.dumps(failures)[:1000])
    for branch, payload in outputs.items():
        (OUT / f"{branch}_best85_rescored.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
