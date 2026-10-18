#!/usr/bin/env python3
"""Run the frozen strict PINO on exact Q48 proxies of printed deliveries.

The protocol JSON supplies one native Q256 delivery, its pipeline manifest, and
the already completed Q48 FEM audit directory for each case.  This script:

1. applies the manifest-selected print orientation;
2. derives Q48 with the deployed adaptive max-pool operator;
3. verifies that the Q48 payload matches the independent FEM audit;
4. evaluates the frozen six-channel PINO checkpoint; and
5. records hashes, timings, predictions, and FEM comparisons.

PINO is reported as a ranking/screening prediction.  The FEM decision remains
the final structural authority.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for module_path in (ROOT, ROOT / "experiments"):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from audit_q48_q256_cross_resolution_20260729 import (  # noqa: E402
    adaptive_max_pool_q48,
    load_binary_q256,
    orient_for_print,
    selected_orientation,
    sha256_binary_occupancy,
)
from run_all_baselines import load_pino  # noqa: E402
from shadowforge_pipeline.run import strict_pino_sf  # noqa: E402
from structural.analysis.load_generator import LoadGenerator  # noqa: E402


SCHEMA = "shadowforge_physical_print_pino_q48_audit_v1"
TAU = 1.3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cases = protocol.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Protocol must contain a non-empty cases list")

    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(
        args.device if args.device.startswith("cpu") or torch.cuda.is_available()
        else "cpu"
    )
    started = time.time()
    model = load_pino(str(checkpoint), device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    load_generator = LoadGenerator()
    rows: list[dict[str, Any]] = []

    for record in cases:
        case_id = str(record["case_id"])
        input_path = Path(record["input"]).resolve()
        manifest_path = Path(record["manifest"]).resolve()
        fem_dir = Path(record["q48_fem_dir"]).resolve()
        for required in (input_path, manifest_path, fem_dir / "metadata.json"):
            if not required.exists():
                raise FileNotFoundError(required)

        display_q256 = load_binary_q256(input_path)
        orientation = selected_orientation(manifest_path, "auto")
        print_q256 = orient_for_print(display_q256, orientation)
        q48 = adaptive_max_pool_q48(print_q256)
        q48_hash = sha256_binary_occupancy(q48)
        q256_hash = sha256_binary_occupancy(print_q256)

        fem_metadata = json.loads(
            (fem_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if fem_metadata.get("status") != "Q48_ONLY_COMPLETE":
            raise RuntimeError(f"{case_id}: Q48 FEM audit is not complete")
        if fem_metadata.get("q48_payload_sha256") != q48_hash:
            raise RuntimeError(f"{case_id}: PINO and FEM Q48 payloads differ")
        if fem_metadata.get("oriented_q256_payload_sha256") != q256_hash:
            raise RuntimeError(f"{case_id}: oriented Q256 payloads differ")

        case_dir = output / case_id
        case_dir.mkdir()
        q48_path = case_dir / "q48_proxy.npy"
        np.save(q48_path, q48.astype(np.uint8))

        load = load_generator.standing_load(q48.astype(np.float32))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pino_started = time.perf_counter()
        prediction = strict_pino_sf(
            model,
            q48.astype(np.float32),
            np.asarray(load["force"], dtype=np.float32),
            np.asarray(load["fixed_mask"], dtype=np.float32),
            device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pino_elapsed = float(time.perf_counter() - pino_started)

        fem = fem_metadata["q48_result"]
        sf_pino = float(prediction["sf_pino"])
        sf_fem = float(fem["safety_factor"])
        row = {
            "case_id": case_id,
            "print_orientation": orientation,
            "q256_input_sha256": sha256_file(input_path),
            "manifest_sha256": sha256_file(manifest_path),
            "oriented_q256_payload_sha256": q256_hash,
            "q48_payload_sha256": q48_hash,
            "q48_file_sha256": sha256_file(q48_path),
            "occupied_q256": int(print_q256.sum()),
            "occupied_q48": int(q48.sum()),
            "fixed_q48": int(np.count_nonzero(load["fixed_mask"])),
            "sf_pino": sf_pino,
            "pino_safe_at_tau_1p3": bool(sf_pino >= TAU),
            "pino_elapsed_s": pino_elapsed,
            "pino_diagnostics": prediction,
            "sf_fem_q48": sf_fem,
            "fem_safe_at_tau_1p3": bool(sf_fem >= TAU),
            "fem_solver_converged": bool(fem["solver_converged"]),
            "fem_wall_time_s": float(fem["wall_time_s"]),
            "pino_minus_fem_sf": float(sf_pino - sf_fem),
            "absolute_log10_sf_error": float(
                abs(np.log10(max(sf_pino, 1.0e-12))
                    - np.log10(max(sf_fem, 1.0e-12)))
            ),
            "decision_agreement": bool((sf_pino >= TAU) == (sf_fem >= TAU)),
            "final_authority": "Q48 voxel FEM",
        }
        write_json(case_dir / "result.json", row)
        rows.append(row)

    csv_fields = (
        "case_id",
        "print_orientation",
        "occupied_q48",
        "sf_pino",
        "pino_safe_at_tau_1p3",
        "sf_fem_q48",
        "fem_safe_at_tau_1p3",
        "decision_agreement",
        "pino_elapsed_s",
        "fem_wall_time_s",
        "absolute_log10_sf_error",
    )
    with (output / "per_case.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in csv_fields})

    summary = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "completed_at_utc": utc_now(),
        "elapsed_s": float(time.time() - started),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_parameter_count": parameter_count,
        "threshold_sf": TAU,
        "n_cases": len(rows),
        "n_pino_fem_decision_agreements": int(
            sum(row["decision_agreement"] for row in rows)
        ),
        "n_fem_safe": int(sum(row["fem_safe_at_tau_1p3"] for row in rows)),
        "mean_pino_elapsed_s": float(
            np.mean([row["pino_elapsed_s"] for row in rows])
        ),
        "mean_fem_wall_time_s": float(
            np.mean([row["fem_wall_time_s"] for row in rows])
        ),
        "interpretation": (
            "Exact same-proxy audit for the printed deliveries listed in the "
            "protocol. Raw PINO is a screening/ranking prediction; Q48 voxel "
            "FEM remains the final structural authority."
        ),
        "rows": rows,
    }
    write_json(output / "summary.json", summary)
    manifest = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "outputs": {
            "summary": "summary.json",
            "per_case": "per_case.csv",
            "case_results": [f"{row['case_id']}/result.json" for row in rows],
            "q48_proxies": [f"{row['case_id']}/q48_proxy.npy" for row in rows],
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
        },
    }
    write_json(output / "RUN_MANIFEST.json", manifest)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
