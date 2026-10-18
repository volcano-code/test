#!/usr/bin/env python3
"""Run Sadekar author-code mesh and voxel baselines under the frozen TFS setup.

The launcher refuses to compete with an active PINO process.  It only produces
raw author artifacts; a separate common-renderer evaluator must audit scores.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/lin/xvxinyao/ShadowForge")
SAR = Path("/data/xvxinyao/ShadowArt-Revisited")
PY = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python")


def pino_active() -> bool:
    result = subprocess.run(["systemctl", "--user", "is-active", "shadowforge-pino-v10-v7.service"], text=True, capture_output=True)
    return result.stdout.strip() == "active"


def command(kind: str, run_id: str, targets: list[str], seed: int, mirror: bool) -> list[str]:
    args = [
        "cuda:0", run_id,
        "600" if kind == "voxel" else "2000",
        "0.01" if kind == "voxel" else "0.15",
    ]
    if kind == "mesh":
        args += ["0", "-swt", "1.6", "-l1wt", "1.6", "-mwt", "0.0", "-i2vwt", "0.0", "-ewt", "1.6", "-nwt", "0.6", "-lwt", "1.2"]
    else:
        args += ["-swt", "10.0", "-l1wt", "10.0", "-mwt", "0.0", "-th", "0.05"]
    args += ["-mr", str(bool(mirror)).lower(), "-tsf", "true", "-tsf4", "false", "-cam", "ortho", "-imsz", "256", "-zd", "1.7", "-sdlist", *targets]
    code = (
        "import os,random,runpy,numpy as np,torch;"
        f"seed={seed};os.environ['PYTHONHASHSEED']=str(seed);"
        "random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);"
        "runpy.run_path('val.py',run_name='__main__')"
    )
    return [str(PY), "-c", code, *args]


def expected(kind: str, run_id: str) -> Path:
    name = "sample_0_final_voxels.npy" if kind == "voxel" else "sample_0_output.obj"
    return SAR / f"{kind}_results" / run_id / name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/controlled_fixed3_author_tsf_20260716/benchmark_manifest.json")
    parser.add_argument("--out-tag", default="shadowforge_fixed3_author_tsf_20260716")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--keep-input-frame",
        action="store_true",
        help="Disable Sadekar's default 70%% target normalization for a declared common-frame study.",
    )
    args = parser.parse_args()
    if pino_active():
        raise RuntimeError("PINO service is active; baseline GPU runs are intentionally blocked.")
    payload = json.loads(args.manifest.read_text())
    mirror = bool(payload.get("protocol", {}).get("mr", False))
    selected = [row for row in payload["cases"] if not args.case or row["case_id"] in args.case]
    seeds = args.seed or payload["protocol"].get("seeds", [17, 29, 43])
    output_root = ROOT / "results" / args.out_tag
    run_env = os.environ.copy()
    if args.keep_input_frame:
        run_env["SHADOWFORGE_KEEP_INPUT_FRAME"] = "1"
    summary = {"manifest": str(args.manifest), "runs": []}
    for row in selected:
        for seed in seeds:
            for kind in ("voxel", "mesh"):
                run_id = f"{args.out_tag}/{kind}/{row['case_id']}/seed_{seed}"
                artifact = expected(kind, run_id)
                rec = {"case_id": row["case_id"], "kind": kind, "seed": seed, "artifact": str(artifact)}
                if artifact.exists() and not args.force:
                    rec["status"] = "reused"
                else:
                    # The upstream Sadekar launcher uses single-level os.mkdir
                    # calls and silently ignores missing-parent failures.  The
                    # controlled benchmark uses nested run identifiers, so
                    # create the complete artifact directory explicitly.
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    log = output_root / "logs" / row["case_id"] / f"{kind}_seed{seed}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    started = time.time()
                    with log.open("w") as handle:
                        subprocess.run(command(kind, run_id, row["sadekar_tfs_targets"], seed, mirror), cwd=SAR / kind, stdout=handle, stderr=subprocess.STDOUT, check=True, env=run_env)
                    rec.update({"status": "completed", "seconds": time.time() - started, "log": str(log)})
                summary["runs"].append(rec)
                output_root.mkdir(parents=True, exist_ok=True)
                (output_root / "baseline_run_summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
