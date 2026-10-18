#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from summarize import summarize


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def manifest_is_complete(case_dir: Path) -> bool:
    manifest_path = case_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = load_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return False
    if not str(manifest.get("status", "")).startswith("SUCCESS"):
        return False
    stl_paths = [
        manifest.get("display_stl"),
        manifest.get("print_stl"),
    ]
    return all(raw and Path(raw).is_file() for raw in stl_paths)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def build_command(config: dict, case: dict, case_dir: Path) -> list[str]:
    command = [
        config["python"],
        "-u",
        config["pipeline"],
        "--silhouettes",
        *case["silhouettes"],
        "--output",
        str(case_dir),
        "--branch",
        "soft-geometry",
        "--tsf",
        str(case["tsf"]),
        "--mr",
        str(case["mr"]),
        "--soft-resolution",
        "256",
        "--soft-control-res",
        "0",
        "--soft-proposal-mode",
        "proposal",
        "--soft-disable-losses",
        "volume",
        "--soft-iterations",
        "2000",
        "--soft-lr",
        "0.007",
        "--soft-save-every",
        "100",
        "--soft-seed",
        "42",
        "--soft-seed-thresholds",
        "0.05,0.10,0.15,0.20,0.25,0.35,0.45,0.55,0.65",
        "--bridge-radius",
        "2",
        "--bridge-passes",
        "48",
        "--core-resolution",
        "48",
        "--delivery-shell-voxels",
        "2",
        "--min-iou",
        "0.95",
        "--min-view-iou",
        "0.90",
        "--min-saving",
        "0.000001",
        "--max-saving",
        "0.85",
        "--min-erosion-survival",
        "0.08",
        "--selection-iou-tolerance",
        "0.01",
        "--selection-policy",
        "material",
        "--print-mm",
        "70",
        "--screen",
        "pino_ranked_fem",
        "--structural-resolution",
        "48",
        "--sf-threshold",
        "1.3",
        "--pino-checkpoint",
        config["checkpoint"],
        "--pino-ranked-policy",
        config["policy"],
        "--pino-top-k",
        "3",
        "--pino-top-k-fallback",
        "expand",
        "--allow-best-effort-fidelity-fallback",
        "--export-best-rejected",
        "--voxel-python",
        config["python"],
        "--device",
        "cuda:0",
    ]
    if case["views"] == 3:
        command.extend(
            [
                "--v256-geometry-owner",
                "--delivery-shortlist-size",
                "5",
                "--delivery-bridge-passes",
                "8",
            ]
        )
    return command


def write_summary(config_path: Path, config: dict) -> None:
    rows = summarize(config_path)
    output_root = Path(config["output_root"])
    atomic_json(
        output_root / "SUMMARY.json",
        {"batch_id": config["batch_id"], "updated_at": now(), "cases": rows},
    )


def wait_for_pids(pids: list[int], log) -> None:
    remaining = [pid for pid in pids if process_alive(pid)]
    while remaining:
        message = f"[wait] {now()} waiting for blocker PIDs: {remaining}"
        print(message, flush=True)
        log.write(message + "\n")
        log.flush()
        time.sleep(60)
        remaining = [pid for pid in remaining if process_alive(pid)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("cases.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--wait-for-pid", action="append", type=int, default=[])
    args = parser.parse_args()

    config = load_json(args.config)
    output_root = Path(config["output_root"])
    log_root = Path(config["log_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    lock_path = log_root / "RUNNER.lock"
    lock_stream = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another batch runner owns {lock_path}", file=sys.stderr)
        return 2

    master_log_path = log_root / "MASTER.log"
    with master_log_path.open("a", encoding="utf-8", buffering=1) as master_log:
        master_log.write(f"[runner-start] {now()} pid={os.getpid()}\n")
        atomic_json(
            log_root / "RUNNER.json",
            {"pid": os.getpid(), "status": "RUNNING", "started_at": now()},
        )
        wait_for_pids(args.wait_for_pid, master_log)

        selected_cases = config["cases"]
        if args.only:
            wanted = set(args.only)
            selected_cases = [
                case for case in selected_cases if case["id"] in wanted
            ]
        if args.limit is not None:
            selected_cases = selected_cases[: args.limit]

        if args.dry_run:
            for case in selected_cases:
                command = build_command(
                    config,
                    case,
                    output_root / case["id"],
                )
                print(f"[dry-run] {case['id']}")
                print(shlex.join(command))
            atomic_json(
                log_root / "RUNNER.json",
                {"pid": os.getpid(), "status": "DRY_RUN_COMPLETE", "ended_at": now()},
            )
            return 0

        failures = 0
        for index, case in enumerate(selected_cases, start=1):
            case_dir = output_root / case["id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            status_path = case_dir / "STATUS.json"
            if manifest_is_complete(case_dir):
                atomic_json(
                    status_path,
                    {
                        "case": case["id"],
                        "status": "SKIPPED_COMPLETE",
                        "updated_at": now(),
                    },
                )
                master_log.write(f"[skip] {now()} {case['id']}\n")
                write_summary(args.config, config)
                continue

            free_gib = shutil.disk_usage(output_root.parent).free / (1024**3)
            if free_gib < config["minimum_free_gb"]:
                payload = {
                    "case": case["id"],
                    "status": "DISK_BLOCKED",
                    "free_gib": free_gib,
                    "minimum_free_gb": config["minimum_free_gb"],
                    "updated_at": now(),
                }
                atomic_json(status_path, payload)
                atomic_json(
                    log_root / "RUNNER.json",
                    {
                        "pid": os.getpid(),
                        "status": "DISK_BLOCKED",
                        "case": case["id"],
                        "free_gib": free_gib,
                        "updated_at": now(),
                    },
                )
                master_log.write(
                    f"[disk-blocked] {now()} {case['id']} "
                    f"free={free_gib:.2f}GiB\n"
                )
                write_summary(args.config, config)
                return 3

            command = build_command(config, case, case_dir)
            atomic_json(
                case_dir / "COMMAND.json",
                {
                    "case": case,
                    "command": command,
                    "shell": shlex.join(command),
                    "created_at": now(),
                },
            )
            started_at = now()
            atomic_json(
                status_path,
                {
                    "case": case["id"],
                    "status": "RUNNING",
                    "index": index,
                    "total": len(selected_cases),
                    "started_at": started_at,
                    "free_gib_at_start": free_gib,
                },
            )
            master_log.write(
                f"[start] {started_at} {index}/{len(selected_cases)} "
                f"{case['id']} views={case['views']} "
                f"category={case['category']}\n"
            )

            with (case_dir / "RUN.log").open(
                "a",
                encoding="utf-8",
                buffering=1,
            ) as case_log:
                case_log.write(
                    f"\n[batch-start] {started_at}\n"
                    f"[command] {shlex.join(command)}\n"
                )
                completed = subprocess.run(
                    command,
                    cwd=config["root"],
                    stdout=case_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    start_new_session=False,
                )

            ended_at = now()
            manifest_complete = manifest_is_complete(case_dir)
            case_status = (
                "SUCCESS"
                if completed.returncode == 0 and manifest_complete
                else "FAILED"
            )
            if case_status == "FAILED":
                failures += 1
            atomic_json(
                status_path,
                {
                    "case": case["id"],
                    "status": case_status,
                    "returncode": completed.returncode,
                    "manifest_complete": manifest_complete,
                    "started_at": started_at,
                    "ended_at": ended_at,
                },
            )
            master_log.write(
                f"[{case_status.lower()}] {ended_at} {case['id']} "
                f"returncode={completed.returncode} "
                f"manifest_complete={manifest_complete}\n"
            )
            write_summary(args.config, config)

        final_status = "COMPLETE" if failures == 0 else "COMPLETE_WITH_FAILURES"
        atomic_json(
            log_root / "RUNNER.json",
            {
                "pid": os.getpid(),
                "status": final_status,
                "failures": failures,
                "ended_at": now(),
            },
        )
        master_log.write(
            f"[runner-end] {now()} status={final_status} "
            f"failures={failures}\n"
        )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(143))
    raise SystemExit(main())
