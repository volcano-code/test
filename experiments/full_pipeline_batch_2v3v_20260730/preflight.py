#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageStat


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict:
    with Image.open(path) as image:
        gray = image.convert("L")
        histogram = gray.histogram()
        total = max(1, sum(histogram))
        foreground = sum(histogram[128:]) / total
        extrema = gray.getextrema()
        mean = ImageStat.Stat(gray).mean[0]
        return {
            "path": str(path),
            "size": list(image.size),
            "mode": image.mode,
            "gray_extrema": list(extrema),
            "gray_mean": mean,
            "foreground_fraction": foreground,
            "sha256": sha256(path),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("cases.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = read_config(args.config)
    report_path = args.output or Path(config["output_root"]) / "PREFLIGHT.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    required = {
        key: Path(config[key])
        for key in ("python", "pipeline", "checkpoint", "policy")
    }
    errors: list[str] = []
    warnings: list[str] = []
    required_report = {}
    for name, path in required.items():
        exists = path.is_file()
        required_report[name] = {"path": str(path), "exists": exists}
        if not exists:
            errors.append(f"missing required file: {name}={path}")

    case_reports = []
    hashes: dict[str, list[str]] = {}
    for case in config["cases"]:
        silhouettes = []
        if case["views"] != len(case["silhouettes"]):
            errors.append(
                f"{case['id']}: views={case['views']} but "
                f"{len(case['silhouettes'])} silhouettes were declared"
            )
        if case["views"] == 2 and (case["tsf"], case["mr"]) != (0, 0):
            warnings.append(
                f"{case['id']}: two-view case will not use the mature "
                "two-wall dispatch because tsf/mr is not 0/0"
            )
        for raw_path in case["silhouettes"]:
            path = Path(raw_path)
            if not path.is_file():
                errors.append(f"{case['id']}: missing silhouette {path}")
                silhouettes.append({"path": str(path), "exists": False})
                continue
            info = inspect_image(path)
            info["exists"] = True
            silhouettes.append(info)
            hashes.setdefault(info["sha256"], []).append(str(path))
            fraction = info["foreground_fraction"]
            if not 0.002 <= fraction <= 0.75:
                warnings.append(
                    f"{case['id']}: unusual foreground fraction "
                    f"{fraction:.5f} for {path}"
                )
        case_reports.append(
            {
                "id": case["id"],
                "views": case["views"],
                "category": case["category"],
                "tsf": case["tsf"],
                "mr": case["mr"],
                "silhouettes": silhouettes,
            }
        )

    duplicates = [
        {"sha256": digest, "paths": paths}
        for digest, paths in hashes.items()
        if len(set(paths)) > 1
    ]
    disk = shutil.disk_usage(Path(config["output_root"]).parent)
    free_gb = disk.free / (1024**3)
    if free_gb < config["minimum_free_gb"]:
        errors.append(
            f"free disk {free_gb:.1f} GiB is below the configured "
            f"{config['minimum_free_gb']} GiB guard"
        )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "batch_id": config["batch_id"],
        "required": required_report,
        "disk_free_gib": free_gb,
        "minimum_free_gb": config["minimum_free_gb"],
        "cases": case_reports,
        "duplicates": duplicates,
        "warnings": warnings,
        "errors": errors,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
