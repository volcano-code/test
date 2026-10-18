#!/usr/bin/env python3
"""Fail-closed summaries and paired inference for P-CRAFT soft-loss-v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from experiments.run_soft_loss_v2 import (
    APPROVED_STATUS,
    BENCHMARK,
    OUTPUT_ROOT,
    atomic_json,
    is_completed_run,
    load_frozen_config,
)


PRIMARY_THRESHOLD = "0.05"
METRIC_FIELDS = (
    "mean_binary_iou", "worst_view_iou", "fpr", "recall", "dice",
    "ssim", "boundary_f1", "continuous_iou", "soft_binary_iou_gap",
    "final_loss", "gradient_norm",
)
BASE_FORMAL_VARIANTS = (
    "legacy_full", "l12_plain", "l12_bal", "new_full",
    "new_full_no_proxy", "new_full_no_gate", "new_full_no_boundary",
    "new_full_mean", "new_full_no_l1", "new_full_no_l2",
)
ENDPOINTS = {
    "proxy": ("new_full_no_proxy", "worst_view_iou", "higher"),
    "gate": ("new_full_no_gate", "fpr", "lower"),
    "boundary": ("new_full_no_boundary", "boundary_f1", "higher"),
    "mean_view": ("new_full_mean", "worst_view_iou", "higher"),
    "l1": ("new_full_no_l1", "recall", "higher"),
    "l2": ("new_full_no_l2", "mean_binary_iou", "higher"),
    "ssim": ("new_full_no_ssim", "ssim", "higher"),
}
STAGE_RUN_COUNTS = {
    "smoke128": 8,
    "dev128_screen": 27,
    "dev128_confirm": 9,
    "transfer256": 9,
}
STAGE_VARIANTS = {
    "smoke128": {
        "legacy_full", "l12_plain", "l12_bal", "new_full",
        "new_full_no_proxy", "new_full_no_gate", "new_full_no_boundary",
        "new_full_mean",
    },
    "dev128_confirm": {"l12_bal", "new_full_no_ssim", "new_full_ssim"},
    "transfer256": {"l12_bal", "new_full", "new_full_ssim"},
}


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_registered_input_hashes(registration: Mapping[str, Any]) -> None:
    """Rehash each current input so in-place target replacement cannot pass."""

    entries = registration.get("input_sha256")
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("registered run must contain three input SHA256 records")
    registered_paths = []
    for entry in entries:
        if not isinstance(entry, Mapping) or not entry.get("path") or not entry.get("sha256"):
            raise ValueError("malformed registered input SHA256 record")
        path = Path(str(entry["path"]))
        try:
            current = _file_sha256(path)
        except FileNotFoundError as exc:
            raise ValueError(f"registered input is missing: {path}") from exc
        if current != entry["sha256"]:
            raise ValueError(f"input SHA256 changed after registration: {path}")
        registered_paths.append(str(path))
    targets = registration.get("targets")
    if targets is not None and list(targets) != registered_paths:
        raise ValueError("input SHA256 records do not match the registered target order")


def validate_formal_provenance(
    master: Mapping[str, Any],
    stage: Mapping[str, Any],
    benchmark_path: Path,
) -> None:
    """Require master, stage, and current frozen benchmark bytes to agree."""

    benchmark_path = Path(benchmark_path).resolve()
    try:
        current = _file_sha256(benchmark_path)
    except FileNotFoundError as exc:
        raise ValueError(f"formal benchmark is missing: {benchmark_path}") from exc
    master_path = Path(str(master.get("benchmark_path", ""))).resolve()
    stage_path = Path(str(stage.get("benchmark_path", ""))).resolve()
    if master_path != benchmark_path or stage_path != benchmark_path:
        raise ValueError("master/stage benchmark paths differ from the formal benchmark")
    if master.get("benchmark_sha256") != current or stage.get("benchmark_sha256") != current:
        raise ValueError("benchmark SHA256 differs from master/stage registration")


def _dig(mapping: Mapping[str, Any], *paths: Sequence[str]) -> Any:
    for path in paths:
        value: Any = mapping
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            return value
    raise KeyError(" / ".join(".".join(path) for path in paths))


def extract_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the trainer's final/aggregate schema in one audited helper."""

    aggregate = _dig(
        metrics,
        ("thresholds", PRIMARY_THRESHOLD, "aggregate"),
        ("aggregate",),
        ("final", "aggregate"),
    )
    if not isinstance(aggregate, Mapping):
        raise ValueError("aggregate metrics is not a mapping")

    def number(*names: str) -> float:
        for name in names:
            if name in aggregate and aggregate[name] is not None:
                value = float(aggregate[name])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite metric {name}")
                return value
        raise ValueError(f"missing aggregate metric: {names}")

    normalized = {
        "mean_binary_iou": number("mean_binary_iou", "binary_iou"),
        "worst_view_iou": number("worst_view_binary_iou", "worst_view_iou"),
        "fpr": number("mean_fpr", "fpr"),
        "recall": number("mean_recall", "recall"),
        "dice": number("mean_dice", "dice"),
        "ssim": number("mean_ssim", "ssim"),
        "boundary_f1": number("mean_boundary_f1_2px", "boundary_f1_2px", "boundary_f1"),
        "continuous_iou": number("mean_raw_continuous_iou", "mean_continuous_iou", "continuous_iou"),
        "soft_binary_iou_gap": number("mean_soft_binary_iou_gap", "soft_binary_iou_gap"),
        "final_loss": float(metrics.get("final_loss", metrics.get("loss", float("nan")))),
        "gradient_norm": float(metrics.get("final_gradient_norm_pre_clip", metrics.get("gradient_norm", float("nan")))),
    }
    thresholds = metrics.get("thresholds", {})
    normalized["threshold_stability"] = {
        key: float(value["aggregate"]["mean_binary_iou"])
        for key, value in thresholds.items()
        if key in {"0.04", "0.05", "0.06"}
        and isinstance(value, Mapping)
        and isinstance(value.get("aggregate"), Mapping)
        and value["aggregate"].get("mean_binary_iou") is not None
    }
    if set(normalized["threshold_stability"]) != {"0.04", "0.05", "0.06"}:
        raise ValueError("metrics must contain density thresholds 0.04, 0.05, and 0.06")
    return normalized


def _stage_steps(stage: str) -> int:
    return {
        "smoke128": 200,
        "dev128_screen": 1000,
        "dev128_confirm": 2000,
        "transfer256": 2000,
        "formal256": 2000,
    }[stage]


def validate_stage_registration(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete task-by-variant matrix before reading any result."""

    stage = str(manifest.get("stage", ""))
    if (
        manifest.get("schema") != "pcraft_soft_loss_v2_stage_manifest_v1"
        or manifest.get("soft_only") is not True
    ):
        raise ValueError("invalid soft-loss-v2 stage registration")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not all(isinstance(run, Mapping) for run in runs):
        raise ValueError("stage registration has no run matrix")
    expected_count = STAGE_RUN_COUNTS.get(stage)
    if expected_count is not None and len(runs) != expected_count:
        raise ValueError(f"{stage} must register exactly {expected_count} runs")
    run_dirs = [str(run.get("run_dir", "")) for run in runs]
    keys = [(str(run.get("task", "")), str(run.get("variant", ""))) for run in runs]
    if "" in run_dirs or len(run_dirs) != len(set(run_dirs)) or len(keys) != len(set(keys)):
        raise ValueError(f"{stage} registration contains missing or duplicate runs")
    variants = {variant for _, variant in keys}
    if stage in STAGE_VARIANTS and variants != STAGE_VARIANTS[stage]:
        raise ValueError(f"{stage} registered variants differ from the preregistered matrix")
    if stage == "dev128_screen":
        if "l12_bal" not in variants or len(variants) != 9:
            raise ValueError("dev128_screen must register l12_bal plus eight candidates")
    if stage == "formal256":
        allowed = [set(BASE_FORMAL_VARIANTS), set(BASE_FORMAL_VARIANTS) | {"new_full_no_ssim"}]
        if variants not in allowed:
            raise ValueError("formal256 registered variants differ from the frozen formal matrix")
        expected_tasks = {f"connected_{index:02d}" for index in range(1, 11)}
        for variant in variants:
            tasks = {task for task, registered_variant in keys if registered_variant == variant}
            if tasks != expected_tasks:
                raise ValueError(f"formal256 variant {variant} lacks its ten registered tasks")
        if len(runs) != 10 * len(variants):
            raise ValueError("formal256 registration is not a complete task-by-variant matrix")
    else:
        tasks_by_variant = {
            variant: {task for task, registered_variant in keys if registered_variant == variant}
            for variant in variants
        }
        if any(len(tasks) * len(variants) != len(runs) for tasks in tasks_by_variant.values()):
            raise ValueError(f"{stage} registration is not a complete task-by-variant matrix")
    return {
        "stage": stage,
        "run_count": len(runs),
        "required_variants": sorted(variants),
        "registered_run_dirs": run_dirs,
    }


def collect_stage_rows(root: Path, stage: str) -> list[dict[str, Any]]:
    """Collect one exact stage depth; never recursively mix development runs."""

    steps = _stage_steps(stage)
    stage_root = Path(root) / stage / f"steps_{steps}"
    stage_manifest = _read_json(Path(root) / stage / "stage_manifest.json")
    matrix = validate_stage_registration(stage_manifest)
    if matrix["stage"] != stage:
        raise ValueError(f"mismatched stage registration for {stage}")
    if (
        stage_manifest.get("determinism") != "fixed-seed single-run best-effort"
        or stage_manifest.get("bitwise_reproducible") is not False
        or "grid_sampler_3d backward" not in str(stage_manifest.get("nondeterminism_reason", ""))
        or not stage_manifest.get("benchmark_sha256")
    ):
        raise ValueError(f"{stage} registration overstates or omits determinism provenance")
    stage_benchmark = Path(str(stage_manifest.get("benchmark_path", "")))
    try:
        current_benchmark_hash = _file_sha256(stage_benchmark)
    except FileNotFoundError as exc:
        raise ValueError(f"registered stage benchmark is missing: {stage_benchmark}") from exc
    if current_benchmark_hash != stage_manifest.get("benchmark_sha256"):
        raise ValueError(f"registered benchmark SHA256 changed before summarizing {stage}")
    frozen_path_text = stage_manifest.get("frozen_config_path")
    if frozen_path_text is not None:
        frozen_path = Path(str(frozen_path_text))
        try:
            current_frozen_sha = _file_sha256(frozen_path)
        except FileNotFoundError as exc:
            raise ValueError(f"registered frozen config is missing: {frozen_path}") from exc
        if current_frozen_sha != stage_manifest.get("frozen_config_file_sha256"):
            raise ValueError(f"frozen config bytes changed before summarizing {stage}")
        expected_source = "dev128_confirm" if stage == "transfer256" else "transfer256"
        frozen = load_frozen_config(frozen_path, expected_source=expected_source)
        if frozen.get("config_hash") != stage_manifest.get("frozen_config_hash"):
            raise ValueError(f"frozen config semantic hash changed before summarizing {stage}")
    registered_runs = {
        str(Path(record["run_dir"]).resolve()): record
        for record in stage_manifest.get("runs", [])
        if isinstance(record, Mapping) and "run_dir" in record
    }
    actual_run_dirs = {
        str(path.parent.resolve())
        for path in stage_root.glob("*/*/seed_17/metrics.json")
    }
    if actual_run_dirs != set(matrix["registered_run_dirs"]):
        missing = sorted(set(matrix["registered_run_dirs"]) - actual_run_dirs)
        extra = sorted(actual_run_dirs - set(matrix["registered_run_dirs"]))
        raise ValueError(f"stage result matrix is incomplete or unregistered; missing={missing}, extra={extra}")
    rows: list[dict[str, Any]] = []
    # Iterate the immutable registration, not whatever happens to be on disk.
    for run_dir_text in sorted(matrix["registered_run_dirs"]):
        run_dir = Path(run_dir_text)
        registration = registered_runs[run_dir_text]
        validate_registered_input_hashes(registration)
        expected_digest = str(registration.get("expected_config_hash", ""))
        if not expected_digest or not is_completed_run(run_dir, expected_digest):
            raise ValueError(f"registered run is incomplete or incompatible: {run_dir}")
        metrics_path = run_dir / "metrics.json"
        task, variant = run_dir.parents[1].name, run_dir.parent.name
        metrics = _read_json(metrics_path)
        command = _read_json(run_dir / "COMMAND.json")
        status = _read_json(run_dir / "STATUS.json")
        manifest = _read_json(run_dir / "run_manifest.json")
        digests = {metrics.get("config_hash"), command.get("config_hash"), status.get("config_hash"), manifest.get("config_hash")}
        if len(digests) != 1 or None in digests:
            raise ValueError(f"config hash mismatch: {run_dir}")
        digest = next(iter(digests))
        if registration.get("expected_config_hash") != digest:
            raise ValueError(f"run differs from its registered configuration: {run_dir}")
        if status.get("state") != "completed":
            raise ValueError(f"run is not completed: {run_dir}")
        config = command.get("resolved_config", manifest.get("config"))
        if not isinstance(config, Mapping):
            raise ValueError(f"missing resolved config: {run_dir}")
        scope = manifest.get("scope", {})
        determinism = manifest.get("determinism", {})
        if (
            determinism.get("seed") != 17
            or determinism.get("mode") != "best_effort"
            or determinism.get("bitwise_reproducibility_guaranteed") is not False
        ):
            raise ValueError(f"run determinism disclosure is missing or overstated: {run_dir}")
        row = {
            "stage": stage,
            "task": task,
            "variant": variant,
            "run_dir": str(run_dir),
            "config_hash": digest,
            "resolution": int(config.get("res", -1)),
            "export_resolution": int(config.get("export_res", -1)),
            "control_resolution": int(config.get("control_res", -1)),
            "steps": int(metrics.get("steps", config.get("steps", steps))),
            "seed": int(config.get("seed", -1)),
            "final_step_exported": bool(metrics.get("final_step_exported")),
            "soft_only": bool(scope.get("soft_only")),
            "paper_eligible": registration.get("paper_eligible") is True,
            "frozen_status": registration.get("frozen_status"),
            "input_order": registration.get("input_order"),
            "profile": str(metrics.get("profile", config.get("profile", ""))),
        }
        row.update(extract_metrics(metrics))
        rows.append(row)
    if not rows:
        raise ValueError(f"no completed results at exact stage path: {stage_root}")
    if len(rows) != matrix["run_count"]:
        raise ValueError(f"summarized row count differs from registered matrix for {stage}")
    return rows


def validate_formal_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_variants: Sequence[str] = BASE_FORMAL_VARIANTS,
) -> None:
    """Fail closed on any deviation from the ten-task frozen formal design."""

    expected_tasks = {f"connected_{index:02d}" for index in range(1, 11)}
    variants_present = {str(row.get("variant")) for row in rows}
    if variants_present != set(required_variants):
        raise ValueError(
            f"formal variants differ from registration: expected {sorted(required_variants)}, "
            f"found {sorted(variants_present)}"
        )
    for row in rows:
        fixed = (
            row.get("stage") == "formal256"
            and int(row.get("resolution", -1)) == 256
            and int(row.get("export_resolution", -1)) == 256
            and int(row.get("control_resolution", -1)) == 0
            and int(row.get("steps", -1)) == 2000
            and int(row.get("seed", -1)) == 17
            and row.get("final_step_exported") is True
            and row.get("soft_only") is True
            and row.get("frozen_status") == APPROVED_STATUS
            and row.get("input_order") == "xyz"
        )
        if not fixed:
            raise ValueError(f"formal protocol violation: {row}")
    for variant in required_variants:
        matching = [row for row in rows if row.get("variant") == variant]
        tasks = [str(row.get("task")) for row in matching]
        if len(tasks) != 10 or set(tasks) != expected_tasks or len(tasks) != len(set(tasks)):
            raise ValueError(f"formal variant {variant} does not have exactly ten unique tasks")


def validate_frozen_benchmark(path: Path) -> None:
    manifest = _read_json(path)
    if manifest.get("status") != APPROVED_STATUS:
        raise ValueError("formal benchmark is not approved and frozen")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 10:
        raise ValueError("formal benchmark does not have ten cases")
    for case in cases:
        paths = case.get("target_paths_xyz") if isinstance(case, Mapping) else None
        if not isinstance(paths, list) or len(paths) != 3:
            raise ValueError("formal benchmark lacks target_paths_xyz")
        if not all(f"view_{axis}" in Path(str(value)).stem.lower() for value, axis in zip(paths, "xyz")):
            raise ValueError("formal benchmark target_paths_xyz is not x,y,z ordered")


def transfer_gate(
    candidate_by_task: Mapping[str, Mapping[str, float]],
    baseline_by_task: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    tasks = sorted(set(candidate_by_task) & set(baseline_by_task))
    if not tasks or set(tasks) != set(candidate_by_task) or set(tasks) != set(baseline_by_task):
        raise ValueError("candidate and l12_bal must contain identical task sets")
    candidate_mean_iou = float(np.mean([candidate_by_task[t]["mean_binary_iou"] for t in tasks]))
    baseline_mean_iou = float(np.mean([baseline_by_task[t]["mean_binary_iou"] for t in tasks]))
    recall_deltas = [candidate_by_task[t]["recall"] - baseline_by_task[t]["recall"] for t in tasks]
    worst_gain = float(np.mean([candidate_by_task[t]["worst_view_iou"] - baseline_by_task[t]["worst_view_iou"] for t in tasks]))
    candidate_fpr = float(np.mean([candidate_by_task[t]["fpr"] for t in tasks]))
    baseline_fpr = float(np.mean([baseline_by_task[t]["fpr"] for t in tasks]))
    relative_fpr_reduction = (baseline_fpr - candidate_fpr) / baseline_fpr if baseline_fpr > 0 else 0.0
    boundary_gain = float(np.mean([candidate_by_task[t]["boundary_f1"] - baseline_by_task[t]["boundary_f1"] for t in tasks]))
    prerequisites = {
        "mean_iou_noninferior": candidate_mean_iou >= baseline_mean_iou - 0.001,
        "all_task_recall_noninferior": min(recall_deltas) >= -0.005,
    }
    improvements = {
        "worst_view_iou": worst_gain,
        "relative_fpr_reduction": relative_fpr_reduction,
        "boundary_f1": boundary_gain,
    }
    alternative = worst_gain >= 0.002 or relative_fpr_reduction >= 0.10 or boundary_gain >= 0.005
    return {
        "passed": bool(all(prerequisites.values()) and alternative),
        "tasks": tasks,
        "prerequisites": prerequisites,
        "primary_improvement": improvements,
        "mean_iou_delta": candidate_mean_iou - baseline_mean_iou,
        "worst_task_recall_delta": min(recall_deltas),
    }


def paired_bootstrap(
    differences: Sequence[float], *, draws: int = 100_000, seed: int = 20260826,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("paired differences must be a finite non-empty vector")
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    generator = np.random.default_rng(seed)
    sampled_means = np.empty(draws, dtype=np.float64)
    chunk = min(draws, 20_000)
    for start in range(0, draws, chunk):
        count = min(chunk, draws - start)
        indices = generator.integers(0, values.size, size=(count, values.size))
        sampled_means[start : start + count] = values[indices].mean(axis=1)
    low, high = np.quantile(sampled_means, [0.025, 0.975])
    return {
        "n": int(values.size), "draws": int(draws),
        "mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high),
    }


def exact_sign_flip_test(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or values.size > 20 or not np.isfinite(values).all():
        raise ValueError("exact sign-flip requires 1--20 finite paired differences")
    observed = abs(float(values.mean()))
    extreme = 0
    total = 1 << values.size
    for bits in range(total):
        signs = np.fromiter((1.0 if bits & (1 << i) else -1.0 for i in range(values.size)), dtype=np.float64)
        if abs(float((values * signs).mean())) >= observed - 1e-15:
            extreme += 1
    return float(extreme / total)


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(((name, float(value)) for name, value in p_values.items()), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def assess_component(
    *, component: str, mean_difference: float, ci_low: float, ci_high: float,
    holm_p: float, direction: str, safety_ok: bool,
) -> dict[str, Any]:
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be higher or lower")
    # Differences passed here are normalized so positive always means full is better.
    direction_ok = mean_difference > 0 and ci_low > 0
    return {
        "component": component,
        "direction": direction,
        "mean_difference_full_better": float(mean_difference),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "holm_p": float(holm_p),
        "safety_ok": bool(safety_ok),
        "effective": bool(direction_ok and holm_p < 0.05 and safety_ok),
    }


def _by_variant_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        variant, task = str(row["variant"]), str(row["task"])
        if task in result[variant]:
            raise ValueError(f"duplicate task {task} for variant {variant}")
        result[variant][task] = row
    return dict(result)


def paired_inference(rows: Sequence[Mapping[str, Any]], *, draws: int = 100_000) -> dict[str, Any]:
    by_variant = _by_variant_task(rows)
    full = by_variant.get("new_full")
    if not full:
        raise ValueError("new_full rows are required for paired inference")
    raw: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}
    for component, (ablation_variant, metric, direction) in ENDPOINTS.items():
        if ablation_variant not in by_variant:
            if component == "ssim":
                continue
            raise ValueError(f"missing preregistered ablation {ablation_variant}")
        ablation = by_variant[ablation_variant]
        tasks = sorted(set(full) & set(ablation))
        if set(tasks) != set(full) or set(tasks) != set(ablation):
            raise ValueError(f"unpaired tasks for {component}")
        if direction == "higher":
            differences = [float(full[t][metric]) - float(ablation[t][metric]) for t in tasks]
        else:
            differences = [float(ablation[t][metric]) - float(full[t][metric]) for t in tasks]
        bootstrap = paired_bootstrap(differences, draws=draws, seed=20260826)
        p_value = exact_sign_flip_test(differences)
        safety_ok = True
        if component == "gate":
            safety_ok = min(float(full[t]["recall"]) - float(ablation[t]["recall"]) for t in tasks) >= -0.005
        raw[component] = {
            "component": component, "ablation_variant": ablation_variant,
            "endpoint": metric, "direction": direction, "tasks": tasks,
            "differences_full_better": differences, "bootstrap": bootstrap,
            "exact_sign_flip_p": p_value, "safety_ok": safety_ok,
        }
        p_values[component] = p_value
    adjusted = holm_adjust(p_values)
    for component, record in raw.items():
        boot = record["bootstrap"]
        record["holm_p"] = adjusted[component]
        record["effectiveness"] = assess_component(
            component=component, mean_difference=boot["mean"],
            ci_low=boot["ci_low"], ci_high=boot["ci_high"],
            holm_p=adjusted[component], direction=record["direction"],
            safety_ok=record["safety_ok"],
        )
    return {"schema": "pcraft_soft_loss_v2_paired_v1", "draws": draws, "results": raw}


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    result = {}
    for variant, values in sorted(grouped.items()):
        result[variant] = {
            "n_tasks": len(values),
            "metrics": {
                metric: {
                    "mean": float(np.mean([float(row[metric]) for row in values])),
                    "std": float(np.std([float(row[metric]) for row in values], ddof=1)) if len(values) > 1 else 0.0,
                }
                for metric in METRIC_FIELDS
                if all(math.isfinite(float(row[metric])) for row in values)
            },
        }
    return result


def _atomic_tsv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def export_paper_table(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    required_variants: Sequence[str] = BASE_FORMAL_VARIANTS,
) -> None:
    if not rows or any(
        row.get("stage") != "formal256"
        or row.get("paper_eligible") is not True
        or int(row.get("resolution", -1)) != 256
        for row in rows
    ):
        raise ValueError("paper tables may contain only formal256 paper-eligible rows")
    validate_formal_rows(rows, required_variants=required_variants)
    fields = ("task", "variant", *METRIC_FIELDS)
    _atomic_tsv(Path(path), rows, fields)


def _write_paired_csv(path: Path, paired: Mapping[str, Any]) -> None:
    rows = []
    for component, record in paired["results"].items():
        effect = record["effectiveness"]
        rows.append(
            {
                "component": component, "variant": record["ablation_variant"],
                "endpoint": record["endpoint"], "mean_difference_full_better": effect["mean_difference_full_better"],
                "ci_low": effect["ci_low"], "ci_high": effect["ci_high"],
                "exact_sign_flip_p": record["exact_sign_flip_p"], "holm_p": effect["holm_p"],
                "safety_ok": effect["safety_ok"], "effective": effect["effective"],
            }
        )
    _atomic_tsv(path, rows, tuple(rows[0]) if rows else ("component",))


def _task_metric_map(rows: Sequence[Mapping[str, Any]], variant: str) -> dict[str, Mapping[str, float]]:
    return {str(row["task"]): row for row in rows if row["variant"] == variant}


def write_dev_selection(rows: Sequence[Mapping[str, Any]], output: Path) -> dict[str, Any]:
    baseline = _task_metric_map(rows, "l12_bal")
    candidates = sorted({str(row["variant"]) for row in rows if str(row["variant"]).startswith("new_full_g")})

    def parsed_weights(variant: str) -> dict[str, float]:
        tokens = variant.split("_")

        def parsed(prefix: str) -> float:
            token = next((value for value in tokens if value.startswith(prefix)), None)
            if token is None:
                raise ValueError(f"screened variant is missing {prefix}: {variant}")
            return float(token[len(prefix):].replace("p", "."))

        return {
            "lambda_g": parsed("g"),
            "lambda_b": parsed("b"),
            "late_ssim_weight": parsed("ssim"),
        }

    records = []
    for variant in candidates:
        candidate = _task_metric_map(rows, variant)
        report = transfer_gate(candidate, baseline)
        improvement = report["primary_improvement"]
        # Each term is normalized by its preregistered meaningful-improvement
        # threshold.  A score >= 1 means at least one improvement gate passed.
        score = max(
            float(improvement["worst_view_iou"]) / 0.002,
            float(improvement["relative_fpr_reduction"]) / 0.10,
            float(improvement["boundary_f1"]) / 0.005,
        )
        weights = parsed_weights(variant)
        records.append(
            {
                "variant": variant,
                "report": report,
                "score": float(score),
                "weights": weights,
                "noninferior": bool(all(report["prerequisites"].values())),
            }
        )

    passed = [record for record in records if record["report"]["passed"]]
    noninferior = [record for record in records if record["noninferior"]]
    pool = passed if passed else noninferior

    def select_record(available: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        best_score = max(float(record["score"]) for record in available)
        # Within numerical screening tolerance, prefer the simpler/lower-weight
        # auxiliary objective, then a fixed variant-name ordering.
        near_best = [
            record for record in available
            if best_score - float(record["score"]) <= 1e-3 + 1e-12
        ]
        return min(
            near_best,
            key=lambda record: (
                float(record["weights"]["lambda_g"])
                + float(record["weights"]["lambda_b"]),
                str(record["variant"]),
            ),
        )

    payload: dict[str, Any] = {
        "schema": "pcraft_soft_loss_v2_dev128_selection_v1",
        "paper_eligible": False,
        "screen_improvement_gate_passed": bool(passed),
        "screen_score_definition": (
            "max(worst_view_gain/0.002, relative_fpr_reduction/0.10, "
            "boundary_f1_gain/0.005)"
        ),
        "screen_score_tolerance": 1e-3,
        "candidate_reports": {
            str(record["variant"]): {
                "normalized_improvement_score": record["score"],
                "noninferior": record["noninferior"],
                "transfer_gate": record["report"],
            }
            for record in records
        },
    }
    if pool:
        selected = select_record(pool)
        variant = str(selected["variant"])
        report = selected["report"]
        weights = selected["weights"]
        ordered_pool = sorted(
            pool,
            key=lambda record: (
                -float(record["score"]),
                float(record["weights"]["lambda_g"])
                + float(record["weights"]["lambda_b"]),
                str(record["variant"]),
            ),
        )
        payload.update(
            {
                "status": (
                    "SELECTED_FOR_CONFIRMATION"
                    if passed
                    else "SELECTED_NONINFERIOR_FOR_CONFIRMATION"
                ),
                "eligible_candidates": [record["variant"] for record in ordered_pool],
                "selected_candidate": {
                    "variant": variant,
                    "lambda_g": weights["lambda_g"],
                    "lambda_b": weights["lambda_b"],
                    # Recorded for provenance only: confirm still runs both 0/.05.
                    "late_ssim_weight": weights["late_ssim_weight"],
                },
                "selection_gate": report,
                "screen_normalized_improvement_score": selected["score"],
                "ssim_decision_deferred_to_confirm": True,
            }
        )
    else:
        payload["status"] = "NO_ELIGIBLE_CANDIDATE"
        payload["eligible_candidates"] = []
    atomic_json(output, payload)
    return payload


def choose_transfer_configuration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the 256 migration gate to both registered SSIM alternatives."""

    baseline = _task_metric_map(rows, "l12_bal")
    reports = {
        variant: transfer_gate(_task_metric_map(rows, variant), baseline)
        for variant in ("new_full", "new_full_ssim")
    }
    eligible = []
    for variant, report in reports.items():
        if report["passed"]:
            improvement = report["primary_improvement"]
            score = (
                float(improvement["worst_view_iou"]),
                float(improvement["relative_fpr_reduction"]),
                float(improvement["boundary_f1"]),
                float(report["mean_iou_delta"]),
            )
            eligible.append((score, variant))
    if not eligible:
        return {
            "status": "NO_TRANSFER_CANDIDATE_PASSED",
            "passed": False,
            "candidate_reports": reports,
        }
    if {variant for _, variant in eligible} == {"new_full", "new_full_ssim"}:
        by_variant = _by_variant_task(rows)
        comparison_metrics = (
            "mean_binary_iou", "recall", "worst_view_iou", "fpr", "boundary_f1"
        )
        summaries = {
            variant: {
                metric: float(
                    np.mean([float(row[metric]) for row in by_variant[variant].values()])
                )
                for metric in comparison_metrics
            }
            for variant in ("new_full", "new_full_ssim")
        }
        tolerance = 1e-6
        within_tolerance = all(
            abs(summaries["new_full"][metric] - summaries["new_full_ssim"][metric])
            <= tolerance
            for metric in comparison_metrics
        )
        selected = "new_full" if within_tolerance else max(eligible)[1]
    else:
        _, selected = max(eligible)
    return {
        "status": "TRANSFER_PASSED",
        "passed": True,
        "selected_variant": selected,
        "late_ssim_weight": 0.05 if selected == "new_full_ssim" else 0.0,
        "candidate_reports": reports,
        "ssim_tie_tolerance": 1e-6,
    }


def confirm_development_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require a 2000-step development candidate to pass before transfer."""

    baseline = _task_metric_map(rows, "l12_bal")
    reports = {
        variant: transfer_gate(_task_metric_map(rows, variant), baseline)
        for variant in ("new_full_no_ssim", "new_full_ssim")
    }
    passing = [variant for variant, report in reports.items() if report["passed"]]
    return {
        "passed": bool(passing),
        "passing_variants": passing,
        "candidate_reports": reports,
    }


def summarize_stage(
    root: Path, stage: str, *, benchmark: Path = BENCHMARK,
    draws: int = 100_000, approve_freeze: bool = False,
) -> dict[str, Any]:
    rows = collect_stage_rows(root, stage)
    output = Path(root) / "analysis" / stage
    output.mkdir(parents=True, exist_ok=True)
    if stage == "formal256":
        validate_frozen_benchmark(benchmark)
        master = _read_json(Path(root) / "experiment_manifest.json")
        stage_registration = _read_json(Path(root) / stage / "stage_manifest.json")
        validate_formal_provenance(master, stage_registration, benchmark)
        if (
            master.get("schema") != "pcraft_soft_loss_v2_experiment_manifest_v1"
            or master.get("benchmark_status") != APPROVED_STATUS
            or master.get("input_order") != "target_paths_xyz:x,y,z"
            or master.get("paper_eligible_stage") != "formal256"
        ):
            raise ValueError("formal summary lacks an approved P-CRAFT xyz experiment registration")
        required = validate_stage_registration(stage_registration)["required_variants"]
        validate_formal_rows(rows, required_variants=required)
    fields = tuple(
        dict.fromkeys(
            ("stage", "task", "variant", "profile", "resolution", "export_resolution",
             "control_resolution", "steps", "seed", "paper_eligible", *METRIC_FIELDS,
             "run_dir", "config_hash")
        )
    )
    _atomic_tsv(output / "task_rows.tsv", rows, fields)
    aggregate = {
        "schema": "pcraft_soft_loss_v2_aggregate_v1",
        "stage": stage,
        "paper_eligible": stage == "formal256",
        "variants": aggregate_rows(rows),
    }
    atomic_json(output / "aggregate_metrics.json", aggregate)
    result: dict[str, Any] = {"rows": len(rows), "aggregate": aggregate}
    if stage == "dev128_screen":
        result["selection"] = write_dev_selection(rows, Path(root) / "analysis/dev128_selection.json")
    if stage == "dev128_confirm":
        no_ssim_rows = [row for row in rows if row["variant"] == "new_full_no_ssim"]
        ssim_rows = [row for row in rows if row["variant"] == "new_full_ssim"]
        if len(no_ssim_rows) != 3 or len(ssim_rows) != 3:
            raise ValueError("confirmation requires three tasks for both late-SSIM alternatives")
        manifests = [
            _read_json(Path(row["run_dir"]) / "run_manifest.json")
            for row in no_ssim_rows + ssim_rows
        ]
        gate_boundary = {
            (manifest["loss_weights"]["gate"], manifest["loss_weights"]["boundary"])
            for manifest in manifests
        }
        if len(gate_boundary) != 1:
            raise ValueError("confirmation tasks used different selected Gate/Boundary weights")
        lambda_g, lambda_b = next(iter(gate_boundary))
        confirmation = confirm_development_gate(rows)
        result["confirmation_gate"] = confirmation
        atomic_json(output / "confirmation_gate.json", confirmation)
        if approve_freeze and not confirmation["passed"]:
            raise ValueError(
                "cannot approve transfer configuration: no 2000-step development candidate passed"
            )
        frozen = {
            "schema": "pcraft_soft_loss_v2_frozen_config_v1",
            "status": APPROVED_STATUS if approve_freeze else "PENDING_APPROVAL",
            "source_stage": "dev128_confirm",
            "lambda_g": lambda_g, "lambda_b": lambda_b,
            "late_ssim_weight": 0.0,
            "ssim_decision": "DEFERRED_TO_TRANSFER256",
            "paper_eligible": False,
            "confirmation_gate_passed": confirmation["passed"],
        }
        target = Path(root) / "analysis" / (
            "transfer_config.json" if approve_freeze else "transfer_config_candidate.json"
        )
        atomic_json(target, frozen)
        result["frozen_config"] = frozen
    if stage == "transfer256":
        transfer = choose_transfer_configuration(rows)
        atomic_json(output / "transfer_gate.json", transfer)
        result["transfer_gate"] = transfer
        if approve_freeze:
            if not transfer["passed"]:
                raise ValueError("cannot freeze a formal configuration: no 256 candidate passed")
            chosen = next(row for row in rows if row["variant"] == transfer["selected_variant"])
            manifest = _read_json(Path(chosen["run_dir"]) / "run_manifest.json")
            frozen = {
                "schema": "pcraft_soft_loss_v2_frozen_config_v1",
                "status": APPROVED_STATUS,
                "source_stage": "transfer256",
                "lambda_g": manifest["loss_weights"]["gate"],
                "lambda_b": manifest["loss_weights"]["boundary"],
                "late_ssim_weight": transfer["late_ssim_weight"],
                "selected_variant": transfer["selected_variant"],
                "paper_eligible": False,
            }
            atomic_json(Path(root) / "analysis/frozen_config.json", frozen)
            result["frozen_config"] = frozen
    if stage == "formal256":
        paired = paired_inference(rows, draws=draws)
        atomic_json(output / "paired_results.json", paired)
        _write_paired_csv(output / "paired_results.csv", paired)
        export_paper_table(
            rows, output / "paper_task_rows.tsv", required_variants=required
        )
        result["paired"] = paired
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke128", "dev128_screen", "dev128_confirm", "transfer256", "formal256"))
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--bootstrap-draws", type=int, default=100_000)
    parser.add_argument("--approve-freeze", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = summarize_stage(
        args.root, args.stage, benchmark=args.benchmark,
        draws=args.bootstrap_draws, approve_freeze=args.approve_freeze,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
