"""Authenticated two-phase analyzer for the P-CRAFT v7 soft ablation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from types import MappingProxyType, SimpleNamespace
from typing import Mapping, Sequence


if __package__ in (None, ""):
    _BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
    if str(_BOOTSTRAP_ROOT) not in sys.path:
        sys.path.insert(0, str(_BOOTSTRAP_ROOT))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_OUTPUT_ROOT = Path("/data/xvxinyao/pcraft_soft_loss_v7_hit_barrier_256_20260829")
LOCKED_INTERPRETER = Path("/home/lin/anaconda3/envs/shadowart-revisited/bin/python3.10")
LOCKED_INTERPRETER_SHA256 = "7365743bea8011d4ada1d0379699045fe411de80f620c39e4c4625855f0b1772"
REQUIRED_ENVIRONMENT = {
    "PYTHONHASHSEED": "17", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0", "NVIDIA_VISIBLE_DEVICES": "0",
}
TASK_IDS = tuple(f"v7_formal_{index:02d}" for index in range(1, 11))
PROFILES = ("base", "hit_only", "barrier_only", "full")
CSV_FIELDS = (
    "row_index", "task_id", "profile", "run_path", "config_sha256",
    "checkpoint_sha256", "artifact_manifest_sha256", "final_density_sha256",
    "target_stack_sha256", "mean_iou", "mean_dice", "mean_recall", "mean_fpr",
)
_AUTH = {
    "development-0300": Path("authorizations/development_0300.json"),
    "development-1000": Path("authorizations/development_1000.json"),
    "development-3000": Path("authorizations/development_3000.json"),
}
_EVIDENCE = {
    stage: Path(str(path).replace(".json", ".evidence.json"))
    for stage, path in _AUTH.items()
}
_GATES = (
    ("development-0300", 2, Path("runs/RUNNER_GATE_development-0300_step-0002.json")),
    ("development-0300", 200, Path("runs/RUNNER_GATE_development-0300_step-0200.json")),
    ("development-1000", 2, Path("runs/RUNNER_GATE_development-1000_step-0002.json")),
    ("formal-3000", 2, Path("runs/RUNNER_GATE_formal-3000_step-0002.json")),
    ("formal-3000", 200, Path("runs/RUNNER_GATE_formal-3000_step-0200.json")),
)
_ANALYZE_USED = False
_STATISTICS_MODULE = "shadowforge_pipeline.soft_voxel_v7.statistics"


def _immutable(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _immutable(nested) for key, nested in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable(nested) for nested in value)
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(nested) for nested in value]
    return value


@dataclass(frozen=True)
class AuthenticatedSoftReport:
    root: Path
    report_path: Path
    report_sha256: str
    csv_path: Path
    csv_sha256: str
    analysis_status: str
    soft_pass: str
    both_key: str
    claim_allowed: bool
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).absolute())
        object.__setattr__(self, "report_path", Path(self.report_path).absolute())
        object.__setattr__(self, "csv_path", Path(self.csv_path).absolute())
        object.__setattr__(self, "payload", _immutable(self.payload))


@dataclass(frozen=True)
class _PhaseA:
    root: Path
    status: str
    plain: Mapping[str, object]
    root_seals: tuple[Mapping[str, object], ...]
    external_seals: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        _reject_retained_payload(self.plain)
        for collection in (self.root_seals, self.external_seals):
            for seal in collection:
                _reject_retained_payload(seal)
                if type(seal) not in (dict, MappingProxyType) or set(seal) != _SEAL_KEYS:
                    raise ValueError("Phase A may retain compact seals only")
        object.__setattr__(self, "plain", _immutable(self.plain))
        object.__setattr__(self, "root_seals", tuple(_immutable(row) for row in self.root_seals))
        object.__setattr__(self, "external_seals", tuple(_immutable(row) for row in self.external_seals))


_SEAL_KEYS = {
    "path", "sha256", "size", "device", "inode", "nlink", "mtime_ns",
    "ctime_ns", "exclusive",
}


def _reject_retained_payload(value: object) -> None:
    if isinstance(value, (bytes, bytearray, memoryview)) or hasattr(value, "payload"):
        raise TypeError("Phase A compact seals may not retain payload bytes or snapshot objects")
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_retained_payload(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_retained_payload(nested)


def _compact_snapshot(snapshot: object) -> Mapping[str, object]:
    """Discard authenticated bytes immediately and retain only inode-bound identity."""

    return _immutable({
        "path": str(snapshot.path), "sha256": snapshot.sha256,
        "size": snapshot.size, "device": snapshot.device,
        "inode": snapshot.inode, "nlink": snapshot.nlink,
        "mtime_ns": snapshot.mtime_ns, "ctime_ns": snapshot.ctime_ns,
        "exclusive": snapshot.exclusive,
    })  # type: ignore[return-value]


def _snapshot_compact(provenance, path: Path, *, root: Path | None = None,
                      exclusive: bool = True) -> Mapping[str, object]:
    snapshot = provenance.snapshot_regular_file(path, root=root, exclusive=exclusive)
    return _compact_snapshot(snapshot)


def _revalidate_compact(provenance, seal: Mapping[str, object], *, root: Path | None) -> None:
    if type(seal) not in (dict, MappingProxyType) or set(seal) != _SEAL_KEYS:
        raise ValueError("compact evidence seal has the wrong exact schema")
    current = provenance.snapshot_regular_file(
        Path(seal["path"]), root=root, exclusive=seal["exclusive"]
    )
    if _plain(_compact_snapshot(current)) != _plain(seal):
        raise ValueError("authenticated compact evidence changed")


def _snapshot_set(provenance, paths: Sequence[tuple[Path, Path | None, bool]]) -> tuple[Mapping[str, object], ...]:
    return tuple(_snapshot_compact(provenance, path, root=root, exclusive=exclusive)
                 for path, root, exclusive in paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate and analyze the registered v7 soft ablation",
        allow_abbrev=False,
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    return parser


def _runtime_authority(provenance) -> Mapping[str, object]:
    executable = Path(sys.executable)
    if executable != LOCKED_INTERPRETER:
        raise RuntimeError("analyzer requires the exact locked Python interpreter")
    snapshot = provenance.snapshot_regular_file(executable, exclusive=False)
    if snapshot.sha256 != LOCKED_INTERPRETER_SHA256:
        raise RuntimeError("locked interpreter bytes changed")
    for name, value in REQUIRED_ENVIRONMENT.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"analyzer requires exact environment {name}={value}")
    return _compact_snapshot(snapshot)


def _canonical_root(value: Path, *, production: bool = True) -> Path:
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.abspath(path)):
        raise ValueError("output root must be canonical absolute")
    if production and path != REAL_OUTPUT_ROOT:
        raise ValueError(f"production output root must be exactly {REAL_OUTPUT_ROOT}")
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("output root must be an ordinary directory")
    return path


def _exists_regular(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"registered evidence is not an ordinary file: {path}")
    return True


def _analysis_namespace(root: Path) -> str:
    directory = root / "analysis"
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        return "absent"
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("analysis namespace is not an ordinary directory")
    names = set(os.listdir(directory))
    if not names:
        return "empty"
    if names == {"soft_rows.csv"} and _exists_regular(directory / "soft_rows.csv"):
        return "csv_only"
    if names == {"soft_rows.csv", "soft_report.json"} and all(
        _exists_regular(directory / name) for name in names
    ):
        return "committed"
    raise ValueError("analysis namespace has an unregistered state")


def _sealed_json(payload: Mapping[str, object]) -> bytes:
    unsigned = _plain(payload)
    if type(unsigned) is not dict:
        raise TypeError("sealed report must be a mapping")
    unsigned.pop("semantic_sha256", None)
    raw_unsigned = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    unsigned["semantic_sha256"] = hashlib.sha256(raw_unsigned).hexdigest()
    return json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _local_canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _local_semantic_sha(value: Mapping[str, object]) -> str:
    unsigned = _plain(value)
    if type(unsigned) is not dict:
        raise ValueError("semantic payload must be an exact mapping")
    unsigned.pop("semantic_sha256", None)
    return hashlib.sha256(_local_canonical_bytes(unsigned)).hexdigest()


def _authorization_evidence_path(stage: str) -> str:
    if type(stage) is not str or stage not in _EVIDENCE:
        raise ValueError("authorization evidence stage is invalid")
    return _EVIDENCE[stage].as_posix()


def _validate_local_stage_evidence(
    value: object, *, stage: str, campaign_sha: str, endpoint: int,
) -> dict[str, object]:
    keys = {
        "schema", "stage", "campaign_sha256", "endpoint",
        "physical_matrix_sha256", "logical_matrix_sha256",
        "engineering_rows", "metric_rows", "evidence_sha256",
        "semantic_sha256",
    }
    if type(value) is not dict or set(value) != keys:
        raise ValueError("stage evidence has the wrong exact schema")
    row = value
    if (
        row["schema"] != "pcraft_soft_loss_v7_stage_evidence_v1"
        or row["stage"] != stage or row["campaign_sha256"] != campaign_sha
        or type(row["endpoint"]) is not int or row["endpoint"] != endpoint
        or type(row["engineering_rows"]) is not list
        or type(row["metric_rows"]) is not list
    ):
        raise ValueError("stage evidence identity differs")
    for key in (
        "campaign_sha256", "physical_matrix_sha256", "logical_matrix_sha256",
        "evidence_sha256", "semantic_sha256",
    ):
        _digest(row[key], label=f"stage evidence {key}")
    expected_evidence = hashlib.sha256(_local_canonical_bytes({
        "engineering_rows":row["engineering_rows"],
        "metric_rows":row["metric_rows"],
    })).hexdigest()
    if row["evidence_sha256"] != expected_evidence:
        raise ValueError("stage evidence rows digest mismatch")
    if row["semantic_sha256"] != _local_semantic_sha(row):
        raise ValueError("stage evidence semantic digest mismatch")
    return row


def _required_gate_sequence(
    *, status: str, stage: str, target: int | None,
) -> tuple[tuple[str, int], ...]:
    development = (("development-0300", 2), ("development-0300", 200))
    stage2 = development + (("development-1000", 2),)
    formal2 = stage2 + (("formal-3000", 2),)
    all_five = formal2 + (("formal-3000", 200),)
    key = (status, stage, target)
    table = {
        ("DEVELOPMENT_ENGINEERING_FAILED", "development-0300", 2): (),
        ("DEVELOPMENT_ENGINEERING_FAILED", "development-0300", 200): development[:1],
        ("DEVELOPMENT_ENGINEERING_FAILED", "development-1000", 2): development,
        ("DEVELOPMENT_NOT_CONFIRMED", "development-0300", None): development,
        ("DEVELOPMENT_NOT_CONFIRMED", "development-1000", None): stage2,
        ("DEVELOPMENT_NOT_CONFIRMED", "development-3000", None): stage2,
        ("FORMAL_ENGINEERING_FAILED", "formal-3000", 2): stage2,
        ("FORMAL_ENGINEERING_FAILED", "formal-3000", 200): formal2,
        ("FORMAL_ENGINEERING_FAILED", "formal-3000", 3000): all_five,
        ("FORMAL_COMPLETE", "formal-3000", None): all_five,
    }
    if key not in table:
        raise ValueError("terminal state has no registered gate sequence")
    return table[key]


_PREFIX_KEYS = {
    "schema", "stage", "physical_index", "task_id", "budget_id", "profile",
    "config_sha256", "run_path", "completed_updates", "checkpoint_path",
    "checkpoint_sha256", "stage_matrix_sha256", "authorization_sha256",
    "device", "loss_finite", "gradient_finite", "parameters_finite",
    "cuda_executed", "clip_count", "optimizer_step_count", "slot_valid",
    "disabled_not_constructed", "step_time_count", "step_times_finite",
    "min_step_seconds", "peak_allocated_bytes", "device_total_bytes",
    "hit_eligible_slots", "hit_active_slots", "barrier_eligible_slots",
    "barrier_active_slots", "semantic_sha256",
}


def _validate_local_pass_prefix(
    value: object, config: Mapping[str, object], matrix: object, *,
    physical_index: int, cap: int, checkpoint_sha: str,
) -> None:
    if type(value) not in (dict, MappingProxyType) or set(value) != _PREFIX_KEYS:
        raise ValueError("engineering prefix has the wrong exact schema")
    row = _plain(value)
    expected_physical = matrix.physical_rows[physical_index]
    if (
        row["schema"] != "pcraft_soft_loss_v7_engineering_prefix_v1"
        or row["stage"] != matrix.stage or row["physical_index"] != physical_index
        or row["config_sha256"] != config["config_sha256"]
        or row["config_sha256"] != expected_physical["config_sha256"]
        or row["completed_updates"] != cap
        or row["checkpoint_sha256"] != checkpoint_sha
        or row["stage_matrix_sha256"] != matrix.semantic_sha256
        or row["checkpoint_path"] != f"{config['run_path']}/checkpoints/step_{cap:04d}.pt"
    ):
        raise ValueError("engineering prefix identity differs")
    for key in ("loss_finite", "gradient_finite", "parameters_finite", "cuda_executed",
                "slot_valid", "disabled_not_constructed", "step_times_finite"):
        if row[key] is not True:
            raise ValueError("engineering prefix is not exact PASS evidence")
    if (
        row["device"] != "cuda:0" or row["clip_count"] != cap
        or row["optimizer_step_count"] != cap or row["step_time_count"] != cap
        or type(row["min_step_seconds"]) is not float
        or not math.isfinite(row["min_step_seconds"]) or row["min_step_seconds"] <= 0
        or type(row["peak_allocated_bytes"]) is not int
        or type(row["device_total_bytes"]) is not int
        or not 0 < row["peak_allocated_bytes"] < row["device_total_bytes"]
    ):
        raise ValueError("engineering prefix PASS counters are invalid")
    if row["semantic_sha256"] != _local_semantic_sha(row):
        raise ValueError("engineering prefix semantic digest mismatch")


def _cap_equality_gate_failure(
    trainer: object, config: Mapping[str, object], matrix: object, *,
    physical_index: int, cap: int, failed: object,
) -> bool:
    """Disambiguate failure while constructing a cap from failure after it."""

    try:
        authenticated = trainer.validate_failed_run(config, stage_matrix=matrix)
        if (
            authenticated.last_checkpoint.completed_updates != cap
            or authenticated.last_checkpoint.checkpoint_sha256
               != failed.last_checkpoint.checkpoint_sha256
            or matrix.physical_rows[physical_index]["config_sha256"]
               != config["config_sha256"]
        ):
            raise ValueError("cap-equality failed-run identity differs")
        expected_failure = Path(str(config["run_path"])) / "failure/failure.json"
        expected_failure = (
            Path(str(config["root_path"])) / expected_failure
            if "root_path" in config else expected_failure
        )
        if Path(authenticated.failure_record_path) != expected_failure:
            raise ValueError("cap-equality failure record path differs")
        engineering = trainer.project_engineering_row(
            config, stage_matrix=matrix, physical_index=physical_index
        )
    except Exception as error:
        raise ValueError("cap-equality corruption before exact FAIL evidence") from error
    required = {
        "status":"FAIL", "completed_updates":cap,
        "checkpoint_sha256":authenticated.last_checkpoint.checkpoint_sha256,
        "failure_record_path":f"{config['run_path']}/failure/failure.json",
        "failure_record_sha256":authenticated.failure_record_sha256,
        "config_sha256":config["config_sha256"], "physical_index":physical_index,
    }
    if any(engineering.get(key) != expected for key, expected in required.items()):
        raise ValueError("cap-equality engineering FAIL identity differs")
    if engineering.get("failure_code") in (None, "NONE"):
        raise ValueError("cap-equality engineering row is not a registered FAIL")
    try:
        prefix = trainer.project_engineering_prefix(
            config, stage_matrix=matrix, physical_index=physical_index,
            completed_updates=cap,
        )
    except trainer.TrainingStateError:
        return True
    except Exception as error:
        raise ValueError("cap-equality corruption from non-training exception") from error
    _validate_local_pass_prefix(
        prefix, config, matrix, physical_index=physical_index, cap=cap,
        checkpoint_sha=authenticated.last_checkpoint.checkpoint_sha256,
    )
    return False


def _task2_dependency_paths(formal_suite_payload: Mapping[str, object]) -> set[str]:
    selector = formal_suite_payload.get("selector")
    if not isinstance(selector, Mapping):
        raise ValueError("formal selector evidence is missing")
    executed = selector.get("executed_sources")
    anchors = selector.get("anchor_paths")
    if not isinstance(executed, Mapping) or type(anchors) is not list or len(anchors) != 39:
        raise ValueError("formal Task2 dependency evidence is incomplete")
    result = {str(PROJECT_ROOT / str(relative)) for relative in executed}
    for raw in anchors:
        if type(raw) is not str:
            raise ValueError("formal anchor path has the wrong type")
        path = Path(raw)
        result.add(str(path if path.is_absolute() else PROJECT_ROOT / path))
    return result


def _snapshot_task2_dependencies(
    provenance: object, formal_suite_payload: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    selector = formal_suite_payload["selector"]
    executed = selector["executed_sources"]
    anchors = selector["anchor_paths"]
    anchor_sha = selector["anchor_sha256"]
    if type(anchor_sha) is not list or len(anchor_sha) != 39:
        raise ValueError("formal anchor digest vector is incomplete")
    expected: dict[str,str] = {
        str(PROJECT_ROOT / str(relative)):str(digest)
        for relative,digest in executed.items()
    }
    for raw,digest in zip(anchors,anchor_sha,strict=True):
        path=Path(raw); canonical=path if path.is_absolute() else PROJECT_ROOT/path
        expected[str(canonical)]=str(digest)
    observed=[]
    for raw_path,digest in sorted(expected.items(),key=lambda item:item[0].encode()):
        _digest(digest,label="Task2 dependency SHA")
        snapshot=provenance.snapshot_regular_file(Path(raw_path),exclusive=False)
        if snapshot.sha256 != digest:
            raise ValueError("Task2 binder/anchor dependency bytes changed")
        observed.append(_compact_snapshot(snapshot))
    return tuple(observed)


def _snapshot_fixed_task2_dependencies(
    provenance: object, *, project_root: Path,
    executed_relatives: Sequence[str], anchor_rows: Sequence[Sequence[str]],
) -> tuple[Mapping[str, object], ...]:
    """Snapshot the Task2 producer closure from public, code-fixed constants."""

    expected: dict[Path, str | None] = {}
    for relative in executed_relatives:
        if type(relative) is not str:
            raise ValueError("Task2 executed source path has the wrong type")
        expected[project_root / _root_relative_path(relative, label="Task2 source path")] = None
    for raw in anchor_rows:
        if type(raw) not in (tuple, list) or len(raw) != 2:
            raise ValueError("Task2 anchor row has the wrong shape")
        relative, digest = raw
        if type(relative) is not str or type(digest) is not str:
            raise ValueError("Task2 anchor row has the wrong type")
        _digest(digest, label="Task2 anchor SHA")
        anchor=Path(relative)
        if anchor.is_absolute():
            try: anchor.relative_to(project_root)
            except ValueError as error:
                raise ValueError("Task2 absolute anchor escapes project root") from error
        else:
            anchor=project_root / _root_relative_path(relative,label="Task2 anchor path")
        expected[anchor] = digest
    observed = []
    for path, digest in sorted(expected.items(), key=lambda item:item[0].as_posix().encode()):
        snapshot = provenance.snapshot_regular_file(path, exclusive=False)
        if digest is not None and snapshot.sha256 != digest:
            raise ValueError("Task2 fixed anchor bytes changed")
        observed.append(_compact_snapshot(snapshot))
    return tuple(observed)


def _csv_row(cell: Mapping[str, object]) -> Mapping[str, object]:
    metric = cell["metric_row"]
    if not isinstance(metric, Mapping):
        raise ValueError("cell metric row is invalid")
    return {
        "row_index": metric["logical_index"], "task_id": metric["task_id"],
        "profile": metric["profile"], "run_path": metric["run_path"],
        "config_sha256": metric["config_sha256"],
        "checkpoint_sha256": metric["checkpoint_sha256"],
        "artifact_manifest_sha256": cell["artifact_manifest_sha256"],
        "final_density_sha256": cell["final_density_sha256"],
        "target_stack_sha256": cell["target_stack_sha256"],
        "mean_iou": metric["iou"], "mean_dice": metric["dice"],
        "mean_recall": metric["recall"], "mean_fpr": metric["fpr"],
    }


def _csv_bytes(cells: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    for cell in cells:
        writer.writerow(_csv_row(cell))
    return stream.getvalue().encode("utf-8")


def _relative(root: Path, path: Path) -> str:
    try:
        result = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("root evidence path escaped campaign root") from error
    parsed = PurePosixPath(result)
    if parsed.is_absolute() or any(part in ("", ".", "..") for part in parsed.parts):
        raise ValueError("invalid report-relative path")
    return result


def _seal_record(snapshot: Mapping[str, object], *, root: Path | None) -> dict[str, object]:
    path = Path(snapshot["path"])
    return {
        "path": _relative(root, path) if root is not None else str(path),
        "sha256": snapshot["sha256"], "size": snapshot["size"],
        "device": snapshot["device"], "inode": snapshot["inode"], "nlink": snapshot["nlink"],
        "mtime_ns": snapshot["mtime_ns"], "ctime_ns": snapshot["ctime_ns"],
        "exclusive": snapshot["exclusive"],
    }


def _deduplicate_seals(seals: Sequence[Mapping[str, object]], *, root: Path | None) -> tuple[dict[str, object], ...]:
    merged: dict[str, dict[str, object]] = {}
    for snapshot in seals:
        if set(snapshot) != _SEAL_KEYS:
            raise ValueError("compact seal schema mismatch")
        canonical = str(Path(snapshot["path"]))
        prior = merged.get(canonical)
        if prior is not None:
            same = all(prior[name] == snapshot[name] for name in _SEAL_KEYS - {"exclusive"})
            if not same:
                raise ValueError("overlapping evidence roles disagree")
            prior["exclusive"] = bool(prior["exclusive"] or snapshot["exclusive"])
        else:
            merged[canonical] = dict(snapshot)
    return tuple(
        _seal_record(merged[key], root=root)
        for key in sorted(merged, key=lambda item: (_relative(root, Path(item)) if root else item).encode())
    )


def _config_map(campaign: Mapping[str, object], formal: object | None = None) -> dict[str, Mapping[str, object]]:
    rows = formal.payload["formal_configs"] if formal is not None else campaign["development_configs"]
    return {str(row["config_sha256"]): row for row in rows}


def _stage_gate_rows(matrix: object, *, cap: int) -> tuple[Mapping[str, object], ...]:
    rows = tuple(matrix.physical_rows)
    if matrix.stage == "development-0300" and cap == 2:
        return rows
    if matrix.stage == "development-0300" and cap == 200:
        return tuple(row for row in rows if row["profile"] in ("base", "full"))
    if matrix.stage == "development-1000" and cap == 2:
        return tuple(row for row in rows if row["task_id"] != "arcane_casters")
    if matrix.stage == "formal-3000" and cap in (2, 200):
        return rows
    raise ValueError("unregistered analyzer engineering gate")


def _gate_payload(matrix: object, *, cap: int,
                  covered_rows: Sequence[Mapping[str, object]],
                  engineering_rows: Sequence[Mapping[str, object]],
                  canonical_sha256) -> dict[str, object]:
    if len(covered_rows) != len(engineering_rows):
        raise ValueError("engineering gate row count mismatch")
    rows = []
    for expected, raw in zip(covered_rows, engineering_rows, strict=True):
        row = _plain(raw)
        if (
            type(row) is not dict
            or row.get("schema") != "pcraft_soft_loss_v7_engineering_prefix_v1"
            or row.get("stage") != matrix.stage
            or row.get("completed_updates") != cap
            or row.get("config_sha256") != expected["config_sha256"]
            or row.get("physical_index") != expected["physical_index"]
            or row.get("checkpoint_path") != f"{expected['run_path']}/checkpoints/step_{cap:04d}.pt"
            or row.get("stage_matrix_sha256") != matrix.semantic_sha256
            or row.get("authorization_sha256") != matrix.payload["authorization_sha256"]
        ):
            raise ValueError("engineering prefix identity mismatch")
        unsigned = dict(row); semantic = unsigned.pop("semantic_sha256", None)
        if semantic != canonical_sha256(unsigned):
            raise ValueError("engineering prefix semantic SHA mismatch")
        rows.append(row)
    payload = {
        "schema":"pcraft_soft_loss_v7_runner_engineering_gate_v1",
        "stage":matrix.stage, "cap":cap,
        "campaign_sha256":matrix.payload["campaign_sha256"],
        "stage_matrix_sha256":matrix.semantic_sha256,
        "authorization_sha256":matrix.payload["authorization_sha256"],
        "ordered_config_sha256":[row["config_sha256"] for row in covered_rows],
        "engineering_rows":rows,
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    return payload


def _stage_evidence(matrix: object, engineering_rows: Sequence[Mapping[str, object]],
                    metric_rows: Sequence[Mapping[str, object]], canonical_sha256) -> dict[str, object]:
    engineering = [_plain(row) for row in engineering_rows]
    metrics = [_plain(row) for row in metric_rows]
    payload = {
        "schema":"pcraft_soft_loss_v7_stage_evidence_v1", "stage":matrix.stage,
        "campaign_sha256":matrix.payload["campaign_sha256"], "endpoint":matrix.endpoint,
        "physical_matrix_sha256":canonical_sha256([_plain(row) for row in matrix.physical_rows]),
        "logical_matrix_sha256":canonical_sha256([_plain(row) for row in matrix.logical_rows]),
        "engineering_rows":engineering, "metric_rows":metrics,
        "evidence_sha256":canonical_sha256({"engineering_rows":engineering,"metric_rows":metrics}),
    }
    payload["semantic_sha256"] = canonical_sha256(payload)
    return payload


def _classify_terminal_vector(*, stage: str, gates: Sequence[Mapping[str, object]],
                              states: Sequence[Mapping[str, object]]) -> tuple[str, int | None, int | None]:
    """Classify only exact runner barrier residues; return status/failure-index/target."""

    if type(states) not in (list, tuple) or not states:
        raise ValueError("terminal vector is empty")
    caps = tuple(int(row["cap"]) for row in gates if row.get("stage") == stage)
    failures = [index for index, row in enumerate(states) if row.get("state") == "FAILED"]
    if stage == "formal-3000":
        if len(states) != 40:
            raise ValueError("formal terminal vector must contain 40 states")
        if not failures and caps == (2, 200) and all(
            row.get("state") == "COMPLETED" and row.get("completed_updates") == 3000 for row in states
        ):
            return "FORMAL_COMPLETE", None, None
        if len(failures) != 1:
            raise ValueError("formal failure vector must contain one FailedRun")
        failed_index = failures[0]; completed = states[failed_index].get("completed_updates")
        if caps == ():
            valid = all(row.get("state") == "RUNNING" and row.get("completed_updates") == 2
                        for row in states[:failed_index]) and completed in (0, 2) and all(
                            row.get("state") == "ABSENT" for row in states[failed_index + 1:])
            target = 2
        elif caps == (2,):
            valid = all(row.get("state") == "RUNNING" and row.get("completed_updates") == 200
                        for row in states[:failed_index]) and completed in (2, 200) and all(
                            row.get("state") == "RUNNING" and row.get("completed_updates") == 2
                            for row in states[failed_index + 1:])
            target = 200
        elif caps == (2, 200):
            durable = {200,300,400,600,800,1000,1200,1400,1600,1800,2000,2200,
                       2400,2600,2700,2800,3000}
            valid = all(row.get("state") == "COMPLETED" and row.get("completed_updates") == 3000
                        for row in states[:failed_index]) and completed in durable and all(
                            row.get("state") == "RUNNING" and row.get("completed_updates") == 200
                            for row in states[failed_index + 1:])
            target = 3000
        else:
            raise ValueError("formal gate prefix is invalid")
        if not valid:
            raise ValueError("formal terminal state vector violates runner ordering")
        return "FORMAL_ENGINEERING_FAILED", failed_index, target
    if stage == "development-0300":
        if len(states) != 10 or len(failures) != 1:
            raise ValueError("development-0300 failure vector is invalid")
        failed_index = failures[0]; completed = states[failed_index].get("completed_updates")
        if caps == ():
            valid = all(row.get("state") == "RUNNING" and row.get("completed_updates") == 2
                        for row in states[:failed_index]) and completed in (0,2) and all(
                            row.get("state") == "ABSENT" for row in states[failed_index+1:])
            target = 2
        elif caps == (2,):
            covered = [i for i,row in enumerate(states) if row.get("profile") in ("base","full")]
            if failed_index not in covered: raise ValueError("step-200 failure is not covered")
            position = covered.index(failed_index)
            valid = completed in (2,200)
            valid = valid and all(states[i].get("state") == "RUNNING" and states[i].get("completed_updates") == 200 for i in covered[:position])
            valid = valid and all(states[i].get("state") == "RUNNING" and states[i].get("completed_updates") == 2 for i in covered[position+1:])
            valid = valid and all(states[i].get("state") == "RUNNING" and states[i].get("completed_updates") == 2 for i in range(10) if i not in covered)
            target = 200
        else: raise ValueError("development-0300 gate prefix is invalid")
        if not valid: raise ValueError("development-0300 terminal vector violates runner ordering")
        return "DEVELOPMENT_ENGINEERING_FAILED", failed_index, target
    if stage == "development-1000":
        if len(states) != 21 or caps or len(failures) != 1:
            raise ValueError("development-1000 gate-failure vector is invalid")
        failed_index = failures[0]
        if failed_index < 7 or states[failed_index].get("completed_updates") not in (0,2):
            raise ValueError("development-1000 failure is not a fresh cell")
        valid = all(row.get("state") == "RUNNING" and row.get("completed_updates") == 300 for row in states[:7])
        valid = valid and all(row.get("state") == "RUNNING" and row.get("completed_updates") == 2 for row in states[7:failed_index])
        valid = valid and all(row.get("state") == "ABSENT" for row in states[failed_index+1:])
        if not valid: raise ValueError("development-1000 terminal vector violates runner ordering")
        return "DEVELOPMENT_ENGINEERING_FAILED", failed_index, 2
    raise ValueError("stage has no engineering terminal vector")


def _validate_mnc_vector(stage: str, authorization: object,
                         states: Sequence[Mapping[str, object]]) -> None:
    engineering = tuple(authorization.evidence["engineering_rows"])
    if len(engineering) != len(states):
        raise ValueError("MNC engineering/state vector length mismatch")
    endpoint = {"development-0300":300,"development-1000":1000,
                "development-3000":3000}[stage]
    for index, (evidence, state) in enumerate(zip(engineering, states, strict=True)):
        if evidence["physical_index"] != index or evidence["task_id"] != state["task_id"] or evidence["profile"] != state["profile"] or evidence["budget_id"] != state["budget_id"]:
            raise ValueError("MNC state identity differs from authorization evidence")
        if evidence["status"] == "PASS":
            expected_state = "COMPLETED" if endpoint == 3000 else "RUNNING"
            if state["state"] != expected_state or state["completed_updates"] != endpoint:
                raise ValueError("MNC PASS row is not at its exact endpoint")
        elif evidence["status"] == "FAIL":
            if state["state"] != "FAILED" or state["completed_updates"] != evidence["completed_updates"]:
                raise ValueError("MNC FAIL row differs from durable endpoint evidence")
        else:
            raise ValueError("MNC engineering row status is invalid")


def _snapshot_selector_universe(provenance) -> tuple[Mapping[str, object], ...]:
    roots = (
        PROJECT_ROOT / "data/silhouettes/bench3",
        PROJECT_ROOT / "data/silhouettes/complex_v2_filled",
        PROJECT_ROOT / "data/silhouettes/original_anime_30_20260704",
        PROJECT_ROOT / "data/silhouettes/anime_sheet_100_20260704",
    )
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = []
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("selector root is missing or unsafe")
        paths.extend(path for path in root.rglob("*") if path.is_file() and
                     path.name != "CONTACT_SHEET.png" and path.suffix.lower() in suffixes)
    return tuple(_snapshot_compact(provenance, path, exclusive=False)
                 for path in sorted(paths, key=lambda item: item.as_posix()))


def _validate_exact_reached_namespace(root: Path, *, expected_root_files: set[str],
                                      absent_run_paths: Sequence[str] = ()) -> None:
    for relative in absent_run_paths:
        if os.path.lexists(root / relative):
            raise ValueError("ABSENT run path exists")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file() or path.is_symlink()
    }
    # Analysis/postprocess are separate registered consumers.
    actual = {row for row in actual if not row.startswith("analysis/") and not row.startswith("postprocess/")}
    if actual != expected_root_files:
        missing=sorted(expected_root_files-actual,key=lambda value:value.encode())
        extra=sorted(actual-expected_root_files,key=lambda value:value.encode())
        raise ValueError(f"reached root namespace has missing or extra files: missing={missing!r}, extra={extra!r}")
    allowed_runs = {
        relative.split("/CONFIG.json", 1)[0]
        for relative in expected_root_files if relative.endswith("/CONFIG.json")
    }
    allowed_directories = {"runs"}
    for run in allowed_runs:
        parts = PurePosixPath(run).parts
        allowed_directories.update(PurePosixPath(*parts[:index]).as_posix()
                                   for index in range(1, len(parts) + 1))
        allowed_directories.update({f"{run}/checkpoints", f"{run}/artifacts",
                                    f"{run}/artifacts/metrics", f"{run}/failure"})
    runs = root / "runs"
    if runs.is_dir():
        actual_directories = {path.relative_to(root).as_posix() for path in runs.rglob("*") if path.is_dir()}
        if not actual_directories.issubset(allowed_directories):
            raise ValueError("runs namespace contains an unregistered directory")


def _registered_checkpoint_steps_at(completed_updates: int) -> tuple[int, ...]:
    if type(completed_updates) is not int or completed_updates < 0 or completed_updates > 3000:
        raise ValueError("terminal checkpoint update is outside the registered range")
    values = {0, completed_updates}
    if completed_updates >= 2:
        values.add(2)
    values.update(range(200, completed_updates + 1, 200))
    values.update(step for step in (300, 1000, 2400, 2700, 3000)
                  if step <= completed_updates)
    return tuple(sorted(values))


def _root_relative_path(value: object, *, label: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{label} must be a root-relative string")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} is not normalized root-relative POSIX")
    if path.as_posix() != value:
        raise ValueError(f"{label} is not normalized root-relative POSIX")
    return value


def _validate_terminal_state_paths(state: Mapping[str, object]) -> None:
    run = _root_relative_path(state["run_path"], label="terminal run path")
    if not run.startswith("runs/"):
        raise ValueError("terminal run path is outside runs")
    if state["state"] == "ABSENT":
        return
    completed = state["completed_updates"]
    if type(completed) is not int:
        raise ValueError("terminal completed update is not an exact integer")
    if state["status_path"] != f"{run}/STATUS.json":
        raise ValueError("terminal STATUS path is not derived from run path")
    if state["checkpoint_path"] != f"{run}/checkpoints/step_{completed:04d}.pt":
        raise ValueError("terminal checkpoint path is not derived from completed update")
    if state["state"] == "FAILED":
        if (state["observation_path"] != f"{run}/failure/observation.json"
            or state["failure_record_path"] != f"{run}/failure/failure.json"):
            raise ValueError("terminal failure path is not derived from run path")


def _validate_terminal_failure_crosslink(
    terminal: Mapping[str, object], states: Sequence[Mapping[str, object]], *, target: int,
) -> None:
    failed = [state for state in states if state["state"] == "FAILED"]
    if len(failed) != 1:
        raise ValueError("terminal failure does not identify one FailedRun")
    state = failed[0]
    expected_phase = {2:"GATE_STEP_0002", 200:"GATE_STEP_0200", 3000:"ENDPOINT_ADVANCE"}[target]
    if terminal["target_updates"] != target or terminal["phase"] != expected_phase:
        raise ValueError("terminal failure phase is not derived from the terminal vector")
    identity = {
        "cohort", "stage", "task_id", "profile", "budget_id", "run_path",
        "completed_updates", "status_path", "status_sha256", "checkpoint_path",
        "checkpoint_sha256", "observation_path", "observation_sha256",
        "failure_record_path", "failure_record_sha256",
    }
    if any(terminal[key] != state[key] for key in identity):
        raise ValueError("terminal failure cross-link differs from unique FailedRun")


def _registered_metric_steps(state: Mapping[str, object]) -> tuple[int, ...]:
    completed = int(state["completed_updates"])
    successful_limit = completed if state["state"] != "FAILED" else completed - 1
    if state["cohort"] == "formal":
        return (3000,) if successful_limit >= 3000 else ()
    endpoints = (300, 1000, 3000) if state["task_id"] == "arcane_casters" else (1000, 3000)
    return tuple(step for step in endpoints if step <= successful_limit)


def _expected_run_relative_files(state: Mapping[str, object]) -> set[str]:
    if state["state"] == "ABSENT":
        return set()
    run_path = str(state["run_path"])
    completed = int(state["completed_updates"])
    local = {"CONFIG.json", "STATUS.json", "checkpoints/LATEST.json"}
    local.update(
        f"checkpoints/step_{step:04d}.pt"
        for step in _registered_checkpoint_steps_at(completed)
    )
    local.update(
        f"artifacts/metrics/step_{step:04d}.json"
        for step in _registered_metric_steps(state)
    )
    if state["state"] == "FAILED":
        local.update({"failure/observation.json", "failure/failure.json"})
    elif state["state"] == "COMPLETED":
        local.update({
            "artifacts/final_logits.npy", "artifacts/final_density.npy",
            "artifacts/final_hard_005.npy", "artifacts/audit.json",
            "artifacts/resource.json", "artifacts/manifest.json",
        })
    return {f"{run_path}/{relative}" for relative in local}


def _expected_reached_run_files(states: Sequence[Mapping[str, object]]) -> set[str]:
    reached: set[str] = set()
    for state in states:
        reached.update(_expected_run_relative_files(state))
    if reached:
        reached.add("runs/.pcraft-v7-trainer.lock")
    return reached


def _development_terminal_states_from_evidence(
    evidence_payloads: Mapping[str, Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Project the final durable state of every reached development run."""

    latest: dict[str,dict[str,object]]={}
    for stage in ("development-0300","development-1000","development-3000"):
        evidence=evidence_payloads.get(stage)
        if evidence is None:
            break
        endpoint=evidence.get("endpoint")
        rows=evidence.get("engineering_rows")
        if type(endpoint) is not int or type(rows) is not list:
            raise ValueError("development evidence cannot project reached namespace")
        for row in rows:
            if type(row) is not dict or row.get("status") not in ("PASS","FAIL"):
                raise ValueError("development engineering evidence cannot project run state")
            completed=row.get("completed_updates")
            if type(completed) is not int or completed<0 or completed>endpoint:
                raise ValueError("development engineering durable update is invalid")
            if row["status"]=="PASS" and completed!=endpoint:
                raise ValueError("development PASS evidence is not at the stage endpoint")
            state="FAILED" if row["status"]=="FAIL" else ("COMPLETED" if endpoint==3000 else "RUNNING")
            latest[str(row["run_path"])]=dict(
                cohort="development",stage=stage,task_id=row["task_id"],
                profile=row["profile"],budget_id=row["budget_id"],
                config_sha256=row["config_sha256"],run_path=row["run_path"],
                state=state,completed_updates=completed,
            )
    return tuple(latest[key] for key in sorted(latest,key=lambda value:value.encode()))


def _authenticate_cap_equality_candidate(
    *, root: Path, state: Mapping[str, object], trainer: object,
    config: Mapping[str, object], matrix: object, physical_index: int, cap: int,
    failed: object, provenance: object,
    root_seals: Sequence[Mapping[str, object]],
    external_seals: Sequence[Mapping[str, object]],
) -> bool:
    for seal in root_seals:
        _revalidate_compact(provenance, seal, root=root)
    for seal in external_seals:
        _revalidate_compact(provenance, seal, root=None)
    relatives = sorted(_expected_run_relative_files(state), key=lambda item:item.encode())
    before = tuple(
        _snapshot_compact(provenance, root / relative, root=root, exclusive=True)
        for relative in relatives
    )
    result = _cap_equality_gate_failure(
        trainer, config, matrix, physical_index=physical_index, cap=cap,
        failed=failed,
    )
    after = tuple(
        _snapshot_compact(provenance, root / relative, root=root, exclusive=True)
        for relative in relatives
    )
    if tuple(map(_plain,before)) != tuple(map(_plain,after)):
        raise ValueError("cap-equality dependency set changed across public lifecycle")
    for seal in root_seals:
        _revalidate_compact(provenance, seal, root=root)
    for seal in external_seals:
        _revalidate_compact(provenance, seal, root=None)
    return result


def _validate_downstream_run_namespaces(
    root: Path, states: Sequence[Mapping[str, object]],
    cells: Sequence[Mapping[str, object]],
) -> None:
    """Reject same-run progress or files not admitted by the terminal vector."""

    state_by_run: dict[str, Mapping[str, object]] = {}
    for state in states:
        if state["state"] == "ABSENT":
            if os.path.lexists(root / str(state["run_path"])):
                raise ValueError("ABSENT run namespace exists")
            continue
        run_path = str(state["run_path"])
        if run_path in state_by_run:
            raise ValueError("terminal vector duplicates a physical run")
        state_by_run[run_path] = state
    if any(str(cell["run_path"]) not in state_by_run for cell in cells):
        raise ValueError("formal cell is outside the terminal run namespace")
    for run_path, state in state_by_run.items():
        run = root / run_path
        if run.is_symlink() or not run.is_dir():
            raise ValueError("reached run namespace is missing or unsafe")
        expected_files = {
            PurePosixPath(relative).relative_to(run_path).as_posix()
            for relative in _expected_run_relative_files(state)
        }
        expected_directories = {"checkpoints", "artifacts", "artifacts/metrics"}
        if state["state"] == "FAILED":
            expected_directories.add("failure")
        actual_files = {
            path.relative_to(run).as_posix() for path in run.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        actual_directories = {
            path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_dir()
        }
        if actual_files != expected_files or actual_directories != expected_directories:
            raise ValueError("run namespace contains an extra generation, artifact or failure file")


def _reconstruct_expected_evidence_membership(report: Mapping[str, object], root: Path,
                                              provenance) -> tuple[dict[str, bool], dict[str, bool]]:
    """Rebuild exact seal path membership without registry/trainer imports."""

    root_expected = {"V7_ROOT.json", "manifests/development_suite.json",
                     "manifests/formal_suite.json", "manifests/campaign.json"}
    for row in report["development_authorizations"]:
        root_expected.add(row["path"])
        root_expected.add(row["evidence_path"])
    development_evidence={}
    for record in report["development_authorizations"]:
        snapshot=provenance.snapshot_regular_file(
            root / record["evidence_path"],root=root,exclusive=True,
        )
        development_evidence[record["stage"]]=json.loads(snapshot.payload)
    root_expected.update(_expected_reached_run_files(
        _development_terminal_states_from_evidence(development_evidence)
    ))
    if report["formal_matrix"] is not None:
        root_expected.add(report["formal_matrix"]["path"])
    for row in report["runner_gates"]:
        root_expected.add(row["path"])
    for row in report["terminal_run_states"]:
        for key in ("status_path","checkpoint_path","observation_path","failure_record_path"):
            if row[key] is not None: root_expected.add(row[key])
    root_expected.update(_expected_reached_run_files(report["terminal_run_states"]))
    for cell in report["cells"]:
        run = cell["run_path"]
        root_expected.update({
            f"{run}/artifacts/manifest.json", f"{run}/artifacts/final_logits.npy",
            f"{run}/artifacts/final_density.npy", f"{run}/artifacts/final_hard_005.npy",
            cell["metric_row"]["metrics_path"],
        })
    campaign_snapshot = provenance.snapshot_regular_file(root / "manifests/campaign.json", root=root, exclusive=True)
    campaign = json.loads(campaign_snapshot.payload)
    external: dict[str, bool] = {
        str(Path(report["implementation"]["analyzer_path"])): True,
        str(Path(report["implementation"]["statistics_path"])): True,
        str(PROJECT_ROOT / ".superpowers/sdd/2026-08-29-soft-loss-v7-hit-barrier/spec.md"): True,
        str(PROJECT_ROOT / ".superpowers/sdd/2026-08-29-soft-loss-v7-hit-barrier/v3-v6-baseline.sha256"): True,
        str(LOCKED_INTERPRETER): False,
    }
    bundle = campaign["core"]["source_bundle"]
    external.update({str(Path(bundle["project_root"]) / relative): True for relative in bundle["files_sha256"]})
    for task in campaign["core"]["tasks"]:
        external.update({str(Path(item["resolved_path"])): False for item in task["inputs"]})
    development_snapshot = provenance.snapshot_regular_file(
        root / "manifests/development_suite.json", root=root, exclusive=True
    )
    development_payload = json.loads(development_snapshot.payload)
    external[str(Path(development_payload["source_suite"]["path"]))] = False
    formal_snapshot = provenance.snapshot_regular_file(
        root / "manifests/formal_suite.json", root=root, exclusive=True
    )
    formal_payload = json.loads(formal_snapshot.payload)
    for dependency in _task2_dependency_paths(formal_payload):
        external.setdefault(dependency, False)
    inventory = PROJECT_ROOT / ".superpowers/sdd/2026-08-29-soft-loss-v7-hit-barrier/v3-v6-baseline.sha256"
    inventory_snapshot = provenance.snapshot_regular_file(inventory, exclusive=True)
    for line in inventory_snapshot.payload.decode("utf-8").splitlines():
        if line:
            path = str(PROJECT_ROOT / line.split("  ",1)[1])
            external.setdefault(path, False)
    for row in _snapshot_selector_universe(provenance):
        external.setdefault(str(row["path"]), False)
    return {path: True for path in root_expected}, external


def _snapshot_dependency(snapshot_fn, path: Path, *, root: Path | None, exclusive: bool) -> object:
    return snapshot_fn(path, root=root, exclusive=exclusive)


def _terminal_state(evidence: object, root: Path, row: Mapping[str, object], *,
                    order_index: int, cohort: str, stage: str) -> dict[str, object]:
    base = {
        "order_index": order_index, "cohort": cohort, "stage": stage,
        "task_id": row["task_id"], "profile": row["profile"],
        "budget_id": row["budget_id"], "run_path": row["run_path"],
    }
    if evidence is None:
        return {**base, "state": "ABSENT", "completed_updates": None,
                "status_path": None, "status_sha256": None, "checkpoint_path": None,
                "checkpoint_sha256": None, "observation_path": None,
                "observation_sha256": None, "failure_record_path": None,
                "failure_record_sha256": None}
    if hasattr(evidence, "last_checkpoint"):
        checkpoint = evidence.last_checkpoint
        state = "FAILED"
    else:
        checkpoint = evidence.current_checkpoint
        state = "COMPLETED" if hasattr(evidence, "artifact_manifest_path") else "RUNNING"
    result = {**base, "state": state, "completed_updates": checkpoint.completed_updates,
              "status_path": _relative(root, evidence.status_path), "status_sha256": evidence.status_sha256,
              "checkpoint_path": _relative(root, checkpoint.checkpoint_path),
              "checkpoint_sha256": checkpoint.checkpoint_sha256,
              "observation_path": None, "observation_sha256": None,
              "failure_record_path": None, "failure_record_sha256": None}
    if state == "FAILED":
        result.update(observation_path=_relative(root, evidence.observation_path),
                      observation_sha256=evidence.observation_sha256,
                      failure_record_path=_relative(root, evidence.failure_record_path),
                      failure_record_sha256=evidence.failure_record_sha256)
    return result


def _discover_state(root: Path, config: Mapping[str, object], matrix: object, trainer, provenance) -> object | None:
    status = root / str(config["run_path"]) / "STATUS.json"
    if not _exists_regular(status):
        return None
    status_snapshot = provenance.snapshot_regular_file(status, root=root, exclusive=True)
    raw = provenance.decode_canonical_json(status_snapshot.payload, label="run STATUS")
    name = raw.get("status")
    if name == "running": return trainer.validate_resumable_run(config, stage_matrix=matrix)
    if name == "completed": return trainer.validate_completed_run(config, stage_matrix=matrix)
    if name == "failed": return trainer.validate_failed_run(config, stage_matrix=matrix)
    raise ValueError("run STATUS has an invalid state")


def _validate_gate(root: Path, campaign: Mapping[str, object], matrix: object,
                   configs: Mapping[str, Mapping[str, object]], cap: int,
                   relative: Path, trainer, provenance) -> tuple[dict[str, object], Mapping[str, object]]:
    covered = _stage_gate_rows(matrix, cap=cap)
    projected = tuple(
        trainer.project_engineering_prefix(
            configs[str(row["config_sha256"])], stage_matrix=matrix,
            physical_index=int(row["physical_index"]), completed_updates=cap,
        ) for row in covered
    )
    expected = _gate_payload(
        matrix, cap=cap, covered_rows=covered, engineering_rows=projected,
        canonical_sha256=provenance.canonical_sha256,
    )
    snapshot = provenance.snapshot_regular_file(root / relative, root=root, exclusive=True)
    observed = provenance.decode_canonical_json(snapshot.payload, label="runner gate")
    if snapshot.payload != provenance.canonical_json_bytes(expected) or observed != expected:
        raise ValueError("runner gate differs from exact reconstruction")
    return {"stage": matrix.stage, "cap": cap, "path": relative.as_posix(),
            "raw_sha256": snapshot.sha256, "semantic_sha256": expected["semantic_sha256"]}, _compact_snapshot(snapshot)


def _require_exact_gate_prefix(root: Path, runner_gates: Sequence[Mapping[str, object]],
                               expected: Sequence[tuple[str, int]]) -> None:
    observed = tuple((row["stage"], row["cap"]) for row in runner_gates)
    if observed != tuple(expected):
        raise ValueError("required runner gate prefix is missing or reordered")
    actual = {
        path.name for path in (root / "runs").iterdir()
        if path.name.startswith("RUNNER_GATE_")
    } if (root / "runs").is_dir() else set()
    expected_names = {
        relative.name for stage, cap, relative in _GATES if (stage, cap) in expected
    }
    if actual != expected_names:
        raise ValueError("runner gate namespace has missing or extra entries")


def _phase_a(output_root: Path, *, verify: bool = False) -> _PhaseA:
    if _STATISTICS_MODULE in sys.modules:
        raise RuntimeError("statistics was loaded before the final training validator")
    provenance = importlib.import_module("shadowforge_pipeline.soft_voxel_v7.provenance")
    interpreter_seal = _runtime_authority(provenance)
    root = _canonical_root(output_root)
    selector_before = _snapshot_selector_universe(provenance)

    registry = importlib.import_module("shadowforge_pipeline.soft_voxel_v7.registry")
    trainer = importlib.import_module("shadowforge_pipeline.soft_voxel_v7.trainer")
    suite_module = importlib.import_module("shadowforge_pipeline.soft_voxel_v7.suite")
    task2_before = _snapshot_fixed_task2_dependencies(
        provenance, project_root=PROJECT_ROOT,
        executed_relatives=suite_module.EXECUTED_SOURCE_RELATIVES,
        anchor_rows=suite_module.RAW_ANCHOR_ROWS,
    )
    root_identity = provenance.validate_v7_root(root)
    campaign = registry.validate_campaign_manifest(root / "manifests/campaign.json")
    registry.validate_training_source_bundle(campaign["core"]["source_bundle"], PROJECT_ROOT)
    development = suite_module.validate_v7_suite(
        root / "manifests/development_suite.json", expected_kind="development")
    formal_suite = suite_module.validate_v7_suite(
        root / "manifests/formal_suite.json", expected_kind="formal")
    task2_after = _snapshot_fixed_task2_dependencies(
        provenance, project_root=PROJECT_ROOT,
        executed_relatives=suite_module.EXECUTED_SOURCE_RELATIVES,
        anchor_rows=suite_module.RAW_ANCHOR_ROWS,
    )
    if tuple(map(_plain,task2_before)) != tuple(map(_plain,task2_after)):
        raise ValueError("Task2 producer dependencies changed across public validators")
    declared_task2 = _snapshot_task2_dependencies(provenance, formal_suite.payload)
    if tuple(map(_plain,declared_task2)) != tuple(map(_plain,task2_after)):
        raise ValueError("formal suite Task2 dependency evidence differs from fixed producer closure")
    selector_after = _snapshot_selector_universe(provenance)
    if tuple(_plain(row) for row in selector_before) != tuple(_plain(row) for row in selector_after):
        raise ValueError("selector universe changed across suite validation")

    root_seals: list[Mapping[str, object]] = [
        _snapshot_compact(provenance, root / "V7_ROOT.json", root=root, exclusive=True),
        _snapshot_compact(provenance, root / "manifests/campaign.json", root=root, exclusive=True),
        _snapshot_compact(provenance, root / "manifests/development_suite.json", root=root, exclusive=True),
        _snapshot_compact(provenance, root / "manifests/formal_suite.json", root=root, exclusive=True),
    ]
    external_seals: list[Mapping[str, object]] = [interpreter_seal]
    source_root = Path(campaign["core"]["source_bundle"]["project_root"])
    source_rows = campaign["core"]["source_bundle"]["files_sha256"]
    for relative, digest in source_rows.items():
        source_snapshot = provenance.snapshot_regular_file(
            source_root / relative, root=source_root, exclusive=True
        )
        if source_snapshot.sha256 != digest:
            raise ValueError("training source digest changed")
        external_seals.append(_compact_snapshot(source_snapshot))
    for path in (PROJECT_ROOT / ".superpowers/sdd/2026-08-29-soft-loss-v7-hit-barrier/spec.md",
                 PROJECT_ROOT / ".superpowers/sdd/2026-08-29-soft-loss-v7-hit-barrier/v3-v6-baseline.sha256",
                 Path(__file__).resolve(), PROJECT_ROOT / "shadowforge_pipeline/soft_voxel_v7/statistics.py"):
        external_seals.append(_snapshot_compact(provenance, path, exclusive=True))
    for suite in (development, formal_suite):
        for task in suite.tasks:
            for item in task.inputs:
                external_seals.append(_snapshot_compact(provenance, Path(item.resolved_path), exclusive=False))
    external_seals.append(_snapshot_compact(
        provenance, Path(development.payload["source_suite"]["path"]), exclusive=False
    ))
    external_seals.extend(task2_after)
    inventory_path = PROJECT_ROOT / ".superpowers/sdd/2026-08-29-soft-loss-v7-hit-barrier/v3-v6-baseline.sha256"
    inventory_snapshot = provenance.snapshot_regular_file(inventory_path, exclusive=True)
    for line in inventory_snapshot.payload.decode("utf-8").splitlines():
        if not line:
            continue
        digest, relative = line.split("  ", 1)
        protected_path=Path(relative) if Path(relative).is_absolute() else PROJECT_ROOT/relative
        protected = provenance.snapshot_regular_file(
            protected_path, root=None if Path(relative).is_absolute() else PROJECT_ROOT,
            exclusive=False,
        )
        if protected.sha256 != digest:
            raise ValueError("protected inventory member changed")
        external_seals.append(_compact_snapshot(protected))
    external_seals.extend(selector_after)

    authorizations = []
    auth_objects: dict[str, object] = {}
    predecessor = None
    terminal_mnc: str | None = None
    runner_gates = []
    terminal_states: list[dict[str, object]] = []
    reached_run_paths: set[str] = set()
    status = ""
    for stage in ("development-0300", "development-1000", "development-3000"):
        path = root / _AUTH[stage]
        if not _exists_regular(path):
            break
        evidence_path = root / _EVIDENCE[stage]
        if not _exists_regular(evidence_path):
            raise ValueError("development authorization evidence file is missing")
        authorization = registry.validate_stage_authorization(path, expected_stage=stage, campaign=campaign)
        auth_objects[stage] = authorization
        snapshot = provenance.snapshot_regular_file(path, root=root, exclusive=True)
        evidence_snapshot = provenance.snapshot_regular_file(
            evidence_path, root=root, exclusive=True
        )
        evidence_payload = provenance.decode_canonical_json(
            evidence_snapshot.payload, label="development stage evidence"
        )
        if evidence_snapshot.payload != provenance.canonical_json_bytes(evidence_payload):
            raise ValueError("development stage evidence is not canonical JSON")
        validated_evidence = _validate_local_stage_evidence(
            evidence_payload, stage=stage, campaign_sha=root_seals[1]["sha256"],
            endpoint={"development-0300":300,"development-1000":1000,
                      "development-3000":3000}[stage],
        )
        if (
            authorization.evidence is None
            or _local_canonical_bytes(authorization.evidence) != evidence_snapshot.payload
            or authorization.payload["evidence_sha256"] != validated_evidence["evidence_sha256"]
        ):
            raise ValueError("development authorization/evidence cross-link differs")
        root_seals.extend((_compact_snapshot(evidence_snapshot), _compact_snapshot(snapshot)))
        authorizations.append({"stage": stage, "path": _AUTH[stage].as_posix(),
                               "raw_sha256": authorization.raw_sha256,
                               "semantic_sha256": authorization.semantic_sha256,
                               "evidence_sha256": authorization.evidence["evidence_sha256"],
                               "evidence_path": _EVIDENCE[stage].as_posix(),
                               "evidence_raw_sha256": evidence_snapshot.sha256,
                               "evidence_semantic_sha256": evidence_payload["semantic_sha256"],
                               "status": authorization.status,
                               "selected_budget_id": authorization.selected_budget_id})
        matrix = registry.resolve_stage_matrix(campaign, stage=stage, authorization=predecessor)
        configs = _config_map(campaign)
        eng = [trainer.project_engineering_row(configs[str(row["config_sha256"])], stage_matrix=matrix,
                                               physical_index=int(row["physical_index"])) for row in matrix.physical_rows]
        reached_run_paths.update(str(row["run_path"]) for row in eng)
        failed_budgets = {row["budget_id"] for row in eng if row["status"] == "FAIL" and row["budget_id"] is not None}
        base_failed = any(row["profile"] == "base" and row["status"] == "FAIL" for row in eng)
        metrics = []
        if not base_failed and not (stage == "development-3000" and any(row["status"] == "FAIL" for row in eng)):
            for logical in matrix.logical_rows:
                if logical["budget_id"] not in failed_budgets:
                    metrics.append(trainer.project_metric_row(configs[str(logical["config_sha256"])], stage_matrix=matrix,
                                                              logical_index=int(logical["logical_index"])))
        evidence = _stage_evidence(matrix, eng, metrics, provenance.canonical_sha256)
        rebuilt = registry.build_stage_authorization(campaign, stage=stage, evidence=evidence)
        if provenance.canonical_json_bytes(_plain(rebuilt.payload)) != snapshot.payload:
            raise ValueError("development authorization differs from endpoint reconstruction")
        for gate_stage, cap, gate_path in _GATES:
            if gate_stage == stage and _exists_regular(root / gate_path):
                gate, seal = _validate_gate(root, campaign, matrix, configs, cap, gate_path,
                                            trainer, provenance)
                runner_gates.append(gate); root_seals.append(seal)
        predecessor = authorization
        if authorization.status == "MECHANISM_NOT_CONFIRMED":
            terminal_mnc = stage
            break
    if terminal_mnc is not None:
        if _exists_regular(root / "manifests/formal_matrix.json"):
            raise ValueError("formal matrix exists after development MNC")
        matrix = registry.resolve_stage_matrix(campaign, stage=terminal_mnc,
                                               authorization=auth_objects.get({"development-1000":"development-0300","development-3000":"development-1000"}.get(terminal_mnc,"")))
        configs = _config_map(campaign)
        for index, row in enumerate(matrix.physical_rows):
            evidence = _discover_state(root, configs[str(row["config_sha256"])], matrix, trainer, provenance)
            terminal_states.append(_terminal_state(evidence, root, row, order_index=index,
                                                   cohort="development", stage=terminal_mnc))
            if evidence is not None: reached_run_paths.add(str(row["run_path"]))
        _validate_mnc_vector(terminal_mnc, auth_objects[terminal_mnc], terminal_states)
        status = "DEVELOPMENT_NOT_CONFIRMED"
        terminal_failure = None
        formal_manifest = None
        cells = []
        required = [("development-0300",2),("development-0300",200)]
        if terminal_mnc in ("development-1000","development-3000"):
            required.append(("development-1000",2))
        _require_exact_gate_prefix(root, runner_gates, required)
    elif len(authorizations) < 3:
        # Only a durable, exactly one-failure current stage is publishable.
        current = ("development-0300", "development-1000", "development-3000")[len(authorizations)]
        matrix = registry.resolve_stage_matrix(campaign, stage=current, authorization=predecessor)
        configs = _config_map(campaign)
        failures = []
        for index, row in enumerate(matrix.physical_rows):
            evidence = _discover_state(root, configs[str(row["config_sha256"])], matrix, trainer, provenance)
            terminal_states.append(_terminal_state(evidence, root, row, order_index=index,
                                                   cohort="development", stage=current))
            if terminal_states[-1]["state"] == "FAILED": failures.append((row, evidence))
            if evidence is not None: reached_run_paths.add(str(row["run_path"]))
        status, failure_index, target = _classify_terminal_vector(
            stage=current, gates=runner_gates, states=terminal_states
        )
        assert failure_index is not None and target is not None
        row = matrix.physical_rows[failure_index]
        failed = failures[0][1]
        if terminal_states[failure_index]["completed_updates"] == target and target in (2,200):
            if not _authenticate_cap_equality_candidate(
                root=root, state=terminal_states[failure_index], trainer=trainer,
                config=configs[str(row["config_sha256"])], matrix=matrix,
                physical_index=failure_index, cap=target, failed=failed,
                provenance=provenance, root_seals=root_seals,
                external_seals=external_seals,
            ):
                raise ValueError("failed run had already achieved the missing gate cap")
        terminal_failure, failure_seals = _failure_payload(
            root, "development", current, row, failed, target, provenance
        )
        root_seals.extend(failure_seals)
        formal_manifest = None
        cells = []
    else:
        if predecessor is None or predecessor.status != "PASS":
            raise ValueError("development-3000 did not publish PASS")
        formal = registry.validate_formal_matrix_manifest(root / "manifests/formal_matrix.json", campaign)
        formal_snapshot = provenance.snapshot_regular_file(root / "manifests/formal_matrix.json", root=root, exclusive=True)
        root_seals.append(_compact_snapshot(formal_snapshot))
        formal_manifest = {"path": "manifests/formal_matrix.json", "raw_sha256": formal_snapshot.sha256,
                           "semantic_sha256": formal.semantic_sha256,
                           "selected_budget_id": formal.payload["core"]["selected_budget_id"]}
        matrix = registry.resolve_stage_matrix(campaign, stage="formal-3000", authorization=predecessor)
        configs = _config_map(campaign, formal)
        for gate_stage, cap, gate_path in _GATES:
            if gate_stage == "formal-3000" and _exists_regular(root / gate_path):
                gate, seal = _validate_gate(root, campaign, matrix, configs, cap, gate_path,
                                            trainer, provenance)
                runner_gates.append(gate); root_seals.append(seal)
        states = []
        for index, row in enumerate(matrix.physical_rows):
            evidence = _discover_state(root, configs[str(row["config_sha256"])], matrix, trainer, provenance)
            states.append(evidence)
            terminal_states.append(_terminal_state(evidence, root, row, order_index=index,
                                                   cohort="formal", stage="formal-3000"))
            if evidence is not None: reached_run_paths.add(str(row["run_path"]))
        if all(state is not None and hasattr(state, "artifact_manifest_path") and
               state.current_checkpoint.completed_updates == 3000 for state in states):
            status, _, _ = _classify_terminal_vector(
                stage="formal-3000", gates=runner_gates, states=terminal_states
            )
            terminal_failure = None
            cells = []
            for logical in matrix.logical_rows:
                config = configs[str(logical["config_sha256"])]
                completed = trainer.validate_completed_run(config, stage_matrix=matrix)
                engineering = trainer.project_engineering_row(config, stage_matrix=matrix,
                                                              physical_index=int(logical["physical_index"]))
                metric = trainer.project_metric_row(config, stage_matrix=matrix,
                                                    logical_index=int(logical["logical_index"]))
                cell, seals = _recompute_cell(root, config, completed, engineering, metric, provenance)
                cells.append(cell); root_seals.extend(seals)
        else:
            status, failure_index, target = _classify_terminal_vector(
                stage="formal-3000", gates=runner_gates, states=terminal_states
            )
            assert failure_index is not None and target is not None
            row, failed = matrix.physical_rows[failure_index], states[failure_index]
            if terminal_states[failure_index]["completed_updates"] == target and target in (2,200):
                if not _authenticate_cap_equality_candidate(
                    root=root, state=terminal_states[failure_index], trainer=trainer,
                    config=configs[str(row["config_sha256"])], matrix=matrix,
                    physical_index=failure_index, cap=target, failed=failed,
                    provenance=provenance, root_seals=root_seals,
                    external_seals=external_seals,
                ):
                    raise ValueError("failed run had already achieved the missing formal gate cap")
            terminal_failure, failure_seals = _failure_payload(
                root, "formal", "formal-3000", row, failed, target, provenance
            )
            root_seals.extend(failure_seals); cells = []

    # Last training calls have now occurred. Snapshot every report-driving root file.
    for state_row in terminal_states:
        for key in ("status_path", "checkpoint_path", "observation_path", "failure_record_path"):
            if state_row[key] is not None:
                snap = provenance.snapshot_regular_file(root / state_row[key], root=root, exclusive=True)
                if snap.sha256 != state_row[key.replace("_path", "_sha256")]:
                    raise ValueError("terminal state digest changed")
                root_seals.append(_compact_snapshot(snap))

    expected_gate_prefix = _required_gate_sequence(
        status=status, stage=str(terminal_states[0]["stage"]),
        target=None if terminal_failure is None else int(terminal_failure["target_updates"]),
    )
    _require_exact_gate_prefix(root, runner_gates, expected_gate_prefix)

    # Freeze the complete reached root file namespace, not only report summaries.
    fixed_root_files = {"V7_ROOT.json", "manifests/development_suite.json",
                        "manifests/formal_suite.json", "manifests/campaign.json"}
    fixed_root_files.update(row["path"] for row in authorizations)
    fixed_root_files.update(row["evidence_path"] for row in authorizations)
    if formal_manifest is not None: fixed_root_files.add(formal_manifest["path"])
    fixed_root_files.update(row["path"] for row in runner_gates)
    fixed_root_files.update(_expected_reached_run_files(
        _development_terminal_states_from_evidence({
            stage:_plain(authorization.evidence)
            for stage,authorization in auth_objects.items()
        })
    ))
    fixed_root_files.update(_expected_reached_run_files(terminal_states))
    absent_paths = [row["run_path"] for row in terminal_states if row["state"] == "ABSENT"]
    _validate_exact_reached_namespace(root, expected_root_files=fixed_root_files,
                                      absent_run_paths=absent_paths)
    for relative in sorted(fixed_root_files, key=lambda value: value.encode()):
        root_seals.append(_snapshot_compact(provenance, root / relative, root=root, exclusive=True))
    # Re-enumeration is part of every Phase-A closure boundary.
    selector_again = _snapshot_selector_universe(provenance)
    expected_selector_tail=(external_seals[-len(selector_again):] if selector_again else ())
    if tuple(_plain(row) for row in selector_again) != tuple(
        _plain(row) for row in expected_selector_tail
    ):
        raise ValueError("selector universe changed during Phase A")
    plain = {
        "campaign_record": {"path":"manifests/campaign.json", "raw_sha256":root_seals[1]["sha256"],
                            "semantic_sha256":campaign["semantic_sha256"],
                            "training_source_bundle_semantic_sha256":campaign["core"]["source_bundle"]["semantic_sha256"]},
        "development_authorizations": authorizations, "formal_matrix": formal_manifest,
        "runner_gates": runner_gates, "terminal_failure": terminal_failure,
        "terminal_run_states": terminal_states, "cells": cells,
    }
    return _PhaseA(root, status, _immutable(plain), tuple(root_seals), tuple(external_seals))


def _failure_payload(root: Path, cohort: str, stage: str, row: Mapping[str, object],
                     failed: object, target: int, provenance,
                     ) -> tuple[dict[str, object], tuple[Mapping[str, object], ...]]:
    checkpoint = failed.last_checkpoint
    phase = {2:"GATE_STEP_0002",200:"GATE_STEP_0200",3000:"ENDPOINT_ADVANCE"}[target]
    record_snapshot = provenance.snapshot_regular_file(
        failed.failure_record_path, root=root, exclusive=True
    )
    observation_snapshot = provenance.snapshot_regular_file(
        failed.observation_path, root=root, exclusive=True
    )
    if record_snapshot.sha256 != failed.failure_record_sha256 or observation_snapshot.sha256 != failed.observation_sha256:
        raise ValueError("failure evidence digest changed")
    record = provenance.decode_canonical_json(record_snapshot.payload, label="failure record")
    if record_snapshot.payload != provenance.canonical_json_bytes(record):
        raise ValueError("failure record is not canonical")
    required = {"schema","config_sha256","stage_matrix_sha256",
                "authorization_sha256","last_checkpoint_path",
                "last_checkpoint_sha256","completed_updates","failure_code",
                "training_source_bundle_sha256","semantic_sha256"}
    if type(record) is not dict or set(record) != required or record["schema"] != "pcraft_soft_loss_v7_failure_record_v1":
        raise ValueError("failure record has the wrong exact schema")
    unsigned = dict(record); semantic = unsigned.pop("semantic_sha256")
    if semantic != provenance.canonical_sha256(unsigned):
        raise ValueError("failure record semantic SHA mismatch")
    payload = {
        "cohort": cohort, "stage": stage, "phase": phase, "target_updates": target,
        "task_id": row["task_id"], "profile": row["profile"], "budget_id": row["budget_id"],
        "run_path": row["run_path"], "completed_updates": checkpoint.completed_updates,
        "failure_code": record["failure_code"],
        "status_path": _relative(root, failed.status_path), "status_sha256": failed.status_sha256,
        "checkpoint_path": _relative(root, checkpoint.checkpoint_path), "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "observation_path": _relative(root, failed.observation_path), "observation_sha256": failed.observation_sha256,
        "failure_record_path": _relative(root, failed.failure_record_path), "failure_record_sha256": failed.failure_record_sha256,
    }
    return payload, (_compact_snapshot(record_snapshot), _compact_snapshot(observation_snapshot))


def _tensor_sha(value) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _recompute_cell(root: Path, config: Mapping[str, object], completed: object,
                    engineering: Mapping[str, object], metric: Mapping[str, object],
                    provenance) -> tuple[dict[str, object], tuple[object, ...]]:
    import cv2
    import numpy as np
    import torch
    from shadowforge_pipeline.soft_voxel_v3.data import normalize_target
    from shadowforge_pipeline.soft_voxel_v3.thresholding import render_thresholded
    from shadowforge_pipeline.soft_voxel_v7.core import build_registered_target_stack
    from shadowforge_pipeline.soft_voxel_v7.protocol import RegisteredEARProtocol, registered_camera_payload
    from shadowforge_pipeline.soft_voxel_v7.trainer import evaluate_six_view_metrics

    paths = {
        "checkpoint": completed.current_checkpoint.checkpoint_path,
        "manifest": completed.artifact_manifest_path,
        "logits": completed.run_dir / "artifacts/final_logits.npy",
        "density": completed.run_dir / "artifacts/final_density.npy",
        "hard": completed.run_dir / "artifacts/final_hard_005.npy",
        "metrics": completed.run_dir / str(metric["metrics_path"]).split(str(config["run_path"]) + "/", 1)[-1],
    }
    snapshots = {name: provenance.snapshot_regular_file(path, root=root, exclusive=True)
                 for name, path in paths.items()}
    if snapshots["checkpoint"].sha256 != completed.current_checkpoint.checkpoint_sha256:
        raise ValueError("checkpoint digest changed")
    if snapshots["manifest"].sha256 != completed.artifact_manifest_sha256:
        raise ValueError("artifact manifest digest changed")
    if snapshots["metrics"].sha256 != metric["metrics_sha256"]:
        raise ValueError("endpoint metric digest differs from metric row")
    expected_artifacts = completed.artifacts
    for name, relative in (("logits","artifacts/final_logits.npy"),
                           ("density","artifacts/final_density.npy"),
                           ("hard","artifacts/final_hard_005.npy")):
        if snapshots[name].sha256 != expected_artifacts[relative]:
            raise ValueError("terminal array digest differs from manifest")
    arrays = {}
    for name, dtype in (("logits",np.dtype("float32")),("density",np.dtype("float32")),("hard",np.dtype("bool"))):
        stream = io.BytesIO(snapshots[name].payload)
        value = np.load(stream, allow_pickle=False)
        if stream.tell() != len(snapshots[name].payload) or value.dtype != dtype or value.shape != (256,256,256) or not value.flags.c_contiguous:
            raise ValueError("terminal NPY shape/dtype/layout mismatch")
        if name != "hard" and not np.isfinite(value).all(): raise ValueError("terminal NPY is nonfinite")
        arrays[name] = value
    safe = [np._core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]
    with torch.serialization.safe_globals(safe):
        checkpoint = torch.load(io.BytesIO(snapshots["checkpoint"].payload), map_location="cpu", weights_only=True)
    logits = checkpoint["logits"]
    if not torch.equal(logits[0,0], torch.from_numpy(arrays["logits"])):
        raise ValueError("final logits differ from checkpoint")
    density = torch.sigmoid(logits.to("cuda:0", dtype=torch.float32))[0,0].cpu()
    if not torch.equal(density, torch.from_numpy(arrays["density"])):
        raise ValueError("final density differs from sigmoid checkpoint")
    if not torch.equal(density >= .05, torch.from_numpy(arrays["hard"])):
        raise ValueError("final hard mask differs from density threshold")
    input_seals = []
    bases = []
    for expected_axis, input_row in zip("xyz", config["inputs"], strict=True):
        if input_row["axis"] != expected_axis: raise ValueError("input axis order mismatch")
        snap = provenance.snapshot_regular_file(Path(input_row["resolved_path"]), exclusive=False)
        if snap.sha256 != input_row["raw_sha256"]: raise ValueError("raw input digest changed")
        image = cv2.imdecode(np.frombuffer(snap.payload, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError("raw input cannot be decoded")
        bases.append(torch.from_numpy(normalize_target(image, 256, False) > 0))
        input_seals.append(_compact_snapshot(snap))
    target = build_registered_target_stack(tuple(zip("xyz", bases))).contiguous()
    target_sha = _tensor_sha(target)
    if target_sha != config["target_stack_sha256"]: raise ValueError("registered target digest changed")
    camera = registered_camera_payload()
    for key, config_key in (("rotation_sha256","rotation_sha256"),("translation_sha256","translation_sha256"),
                            ("scale_sha256","scale_sha256"),("projection_sha256","projection_sha256")):
        if camera[key] != config["cameras"][config_key]: raise ValueError("registered camera digest changed")
    protocol = RegisteredEARProtocol(torch.device("cuda:0"))
    density_tensor = torch.from_numpy(arrays["density"])[None,None].contiguous().to("cuda:0", dtype=torch.float32)
    rendered = render_thresholded(protocol, density_tensor, .05).projection
    recomputed = evaluate_six_view_metrics(rendered.float(), target.to("cuda:0"))
    endpoint = provenance.decode_canonical_json(
        snapshots["metrics"].payload, label="formal endpoint metrics"
    )
    if snapshots["metrics"].payload != provenance.canonical_json_bytes(endpoint):
        raise ValueError("formal endpoint metrics are not canonical JSON")
    if _plain(recomputed["per_view"]) != endpoint["per_view"] or _plain(recomputed["aggregate"]) != endpoint["aggregate"]:
        raise ValueError("independent rerender differs from endpoint metric")
    for name, field in (("iou","mean_iou"),("dice","mean_dice"),("recall","mean_recall"),("fpr","mean_fpr")):
        if metric[name] != recomputed["aggregate"][field]: raise ValueError("metric row differs from rerender")
    cell = {
        "row_index": metric["logical_index"], "task_id": metric["task_id"], "profile": metric["profile"],
        "run_path": metric["run_path"], "config_sha256": metric["config_sha256"],
        "checkpoint_sha256": metric["checkpoint_sha256"],
        "artifact_manifest_sha256": completed.artifact_manifest_sha256,
        "final_logits_sha256": snapshots["logits"].sha256,
        "final_density_sha256": snapshots["density"].sha256,
        "final_hard_005_sha256": snapshots["hard"].sha256,
        "target_stack_sha256": target_sha, "endpoint_metrics_sha256": snapshots["metrics"].sha256,
        "per_view_metrics": _plain(recomputed["per_view"]),
        "engineering_row": _plain(engineering), "metric_row": _plain(metric),
    }
    compact_snapshots = tuple(_compact_snapshot(snapshot) for snapshot in snapshots.values())
    del checkpoint, logits, density, density_tensor, rendered, arrays, snapshots
    torch.cuda.synchronize(torch.device("cuda:0"))
    torch.cuda.empty_cache()
    return cell, compact_snapshots + tuple(input_seals)


def _terminal_revalidate(phase: _PhaseA, provenance, *, csv_raw: bytes | None = None,
                         report_raw: bytes | None = None) -> None:
    for seal in phase.root_seals:
        _revalidate_compact(provenance, seal, root=phase.root)
    for seal in phase.external_seals:
        _revalidate_compact(provenance, seal, root=None)
    selector_paths = {str(row["path"]): _plain(row) for row in _snapshot_selector_universe(provenance)}
    expected_selector = {
        str(row["path"]): _plain(row) for row in phase.external_seals
        if str(row["path"]).startswith(str(PROJECT_ROOT / "data/silhouettes/") )
    }
    if selector_paths != expected_selector:
        raise ValueError("selector universe changed across analysis boundary")
    expected_root_files = {
        _relative(phase.root, Path(row["path"])) for row in phase.root_seals
    }
    absent = [row["run_path"] for row in phase.plain["terminal_run_states"] if row["state"] == "ABSENT"]
    _validate_exact_reached_namespace(
        phase.root, expected_root_files=expected_root_files, absent_run_paths=absent
    )
    if csv_raw is not None:
        csv_snapshot = provenance.snapshot_regular_file(
            phase.root / "analysis/soft_rows.csv", root=phase.root, exclusive=True
        )
        if csv_snapshot.payload != csv_raw:
            raise ValueError("published CSV changed")
    if report_raw is not None:
        report_snapshot = provenance.snapshot_regular_file(
            phase.root / "analysis/soft_report.json", root=phase.root, exclusive=True
        )
        if report_snapshot.payload != report_raw:
            raise ValueError("published report changed")


def _phase_b(phase: _PhaseA, *, verify: bool) -> AuthenticatedSoftReport:
    statistics = importlib.import_module(_STATISTICS_MODULE)
    provenance = importlib.import_module("shadowforge_pipeline.soft_voxel_v7.provenance")
    for seal in (*phase.root_seals, *phase.external_seals):
        _revalidate_compact(
            provenance, seal,
            root=phase.root if Path(seal["path"]).is_relative_to(phase.root) else None,
        )
    plain = _plain(phase.plain)
    external_by_path = {str(row["path"]): row for row in phase.external_seals}
    analyzer_path = str(Path(__file__).resolve())
    statistics_path = str(PROJECT_ROOT / "shadowforge_pipeline/soft_voxel_v7/statistics.py")
    cells = plain["cells"]
    if phase.status == "FORMAL_COMPLETE":
        outcome = statistics.evaluate_formal_outcome(
            [cell["metric_row"] for cell in cells], [cell["engineering_row"] for cell in cells]
        )
        labels = _plain(outcome["labels"]); stats_payload = _plain(outcome)
    elif phase.status == "FORMAL_ENGINEERING_FAILED":
        labels = {"SOFT_PASS":"FAIL", "BOTH_KEY":"NOT_EVALUATED", "claim_allowed":False}; stats_payload = None
    else:
        labels = {"SOFT_PASS":"NOT_EVALUATED", "BOTH_KEY":"NOT_EVALUATED", "claim_allowed":False}; stats_payload = None
    csv_raw = _csv_bytes(cells if phase.status == "FORMAL_COMPLETE" else ())
    csv_sha = hashlib.sha256(csv_raw).hexdigest()
    report = {
        "schema":"pcraft_soft_loss_v7_soft_report_v1", "analysis_status":phase.status,
        "root_path":str(phase.root), "campaign":plain["campaign_record"],
        "development_authorizations":plain["development_authorizations"],
        "formal_matrix":plain["formal_matrix"], "runner_gates":plain["runner_gates"],
        "terminal_failure":plain["terminal_failure"], "cells":cells,
        "csv":{"path":"analysis/soft_rows.csv", "raw_sha256":csv_sha,
               "row_count":len(cells) if phase.status == "FORMAL_COMPLETE" else 0,
               "header":list(CSV_FIELDS)},
        "statistics":stats_payload, "labels":labels,
        "inference_scope":{"task_ids":list(TASK_IDS),"views_per_task":6,"statistical_units":10,
                           "aggregation":"task_equal","dice_role":"corroborative",
                           "iou_hypotheses":2,"postprocess_included":False},
        "implementation":{"analyzer_path":analyzer_path,
                          "analyzer_sha256":external_by_path[analyzer_path]["sha256"],
                          "statistics_path":statistics_path,
                          "statistics_sha256":external_by_path[statistics_path]["sha256"]},
        "terminal_run_states":plain["terminal_run_states"],
        "root_evidence_files":list(_deduplicate_seals(phase.root_seals, root=phase.root)),
        "external_evidence_files":list(_deduplicate_seals(phase.external_seals, root=None)),
        "semantic_sha256":"",
    }
    report_raw = _sealed_json(report)
    report_sha = hashlib.sha256(report_raw).hexdigest()
    state = _analysis_namespace(phase.root)
    if verify:
        if state != "committed": raise ValueError("verify requires a committed analysis")
        csv_snap = provenance.snapshot_regular_file(phase.root / "analysis/soft_rows.csv", root=phase.root, exclusive=True)
        report_snap = provenance.snapshot_regular_file(phase.root / "analysis/soft_report.json", root=phase.root, exclusive=True)
        if csv_snap.payload != csv_raw or report_snap.payload != report_raw:
            raise ValueError("sealed analysis differs from full recomputation")
        _terminal_revalidate(phase, provenance, csv_raw=csv_raw, report_raw=report_raw)
    else:
        if state == "committed": raise FileExistsError("analysis is committed; use --verify")
        with provenance.open_v7_root_transaction(phase.root) as transaction:
            transaction.ensure_registered_directory("analysis")
            _terminal_revalidate(phase, provenance)
            if state == "csv_only":
                if transaction.snapshot(Path("analysis/soft_rows.csv")).payload != csv_raw:
                    raise ValueError("conflicting CSV crash prefix")
            else:
                transaction.publish(Path("analysis/soft_rows.csv"), csv_raw)
                transaction.fsync(Path("analysis"))
            _terminal_revalidate(phase, provenance, csv_raw=csv_raw)
            transaction.publish(Path("analysis/soft_report.json"), report_raw)
            transaction.fsync(Path("analysis"))
            _terminal_revalidate(
                phase, provenance, csv_raw=csv_raw, report_raw=report_raw
            )
    return AuthenticatedSoftReport(
        phase.root, phase.root / "analysis/soft_report.json", report_sha,
        phase.root / "analysis/soft_rows.csv", csv_sha, phase.status,
        labels["SOFT_PASS"], labels["BOTH_KEY"], labels["claim_allowed"], report,
    )


def analyze(output_root: Path, *, verify: bool = False) -> AuthenticatedSoftReport:
    global _ANALYZE_USED
    if _ANALYZE_USED:
        raise RuntimeError("analyze is a one-shot fresh-process operation")
    _ANALYZE_USED = True
    phase = _phase_a(Path(output_root), verify=verify)
    return _phase_b(phase, verify=verify)


def _decode_report(path: Path) -> tuple[dict[str, object], bytes]:
    from shadowforge_pipeline.soft_voxel_v7.provenance import canonical_json_bytes, decode_canonical_json
    snapshot = importlib.import_module("shadowforge_pipeline.soft_voxel_v7.provenance").snapshot_regular_file(path, exclusive=True)
    payload = decode_canonical_json(snapshot.payload, label="sealed soft report")
    if snapshot.payload != canonical_json_bytes(payload): raise ValueError("soft report is not canonical JSON")
    unsigned = dict(payload); semantic = unsigned.pop("semantic_sha256", None)
    if semantic != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest():
        raise ValueError("soft report semantic SHA mismatch")
    return payload, snapshot.payload


def _exact_keys(value: object, keys: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} has the wrong exact schema")
    return value


def _digest(value: object, *, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_local_authorization_chain(
    records: Sequence[Mapping[str, object]],
    authorization_payloads: Mapping[str, Mapping[str, object]],
    evidence_payloads: Mapping[str, Mapping[str, object]], *, campaign_sha: str,
) -> None:
    expected_stages=("development-0300","development-1000","development-3000")
    auth_keys={"schema","stage","campaign_sha256","stage_matrix_sha256",
               "evidence_sha256","status","ranked_budget_ids","surviving_budget_ids",
               "selected_budget_id","next_stage","authorized_physical_config_sha256",
               "authorized_logical_rows","semantic_sha256"}
    for index,record in enumerate(records):
        stage=expected_stages[index]
        auth=authorization_payloads.get(stage); evidence=evidence_payloads.get(stage)
        if auth is None or evidence is None or set(auth) != auth_keys:
            raise ValueError("authorization/evidence contiguous chain is incomplete")
        if (auth["schema"]!="pcraft_soft_loss_v7_stage_authorization_v1"
            or auth["stage"]!=stage or auth["campaign_sha256"]!=campaign_sha
            or auth["evidence_sha256"]!=evidence["evidence_sha256"]
            or auth["semantic_sha256"]!=_local_semantic_sha(auth)
            or record["evidence_sha256"]!=evidence["evidence_sha256"]):
            raise ValueError("authorization/evidence semantic cross-link differs")
        if type(auth["stage_matrix_sha256"]) is not str:
            raise ValueError("authorization matrix identity is invalid")
        allowed=("ADVANCE","MECHANISM_NOT_CONFIRMED") if index<2 else ("PASS","MECHANISM_NOT_CONFIRMED")
        if auth["status"] not in allowed or auth["status"]!=record["status"]:
            raise ValueError("authorization status differs from report chain")
        if auth["selected_budget_id"]!=record["selected_budget_id"]:
            raise ValueError("authorization selected budget differs")
        if auth["status"]=="MECHANISM_NOT_CONFIRMED" and index+1!=len(records):
            raise ValueError("negative authorization has a later successor")


def _validate_local_formal_matrix(
    value: object, *, root: Path, campaign: Mapping[str, object],
    final_authorization: Mapping[str, object], final_authorization_sha: str,
) -> dict[str, object]:
    row=_exact_keys(value,{"schema","core","formal_core_sha256","formal_configs","semantic_sha256"},label="formal matrix payload")
    if row["schema"]!="pcraft_soft_loss_v7_formal_matrix_v1" or row["semantic_sha256"]!=_local_semantic_sha(row):
        raise ValueError("formal matrix schema/semantic identity differs")
    core_keys={"schema","root_path","formal_manifest_path","campaign_semantic_sha256",
               "formal_suite","development_3000_authorization_path",
               "development_3000_authorization_sha256","selected_budget_id",
               "formal_budget_ratio","formal_tasks","formal_policy","campaign_path",
               "cameras","scientific_policy","source_bundle","environment"}
    core=_exact_keys(row["core"],core_keys,label="formal core")
    if (core["schema"]!="pcraft_soft_loss_v7_formal_core_v1"
        or core["root_path"]!=str(root) or core["formal_manifest_path"]!="manifests/formal_matrix.json"
        or core["campaign_semantic_sha256"]!=campaign["semantic_sha256"]
        or core["development_3000_authorization_path"]!=_AUTH["development-3000"].as_posix()
        or core["development_3000_authorization_sha256"]!=final_authorization_sha
        or core["selected_budget_id"]!=final_authorization["selected_budget_id"]
        or row["formal_core_sha256"]!=hashlib.sha256(_local_canonical_bytes(core)).hexdigest()
        or type(row["formal_configs"]) is not list or len(row["formal_configs"])!=40):
        raise ValueError("formal matrix does not bind campaign/final PASS")
    config_keys={"schema","root_path","campaign_path","campaign_schema",
        "campaign_core_sha256","cohort","authorization_paths","formal_matrix_path",
        "formal_manifest_schema","formal_core_sha256",
        "development_3000_authorization_sha256","task_id","profile","flags",
        "budget_id","budget_ratio","seed","run_path","inputs","target_stack_sha256",
        "hull_xyz_sha256","hull_dhw_sha256","hull_occupied_count","initialization",
        "cameras","scientific_policy","source_bundle_sha256","environment","config_sha256"}
    tasks=core["formal_tasks"]
    if type(tasks) is not list or len(tasks)!=10:
        raise ValueError("formal config task registry differs")
    task_by_id={task.get("task_id"):task for task in tasks if type(task) is dict}
    if len(task_by_id)!=10:
        raise ValueError("formal config task registry differs")
    profiles=("base","hit_only","barrier_only","full")
    expected_order=[(task["task_id"],profile) for task in tasks for profile in profiles]
    for index,(config,identity) in enumerate(zip(row["formal_configs"],expected_order,strict=True)):
        if type(config) is not dict or set(config)!=config_keys:
            raise ValueError("formal config has the wrong exact schema")
        unsigned=dict(config); claimed=unsigned.pop("config_sha256")
        if claimed!=hashlib.sha256(_local_canonical_bytes(unsigned)).hexdigest():
            raise ValueError("formal config semantic identity differs")
        task=task_by_id.get(config["task_id"]); profile=identity[1]
        expected_run=(f"runs/formal/{identity[0]}/base/seed_17" if profile=="base"
                      else f"runs/formal/{identity[0]}/{profile}/seed_17")
        expected_budget=None if profile=="base" else core["selected_budget_id"]
        expected_ratio=None if profile=="base" else core["formal_budget_ratio"]
        if (identity!=(config["task_id"],config["profile"])
            or config["schema"]!="pcraft_soft_loss_v7_resolved_config_v1"
            or config["root_path"]!=str(root) or config["campaign_path"]!="manifests/campaign.json"
            or config["campaign_core_sha256"]!=campaign["campaign_core_sha256"]
            or config["cohort"]!="formal" or config["formal_matrix_path"]!="manifests/formal_matrix.json"
            or config["formal_core_sha256"]!=row["formal_core_sha256"]
            or config["development_3000_authorization_sha256"]!=final_authorization_sha
            or config["budget_id"]!=expected_budget or config["budget_ratio"]!=expected_ratio
            or config["seed"]!=17 or config["run_path"]!=expected_run
            or config["inputs"]!=task.get("inputs")
            or config["target_stack_sha256"]!=task.get("registered_target_sha256")
            or config["hull_xyz_sha256"]!=task.get("hull_xyz_sha256")
            or config["hull_dhw_sha256"]!=task.get("hull_dhw_sha256")
            or config["hull_occupied_count"]!=task.get("hull_occupied_count")
            or config["cameras"]!=core["cameras"]
            or config["scientific_policy"]!=core["scientific_policy"]
            or config["source_bundle_sha256"]!=core["source_bundle"]["semantic_sha256"]
            or config["environment"]!=core["environment"]):
            raise ValueError("formal config differs from authenticated formal core")
    return row


def _local_stage_matrix_identity(
    campaign: Mapping[str, object], *, campaign_sha: str, stage: str,
    authorization_payloads: Mapping[str, Mapping[str, object]],
    authorization_records: Mapping[str, Mapping[str, object]],
    formal_payload: Mapping[str, object] | None,
) -> tuple[str, str | None, list[dict[str,object]]]:
    if stage in authorization_payloads:
        auth=authorization_payloads[stage]
        physical=[]
        seen=set()
        for evidence in auth.get("authorized_physical_config_sha256",()):
            seen.add(evidence)
        # Endpoint evidence is authoritative for reached authorizations; callers
        # use the separately validated evidence rows for physical order.
        return str(auth["stage_matrix_sha256"]), (
            None if stage=="development-0300"
            else str(authorization_records[{"development-1000":"development-0300",
                                             "development-3000":"development-1000"}[stage]]["raw_sha256"])
        ), physical
    if stage=="development-0300":
        configs=campaign["development_configs"]
        index={(row["task_id"],row["budget_id"],row["profile"]):row for row in configs}
        identities=[("arcane_casters",None,"base")]
        identities.extend(("arcane_casters",budget,profile)
                          for budget in ("budget_010","budget_015","budget_020")
                          for profile in ("hit_only","barrier_only","full"))
        physical=[]
        for position,identity in enumerate(identities):
            config=index[identity]
            physical.append({"physical_index":position,"task_id":identity[0],
                             "budget_id":identity[1],"profile":identity[2],
                             "config_sha256":config["config_sha256"],
                             "run_path":config["run_path"]})
        logical=[{"logical_index":i,**row} for i,row in enumerate(physical)]
        payload={"schema":"pcraft_soft_loss_v7_stage_matrix_v1","stage":stage,
                 "campaign_sha256":campaign_sha,"endpoint":300,
                 "authorization_path":None,"authorization_sha256":None,
                 "physical_rows":physical,"logical_rows":logical}
        return hashlib.sha256(_local_canonical_bytes(payload)).hexdigest(),None,physical
    if stage=="development-1000":
        previous=authorization_payloads["development-0300"]
        budgets=tuple(previous["ranked_budget_ids"][:2])
        configs=campaign["development_configs"]
        index={(row["task_id"],row["budget_id"],row["profile"]):row for row in configs}
        physical=[]
        for task in ("arcane_casters","blade_heroes","stage_dancers"):
            identities=[(task,None,"base")]
            identities.extend((task,budget,profile) for budget in budgets
                              for profile in ("hit_only","barrier_only","full"))
            for identity in identities:
                config=index[identity]; position=len(physical)
                physical.append({"physical_index":position,"task_id":task,
                                 "budget_id":identity[1],"profile":identity[2],
                                 "config_sha256":config["config_sha256"],
                                 "run_path":config["run_path"]})
        by_identity={(row["task_id"],row["budget_id"],row["profile"]):row for row in physical}
        logical=[]
        for budget in budgets:
            for task in ("arcane_casters","blade_heroes","stage_dancers"):
                for profile in ("base","hit_only","barrier_only","full"):
                    physical_row=by_identity[(task,None if profile=="base" else budget,profile)]
                    logical.append({"logical_index":len(logical),**physical_row,"budget_id":budget})
        auth_sha=str(authorization_records["development-0300"]["raw_sha256"])
        payload={"schema":"pcraft_soft_loss_v7_stage_matrix_v1","stage":stage,
                 "campaign_sha256":campaign_sha,"endpoint":1000,
                 "authorization_path":_AUTH["development-0300"].as_posix(),
                 "authorization_sha256":auth_sha,"physical_rows":physical,
                 "logical_rows":logical}
        return hashlib.sha256(_local_canonical_bytes(payload)).hexdigest(),auth_sha,physical
    if stage=="formal-3000" and formal_payload is not None:
        physical=[]
        for index,config in enumerate(formal_payload["formal_configs"]):
            physical.append({"physical_index":index,"task_id":config["task_id"],
                             "budget_id":config["budget_id"],"profile":config["profile"],
                             "config_sha256":config["config_sha256"],"run_path":config["run_path"]})
        logical=[{"logical_index":i,**row} for i,row in enumerate(physical)]
        auth_sha=str(authorization_records["development-3000"]["raw_sha256"])
        payload={"schema":"pcraft_soft_loss_v7_stage_matrix_v1","stage":stage,
                 "campaign_sha256":campaign_sha,"endpoint":3000,
                 "authorization_path":_AUTH["development-3000"].as_posix(),
                 "authorization_sha256":auth_sha,"physical_rows":physical,
                 "logical_rows":logical}
        return hashlib.sha256(_local_canonical_bytes(payload)).hexdigest(),auth_sha,physical
    raise ValueError("cannot derive local stage matrix identity")


def _validate_local_gate_payload(
    value: object, record: Mapping[str, object], *, campaign_sha: str,
    stage_matrix_sha: str, authorization_sha: str | None,
) -> dict[str, object]:
    keys={"schema","stage","cap","campaign_sha256","stage_matrix_sha256",
          "authorization_sha256","ordered_config_sha256","engineering_rows",
          "semantic_sha256"}
    row=_exact_keys(value,keys,label="runner gate payload")
    if (row["schema"]!="pcraft_soft_loss_v7_runner_engineering_gate_v1"
        or row["stage"]!=record["stage"] or row["cap"]!=record["cap"]
        or row["campaign_sha256"]!=campaign_sha
        or row["stage_matrix_sha256"]!=stage_matrix_sha
        or row["authorization_sha256"]!=authorization_sha
        or row["semantic_sha256"]!=_local_semantic_sha(row)
        or type(row["ordered_config_sha256"]) is not list
        or type(row["engineering_rows"]) is not list
        or len(row["ordered_config_sha256"])!=len(row["engineering_rows"])):
        raise ValueError("runner gate semantic cross-link differs")
    for config_sha,prefix in zip(row["ordered_config_sha256"],row["engineering_rows"],strict=True):
        if (type(prefix) is not dict or set(prefix)!=_PREFIX_KEYS
            or prefix["config_sha256"]!=config_sha or prefix["stage"]!=row["stage"]
            or prefix["completed_updates"]!=row["cap"]
            or prefix["stage_matrix_sha256"]!=stage_matrix_sha
            or prefix["authorization_sha256"]!=authorization_sha
            or prefix["semantic_sha256"]!=_local_semantic_sha(prefix)):
            raise ValueError("runner gate prefix row differs")
    return row


def _validate_local_terminal_matrix(
    states: Sequence[Mapping[str, object]], physical_rows: Sequence[Mapping[str, object]],
    *, cohort: str, stage: str,
) -> None:
    if len(states)!=len(physical_rows):
        raise ValueError("terminal matrix/vector length differs")
    for index,(state,physical) in enumerate(zip(states,physical_rows,strict=True)):
        if (state["order_index"]!=index or state["cohort"]!=cohort or state["stage"]!=stage
            or any(state[key]!=physical[key] for key in ("task_id","profile","budget_id","run_path"))):
            raise ValueError("terminal state differs from authenticated matrix order")


def _derive_local_failure_code(
    observation: Mapping[str, object], *, completed_updates: int, profile: str,
) -> str:
    if not observation["loss_finite"] or not observation["gradient_finite"] or not observation["parameters_finite"]:
        return "NONFINITE_TRAINING"
    active_updates=max(0,completed_updates-100)
    hit_slots,barrier_slots=(active_updates+1)//2,active_updates//2
    expected_hit=hit_slots if profile in ("hit_only","full") else 0
    expected_barrier=barrier_slots if profile in ("barrier_only","full") else 0
    if (
        observation["clip_count"]!=completed_updates
        or observation["optimizer_step_count"]!=completed_updates
        or observation["step_time_count"]!=completed_updates
        or not observation["slot_valid"] or not observation["disabled_not_constructed"]
        or observation["hit_eligible_slots"]!=expected_hit
        or observation["barrier_eligible_slots"]!=expected_barrier
        or observation["hit_active_slots"]>observation["hit_eligible_slots"]
        or observation["barrier_active_slots"]>observation["barrier_eligible_slots"]
    ):
        return "STEP_CONTRACT"
    if not observation["cuda_executed"]:
        return "CUDA_NOT_USED"
    if (
        not observation["step_times_finite"] or observation["min_step_seconds"] is None
        or observation["min_step_seconds"]<=0.0
        or not (0<observation["peak_allocated_bytes"]<observation["device_total_bytes"])
    ):
        return "RESOURCE_INVALID"
    return "NONE"


def _validate_local_failure_payloads(
    terminal: Mapping[str, object], state: Mapping[str, object],
    observation: object, record: object, *, source_bundle_sha: str, profile: str,
) -> None:
    observation_keys={"schema","config_sha256","stage_matrix_sha256","authorization_sha256",
        "last_checkpoint_path","last_checkpoint_sha256","completed_updates","device",
        "loss_finite","gradient_finite","parameters_finite","cuda_executed","clip_count",
        "optimizer_step_count","slot_valid","disabled_not_constructed","step_time_count",
        "step_times_finite","min_step_seconds","peak_allocated_bytes","device_total_bytes",
        "hit_eligible_slots","hit_active_slots","barrier_eligible_slots","barrier_active_slots",
        "training_source_bundle_sha256","semantic_sha256"}
    record_keys={"schema","config_sha256","stage_matrix_sha256","authorization_sha256",
        "last_checkpoint_path","last_checkpoint_sha256","completed_updates","failure_code",
        "training_source_bundle_sha256","semantic_sha256"}
    obs=_exact_keys(observation,observation_keys,label="failure observation")
    rec=_exact_keys(record,record_keys,label="failure record")
    if (obs["schema"]!="pcraft_soft_loss_v7_failure_observation_v1"
        or rec["schema"]!="pcraft_soft_loss_v7_failure_record_v1"
        or obs["semantic_sha256"]!=_local_semantic_sha(obs)
        or rec["semantic_sha256"]!=_local_semantic_sha(rec)
        or obs["training_source_bundle_sha256"]!=source_bundle_sha
        or rec["training_source_bundle_sha256"]!=source_bundle_sha):
        raise ValueError("failure payload schema/source identity differs")
    common=("config_sha256","stage_matrix_sha256","authorization_sha256",
            "last_checkpoint_path","last_checkpoint_sha256","completed_updates")
    if any(obs[key]!=rec[key] for key in common):
        raise ValueError("failure observation/record identity differs")
    derived=_derive_local_failure_code(
        obs,completed_updates=int(obs["completed_updates"]),profile=profile,
    )
    if (derived=="NONE" or rec["failure_code"]!=derived
        or rec["failure_code"]!=terminal["failure_code"]
        or rec["last_checkpoint_sha256"]!=state["checkpoint_sha256"]
        or rec["completed_updates"]!=state["completed_updates"]):
        raise ValueError("terminal failure code/checkpoint differs from failure bytes")


def _validate_local_cell_payloads(cell: Mapping[str, object]) -> None:
    metric=cell["metric_row"]; engineering=cell["engineering_row"]
    if (metric["metrics_sha256"]!=cell["endpoint_metrics_sha256"]
        or metric["config_sha256"]!=cell["config_sha256"]
        or engineering["config_sha256"]!=cell["config_sha256"]
        or metric["run_path"]!=cell["run_path"] or engineering["run_path"]!=cell["run_path"]
        or metric["checkpoint_sha256"]!=cell["checkpoint_sha256"]
        or engineering["checkpoint_sha256"]!=cell["checkpoint_sha256"]):
        raise ValueError("cell engineering/metric identity cross-link differs")
    for row,label in ((metric,"metric row"),(engineering,"engineering row")):
        if type(row) is not dict or row.get("semantic_sha256")!=_local_semantic_sha(row):
            raise ValueError(f"cell {label} semantic digest differs")


def _validate_local_endpoint_metric_values(per_view: object, aggregate: object) -> None:
    keys={"view","tp","fp","fn","tn","iou","dice","recall","fpr"}
    if type(per_view) is not list or len(per_view)!=6:
        raise ValueError("endpoint per-view vector has the wrong shape")
    checked=[]
    for index,value in enumerate(per_view):
        row=_exact_keys(value,keys,label="endpoint per-view row")
        if row["view"]!=index or type(row["view"]) is not int:
            raise ValueError("endpoint per-view order differs")
        if any(type(row[key]) is not int or row[key]<0 for key in ("tp","fp","fn","tn")):
            raise ValueError("endpoint per-view counts have the wrong domain")
        tp,fp,fn,tn=(row[key] for key in ("tp","fp","fn","tn"))
        union,dice_den,foreground,background=tp+fp+fn,2*tp+fp+fn,tp+fn,fp+tn
        if min(union,dice_den,foreground,background)<=0:
            raise ValueError("endpoint per-view counts have a zero denominator")
        expected={"iou":tp/union,"dice":2*tp/dice_den,
                  "recall":tp/foreground,"fpr":fp/background}
        for key,result in expected.items():
            if type(row[key]) is not float or not math.isfinite(row[key]) or row[key]!=result:
                raise ValueError("endpoint per-view metric differs from counts")
        checked.append(row)
    aggregate_row=_exact_keys(
        aggregate,{"mean_iou","mean_dice","mean_recall","mean_fpr"},
        label="endpoint aggregate",
    )
    expected_aggregate={
        f"mean_{key}":sum(float(row[key]) for row in checked)/6.0
        for key in ("iou","dice","recall","fpr")
    }
    if any(type(aggregate_row[key]) is not float or aggregate_row[key]!=value
           for key,value in expected_aggregate.items()):
        raise ValueError("endpoint aggregate differs from six per-view rows")


def _validate_local_artifact_manifest(
    manifest: object, *, config_sha: str, checkpoint_sha: str,
    expected_artifacts: Mapping[str,str],
) -> None:
    row=_exact_keys(manifest,{"schema","config_sha256","terminal_checkpoint_sha256",
                              "grant_history","artifacts","semantic_sha256"},label="artifact manifest")
    if (row["schema"]!="pcraft_soft_loss_v7_artifact_manifest_v1"
        or row["config_sha256"]!=config_sha
        or row["terminal_checkpoint_sha256"]!=checkpoint_sha
        or row["semantic_sha256"]!=_local_semantic_sha(row)
        or type(row["grant_history"]) is not list or type(row["artifacts"]) is not list):
        raise ValueError("artifact manifest identity differs")
    observed={}
    previous=None
    for item in row["artifacts"]:
        if type(item) is not dict or set(item)!={"path","size","sha256"}:
            raise ValueError("artifact manifest record schema differs")
        encoded=item["path"].encode()
        if previous is not None and encoded<=previous:
            raise ValueError("artifact manifest order differs")
        previous=encoded; observed[item["path"]]=item["sha256"]
    if observed!=dict(expected_artifacts):
        raise ValueError("artifact manifest coverage differs from evidence seals")


def _validate_report_shape(payload: dict[str, object], root: Path) -> None:
    statuses = {
        "DEVELOPMENT_ENGINEERING_FAILED", "DEVELOPMENT_NOT_CONFIRMED",
        "FORMAL_ENGINEERING_FAILED", "FORMAL_COMPLETE",
    }
    status = payload["analysis_status"]
    if type(status) is not str or status not in statuses:
        raise ValueError("analysis status is not registered")
    campaign = _exact_keys(payload["campaign"], {
        "path","raw_sha256","semantic_sha256","training_source_bundle_semantic_sha256"
    }, label="campaign record")
    if campaign["path"] != "manifests/campaign.json": raise ValueError("campaign path domain mismatch")
    for key in ("raw_sha256","semantic_sha256","training_source_bundle_semantic_sha256"):
        _digest(campaign[key], label=f"campaign {key}")
    authorizations = payload["development_authorizations"]
    if type(authorizations) is not list:
        raise ValueError("development authorizations must be an array")
    expected_stages = ("development-0300","development-1000","development-3000")
    for index, row_raw in enumerate(authorizations):
        row = _exact_keys(row_raw, {"stage","path","raw_sha256","semantic_sha256",
                                    "evidence_path","evidence_raw_sha256",
                                    "evidence_semantic_sha256","evidence_sha256",
                                    "status","selected_budget_id"},
                          label="authorization record")
        if (index >= 3 or row["stage"] != expected_stages[index]
            or row["path"] != _AUTH[row["stage"]].as_posix()
            or row["evidence_path"] != _EVIDENCE[row["stage"]].as_posix()):
            raise ValueError("authorization prefix is missing or reordered")
        if row["status"] not in ("ADVANCE","PASS","MECHANISM_NOT_CONFIRMED"):
            raise ValueError("authorization status is invalid")
        for key in ("raw_sha256","semantic_sha256","evidence_sha256",
                    "evidence_raw_sha256","evidence_semantic_sha256"):
            _digest(row[key], label=f"authorization {key}")
    formal = payload["formal_matrix"]
    if formal is not None:
        formal = _exact_keys(formal, {"path","raw_sha256","semantic_sha256","selected_budget_id"}, label="formal matrix")
        if formal["path"] != "manifests/formal_matrix.json": raise ValueError("formal matrix path mismatch")
        _digest(formal["raw_sha256"], label="formal raw SHA")
        _digest(formal["semantic_sha256"], label="formal semantic SHA")
    gates = payload["runner_gates"]
    if type(gates) is not list: raise ValueError("runner gates must be an array")
    expected_gate_order = [(stage,cap,path.as_posix()) for stage,cap,path in _GATES]
    last = -1
    for row_raw in gates:
        row = _exact_keys(row_raw, {"stage","cap","path","raw_sha256","semantic_sha256"}, label="runner gate")
        identity = (row["stage"],row["cap"],row["path"])
        if identity not in expected_gate_order: raise ValueError("runner gate identity is invalid")
        position = expected_gate_order.index(identity)
        if position <= last: raise ValueError("runner gates are duplicated or reordered")
        last = position
        _digest(row["raw_sha256"], label="gate raw SHA"); _digest(row["semantic_sha256"], label="gate semantic SHA")
    csv_row = _exact_keys(payload["csv"], {"path","raw_sha256","row_count","header"}, label="CSV record")
    expected_count = 40 if status == "FORMAL_COMPLETE" else 0
    if (csv_row["path"] != "analysis/soft_rows.csv" or type(csv_row["row_count"]) is not int
        or csv_row["row_count"] != expected_count or csv_row["header"] != list(CSV_FIELDS)):
        raise ValueError("CSV identity/header/count mismatch")
    _digest(csv_row["raw_sha256"], label="CSV raw SHA")
    labels = _exact_keys(payload["labels"], {"SOFT_PASS","BOTH_KEY","claim_allowed"}, label="labels")
    expected_label_domains = {
        "DEVELOPMENT_ENGINEERING_FAILED":("NOT_EVALUATED","NOT_EVALUATED"),
        "DEVELOPMENT_NOT_CONFIRMED":("NOT_EVALUATED","NOT_EVALUATED"),
        "FORMAL_ENGINEERING_FAILED":("FAIL","NOT_EVALUATED"),
    }
    if type(labels["claim_allowed"]) is not bool: raise ValueError("claim label must be bool")
    if status != "FORMAL_COMPLETE":
        if (labels["SOFT_PASS"],labels["BOTH_KEY"]) != expected_label_domains[status] or labels["claim_allowed"]:
            raise ValueError("terminal labels contradict truth table")
        if payload["statistics"] is not None or payload["cells"] != []:
            raise ValueError("noncomplete report carries scientific rows")
    else:
        if labels["SOFT_PASS"] not in ("PASS","FAIL") or labels["BOTH_KEY"] not in ("PASS","FAIL"):
            raise ValueError("formal labels are invalid")
        if labels["claim_allowed"] != (labels["SOFT_PASS"] == labels["BOTH_KEY"] == "PASS"):
            raise ValueError("claim label contradicts formal labels")
        if type(payload["cells"]) is not list or len(payload["cells"]) != 40:
            raise ValueError("formal complete report requires 40 cells")
        if len(authorizations) != 3 or authorizations[-1]["status"] != "PASS" or formal is None:
            raise ValueError("formal complete lacks exact development PASS chain")
        if tuple((row["stage"],row["cap"]) for row in gates) != tuple((stage,cap) for stage,cap,_ in _GATES):
            raise ValueError("formal complete lacks all five runner gates")
        cell_keys = {"row_index","task_id","profile","run_path","config_sha256",
                     "checkpoint_sha256","artifact_manifest_sha256","final_logits_sha256",
                     "final_density_sha256","final_hard_005_sha256","target_stack_sha256",
                     "endpoint_metrics_sha256","per_view_metrics","engineering_row","metric_row"}
        for index, cell_raw in enumerate(payload["cells"]):
            cell = _exact_keys(cell_raw, cell_keys, label="formal cell")
            task, profile = TASK_IDS[index // 4], PROFILES[index % 4]
            if (cell["row_index"],cell["task_id"],cell["profile"],cell["run_path"]) != (
                index, task, profile, f"runs/formal/{task}/{profile}/seed_17"
            ):
                raise ValueError("formal cell order/path mismatch")
            for key in ("config_sha256","checkpoint_sha256","artifact_manifest_sha256",
                        "final_logits_sha256","final_density_sha256","final_hard_005_sha256",
                        "target_stack_sha256","endpoint_metrics_sha256"):
                _digest(cell[key], label=f"formal cell {key}")
            if type(cell["per_view_metrics"]) is not list or len(cell["per_view_metrics"]) != 6:
                raise ValueError("formal cell must carry six per-view metrics")
    states = payload["terminal_run_states"]
    if type(states) is not list: raise ValueError("terminal states must be an array")
    expected_lengths = {
        "DEVELOPMENT_ENGINEERING_FAILED": {10,21},
        "DEVELOPMENT_NOT_CONFIRMED": {10,21,12},
        "FORMAL_ENGINEERING_FAILED": {40}, "FORMAL_COMPLETE": {40},
    }
    if len(states) not in expected_lengths[status]: raise ValueError("terminal vector length mismatch")
    state_keys = {"order_index","cohort","stage","task_id","profile","budget_id","run_path",
                  "state","completed_updates","status_path","status_sha256","checkpoint_path",
                  "checkpoint_sha256","observation_path","observation_sha256",
                  "failure_record_path","failure_record_sha256"}
    for index, raw_state in enumerate(states):
        row = _exact_keys(raw_state, state_keys, label="terminal state")
        if row["order_index"] != index or type(row["order_index"]) is not int or row["state"] not in ("ABSENT","RUNNING","COMPLETED","FAILED"):
            raise ValueError("terminal state order/enum mismatch")
        _validate_terminal_state_paths(row)
        if row["state"] == "ABSENT":
            if any(row[key] is not None for key in state_keys - {"order_index","cohort","stage","task_id","profile","budget_id","run_path","state"}):
                raise ValueError("ABSENT terminal state carries evidence")
        else:
            if type(row["completed_updates"]) is not int:
                raise ValueError("present state needs exact completed update")
            for key in ("status_path","status_sha256","checkpoint_path","checkpoint_sha256"):
                if row[key] is None: raise ValueError("present state is missing evidence")
            for key in ("status_sha256","checkpoint_sha256"):
                _digest(row[key], label=f"terminal {key}")
            failure_values = tuple(row[key] for key in ("observation_path","observation_sha256","failure_record_path","failure_record_sha256"))
            if (row["state"] == "FAILED") != all(value is not None for value in failure_values):
                raise ValueError("terminal failure nullability mismatch")
            if row["state"] != "FAILED" and any(value is not None for value in failure_values):
                raise ValueError("nonfailed state carries failure evidence")
    classified_target: int | None = None
    if status in ("FORMAL_COMPLETE","FORMAL_ENGINEERING_FAILED"):
        classified, _, classified_target = _classify_terminal_vector(
            stage="formal-3000", gates=gates, states=states
        )
        if classified != status: raise ValueError("formal vector contradicts analysis status")
    terminal_failure = payload["terminal_failure"]
    needs_failure = status in ("DEVELOPMENT_ENGINEERING_FAILED","FORMAL_ENGINEERING_FAILED")
    if needs_failure != (terminal_failure is not None): raise ValueError("terminal failure nullability mismatch")
    if terminal_failure is not None:
        row = _exact_keys(terminal_failure, {"cohort","stage","phase","target_updates","task_id","profile",
                                            "budget_id","run_path","completed_updates","failure_code","status_path",
                                            "status_sha256","checkpoint_path","checkpoint_sha256","observation_path",
                                            "observation_sha256","failure_record_path","failure_record_sha256"},
                          label="terminal failure")
        if (row["phase"],row["target_updates"]) not in (("GATE_STEP_0002",2),("GATE_STEP_0200",200),("ENDPOINT_ADVANCE",3000)):
            raise ValueError("terminal failure phase/target mismatch")
    if status == "DEVELOPMENT_NOT_CONFIRMED":
        if not authorizations or authorizations[-1]["status"] != "MECHANISM_NOT_CONFIRMED" or formal is not None:
            raise ValueError("development MNC chain is inconsistent")
    elif status == "DEVELOPMENT_ENGINEERING_FAILED":
        if formal is not None or (authorizations and authorizations[-1]["status"] == "MECHANISM_NOT_CONFIRMED"):
            raise ValueError("development engineering failure has later evidence")
        classified, _, target = _classify_terminal_vector(
            stage=states[0]["stage"], gates=gates, states=states
        )
        if classified != status or terminal_failure["target_updates"] != target:
            raise ValueError("development failure vector/target mismatch")
        classified_target = target
    elif status == "FORMAL_ENGINEERING_FAILED":
        if len(authorizations) != 3 or authorizations[-1]["status"] != "PASS" or formal is None:
            raise ValueError("formal failure lacks development PASS/formal matrix")
    if terminal_failure is not None:
        if classified_target is None:
            raise ValueError("terminal failure has no classified target")
        _validate_terminal_failure_crosslink(
            terminal_failure, states, target=classified_target
        )
    current_stage = states[0]["stage"]
    required_gates = _required_gate_sequence(
        status=status, stage=current_stage,
        target=None if terminal_failure is None else int(terminal_failure["target_updates"]),
    )
    if tuple((row["stage"], row["cap"]) for row in gates) != required_gates:
        raise ValueError("report lacks the exact required runner gate sequence")
    scope = _exact_keys(payload["inference_scope"], {"task_ids","views_per_task","statistical_units",
                                                     "aggregation","dice_role","iou_hypotheses","postprocess_included"},
                        label="inference scope")
    if scope != {"task_ids":list(TASK_IDS),"views_per_task":6,"statistical_units":10,
                 "aggregation":"task_equal","dice_role":"corroborative","iou_hypotheses":2,
                 "postprocess_included":False}:
        raise ValueError("inference scope changed")
    implementation = _exact_keys(payload["implementation"], {"analyzer_path","analyzer_sha256",
                                                              "statistics_path","statistics_sha256"},
                                 label="implementation")
    if Path(implementation["analyzer_path"]) != Path(__file__).resolve() or Path(implementation["statistics_path"]) != PROJECT_ROOT / "shadowforge_pipeline/soft_voxel_v7/statistics.py":
        raise ValueError("implementation path identity mismatch")
    _digest(implementation["analyzer_sha256"], label="analyzer SHA")
    _digest(implementation["statistics_sha256"], label="statistics SHA")


def validate_sealed_soft_report(path: Path) -> AuthenticatedSoftReport:
    report_path = Path(path)
    if not report_path.is_absolute() or report_path.name != "soft_report.json" or report_path.parent.name != "analysis":
        raise ValueError("sealed report path must be canonical absolute analysis/soft_report.json")
    payload, raw = _decode_report(report_path)
    required = {"schema","analysis_status","root_path","campaign","development_authorizations","formal_matrix",
                "runner_gates","terminal_failure","cells","csv","statistics","labels","inference_scope",
                "implementation","terminal_run_states","root_evidence_files","external_evidence_files","semantic_sha256"}
    if set(payload) != required or payload["schema"] != "pcraft_soft_loss_v7_soft_report_v1":
        raise ValueError("sealed soft report has the wrong exact schema")
    root = Path(payload["root_path"])
    if report_path != root / "analysis/soft_report.json": raise ValueError("report/root path mismatch")
    provenance = importlib.import_module("shadowforge_pipeline.soft_voxel_v7.provenance")
    provenance.validate_v7_root(root)
    _validate_report_shape(payload, root)
    campaign_snapshot = provenance.snapshot_regular_file(
        root / "manifests/campaign.json", root=root, exclusive=True
    )
    campaign_payload = provenance.decode_canonical_json(
        campaign_snapshot.payload, label="sealed report campaign"
    )
    if (
        campaign_snapshot.payload != provenance.canonical_json_bytes(campaign_payload)
        or campaign_snapshot.sha256 != payload["campaign"]["raw_sha256"]
        or campaign_payload.get("semantic_sha256") != payload["campaign"]["semantic_sha256"]
        or campaign_payload.get("core",{}).get("source_bundle",{}).get("semantic_sha256")
           != payload["campaign"]["training_source_bundle_semantic_sha256"]
    ):
        raise ValueError("report campaign identity differs from authenticated bytes")
    unsigned_campaign = dict(campaign_payload); campaign_semantic = unsigned_campaign.pop("semantic_sha256", None)
    if campaign_semantic != provenance.canonical_sha256(unsigned_campaign):
        raise ValueError("campaign semantic SHA does not recompute")
    core = campaign_payload["core"]
    if core["root_path"] != str(root) or core["campaign_path"] != "manifests/campaign.json" or core["device"] != "cuda:0":
        raise ValueError("campaign root/device identity mismatch")
    if provenance.canonical_sha256(core) != campaign_payload["campaign_core_sha256"]:
        raise ValueError("campaign core SHA does not recompute")
    authenticated_suite_payloads={}
    for kind in ("development","formal"):
        suite_record = core[f"{kind}_suite"]
        suite_snapshot = provenance.snapshot_regular_file(Path(suite_record["path"]), root=root, exclusive=True)
        suite_payload = provenance.decode_canonical_json(suite_snapshot.payload, label=f"sealed {kind} suite")
        suite_unsigned = dict(suite_payload); suite_semantic = suite_unsigned.pop("semantic_sha256", None)
        if (suite_snapshot.sha256 != suite_record["sha256"] or suite_semantic != suite_record["semantic_sha256"]
            or suite_semantic != provenance.canonical_sha256(suite_unsigned)):
            raise ValueError("campaign suite identity changed")
        authenticated_suite_payloads[kind]=suite_payload
    _snapshot_task2_dependencies(provenance,authenticated_suite_payloads["formal"])
    for authority_key in ("specification","protected_inventory"):
        record = core[authority_key]
        snapshot = provenance.snapshot_regular_file(Path(record["path"]), exclusive=True)
        if snapshot.sha256 != record["sha256"]:
            raise ValueError(f"campaign {authority_key} bytes changed")
    bundle = core["source_bundle"]
    bundle_unsigned = dict(bundle); bundle_semantic = bundle_unsigned.pop("semantic_sha256", None)
    if bundle_semantic != provenance.canonical_sha256(bundle_unsigned):
        raise ValueError("training source bundle semantic SHA changed")
    source_root = Path(bundle["project_root"])
    for relative,digest in bundle["files_sha256"].items():
        source_snapshot = provenance.snapshot_regular_file(source_root / relative, root=source_root, exclusive=True)
        if source_snapshot.sha256 != digest: raise ValueError("training source bytes changed")
    inventory_snapshot = provenance.snapshot_regular_file(Path(core["protected_inventory"]["path"]), exclusive=True)
    inventory_lines = inventory_snapshot.payload.decode("utf-8").splitlines()
    if len(inventory_lines) != core["protected_inventory"]["entry_count"]:
        raise ValueError("protected inventory count changed")
    for line in inventory_lines:
        digest, relative = line.split("  ",1)
        protected_path=Path(relative) if Path(relative).is_absolute() else PROJECT_ROOT/relative
        protected = provenance.snapshot_regular_file(
            protected_path,root=None if Path(relative).is_absolute() else PROJECT_ROOT,
            exclusive=False,
        )
        if protected.sha256 != digest: raise ValueError("protected inventory member changed")
    interpreter_snapshot = provenance.snapshot_regular_file(LOCKED_INTERPRETER, exclusive=False)
    if interpreter_snapshot.sha256 != LOCKED_INTERPRETER_SHA256:
        raise ValueError("locked interpreter bytes changed")
    authorization_payloads = {}
    authorization_evidence_payloads = {}
    for record in payload["development_authorizations"]:
        evidence_snapshot = provenance.snapshot_regular_file(
            root / record["evidence_path"], root=root, exclusive=True
        )
        evidence_decoded = provenance.decode_canonical_json(
            evidence_snapshot.payload, label="sealed development evidence"
        )
        if (
            evidence_snapshot.payload != provenance.canonical_json_bytes(evidence_decoded)
            or evidence_snapshot.sha256 != record["evidence_raw_sha256"]
            or evidence_decoded.get("semantic_sha256") != record["evidence_semantic_sha256"]
        ):
            raise ValueError("authorization evidence report identity differs from bytes")
        evidence_decoded = _validate_local_stage_evidence(
            evidence_decoded, stage=record["stage"],
            campaign_sha=payload["campaign"]["raw_sha256"],
            endpoint={"development-0300":300,"development-1000":1000,
                      "development-3000":3000}[record["stage"]],
        )
        snapshot = provenance.snapshot_regular_file(root / record["path"], root=root, exclusive=True)
        decoded = provenance.decode_canonical_json(snapshot.payload, label="sealed development authorization")
        if snapshot.payload != provenance.canonical_json_bytes(decoded) or snapshot.sha256 != record["raw_sha256"] or decoded.get("semantic_sha256") != record["semantic_sha256"] or decoded.get("evidence_sha256") != record["evidence_sha256"] or decoded.get("status") != record["status"] or decoded.get("selected_budget_id") != record["selected_budget_id"]:
            raise ValueError("authorization report identity differs from bytes")
        unsigned = dict(decoded); semantic = unsigned.pop("semantic_sha256", None)
        if semantic != provenance.canonical_sha256(unsigned):
            raise ValueError("authorization semantic SHA mismatch")
        if decoded["evidence_sha256"] != evidence_decoded["evidence_sha256"]:
            raise ValueError("authorization and evidence file are not one chain")
        authorization_payloads[record["stage"]] = decoded
        authorization_evidence_payloads[record["stage"]] = evidence_decoded
    _validate_local_authorization_chain(
        payload["development_authorizations"], authorization_payloads,
        authorization_evidence_payloads,
        campaign_sha=payload["campaign"]["raw_sha256"],
    )
    formal_payload = None
    if payload["formal_matrix"] is not None:
        record = payload["formal_matrix"]
        snapshot = provenance.snapshot_regular_file(root / record["path"], root=root, exclusive=True)
        decoded = provenance.decode_canonical_json(snapshot.payload, label="sealed formal matrix")
        unsigned = dict(decoded); semantic = unsigned.pop("semantic_sha256", None)
        if snapshot.sha256 != record["raw_sha256"] or semantic != record["semantic_sha256"] or semantic != provenance.canonical_sha256(unsigned) or decoded.get("core",{}).get("selected_budget_id") != record["selected_budget_id"]:
            raise ValueError("formal matrix report identity differs from bytes")
        formal_payload = _validate_local_formal_matrix(
            decoded, root=root, campaign=campaign_payload,
            final_authorization=authorization_payloads["development-3000"],
            final_authorization_sha=payload["development_authorizations"][-1]["raw_sha256"],
        )
    authorization_records = {
        row["stage"]:row for row in payload["development_authorizations"]
    }
    for record in payload["runner_gates"]:
        snapshot = provenance.snapshot_regular_file(root / record["path"], root=root, exclusive=True)
        decoded = provenance.decode_canonical_json(snapshot.payload, label="sealed runner gate")
        unsigned = dict(decoded); semantic = unsigned.pop("semantic_sha256", None)
        if snapshot.sha256 != record["raw_sha256"] or semantic != record["semantic_sha256"] or semantic != provenance.canonical_sha256(unsigned) or decoded.get("stage") != record["stage"] or decoded.get("cap") != record["cap"]:
            raise ValueError("runner gate report identity differs from bytes")
        matrix_sha, authorization_sha, _ = _local_stage_matrix_identity(
            campaign_payload, campaign_sha=payload["campaign"]["raw_sha256"],
            stage=record["stage"], authorization_payloads=authorization_payloads,
            authorization_records=authorization_records,
            formal_payload=formal_payload,
        )
        _validate_local_gate_payload(
            decoded, record, campaign_sha=payload["campaign"]["raw_sha256"],
            stage_matrix_sha=matrix_sha, authorization_sha=authorization_sha,
        )
    if payload["analysis_status"] == "DEVELOPMENT_NOT_CONFIRMED":
        stage = payload["terminal_run_states"][0]["stage"]
        _validate_mnc_vector(
            stage, SimpleNamespace(evidence=authorization_evidence_payloads[stage]),
            payload["terminal_run_states"],
        )
    terminal_stage=payload["terminal_run_states"][0]["stage"]
    if terminal_stage=="formal-3000":
        _,_,terminal_physical=_local_stage_matrix_identity(
            campaign_payload,campaign_sha=payload["campaign"]["raw_sha256"],
            stage=terminal_stage,authorization_payloads=authorization_payloads,
            authorization_records=authorization_records,formal_payload=formal_payload,
        )
        terminal_cohort="formal"
    elif payload["analysis_status"]=="DEVELOPMENT_NOT_CONFIRMED":
        terminal_physical=[{key:row[key] for key in (
            "physical_index","task_id","budget_id","profile","config_sha256","run_path"
        )} for row in authorization_evidence_payloads[terminal_stage]["engineering_rows"]]
        terminal_cohort="development"
    else:
        _,_,terminal_physical=_local_stage_matrix_identity(
            campaign_payload,campaign_sha=payload["campaign"]["raw_sha256"],
            stage=terminal_stage,authorization_payloads=authorization_payloads,
            authorization_records=authorization_records,formal_payload=formal_payload,
        )
        terminal_cohort="development"
    _validate_local_terminal_matrix(
        payload["terminal_run_states"],terminal_physical,
        cohort=terminal_cohort,stage=terminal_stage,
    )
    expected_root, expected_external = _reconstruct_expected_evidence_membership(
        payload, root, provenance
    )
    observed_by_domain: dict[str, dict[str, dict[str, object]]] = {"root":{},"external":{}}
    for domain, records in ((root, payload["root_evidence_files"]),(None,payload["external_evidence_files"])):
        if type(records) is not list: raise ValueError("evidence records must be arrays")
        previous = None
        observed_paths = set()
        for row in records:
            if type(row) is not dict or set(row) != {"path","sha256","size","device","inode","nlink","mtime_ns","ctime_ns","exclusive"}:
                raise ValueError("evidence seal schema mismatch")
            candidate = (domain / row["path"]) if domain is not None else Path(row["path"])
            snap = provenance.snapshot_regular_file(candidate, root=domain, exclusive=row["exclusive"])
            observed = _seal_record(_compact_snapshot(snap), root=domain)
            if observed != row: raise ValueError("sealed evidence changed")
            ordering = row["path"].encode()
            if previous is not None and ordering <= previous: raise ValueError("evidence order mismatch")
            previous = ordering
            if row["path"] in observed_paths: raise ValueError("duplicate evidence record")
            observed_paths.add(row["path"])
            observed_by_domain["root" if domain is not None else "external"][row["path"]] = row
        expected = expected_root if domain is not None else expected_external
        if observed_paths != set(expected):
            missing=sorted(set(expected)-observed_paths,key=lambda value:value.encode())
            extra=sorted(observed_paths-set(expected),key=lambda value:value.encode())
            raise ValueError(f"sealed evidence membership is missing or extra: missing={missing!r}, extra={extra!r}")
        if any(row["exclusive"] != expected[row["path"]] for row in records):
            raise ValueError("sealed evidence exclusivity differs from producer role")
    _validate_exact_reached_namespace(
        root, expected_root_files=set(expected_root),
        absent_run_paths=[row["run_path"] for row in payload["terminal_run_states"] if row["state"] == "ABSENT"],
    )
    _validate_downstream_run_namespaces(
        root, payload["terminal_run_states"], payload["cells"]
    )
    root_seals = observed_by_domain["root"]
    external_seals = observed_by_domain["external"]
    def bound(relative: str, digest: str) -> None:
        if relative not in root_seals or root_seals[relative]["sha256"] != digest:
            raise ValueError("report identity is not bound to root evidence seal")
    bound("manifests/campaign.json", payload["campaign"]["raw_sha256"])
    for row in payload["development_authorizations"]:
        bound(row["path"], row["raw_sha256"])
        bound(row["evidence_path"], row["evidence_raw_sha256"])
    if payload["formal_matrix"] is not None: bound(payload["formal_matrix"]["path"], payload["formal_matrix"]["raw_sha256"])
    for row in payload["runner_gates"]: bound(row["path"], row["raw_sha256"])
    for row in payload["terminal_run_states"]:
        for path_key, sha_key in (("status_path","status_sha256"),("checkpoint_path","checkpoint_sha256"),
                                  ("observation_path","observation_sha256"),("failure_record_path","failure_record_sha256")):
            if row[path_key] is not None: bound(row[path_key], row[sha_key])
    if payload["terminal_failure"] is not None:
        terminal=payload["terminal_failure"]
        failed_state=next(row for row in payload["terminal_run_states"] if row["state"]=="FAILED")
        observation_snapshot=provenance.snapshot_regular_file(
            root / terminal["observation_path"],root=root,exclusive=True
        )
        failure_snapshot=provenance.snapshot_regular_file(
            root / terminal["failure_record_path"],root=root,exclusive=True
        )
        observation=provenance.decode_canonical_json(
            observation_snapshot.payload,label="sealed failure observation"
        )
        failure_record=provenance.decode_canonical_json(
            failure_snapshot.payload,label="sealed failure record"
        )
        if (observation_snapshot.payload!=provenance.canonical_json_bytes(observation)
            or failure_snapshot.payload!=provenance.canonical_json_bytes(failure_record)):
            raise ValueError("failure payload is not canonical JSON")
        _validate_local_failure_payloads(
            terminal,failed_state,observation,failure_record,
            source_bundle_sha=campaign_payload["core"]["source_bundle"]["semantic_sha256"],
            profile=str(failed_state["profile"]),
        )
    for cell in payload["cells"]:
        _validate_local_cell_payloads(cell)
        run = cell["run_path"]
        for relative, key in ((f"{run}/artifacts/manifest.json","artifact_manifest_sha256"),
                              (f"{run}/artifacts/final_logits.npy","final_logits_sha256"),
                              (f"{run}/artifacts/final_density.npy","final_density_sha256"),
                              (f"{run}/artifacts/final_hard_005.npy","final_hard_005_sha256"),
                              (cell["metric_row"]["metrics_path"],"endpoint_metrics_sha256")):
            bound(relative, cell[key])
        if cell["checkpoint_sha256"] != cell["metric_row"]["checkpoint_sha256"] or cell["config_sha256"] != cell["metric_row"]["config_sha256"]:
            raise ValueError("cell identity differs from embedded metric row")
        endpoint_snapshot = provenance.snapshot_regular_file(
            root / cell["metric_row"]["metrics_path"], root=root, exclusive=True
        )
        endpoint = provenance.decode_canonical_json(endpoint_snapshot.payload, label="sealed endpoint metrics")
        endpoint_keys = {"schema","config_sha256","stage","stage_matrix_sha256","authorization_sha256",
                         "completed_updates","threshold","checkpoint_path","checkpoint_sha256",
                         "target_stack_sha256","camera_R_sha256","camera_T_sha256","camera_scale_sha256",
                         "camera_K_sha256","per_view","aggregate","semantic_sha256"}
        if (set(endpoint) != endpoint_keys or endpoint["schema"] != "pcraft_soft_loss_v7_endpoint_metrics_v1"
            or endpoint["config_sha256"] != cell["config_sha256"] or endpoint["completed_updates"] != 3000
            or endpoint["threshold"] != .05 or endpoint["checkpoint_sha256"] != cell["checkpoint_sha256"]
            or endpoint["target_stack_sha256"] != cell["target_stack_sha256"]
            or endpoint["per_view"] != cell["per_view_metrics"]):
            raise ValueError("embedded cell differs from endpoint metric bytes")
        aggregate = endpoint["aggregate"]
        _validate_local_endpoint_metric_values(endpoint["per_view"],aggregate)
        metric = cell["metric_row"]
        if aggregate != {"mean_iou":metric["iou"],"mean_dice":metric["dice"],
                         "mean_recall":metric["recall"],"mean_fpr":metric["fpr"]}:
            raise ValueError("embedded metric row differs from endpoint aggregate")
        unsigned_endpoint = dict(endpoint); endpoint_semantic = unsigned_endpoint.pop("semantic_sha256")
        if endpoint_semantic != provenance.canonical_sha256(unsigned_endpoint):
            raise ValueError("endpoint metric semantic SHA mismatch")
        manifest_snapshot = provenance.snapshot_regular_file(
            root / run / "artifacts/manifest.json", root=root, exclusive=True
        )
        manifest = provenance.decode_canonical_json(manifest_snapshot.payload, label="sealed artifact manifest")
        expected_artifacts={
            PurePosixPath(relative).relative_to(run).as_posix():seal["sha256"]
            for relative,seal in root_seals.items()
            if relative.startswith(f"{run}/artifacts/")
            and relative!=f"{run}/artifacts/manifest.json"
        }
        _validate_local_artifact_manifest(
            manifest,config_sha=cell["config_sha256"],
            checkpoint_sha=cell["checkpoint_sha256"],
            expected_artifacts=expected_artifacts,
        )
        artifact_map = {row["path"]:row["sha256"] for row in manifest.get("artifacts",()) if type(row) is dict}
        for relative,key in (("artifacts/final_logits.npy","final_logits_sha256"),
                             ("artifacts/final_density.npy","final_density_sha256"),
                             ("artifacts/final_hard_005.npy","final_hard_005_sha256"),
                             ("artifacts/metrics/step_3000.json","endpoint_metrics_sha256")):
            if artifact_map.get(relative) != cell[key]: raise ValueError("cell artifact SHA differs from manifest")
        if "artifacts/audit.json" not in artifact_map or "artifacts/resource.json" not in artifact_map:
            raise ValueError("artifact manifest omits terminal audit/resource")
    for path_key, sha_key in (("analyzer_path","analyzer_sha256"),("statistics_path","statistics_sha256")):
        path_value = payload["implementation"][path_key]
        if path_value not in external_seals or external_seals[path_value]["sha256"] != payload["implementation"][sha_key]:
            raise ValueError("implementation identity is not bound to external evidence")
    csv_path = root / payload["csv"]["path"]
    csv_snap = provenance.snapshot_regular_file(csv_path, root=root, exclusive=True)
    cells = payload["cells"]
    expected_csv = _csv_bytes(cells if payload["analysis_status"] == "FORMAL_COMPLETE" else ())
    if csv_snap.payload != expected_csv or csv_snap.sha256 != payload["csv"]["raw_sha256"]:
        raise ValueError("sealed CSV differs from embedded evidence")
    statistics = importlib.import_module(_STATISTICS_MODULE)
    if payload["analysis_status"] == "FORMAL_COMPLETE":
        expected = statistics.evaluate_formal_outcome(
            [cell["metric_row"] for cell in cells], [cell["engineering_row"] for cell in cells])
        if _plain(expected) != payload["statistics"] or _plain(expected["labels"]) != payload["labels"]:
            raise ValueError("sealed statistics differ from embedded 40-row evidence")
    elif payload["statistics"] is not None:
        raise ValueError("noncomplete report cannot carry statistics")
    return AuthenticatedSoftReport(
        root, report_path, hashlib.sha256(raw).hexdigest(), csv_path, csv_snap.sha256,
        payload["analysis_status"], payload["labels"]["SOFT_PASS"],
        payload["labels"]["BOTH_KEY"], payload["labels"]["claim_allowed"], payload,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = analyze(args.output_root, verify=args.verify)
        print(json.dumps({"analysis_status":report.analysis_status,
                          "SOFT_PASS":report.soft_pass,"BOTH_KEY":report.both_key,
                          "claim_allowed":report.claim_allowed}, sort_keys=True))
        return 0
    except (ValueError, RuntimeError, OSError) as error:
        if argv is None:
            print(f"error: {error}", file=sys.stderr)
            return 2
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AuthenticatedSoftReport", "analyze", "validate_sealed_soft_report", "build_parser", "main"]
