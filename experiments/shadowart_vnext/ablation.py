"""Frozen experiment identities, artifact-backed launch gates and paired inference.

This module never launches jobs or writes manifests. Root owns immutable
publication. Confirmatory output cannot be reclassified after changing losses.
"""
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import re
import statistics
from zipfile import BadZipFile, ZipFile

import numpy as np

from .losses import LossConfig
from .contracts import CameraContract, build_contract, mask_hash


DEV_SEEDS = (17, 42)
CONFIRM_SEEDS = (17, 42, 2026)
MANUFACTURING_CHECKS = ("one_body", "watertight", "winding_consistent",
    "self_intersections_clear", "thickness_pass", "slicer_pass", "support_removal_pass")
REQUIRED_NONLOSS = {"resolution", "image_size", "steps", "lr", "extent", "pixel_supersampling",
    "optimizer", "initialization", "occupancy_rule", "view_update_mode", "preprocessing",
    "budgets", "export_policy", "renderer_digest", "source_hashes"}


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("manifest values must be finite JSON data") from error


def fingerprint(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _snapshot(value):
    return json.loads(_json(value))


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected SHA-256 hex digest")
    return value


def _tasks(tasks, count):
    parsed = _snapshot(tasks)
    if not isinstance(parsed, list) or len(parsed) != count:
        raise ValueError(f"expected exactly {count} registered tasks")
    for task in parsed:
        if set(task) != {"task_id", "object_id", "view_count", "tsf"}:
            raise ValueError("task identity schema is incomplete")
        if any(not isinstance(task[k], str) or not task[k] for k in ("task_id", "object_id")):
            raise ValueError("tasks require nonempty source/object identities")
        if type(task["view_count"]) is not int or task["view_count"] <= 0 or type(task["tsf"]) is not int or task["tsf"] not in (0, 1):
            raise ValueError("invalid orthographic view-count/preset stratum")
    if len({t["task_id"] for t in parsed}) != count or len({t["object_id"] for t in parsed}) != count:
        raise ValueError("independent task/object identities must be unique")
    return sorted(parsed, key=lambda row: row["task_id"])


def _terms(config):
    terms = []
    if config.core in {"l1", "l2_l1"}: terms.append("l1")
    if config.core in {"l2", "l2_l1"}: terms.append("l2")
    if config.auxiliary != "none": terms.append(config.auxiliary)
    return terms


def _arms(arms):
    if not isinstance(arms, dict) or "full" not in arms or not 2 <= len(arms) <= 5:
        raise ValueError("register full and its deletion arms, at most five arms total")
    parsed = {name: (value if isinstance(value, LossConfig) else LossConfig(**value)) for name, value in arms.items()}
    full = parsed["full"]
    if full.control or not _terms(full):
        raise ValueError("full must be a proposed nonempty objective")
    expected = {"full":full}
    expected.update({"without_" + term:full.without(term) for term in _terms(full)})
    for name, value in expected.items():
        if parsed.get(name) != value:
            raise ValueError("each retained primitive needs a coefficient-preserving deletion arm")
    extras = set(parsed) - set(expected)
    if len(extras) > 1 or any(parsed[name].control != "current_proposal" for name in extras):
        raise ValueError("only one historical control may accompany the deletion matrix")
    return {name: config.to_dict() for name, config in sorted(parsed.items())}


def _nonloss(manifest):
    keys = ("nonloss_settings", "targets", "camera_digests", "camera_contracts",
        "geometry_settings", "dev_tasks", "confirm_tasks", "dev_seeds", "confirm_seeds")
    if any(key not in manifest for key in keys):
        raise ValueError("frozen manifest requires complete camera contracts and nonloss conditions")
    return {key:manifest[key] for key in keys}


def _settings(settings):
    if not isinstance(settings, dict) or not REQUIRED_NONLOSS <= set(settings):
        raise ValueError("nonloss settings omit renderer, preprocessing or training/export identity")
    for name,expected in (("resolution",256),("image_size",256),("steps",2000)):
        if type(settings[name]) is not int or settings[name] != expected:
            raise ValueError("formal protocol requires native256 and 2000 steps")
    if type(settings["pixel_supersampling"]) is not int or settings["pixel_supersampling"] < 2:
        raise ValueError("formal pixel supersampling must be at least 2; camera preflight may require more")
    for name in ("extent","lr"):
        value = settings[name]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("training scales must be finite positive numbers")
    if settings["view_update_mode"] != "joint_view_equal" or settings["export_policy"] != "final_step":
        raise ValueError("joint updates and final-step export are fixed")
    _sha(settings["renderer_digest"])
    if not isinstance(settings["source_hashes"],dict) or not settings["source_hashes"]:
        raise ValueError("source implementation hashes are required")
    for value in settings["source_hashes"].values(): _sha(value)


def _task_sets(dev, confirm):
    dev,confirm = _tasks(dev,4),_tasks(confirm,8)
    if {t["task_id"] for t in dev} & {t["task_id"] for t in confirm} or {t["object_id"] for t in dev} & {t["object_id"] for t in confirm}:
        raise ValueError("development and confirmation identities must not overlap")
    if "heroes" not in {t["task_id"] for t in dev}:
        raise ValueError("Heroes must remain a development task")
    if {(t["view_count"],t["tsf"]) for t in confirm} != {(n,t) for n in (2,3,4,6) for t in (0,1)}:
        raise ValueError("confirmation requires N2/3/4/6 x tsf0/1")
    return dev,confirm


def _task_contracts(settings, targets, camera_digests, camera_contracts, tasks):
    """Bind each registered preset stratum to its complete validated geometry."""
    task_ids = {task["task_id"] for task in tasks}
    if not isinstance(camera_contracts,dict) or set(camera_contracts) != task_ids:
        raise ValueError("camera contracts must bind every registered task exactly once")
    if set(targets) != task_ids or set(camera_digests) != task_ids:
        raise ValueError("targets and cameras must bind every registered task exactly once")
    result = {}
    for task in tasks:
        name = task["task_id"]
        raw = camera_contracts[name]
        try:
            contract = CameraContract.from_dict(raw.to_dict() if isinstance(raw,CameraContract) else raw)
        except (TypeError,KeyError,ValueError,AttributeError) as error:
            raise ValueError("invalid camera contract for " + name) from error
        expected = build_contract(task["view_count"],tsf=task["tsf"],
            image_size=settings["image_size"],resolution=settings["resolution"],extent=settings["extent"])
        if (type(contract.base_count) is not int or contract.base_count != task["view_count"]
                or contract.protocol != expected.protocol or contract.cameras != expected.cameras
                or contract.coordinates != expected.coordinates
                or type(contract.image_size) is not int or contract.image_size != settings["image_size"]
                or type(contract.volume_resolution) is not int or contract.volume_resolution != settings["resolution"]
                or contract.extent != settings["extent"]):
            raise ValueError("camera contract does not match registered preset stratum: " + name)
        if _sha(camera_digests[name]) != contract.digest:
            raise ValueError("camera digest does not match full camera contract: " + name)
        if len(contract.input_hashes) != contract.base_count:
            raise ValueError("camera contract requires original input hashes: " + name)
        for digest in contract.input_hashes + contract.target_hashes: _sha(digest)
        record = targets[name]
        if (not isinstance(record,dict) or record.get("dtype") != "bool"
                or record.get("shape") != [len(contract.cameras),settings["image_size"],settings["image_size"]]
                or not isinstance(record.get("hashes"),list)
                or len(record["hashes"]) != len(contract.cameras)
                or record["hashes"] != list(contract.target_hashes)):
            raise ValueError("targets disagree with camera dimensions, views or hashes: " + name)
        result[name] = contract.to_dict()
    return _snapshot(result)


def build_manifest(nonloss_settings, targets, camera_digest, geometry_settings, arms,
                   dev_tasks, confirm_tasks, *, camera_contracts=None, witness=None):
    settings = _snapshot(nonloss_settings)
    _settings(settings)
    dev, confirm = _task_sets(dev_tasks, confirm_tasks)
    task_ids = {t["task_id"] for t in dev + confirm}
    if set(targets) != task_ids or set(camera_digest) != task_ids:
        raise ValueError("targets and cameras must bind every registered task exactly once")
    target_records = {}
    for name, target in targets.items():
        array = np.asarray(target)
        if array.ndim != 3 or min(array.shape) <= 0 or array.shape[1] != array.shape[2] or not np.isin(array, [0,1]).all():
            raise ValueError("targets require nonempty binary square image stacks")
        array = np.ascontiguousarray(array, dtype=bool)
        target_records[name] = {"shape":list(array.shape), "dtype":"bool",
            "hashes":[mask_hash(mask) for mask in array]}
    contracts = _task_contracts(settings,target_records,camera_digest,camera_contracts,dev+confirm)
    for name,raw in contracts.items():
        array = np.asarray(targets[name],dtype=bool)
        for i,camera in enumerate(raw["cameras"]):
            base = camera["derived_from"]
            if base is not None and not np.array_equal(array[i],array[base,:,::-1]):
                raise ValueError("derived opposite target is not the exact base reflection: " + name)
    manifest = {"schema":"shadowart_vnext_ablation_v1", "nonloss_settings":settings,
        "targets":target_records, "camera_digests":{k:_sha(v) for k,v in camera_digest.items()},
        "camera_contracts":contracts,
        "geometry_settings":_snapshot(geometry_settings), "geometry_digest":fingerprint(geometry_settings),
        "arms":_arms(arms), "dev_tasks":dev, "confirm_tasks":confirm,
        "dev_seeds":list(DEV_SEEDS), "confirm_seeds":list(CONFIRM_SEEDS),
        "primary":"final_mean_mesh_iou_with_failed_delivery_scored_zero",
        "practical_delta":.001, "quality_gates":{"mean_strictly_above":.98,"worst_at_least":.97},
        "inference_unit":"object_task_seed_average", "confirmation_max_jobs":120}
    manifest["nonloss_fingerprint"] = fingerprint(_nonloss(manifest))
    manifest["digest"] = fingerprint(manifest)
    if witness is not None:
        verify_heroes_witness(manifest, witness)
    return manifest


def validate_manifest(manifest):
    data = _snapshot(manifest)
    digest = data.pop("digest", None)
    if digest != fingerprint(data):
        raise ValueError("manifest content changed after freezing")
    data["digest"] = digest
    if data["nonloss_fingerprint"] != fingerprint(_nonloss(data)) or data["geometry_digest"] != fingerprint(data["geometry_settings"]):
        raise ValueError("nonloss conditions changed")
    if data["dev_seeds"] != list(DEV_SEEDS) or data["confirm_seeds"] != list(CONFIRM_SEEDS):
        raise ValueError("registered seed policy changed")
    if data["practical_delta"] != .001 or data["quality_gates"] != {"mean_strictly_above":.98,"worst_at_least":.97}:
        raise ValueError("registered scientific thresholds changed")
    _settings(data["nonloss_settings"])
    dev,confirm = _task_sets(data["dev_tasks"],data["confirm_tasks"])
    task_ids = {t["task_id"] for t in dev + confirm}
    if set(data["targets"]) != task_ids or set(data["camera_digests"]) != task_ids:
        raise ValueError("registered target/camera matrix changed")
    for camera_digest in data["camera_digests"].values(): _sha(camera_digest)
    _task_contracts(data["nonloss_settings"],data["targets"],data["camera_digests"],
                    data["camera_contracts"],dev+confirm)
    _arms(data["arms"])
    return data


def _artifact(path_string, digest, *, allow_empty=False):
    _sha(digest)
    if not isinstance(path_string, str) or not Path(path_string).is_absolute():
        raise ValueError("evidence paths must be absolute")
    try:
        content = Path(path_string).read_bytes()
    except OSError as error:
        raise ValueError("required evidence artifact is missing") from error
    if (not content and not allow_empty) or hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("evidence artifact hash mismatch")
    return content


def _verified_sliced_files(slicing, mesh_sha256):
    """Reopen the bytes that generated the archived audit and human review."""
    paths, hashes = slicing["source_paths"], slicing["source_hashes"]
    if hashes["mesh_sha256"] != mesh_sha256:
        raise ValueError("sliced artifacts refer to a different mesh")
    _artifact(paths["mesh"],mesh_sha256)
    content = _artifact(paths["archive"],hashes["archive_sha256"])
    _artifact(paths["effective_settings"],hashes["effective_settings_sha256"])
    _artifact(paths["slicer_result"],hashes["result_sha256"])
    if paths.get("slicer_log") is not None or "slicer_log_sha256" in hashes:
        _artifact(paths["slicer_log"],hashes["slicer_log_sha256"],allow_empty=True)
    member = paths["gcode_member"]
    try:
        with ZipFile(io.BytesIO(content)) as archive:
            names = archive.namelist()
            gcode_names = [name for name in names if re.fullmatch(r"Metadata/plate_\d+\.gcode",name)]
            if len(names) != len(set(names)) or gcode_names != [member]:
                raise ValueError("sliced artifact must bind its single exact G-code archive member")
            info = archive.getinfo(member)
            if info.file_size > 256*1024**2:
                raise ValueError("oversized G-code evidence artifact")
            gcode = archive.read(member)
    except (BadZipFile,KeyError,RuntimeError,NotImplementedError) as error:
        raise ValueError("invalid sliced G-code archive artifact") from error
    if not gcode or hashlib.sha256(gcode).hexdigest() != _sha(hashes["gcode_sha256"]):
        raise ValueError("G-code archive member hash mismatch")


def _passed(report, name, *, status=False, converged=False):
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError(name + " is incomplete or failed")
    if status and report.get("status") != "PASS":
        raise ValueError(name + " is not PASS")
    if converged and report.get("converged") is not True:
        raise ValueError(name + " is not converged")


def _verified_evaluations(rows, quality, view_count):
    """Recompute gates from per-view evidence, including the final resolution pair."""
    from .delivery import converged_quality
    _passed(quality, "reported projection quality", converged=True)
    evaluations = {}
    for row in rows:
        size = row["image_size"]
        if type(size) is not int or size in evaluations or size not in (256,512,1024):
            raise ValueError("invalid or duplicate final QA image resolution")
        values = row["per_view_iou"]
        if len(values) != view_count or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("final per-view IoU evidence is malformed")
        evaluations[size] = values
    result = converged_quality(rows)
    _passed(result, "Heroes final mean>.98 / worst>=.97 and resolution gate", converged=True)
    return evaluations


def _verified_thickness(report):
    _passed(report, "physical thickness", status=True)
    if report["reasons"] or report["review_reasons"] or report["review_required"] is not False or report["component_count_6"] != 1:
        raise ValueError("physical thickness contains unresolved material or sampling evidence")


def _verified_serialized_qa(qa, mesh_sha256, view_count):
    _passed(qa, "serialized QA", status=True)
    if type(qa["native_resolution"]) is not int or qa["native_resolution"] != 256:
        raise ValueError("confirmation requires native 256 geometry")
    aggregate = _verified_evaluations(qa["evaluations"], qa["quality"], view_count)
    if set(qa["serialized"]) != {"obj", "stl"}:
        raise ValueError("both serialized OBJ and STL QA are required")
    projections = []
    for extension in ("obj", "stl"):
        row = qa["serialized"][extension]
        _passed(row, extension + " serialized QA")
        digest = _sha(row["sha256"])
        if extension == "stl" and digest != mesh_sha256:
            raise ValueError("serialized STL hash differs from witness mesh")
        topology = row["topology"]
        _passed(topology, extension + " topology", status=True)
        if (topology["reasons"] or topology["watertight"] is not True
                or topology["winding_consistent"] is not True
                or topology["boundary_edges"] != 0 or topology["nonmanifold_edges"] != 0
                or topology["outer_shells"] != 1 or topology["containment_status"] != "PASS"
                or topology["self_intersection"]["status"] != "PASS"
                or topology["self_intersection"]["selected_faces"] != 0
                or row["component_count_6"] != 1):
            raise ValueError("serialized topology contains failed or unresolved material evidence")
        mesh_scores = _verified_evaluations(row["evaluations"], row["quality"], view_count)
        if set(mesh_scores) != set(aggregate):
            raise ValueError("serialized mesh and aggregate resolution evidence differ")
        projections.append(mesh_scores)
        _verified_evaluations(row["raster_evaluations"], row["raster_quality"], view_count)
        _verified_thickness(row["thickness"])
        _verified_thickness(row["thickness"]["mesh_revoxelized_512"])
    for size, values in aggregate.items():
        conservative = np.minimum(projections[0][size], projections[1][size])
        if not np.allclose(values, conservative, rtol=0, atol=1e-12):
            raise ValueError("aggregate QA does not match conservative serialized mesh evidence")


def verify_heroes_witness(manifest, witness):
    """Authenticate concrete mesh/QA/manufacturing/review files, then check gates.

    The QA producer remains responsible for actual independent rasterization;
    a hash-bound report is not structural certification or a physical print.
    """
    manifest = validate_manifest(manifest)
    try:
        if witness["task_id"] != "heroes":
            raise ValueError("confirmation witness must be Heroes")
        _artifact(witness["mesh_path"], witness["mesh_sha256"])
        qa = json.loads(_artifact(witness["qa_path"], witness["qa_sha256"]))
        manufacturing = json.loads(_artifact(witness["manufacturing_path"], witness["manufacturing_sha256"]))
        if qa["mesh_sha256"] != witness["mesh_sha256"] or manufacturing["mesh_sha256"] != witness["mesh_sha256"]:
            raise ValueError("reports refer to a different serialized mesh")
        for report in (qa, manufacturing):
            if report["camera_digest"] != manifest["camera_digests"]["heroes"] or report["target_hashes"] != manifest["targets"]["heroes"]["hashes"] or report["geometry_digest"] != manifest["geometry_digest"]:
                raise ValueError("witness target/camera/geometry identity mismatch")
        if qa["independent_serialized_mesh"] is not True:
            raise ValueError("soft scores cannot authorize confirmation")
        _verified_serialized_qa(qa, witness["mesh_sha256"], manifest["targets"]["heroes"]["shape"][0])
        _passed(manufacturing, "digital manufacturing", status=True)
        if any(manufacturing["checks"][name] is not True for name in MANUFACTURING_CHECKS):
            raise ValueError("digital manufacturing is incomplete or failed")
        slicing = manufacturing["slicing"]
        _passed(slicing, "sliced artifacts", status=True)
        if slicing["support_removal_pass"] is not True:
            raise ValueError("sliced artifact support removal did not pass")
        for name in ("effective_settings", "slicer_completion", "archive_integrity", "gcode_parse",
                     "gcode_effective_settings", "archive_effective_settings", "warnings",
                     "archive_model", "toolpath_correspondence", "support_removal"):
            _passed(slicing["checks"][name], "sliced artifact " + name, status=True)
        hashes = slicing["source_hashes"]
        _verified_sliced_files(slicing,witness["mesh_sha256"])
        review = manufacturing["support_removal_review"]
        if not isinstance(review["reviewer"],str) or not review["reviewer"].strip():
            raise ValueError("support-removal review needs a named reviewer")
        content = json.loads(_artifact(review["path"], review["sha256"]))
        if content["reviewer"] != review["reviewer"]:
            raise ValueError("support-removal reviewer identity mismatch")
        from .slicer import audit_support_review
        _passed(audit_support_review(content, hashes["mesh_sha256"], hashes["gcode_sha256"],
                                     hashes["archive_sha256"]), "artifact-bound human support review", status=True)
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("incomplete artifact-backed Heroes witness") from error
    return {"status":"WITNESS_VERIFIED", "witness_digest":fingerprint(witness),
            "mesh_sha256":witness["mesh_sha256"], "manufacturing_status":"digital_only"}


def _schedule(manifest, stage):
    task_key, seeds = ("dev_tasks",DEV_SEEDS) if stage == "development" else ("confirm_tasks",CONFIRM_SEEDS)
    return [{"stage":stage, "task_id":task["task_id"], "object_id":task["object_id"],
        "arm":arm, "seed":seed, "manifest_digest":manifest["digest"],
        "nonloss_fingerprint":manifest["nonloss_fingerprint"], "loss_config_digest":fingerprint(config)}
        for task in manifest[task_key] for arm,config in sorted(manifest["arms"].items()) for seed in seeds]


def development_schedule(manifest):
    return _schedule(validate_manifest(manifest), "development")


def confirmation_schedule(manifest, witness):
    data = validate_manifest(manifest)
    verify_heroes_witness(data, witness)
    return _schedule(data, "confirmation")


def calibrate_auxiliary(records, budget):
    """Fixed coefficient from four tasks x three registered gradient-only probes."""
    if isinstance(budget,bool) or budget not in (.1,.2):
        raise ValueError("calibration budget must be .1 or .2")
    rows = list(records)
    if len(rows) != 12:
        raise ValueError("calibration requires all twelve probes")
    tasks = {row.get("task_id") for row in rows}
    keys = {(row.get("task_id"),row.get("step")) for row in rows}
    if len(tasks) != 4 or any(not isinstance(t,str) or not t for t in tasks) or keys != {(t,s) for t in tasks for s in (500,1000,1500)}:
        raise ValueError("calibration probe matrix mismatch")
    ratios = []
    for row in rows:
        core, aux = row.get("core_norm"), row.get("aux_norm")
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0 for v in (core,aux)):
            raise ValueError("calibration norms must be finite and nonnegative")
        if aux == 0:
            raise ValueError("all twelve gradient ratios must be defined; cannot drop inactive probes")
        ratio = core / aux
        if not math.isfinite(ratio): raise ValueError("nonfinite gradient ratio")
        ratios.append(ratio)
    if not ratios or statistics.median(ratios) <= 0:
        raise ValueError("auxiliary/core calibration is inactive")
    return float(budget * statistics.median(ratios))


def exact_sign_flip(differences):
    """One-sided exact paired randomization p-value; no view/seed pseudo-units."""
    values = [float(v) for v in differences]
    if not values or len(values) > 20 or not all(math.isfinite(v) for v in values):
        raise ValueError("sign-flip test requires 1..20 finite paired task differences")
    observed = math.fsum(values)
    count = sum(math.fsum(s*v for s,v in zip(signs,values)) >= observed
                for signs in itertools.product((-1.,1.),repeat=len(values)))
    return count / (2 ** len(values))


def holm_adjust(p_values):
    if any(isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in p_values.values()):
        raise ValueError("p-values must be finite probabilities")
    adjusted, running = {}, 0.
    for rank,(name,p) in enumerate(sorted(p_values.items(),key=lambda row:(row[1],row[0]))):
        running = max(running, min(1.,(len(p_values)-rank)*p))
        adjusted[name] = running
    return adjusted


def _outcome(row):
    if row.get("status") not in {"DELIVERED","FAILED"} or type(row.get("manufacturing_pass")) is not bool:
        raise ValueError("every run must have an explicit delivery/failure outcome")
    if not isinstance(row.get("raw_stages"),dict):
        raise ValueError("raw stage evidence must remain separate")
    _json(row["raw_stages"])
    for key in ("final_mean_iou","final_worst_iou"):
        value = row.get(key)
        if value is None and row["status"] == "FAILED": continue
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("invalid final IoU outcome")
    fin = row.get("fin_volume_mm3")
    if fin is not None and (isinstance(fin,bool) or not isinstance(fin,(int,float)) or not math.isfinite(fin) or fin < 0):
        raise ValueError("invalid fin volume")
    valid = row["status"] == "DELIVERED" and row["manufacturing_pass"]
    return {"score":row["final_mean_iou"] if valid else 0., "worst_score":row["final_worst_iou"] if valid else 0.,
        "fin_volume_mm3":fin, "valid":valid,
        "gate":valid and row["final_mean_iou"] > .98 and row["final_worst_iou"] >= .97}


def _collect(expected, outcomes):
    keys = lambda row:(row["task_id"],row["arm"],row["seed"])
    expected_map = {keys(row):row for row in expected}
    observed = {}
    raw = []
    for row in outcomes:
        try: key = keys(row)
        except (KeyError,TypeError) as error: raise ValueError("outcome identity is missing") from error
        if key in observed or key not in expected_map or any(row.get(k) != v for k,v in expected_map[key].items()):
            raise ValueError("outcome duplicates or changes frozen conditions")
        observed[key] = _outcome(row)
        raw.append(_snapshot(row))
    if set(observed) != set(expected_map):
        raise ValueError("missing outcomes cannot be dropped from confirmation")
    return observed,raw


def rank_development(manifest, outcomes):
    """Frozen lexicographic ranking of the registered proposed/deletion arms.

    Failures have zero fidelity. Historical control stays a reported comparator.
    Ties favor fewer primitives, then registry order. No manufacturing-valid
    candidate yields no selection, keeping confirmation closed.
    """
    data = validate_manifest(manifest)
    observed,_ = _collect(development_schedule(data),outcomes)
    table = []
    for arm,config in sorted(data["arms"].items()):
        parsed = LossConfig(**config)
        if parsed.control: continue
        rows = [row for (task,name,seed),row in observed.items() if name == arm]
        deficits = [max(0.,.98-r["score"],.97-r["worst_score"]) for r in rows]
        valid = sum(row["valid"] for row in rows)
        passes = sum(row["gate"] for row in rows)
        fins = [row["fin_volume_mm3"] for row in rows]
        fin_mean = math.fsum(fins)/len(fins) if all(v is not None for v in fins) else None
        key = (-valid,-passes,max(deficits),math.fsum(deficits)/len(rows),
               -math.fsum(r["worst_score"] for r in rows)/len(rows),
               math.inf if fin_mean is None else fin_mean,len(_terms(parsed)),arm)
        table.append((key,{"arm":arm,"valid_deliveries":valid,"gate_passes":passes,
                           "max_deficit":max(deficits),"mean_deficit":math.fsum(deficits)/len(rows),"mean_fin_volume_mm3":fin_mean}))
    ordered = sorted(table,key=lambda entry:entry[0])
    selected = ordered[0][1]["arm"] if ordered[0][1]["valid_deliveries"] else None
    return {"status":"SELECTED_FOR_DEVELOPMENT" if selected else "GEOMETRY_GATE_NOT_MET",
            "selected_arm":selected,"ranked_arms":[row for _,row in ordered],
            "confirmation_authorized":False}


def analyze_paired(manifest, outcomes, witness):
    """Strict complete-matrix analysis; scientific failure remains an outcome."""
    expected = confirmation_schedule(manifest, witness)
    observed,raw = _collect(expected,outcomes)
    task_ids = [row["task_id"] for row in manifest["confirm_tasks"]]
    contrasts = {}
    for arm in sorted(manifest["arms"]):
        if not arm.startswith("without_"): continue
        deltas, worst_deltas = [], []
        regressions = 0
        for task in task_ids:
            full = [observed[(task,"full",seed)] for seed in CONFIRM_SEEDS]
            deletion = [observed[(task,arm,seed)] for seed in CONFIRM_SEEDS]
            deltas.append(math.fsum(a["score"]-b["score"] for a,b in zip(full,deletion))/3.)
            worst_deltas.append(math.fsum(a["worst_score"]-b["worst_score"] for a,b in zip(full,deletion))/3.)
            regressions += sum(not a["valid"] and b["valid"] for a,b in zip(full,deletion))
        contrasts[arm] = {"task_deltas":deltas, "mean_delta":math.fsum(deltas)/8.,
            "worst_iou_task_deltas":worst_deltas,"mean_worst_iou_delta":math.fsum(worst_deltas)/8.,
            "p_raw":exact_sign_flip(deltas), "manufacturing_regressions":regressions}
    adjusted = holm_adjust({arm:row["p_raw"] for arm,row in contrasts.items()})
    for arm,row in contrasts.items():
        row["p_holm"] = adjusted[arm]
        row["supported"] = row["mean_delta"] >= .001 and row["p_holm"] <= .05 and row["manufacturing_regressions"] == 0
    full_pass = all(row["gate"] for (task,arm,seed),row in observed.items() if arm == "full")
    return {"schema":"shadowart_vnext_paired_v1", "manifest_digest":manifest["digest"],
        "independent_units":8, "seed_policy":"three_seeds_averaged_within_task",
        "failure_count":sum(not row["valid"] for row in observed.values()),
        "full_delivery_pass":full_pass, "contrasts":contrasts,
        "all_primitives_supported":all(row["supported"] for row in contrasts.values()),
        "claim_allowed":full_pass and all(row["supported"] for row in contrasts.values()),
        "raw_outcomes":raw}
