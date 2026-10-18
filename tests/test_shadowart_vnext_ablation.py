"""Artifact-bound gates and task-level inference; fixtures are not real outcomes."""
import copy
import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest


def api():
    return importlib.import_module("experiments.shadowart_vnext.ablation")


def inputs():
    from experiments.shadowart_vnext.losses import LossConfig
    from experiments.shadowart_vnext.contracts import build_contract, mask_hash
    dev = [dict(task_id=name, object_id=name, view_count=n, tsf=t) for name,n,t in
           [("heroes",3,1),("dev2",2,0),("dev4",4,1),("dev6",6,0)]]
    confirm = [dict(task_id=f"c{n}_{t}", object_id=f"obj{n}_{t}", view_count=n, tsf=t)
               for n in (2,3,4,6) for t in (0,1)]
    tasks = dev + confirm
    targets, contracts = {}, {}
    for task in tasks:
        name = task["task_id"]
        contract = build_contract(task["view_count"], tsf=task["tsf"])
        base = np.zeros((task["view_count"],256,256), dtype=bool)
        base[:,48:208,64:192] = True
        targets[name] = np.stack([base[c.target_index][:,::-1] if c.reflect_x
                                 else base[c.target_index] for c in contract.cameras])
        contracts[name] = replace(contract,
            input_hashes=tuple(hashlib.sha256(f"{name}-{i}".encode()).hexdigest()
                               for i in range(task["view_count"])),
            target_hashes=tuple(mask_hash(mask) for mask in targets[name]))
    return dict(nonloss_settings={"resolution":256,"image_size":256,"steps":2000,
        "lr":.007,"extent":1.7,"pixel_supersampling":2,"optimizer":"Adam",
        "initialization":"hull","occupancy_rule":"reference_opacity>=.5",
        "view_update_mode":"joint_view_equal","preprocessing":{"frozen":True},
        "budgets":{"geometry":4},"export_policy":"final_step","renderer_digest":"a"*64,
        "source_hashes":{"training.py":"b"*64}},
        targets=targets, camera_contracts=contracts,
        camera_digest={name:contract.digest for name,contract in contracts.items()}, geometry_settings={"thickness_mm":1.2},
        arms={"full":LossConfig(), "without_l1":LossConfig().without("l1"),
              "without_l2":LossConfig().without("l2"), "historical":LossConfig(control="current_proposal")},
        dev_tasks=dev, confirm_tasks=confirm)


def bind_changed_targets(kwargs, name):
    from experiments.shadowart_vnext.contracts import mask_hash
    kwargs["camera_contracts"][name] = replace(kwargs["camera_contracts"][name],
        target_hashes=tuple(mask_hash(mask) for mask in kwargs["targets"][name]))
    kwargs["camera_digest"][name] = kwargs["camera_contracts"][name].digest


def witness(tmp_path, manifest, *, mean=.99, worst=.98):
    """Create actual temp files only to exercise evidence binding logic."""
    m = api()
    mesh = tmp_path / "fixture.stl"
    mesh.write_bytes(b"solid unit-test-fixture\nendsolid unit-test-fixture\n")
    mesh_hash = hashlib.sha256(mesh.read_bytes()).hexdigest()
    archive = tmp_path / "fixture.gcode.3mf"
    gcode_member = "Metadata/plate_1.gcode"
    gcode = b"; Synthetic unit-test artifact, not a printable toolpath\nG90\n"
    with ZipFile(archive, "w") as package:
        package.writestr(gcode_member, gcode)
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    gcode_hash = hashlib.sha256(gcode).hexdigest()
    source_paths = {"mesh":str(mesh), "archive":str(archive), "gcode_member":gcode_member}
    source_hashes = {"mesh_sha256":mesh_hash, "archive_sha256":archive_hash, "gcode_sha256":gcode_hash}
    for name,key in (("effective_settings","effective_settings_sha256"),
                     ("slicer_result","result_sha256"),("slicer_log","slicer_log_sha256")):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({"fixture":"Synthetic evidence-binding test"}))
        source_paths[name] = str(path)
        source_hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema":"shadowart_support_review_v1",
        "mesh_sha256":mesh_hash,"gcode_sha256":gcode_hash,"archive_sha256":archive_hash,
        "decision":"accessible_and_removable","supports_accessible":True,
        "supports_removable":True,"all_contacts_reviewed":True,
        "reviewer":"unit-test-fixture","reviewed_at":"2026-09-06T10:00:00Z",
        "notes":"Synthetic unit-test review, not a production print result."}))
    scores = [{"image_size":size,"per_view_iou":[mean,mean,worst]*2} for size in (256,512)]
    report = {"passed":True,"converged":True,"per_view_iou":[mean,mean,worst]*2,
              "mean_iou":(mean*2+worst)/3,"worst_iou":min(mean,worst),"resolution_delta":0.}
    thickness = {"passed":True,"status":"PASS","reasons":[],"review_reasons":[],
                 "review_required":False,"component_count_6":1}
    serialized = {ext:{"sha256":mesh_hash,"passed":True,"component_count_6":1,
        "topology":{"passed":True,"status":"PASS","reasons":[],"watertight":True,
            "winding_consistent":True,"boundary_edges":0,"nonmanifold_edges":0,
            "self_intersection":{"status":"PASS","selected_faces":0},
            "containment_status":"PASS","outer_shells":1},
        "evaluations":copy.deepcopy(scores),"quality":copy.deepcopy(report),
        "raster_evaluations":copy.deepcopy(scores),"raster_quality":copy.deepcopy(report),
        "thickness":{**copy.deepcopy(thickness),"mesh_revoxelized_512":copy.deepcopy(thickness)}}
        for ext in ("obj","stl")}
    quality = {"passed":True,"status":"PASS","native_resolution":256,
        "mesh_sha256":mesh_hash,"camera_digest":manifest["camera_digests"]["heroes"],
        "target_hashes":manifest["targets"]["heroes"]["hashes"],
        "geometry_digest":manifest["geometry_digest"], "independent_serialized_mesh":True,
        "evaluations":scores,"quality":report,"serialized":serialized}
    manufacture = {"mesh_sha256":mesh_hash,"status":"PASS","passed":True,
        "camera_digest":manifest["camera_digests"]["heroes"],"target_hashes":manifest["targets"]["heroes"]["hashes"],
        "geometry_digest":manifest["geometry_digest"],
        "checks":{name:True for name in m.MANUFACTURING_CHECKS},
        "slicing":{"passed":True,"status":"PASS","support_removal_pass":True,
            "source_hashes":source_hashes,"source_paths":source_paths,
            "checks":{name:{"passed":True,"status":"PASS"} for name in (
                "effective_settings","slicer_completion","archive_integrity","gcode_parse",
                "gcode_effective_settings","archive_effective_settings","warnings","archive_model",
                "toolpath_correspondence","support_removal")}},
        "support_removal_review":{"path":str(review),"sha256":hashlib.sha256(review.read_bytes()).hexdigest(),"reviewer":"unit-test-fixture"}}
    qa = tmp_path / "qa.json"
    qa.write_text(json.dumps(quality))
    ma = tmp_path / "manufacturing.json"
    ma.write_text(json.dumps(manufacture))
    return {"task_id":"heroes","mesh_path":str(mesh),"mesh_sha256":mesh_hash,
        "qa_path":str(qa),"qa_sha256":hashlib.sha256(qa.read_bytes()).hexdigest(),
        "manufacturing_path":str(ma),"manufacturing_sha256":hashlib.sha256(ma.read_bytes()).hexdigest()}


def test_manifest_binds_nonloss_settings_targets_and_deletions():
    m = api()
    kwargs = inputs()
    a = m.build_manifest(**kwargs)
    assert a["targets"]["heroes"]["shape"] == [6,256,256]
    assert a["camera_contracts"]["heroes"]["base_count"] == 3
    assert m.build_manifest(**kwargs) == a
    changed = inputs()
    changed["targets"]["heroes"][0,0,0] = True
    changed["targets"]["heroes"][3,0,255] = True
    bind_changed_targets(changed, "heroes")
    b = m.build_manifest(**changed)
    assert a["nonloss_fingerprint"] != b["nonloss_fingerprint"]
    damaged = copy.deepcopy(a)
    damaged["nonloss_settings"]["steps"] += 1
    with pytest.raises(ValueError):
        m.development_schedule(damaged)
    assert len(m.development_schedule(a)) == 32
    assert {r["seed"] for r in m.development_schedule(a)} == {17,42}
    wrong = inputs()
    wrong["arms"]["without_l2"] = wrong["arms"]["without_l2"].__class__(core="l1")
    with pytest.raises(ValueError):
        m.build_manifest(**wrong)


def test_heldout_identity_and_crossed_counts_cannot_be_reused_or_omitted():
    m = api()
    for change in ("object", "count", "source"):
        kwargs = inputs()
        if change == "object":
            kwargs["confirm_tasks"][0]["object_id"] = "heroes"
        elif change == "count":
            kwargs["confirm_tasks"][0]["view_count"] = 3
        else:
            del kwargs["nonloss_settings"]["renderer_digest"]
        with pytest.raises(ValueError):
            m.build_manifest(**kwargs)


def test_confirmation_requires_bound_actual_files_and_three_seeds(tmp_path):
    m = api()
    manifest = m.build_manifest(**inputs())
    with pytest.raises(ValueError):
        m.confirmation_schedule(manifest, {"status":"SUCCESS"})
    proof = witness(tmp_path, manifest)
    rows = m.confirmation_schedule(manifest, proof)
    assert len(rows) == 96
    assert {row["seed"] for row in rows} == {17,42,2026}
    assert len({row["task_id"] for row in rows}) == 8
    (tmp_path / "fixture.stl").write_bytes(b"changed")
    with pytest.raises(ValueError):
        m.confirmation_schedule(manifest, proof)


@pytest.mark.parametrize("change", ["mean_equal", "worst_low", "camera", "review", "self_intersection", "resolution"])
def test_confirmation_never_promotes_soft_scores_or_incomplete_digital_evidence(tmp_path, change):
    m = api()
    manifest = m.build_manifest(**inputs())
    proof = witness(tmp_path, manifest)
    key = "manufacturing" if change in {"review","self_intersection"} else "qa"
    path = tmp_path / ("manufacturing.json" if key == "manufacturing" else "qa.json")
    payload = json.loads(path.read_text())
    if change == "mean_equal":
        for row in payload["evaluations"]: row["per_view_iou"] = [.98]*6
    elif change == "worst_low":
        payload["evaluations"][0]["per_view_iou"] = [1.,1.,.969]*2
    elif change == "camera": payload["camera_digest"] = "d"*64
    elif change == "review": del payload["support_removal_review"]
    elif change == "self_intersection": payload["checks"]["self_intersections_clear"] = False
    elif change == "resolution": payload["evaluations"][1]["per_view_iou"] = [1.]*6
    path.write_text(json.dumps(payload))
    proof[key + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError): m.confirmation_schedule(manifest, proof)


def replace_evidence(tmp_path, proof, key, payload):
    path = tmp_path / ("qa.json" if key == "qa" else "manufacturing.json")
    path.write_text(json.dumps(payload))
    proof[key + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("path,value", [
    (("passed",), False), (("status",), "FAIL"), (("native_resolution",), 128),
    (("quality","passed"), False), (("quality","converged"), False),
    (("serialized","obj","passed"), False),
    (("serialized","stl","sha256"), "f"*64),
    (("serialized","obj","topology","passed"), False),
    (("serialized","obj","topology","watertight"), False),
    (("serialized","obj","topology","winding_consistent"), False),
    (("serialized","obj","topology","boundary_edges"), 1),
    (("serialized","obj","topology","self_intersection","status"), "REVIEW_REQUIRED"),
    (("serialized","obj","topology","containment_status"), "REVIEW_REQUIRED"),
    (("serialized","obj","topology","outer_shells"), 2),
    (("serialized","obj","component_count_6"), 2),
    (("serialized","stl","quality","passed"), False),
    (("serialized","obj","raster_quality","passed"), False),
    (("serialized","stl","thickness","passed"), False),
    (("serialized","stl","thickness","review_required"), True),
    (("serialized","stl","thickness","mesh_revoxelized_512","passed"), False),
])
def test_witness_rejects_failed_or_contradictory_serialized_qa(tmp_path, path, value):
    m = api(); manifest = m.build_manifest(**inputs()); proof = witness(tmp_path, manifest)
    payload = json.loads((tmp_path / "qa.json").read_text())
    field = payload
    for part in path[:-1]: field = field[part]
    field[path[-1]] = value
    replace_evidence(tmp_path, proof, "qa", payload)
    with pytest.raises(ValueError): m.confirmation_schedule(manifest, proof)


@pytest.mark.parametrize("path", [(), ("serialized","obj"), ("serialized","stl","raster")])
def test_witness_rechecks_last_resolution_pair_instead_of_trusting_pass_flags(tmp_path, path):
    m = api(); manifest = m.build_manifest(**inputs()); proof = witness(tmp_path, manifest)
    payload = json.loads((tmp_path / "qa.json").read_text())
    field = payload
    for part in path[:2]: field = field[part]
    name = "raster_evaluations" if len(path) == 3 else "evaluations"
    field[name] = [{"image_size":s,"per_view_iou":[v]*6}
                   for s,v in ((256,.985),(512,.990),(1024,.995))]
    replace_evidence(tmp_path, proof, "qa", payload)
    with pytest.raises(ValueError): m.confirmation_schedule(manifest, proof)


def test_witness_requires_aggregate_scores_to_match_serialized_meshes(tmp_path):
    m = api(); manifest = m.build_manifest(**inputs()); proof = witness(tmp_path, manifest)
    payload = json.loads((tmp_path / "qa.json").read_text())
    for row in payload["evaluations"]: row["per_view_iou"] = [.999]*6
    replace_evidence(tmp_path, proof, "qa", payload)
    with pytest.raises(ValueError): m.confirmation_schedule(manifest, proof)


@pytest.mark.parametrize("path,value", [
    (("passed",), False), (("camera_digest",), "f"*64),
    (("target_hashes",), ["f"*64]*6), (("geometry_digest",), "f"*64),
    (("slicing","passed"), False), (("slicing","support_removal_pass"), False),
    (("slicing","source_hashes","mesh_sha256"), "f"*64),
    (("slicing","checks","archive_integrity","passed"), False),
    (("slicing","checks","toolpath_correspondence","status"), "REVIEW_REQUIRED"),
])
def test_witness_rejects_stale_or_failed_manufacturing_bindings(tmp_path, path, value):
    m = api(); manifest = m.build_manifest(**inputs()); proof = witness(tmp_path, manifest)
    payload = json.loads((tmp_path / "manufacturing.json").read_text())
    field = payload
    for part in path[:-1]: field = field[part]
    field[path[-1]] = value
    replace_evidence(tmp_path, proof, "manufacturing", payload)
    with pytest.raises(ValueError): m.confirmation_schedule(manifest, proof)


@pytest.mark.parametrize("key,value", [("mesh_sha256","f"*64),("gcode_sha256","f"*64),
    ("archive_sha256","f"*64),("supports_removable",False),("all_contacts_reviewed",False),
    ("decision","not_reviewed"),("reviewer","different-reviewer"),("reviewed_at","invalid")])
def test_witness_validates_hashed_review_content_against_sliced_artifacts(tmp_path, key, value):
    m = api(); manifest = m.build_manifest(**inputs()); proof = witness(tmp_path, manifest)
    path = tmp_path / "review.json"
    review = json.loads(path.read_text()); review[key] = value
    path.write_text(json.dumps(review))
    manufacture = json.loads((tmp_path / "manufacturing.json").read_text())
    manufacture["support_removal_review"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    replace_evidence(tmp_path, proof, "manufacturing", manufacture)
    with pytest.raises(ValueError): m.confirmation_schedule(manifest, proof)


def test_calibration_uses_all_twelve_probes_and_rejects_unusable_evidence():
    m = api()
    rows = [{"task_id":task,"step":step,"core_norm":2.,"aux_norm":1.}
            for task in ("heroes","dev2","dev4","dev6") for step in (500,1000,1500)]
    assert m.calibrate_auxiliary(rows, .1) == pytest.approx(.2)
    for bad in (rows[:-1], [dict(row, aux_norm=0.) for row in rows],
                [dict(row, aux_norm=float("nan")) for row in rows]):
        with pytest.raises(ValueError): m.calibrate_auxiliary(bad, .1)
    partially_inactive = [dict(row) for row in rows]
    partially_inactive[0]["aux_norm"] = 0.
    with pytest.raises(ValueError): m.calibrate_auxiliary(partially_inactive, .1)


def test_task_level_inference_has_eight_units_and_includes_failed_deliveries(tmp_path):
    m = api()
    manifest = m.build_manifest(**inputs())
    proof = witness(tmp_path, manifest)
    schedule = m.confirmation_schedule(manifest, proof)
    outcomes = [dict(row, status="DELIVERED", final_mean_iou=.99 if row["arm"] == "full" else .985,
        final_worst_iou=.98, manufacturing_pass=True, fin_volume_mm3=0., raw_stages={"soft":.999}) for row in schedule]
    report = m.analyze_paired(manifest, outcomes, proof)
    assert report["independent_units"] == 8
    assert report["contrasts"]["without_l1"]["p_raw"] == pytest.approx(1/256)
    assert report["contrasts"]["without_l1"]["supported"]
    assert report["contrasts"]["without_l1"]["mean_worst_iou_delta"] == 0.
    bad = copy.deepcopy(outcomes)
    failed = next(row for row in bad if row["arm"] == "full")
    failed.update(status="FAILED", manufacturing_pass=False)
    after = m.analyze_paired(manifest, bad, proof)
    assert after["full_delivery_pass"] is False
    assert after["failure_count"] == 1
    assert after["contrasts"]["without_l1"]["mean_delta"] < 0
    with pytest.raises(ValueError): m.analyze_paired(manifest, outcomes[:-1], proof)
    changed = copy.deepcopy(outcomes)
    changed[0]["nonloss_fingerprint"] = "f"*64
    with pytest.raises(ValueError): m.analyze_paired(manifest, changed, proof)


def test_holm_preserves_ties_and_exact_signflip_counts_zero_differences():
    m = api()
    assert m.holm_adjust({"a":.01,"b":.03,"c":.03}) == {"a":.03,"b":.06,"c":.06}
    assert m.exact_sign_flip([0.] * 8) == 1.


def test_development_selection_penalizes_failure_and_stops_when_geometry_invalid():
    m = api()
    manifest = m.build_manifest(**inputs())
    rows = [dict(row, status="DELIVERED", final_mean_iou=.991 if row["arm"] == "full" else .99,
        final_worst_iou=.98, manufacturing_pass=True, fin_volume_mm3=1., raw_stages={})
        for row in m.development_schedule(manifest)]
    rank = m.rank_development(manifest, rows)
    # All profiles already pass both gates with equal worst-view IoU and fins;
    # the declared parsimony tie break selects the simpler first deletion.
    assert rank["selected_arm"] == "without_l1"
    broken = copy.deepcopy(rows)
    for row in broken:
        row.update(status="FAILED",manufacturing_pass=False)
    stopped = m.rank_development(manifest, broken)
    assert stopped["selected_arm"] is None
    assert stopped["status"] == "GEOMETRY_GATE_NOT_MET"
    with pytest.raises(ValueError): m.rank_development(manifest, rows[:-1])


def test_manifest_rejects_reforged_wrong_strata_or_missing_native_render_settings():
    m = api()
    manifest = m.build_manifest(**inputs())
    bad = copy.deepcopy(manifest)
    bad["confirm_tasks"][0]["view_count"] = 999
    bad["nonloss_fingerprint"] = m.fingerprint({k:bad[k] for k in (
        "nonloss_settings","targets","camera_digests","camera_contracts","geometry_settings","dev_tasks","confirm_tasks","dev_seeds","confirm_seeds")})
    bad.pop("digest")
    bad["digest"] = m.fingerprint(bad)
    with pytest.raises(ValueError): m.development_schedule(bad)
    kwargs = inputs()
    kwargs["nonloss_settings"]["pixel_supersampling"] = 1
    with pytest.raises(ValueError): m.build_manifest(**kwargs)
    kwargs["nonloss_settings"]["pixel_supersampling"] = 3
    assert m.build_manifest(**kwargs)["nonloss_settings"]["pixel_supersampling"] == 3


def test_manifest_requires_actual_camera_contracts():
    kwargs = inputs()
    kwargs.pop("camera_contracts", None)
    with pytest.raises(ValueError, match="camera contracts"):
        api().build_manifest(**kwargs)


def test_witness_requires_sliced_artifact_paths(tmp_path):
    m = api(); manifest = m.build_manifest(**inputs()); proof = witness(tmp_path, manifest)
    payload = json.loads((tmp_path / "manufacturing.json").read_text())
    payload["slicing"].pop("source_paths", None)
    replace_evidence(tmp_path, proof, "manufacturing", payload)
    with pytest.raises(ValueError, match="artifact"):
        m.confirmation_schedule(manifest, proof)


@pytest.mark.parametrize("change", ["dimensions", "views", "target_hash", "digest", "base_count",
    "tsf", "camera_frame", "unsigned", "input_hashes", "missing_task"])
def test_manifest_rejects_targets_or_strata_that_disagree_with_cameras(change):
    kwargs = inputs()
    contract = kwargs["camera_contracts"]["heroes"]
    if change == "dimensions": kwargs["targets"]["heroes"] = np.zeros((6,2,2),dtype=bool)
    elif change == "views": kwargs["targets"]["heroes"] = kwargs["targets"]["heroes"][:1]
    elif change == "target_hash": kwargs["targets"]["heroes"][0,0,0] = True
    elif change == "digest": kwargs["camera_digest"]["heroes"] = "f"*64
    elif change == "base_count": kwargs["dev_tasks"][0]["view_count"] = 4
    elif change == "tsf": kwargs["dev_tasks"][0]["tsf"] = 0
    elif change == "camera_frame":
        kwargs["camera_contracts"]["heroes"] = replace(contract,
            cameras=tuple(replace(camera,scale=1.9) for camera in contract.cameras))
        kwargs["camera_digest"]["heroes"] = kwargs["camera_contracts"]["heroes"].digest
    elif change == "unsigned":
        kwargs["camera_contracts"]["heroes"] = replace(contract,
            cameras=contract.cameras[:3],target_hashes=contract.target_hashes[:3])
        kwargs["targets"]["heroes"] = kwargs["targets"]["heroes"][:3]
        kwargs["camera_digest"]["heroes"] = kwargs["camera_contracts"]["heroes"].digest
    elif change == "input_hashes":
        kwargs["camera_contracts"]["heroes"] = replace(contract,input_hashes=())
        kwargs["camera_digest"]["heroes"] = kwargs["camera_contracts"]["heroes"].digest
    elif change == "missing_task": del kwargs["camera_contracts"]["c6_1"]
    with pytest.raises(ValueError): api().build_manifest(**kwargs)


def test_manifest_rechecks_serialized_contract_after_all_envelope_hashes_change():
    m = api(); manifest = m.build_manifest(**inputs())
    manifest["camera_contracts"]["heroes"]["protocol"] = "fixed3_ring"
    from experiments.shadowart_vnext.contracts import CameraContract
    manifest["camera_digests"]["heroes"] = CameraContract.from_dict(manifest["camera_contracts"]["heroes"]).digest
    manifest["nonloss_fingerprint"] = m.fingerprint({key:manifest[key] for key in (
        "nonloss_settings","targets","camera_digests","camera_contracts","geometry_settings",
        "dev_tasks","confirm_tasks","dev_seeds","confirm_seeds")})
    manifest.pop("digest"); manifest["digest"] = m.fingerprint(manifest)
    with pytest.raises(ValueError): m.development_schedule(manifest)


def test_manifest_rejects_nonreflected_opposite_target_even_if_hashes_match():
    kwargs = inputs(); kwargs["targets"]["heroes"][3,0,0] = True
    bind_changed_targets(kwargs, "heroes")
    with pytest.raises(ValueError): api().build_manifest(**kwargs)


@pytest.mark.parametrize("name", ["mesh", "archive", "effective_settings", "slicer_result", "slicer_log"])
@pytest.mark.parametrize("change", ["missing", "changed"])
def test_witness_reopens_every_recorded_slicer_artifact(tmp_path, name, change):
    m = api(); manifest = m.build_manifest(**inputs()); proof = witness(tmp_path, manifest)
    manufacture = json.loads((tmp_path / "manufacturing.json").read_text())
    path = Path(manufacture["slicing"]["source_paths"][name])
    if change == "missing": path.unlink()
    else: path.write_bytes(b"changed after audit")
    with pytest.raises(ValueError): m.confirmation_schedule(manifest, proof)


@pytest.mark.parametrize("change", ["embedded_gcode", "wrong_member", "second_plate", "invalid_zip"])
def test_witness_checks_gcode_inside_rehashed_archive(tmp_path, change):
    m = api(); manifest = m.build_manifest(**inputs()); proof = witness(tmp_path, manifest)
    manufacture = json.loads((tmp_path / "manufacturing.json").read_text())
    slicing = manufacture["slicing"]; archive = Path(slicing["source_paths"]["archive"])
    member = slicing["source_paths"]["gcode_member"]
    if change == "embedded_gcode":
        with ZipFile(archive, "w") as package: package.writestr(member,b"G1 X999 E999\n")
    elif change == "wrong_member": slicing["source_paths"]["gcode_member"] = "Metadata/missing.gcode"
    elif change == "second_plate":
        with ZipFile(archive, "a") as package: package.writestr("Metadata/plate_2.gcode",b"G90\n")
    elif change == "invalid_zip": archive.write_bytes(b"not a zip")
    slicing["source_hashes"]["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    review_path = tmp_path / "review.json"; review = json.loads(review_path.read_text())
    review["archive_sha256"] = slicing["source_hashes"]["archive_sha256"]
    review_path.write_text(json.dumps(review))
    manufacture["support_removal_review"]["sha256"] = hashlib.sha256(review_path.read_bytes()).hexdigest()
    replace_evidence(tmp_path, proof, "manufacturing", manufacture)
    with pytest.raises(ValueError): m.confirmation_schedule(manifest, proof)
