#!/usr/bin/env python3
"""Opt-in ShadowArt discrete manufacturing extension; never starts a printer.

The unchanged geometric adapter can be run first, or a hash-checked successful
run can be replayed. Reinforcement is performed on the 256^3 conversion field,
before SDF extraction, with exact camera and physical-scale acceptance gates.
This extension is not the paper's unmodified Core48 protocol or FEM validation.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import select
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from experiments import multiview_export_adapter_20260830 as a
from experiments import multiview_contract_delivery_20260902 as d
from experiments import multiview_native_delivery_20260905 as n
from experiments import shadowart_manufacturing_20260905 as m


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--geometry-run", type=Path)
    source.add_argument("--silhouettes", type=Path, nargs="+")
    parser.add_argument("--tsf", type=int, choices=(0, 1), default=1)
    parser.add_argument("--mirror", choices=("auto", "0", "1"), default="auto")
    parser.add_argument("--soft-artifact-dir", type=Path)
    parser.add_argument("--delivery-mode", choices=("auto", "core48", "native256"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--refine-steps", type=int, default=20)
    parser.add_argument("--allow-external-support", action="store_true")
    parser.add_argument("--bambu-appimage", type=Path,
                        default=Path("/home/lin/Applications/BambuStudio_ubuntu24.04-v02.07.01.62.AppImage"))
    args = parser.parse_args(argv)
    if not args.allow_external_support:
        parser.error("this profile requires explicit --allow-external-support")
    if not 0 <= args.refine_steps <= 60:
        parser.error("--refine-steps must be in [0,60]")
    if args.geometry_run and args.soft_artifact_dir:
        parser.error("--geometry-run uses its recorded immutable soft artifact")
    return args


def load_geometry(directory, *, device):
    import trimesh
    directory = Path(directory).resolve()
    report = json.loads((directory / a.TOP_MANIFEST).read_text())
    if report.get("status") != "SUCCESS":
        raise d.DeliveryFailure("GEOMETRY_SOURCE_FAILED", "a successful geometric run is required")
    recorded = report["camera_contract"]
    artifact = Path(report["soft_artifact"])
    silhouettes = tuple(Path(p) for p in recorded["target_order"])
    spec = a.resolve_protocol(n=len(silhouettes), tsf=recorded["requested_tsf"], mirror=recorded["mirror"])
    contract = a.validate_soft_artifact(artifact, spec=spec, resolution=256, silhouettes=silhouettes)
    if json.loads(json.dumps(asdict(contract))) != recorded:
        raise d.DeliveryFailure("FAILED_CAMERA_CONTRACT", "source contract does not match immutable artifact")
    for name in a.CANONICAL_EXPORTS.values():
        if d._hash(directory / name) != report["canonical_sha256"][name]:
            raise d.DeliveryFailure("GEOMETRY_SOURCE_MUTATED", name)
    exports = report["delivery"]["result"]["mesh"]["exports"]
    mesh = trimesh.load(directory / "delivery_postprocessed.stl", force="mesh", process=True)
    mesh.apply_translation(-np.asarray(exports["transform"]["translation"]))
    mesh.apply_scale(1. / float(exports["transform"]["uniform_scale"]))
    d.audit_shell_containment(mesh)
    volume = d.rasterize_mesh_with_cavities(mesh, (256,) * 3)
    targets = d._targets(artifact, contract, 256)
    soft = d.trainer_zyx_to_canonical_xyz(np.load(artifact / "occupancy_soft_proposal.npy").astype(bool))
    reference = d.backproject_contract(targets, contract, 256, device=device) | soft
    metrics = d.evaluate_contract(volume, targets, contract, device=device, hull_voxels=int(reference.sum()))
    if not d._delivery_pass(metrics, d.DeliveryConfig()):
        raise d.DeliveryFailure("GEOMETRY_SOURCE_GATE_FAILED", "reloaded source failed exact gates", metrics)
    return report, contract, artifact, volume, targets, int(reference.sum())


def bundled_profiles(appimage, output):
    """Use resources from the same executable version, not user preset leaves."""
    with (output / "profile_mount.log").open("w") as log:
        process = subprocess.Popen([str(appimage), "--appimage-mount"], stdout=subprocess.PIPE,
                                   stderr=log, text=True)
        try:
            if not select.select([process.stdout], [], [], 30)[0]:
                raise RuntimeError("Bambu AppImage mount timed out")
            mounted = Path(process.stdout.readline().strip())
            resource = mounted / "resources" / "profiles" / "BBL"
            if not mounted.is_absolute() or not resource.is_dir():
                raise RuntimeError("Bambu AppImage profile mount unavailable")
            return m.prepare_slicer_profiles(resource, output / "profiles")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def slice_supported(appimage, mesh_path, stage):
    profiles, provenance = bundled_profiles(appimage, stage)
    folder = stage / "slicer"
    folder.mkdir()
    command = ["xvfb-run", "-a", str(appimage), "--debug", "2", "--load-settings",
               str(profiles["machine"]) + ";" + str(profiles["process"]), "--load-filaments",
               str(profiles["filament"]), "--arrange", "1", "--orient", "0", "--ensure-on-bed",
               "--slice", "0", "--mstpp", "120", "--export-3mf", "shadowart_supported.gcode.3mf",
               "--export-settings", str(folder / "effective_settings.json"),
               "--outputdir", str(folder), str(mesh_path)]
    env = os.environ.copy()
    for key in ("WAYLAND_DISPLAY", "WAYLAND_SOCKET"):
        env.pop(key, None)
    env.update({"GDK_BACKEND": "x11", "XDG_SESSION_TYPE": "x11"})
    with (folder / "slicer.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900)
    if result.returncode != 0:
        raise d.DeliveryFailure("SLICER_FAILED", f"Bambu CLI exit {result.returncode}")
    sliced = json.loads((folder / "result.json").read_text())
    effective = json.loads((folder / "effective_settings.json").read_text())
    audit = m.slicing_audit(sliced)
    settings = m.audit_effective_settings(effective)
    if not audit["print_preparation_passed"] or not settings["passed"]:
        raise d.DeliveryFailure("SLICER_GATE_FAILED", "slicing warning or effective settings mismatch",
                                {"slicing": audit, "effective_settings": settings})
    return {"slicing": audit, "effective_settings": settings, "slicer_command": command,
            "sliced_archive_audit": m.audit_sliced_archive(folder / "shadowart_supported.gcode.3mf"),
            "slicer_profile_provenance": provenance, "slicer_executable_sha256": d._hash(appimage),
            "sliced_3mf_sha256": d._hash(folder / "shadowart_supported.gcode.3mf"),
            "slicer_statistics": sliced}


def run(args, output):
    geometry = args.geometry_run
    if geometry is None:
        geometry = output / "geometry_diagnostic"
        command = [sys.executable, str(Path(a.__file__)), "--silhouettes",
                   *[str(p.resolve()) for p in args.silhouettes], "--tsf", str(args.tsf),
                   "--mirror", args.mirror, "--output", str(geometry), "--device", args.device,
                   "--delivery-mode", args.delivery_mode]
        if args.soft_artifact_dir:
            command += ["--soft-artifact-dir", str(args.soft_artifact_dir.resolve())]
        subprocess.run(command, check=True)
    print("Reloading and verifying the source mesh and immutable camera contract", flush=True)
    base, contract, artifact, source, targets, hull_voxels = load_geometry(geometry, device=args.device)
    stage = output / "diagnostics"
    stage.mkdir()
    d._write(output / a.CONTRACT_MANIFEST, asdict(contract))
    pitch = base["delivery"]["result"]["mesh"]["exports"]["transform"]["uniform_scale"]
    # Fixed manufacturing proxy: at 100 mm, one-voxel erosion must not split
    # surviving lobes >= 1 mm^3. This does not certify every sub-mm feature.
    def neck(v, actual_pitch=pitch):
        return m.neck_audit(v, pitch_mm=actual_pitch, minimum_lobe_mm3=1.)
    fortified, locked, reinforcement = m.fortify_necks(source, pitch_mm=pitch, minimum_lobe_mm3=1.)
    d._write(stage / "reinforcement.json", reinforcement)
    np.save(stage / "protected_material.npy", locked)
    refined, history = n.refine_connected(fortified, targets, contract, device=args.device,
                                         steps=args.refine_steps, protected=locked,
                                         acceptance_gate=lambda v: neck(v)["robust_neck_gate"])
    d._write(stage / "refinement.json", history)
    rows, volumes = [], {"baseline": source, "reinforced": fortified, "reinforced_refined": refined}
    for name, volume in volumes.items():
        metrics = d.evaluate_contract(volume, targets, contract, device=args.device, hull_voxels=hull_voxels)
        rows.append({"id": name, **metrics, "voxels": int(volume.sum()),
                     "robust_neck_gate": neck(volume)["robust_neck_gate"], "neck": neck(volume)})
        np.save(stage / f"candidate_{name}.npy", volume)
    d._write(stage / "candidate_comparison.json", rows)
    selected = m.select_manufacturing_candidate(rows)
    print(f"Selected {selected['id']}: mean={selected['mean_iou']:.6f}; running mesh/reload gates", flush=True)
    volume = volumes[selected["id"]]
    raster, metrics, mesh = n.finalize_native(volume, targets, contract, stage, args.device,
                                             d.DeliveryConfig(), hull_voxels=hull_voxels)
    actual_pitch = mesh["exports"]["transform"]["uniform_scale"]
    mesh_neck = neck(raster, actual_pitch)
    if not mesh_neck["robust_neck_gate"]:
        raise d.DeliveryFailure("MESH_NECK_FAILED", "mesh raster lost robust connectivity", mesh_neck)
    import trimesh
    reload_necks = {}
    for role in ("fabrication_obj", "fabrication_stl"):
        loaded = trimesh.load(stage / mesh["exports"]["files"][role], force="mesh", process=True)
        loaded.apply_translation(-np.asarray(mesh["exports"]["transform"]["translation"]))
        loaded.apply_scale(1. / actual_pitch)
        reload_necks[role] = neck(d.rasterize_mesh_with_cavities(loaded, raster.shape), actual_pitch)
        if not reload_necks[role]["robust_neck_gate"]:
            raise d.DeliveryFailure("RELOADED_MESH_NECK_FAILED", role, reload_necks[role])
    np.save(output / "delivery_voxels_xyz_256.npy", raster)
    import cv2
    for i, prediction in enumerate(d.render_contract(raster, contract, device=args.device)):
        cv2.imwrite(str(output / f"pred_view_{i}.png"), prediction.astype(np.uint8) * 255)
    print("Geometry and local neck gates passed; slicing with removable external supports", flush=True)
    report = {"schema": "shadowart_manufacturing_extension_v1", "status": "STAGING",
              "mode": "shadowart_geometry_and_print_preparation", "structural_status": "NOT_REQUESTED",
              "physical_print_validated": False, "paper_method_equivalence": False,
              "method": "physical_scale_Bridge6_locked_same_camera_refinement_then_SDF",
              "source_geometry_run": str(Path(geometry).resolve()),
              "source_geometry_manifest_sha256": d._hash(Path(geometry) / a.TOP_MANIFEST),
              "source_geometry_method": base.get("delivery_method"),
              "source_sha256": {str(p.relative_to(ROOT)): d._hash(p) for p in
                                (Path(__file__), Path(m.__file__), Path(n.__file__), Path(d.__file__), Path(a.__file__))},
              "camera_contract": asdict(contract), "candidate_comparison": rows, "selection": selected,
              "reinforcement": reinforcement, "mesh": mesh, "mesh_neck": mesh_neck,
              "reload_necks": reload_necks, "voxel_resolution": 256, "max_dimension_mm": 100.,
              "saving_reference": "center_ray_hull_union_immutable_soft_support_NOT_paper_Table1",
              "remaining_limits": ["neck test is a one-voxel erosion proxy, not a minimum-wall certificate",
                                   "sub-1 mm^3 eroded fragments are not individually certified",
                                   "support removal and physical print have not been tested", "no PINO/FEM validation"]}
    d._write(stage / "geometry_and_manufacturing_audit.json", report)
    report.update(slice_supported(args.bambu_appimage.resolve(), stage / mesh["exports"]["files"]["fabrication_stl"], stage))
    if d._snapshot(artifact) != tuple(tuple(row) for row in contract.artifact_sha256):
        raise d.DeliveryFailure("ARTIFACT_MUTATED", "source changed during preparation")
    report["status"] = "PRINT_PREPARATION_VALIDATED"
    report["canonical_sha256"] = m.publish_preparation(stage, output, report)
    # No fallible output I/O after the manifest publication commit.
    return 0


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve()
    # Never overwrite an earlier result, including failed diagnostics.
    output.mkdir(parents=True, exist_ok=False)
    try:
        return run(args, output)
    except Exception as error:
        d._write(output / "MANUFACTURING.json", {"status": "FAILED", "reason": str(error),
                 "failure_code": getattr(error, "status", type(error).__name__),
                 "details": getattr(error, "details", {}), "physical_print_validated": False})
        print(f"FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
