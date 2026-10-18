"""Artifact-bound, fail-closed Bambu slicing audit. Never contacts a printer.

The historical CLI wrapper is reused without editing it. Nominal toolpath
screens and a human support-removal decision are separate checks; neither can
override unknown source/model correspondence or an incomplete G-code parse.
"""
from __future__ import annotations

from array import array
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
from scipy import ndimage as ndi

APPIMAGE = Path("/home/lin/Applications/BambuStudio_ubuntu24.04-v02.07.01.62.AppImage")
NUMERIC = {"nozzle_diameter": .4, "filament_diameter": 1.75,
           "line_width": .42, "initial_layer_line_width": .5,
           "outer_wall_line_width": .42, "inner_wall_line_width": .45,
           "top_surface_line_width": .42, "sparse_infill_line_width": .45,
           "internal_solid_infill_line_width": .42, "support_line_width": .42,
           "layer_height": .2, "initial_layer_print_height": .2,
           "wall_loops": 3., "brim_width": 8., "sparse_infill_density": 15.}
TEXTUAL = {"filament_type": "PLA", "wall_generator": "arachne",
           "brim_type": "outer_only", "sparse_infill_pattern": "grid",
           "enable_support": "1", "support_type": "tree(auto)",
           "support_on_build_plate_only": "1"}
MODEL_FEATURES = {"Outer wall", "Inner wall", "Overhang wall", "Bridge", "Gap infill",
                  "Sparse infill", "Internal solid infill", "Top surface", "Bottom surface"}
PARAMETER = re.compile(r"([A-Z])([-+]?(?:\d+(?:\.\d*)?|\.\d+))")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result(status, **details):
    return {"status": status, "passed": status == "PASS", **details}


def _scalar(value):
    return value[0] if isinstance(value, list) and len(value) == 1 else value


def audit_effective_settings(settings):
    checks = {}
    for key, expected in NUMERIC.items():
        try:
            actual = float(str(_scalar(settings[key])).rstrip("%"))
            checks[key] = math.isfinite(actual) and abs(actual - expected) <= 1e-6
        except (KeyError, ValueError, TypeError):
            checks[key] = False
    checks.update({k: str(_scalar(settings.get(k))) == expected for k, expected in TEXTUAL.items()})
    checks["printer_model"] = _scalar(settings.get("printer_model")) in ("A1", "Bambu Lab A1")
    return _result("PASS" if all(checks.values()) else "FAIL", checks=checks,
                   observed={key: settings.get(key) for key in checks},
                   profile="A1_PLA_0.4_0.2mm_3walls_Arachne_outer8mm_brim_15pct_grid_v1")


def _parse_gcode(text):
    """Parse nominal millimetre XYZ/E state; rows are XYZXYZ,E,width,height,kind."""
    position = np.zeros(3); extrusion = 0.; retraction_debt = 0.; absolute_xyz = True; absolute_e = False
    feature = "Custom"; layer_height = .2; filament_area = math.pi * 1.75 ** 2 / 4
    rows = array("d"); problems = []; config = {}; stationary = 0; config_active = False
    known_e_mode = False
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if line == "; CONFIG_BLOCK_START":
            config_active = True
        elif line == "; CONFIG_BLOCK_END":
            config_active = False
        if config_active and line.startswith("; ") and " = " in line:
            key, value = line[2:].split(" = ", 1)
            config[key] = value.strip().strip('"')
        if line.startswith("; FEATURE:"):
            feature = line.split(":", 1)[1].strip()
        if line.startswith("; LAYER_HEIGHT:"):
            try:
                layer_height = float(line.split(":", 1)[1])
            except ValueError:
                problems.append(f"line {line_number}: invalid layer height")
        command = line.split(";", 1)[0].strip()
        if not command:
            continue
        opcode = command.split()[0].upper()
        parameters = {k: float(v) for k, v in PARAMETER.findall(command)[1:]}
        if opcode == "G90":
            absolute_xyz = True
        elif opcode == "G91":
            absolute_xyz = False
        elif opcode in ("M82", "M83"):
            absolute_e = opcode == "M82"; known_e_mode = True
        elif opcode == "G92":
            if any(key in parameters for key in 'XYZ'):
                problems.append(f"line {line_number}: unsupported physical XYZ G92 offset")
            for axis, key in enumerate("XYZ"):
                if key in parameters:
                    position[axis] = parameters[key]
            extrusion = parameters.get("E", extrusion)
        elif opcode == 'M221':
            if 'S' in parameters and parameters['S'] != 100.:
                problems.append(f"line {line_number}: unsupported extrusion flow multiplier M221")
        elif opcode in ("G20", "G90.1", "G91.1", "M200", "G53", "G54", "G55", "G56", "G57", "G58", "G59", "G68", "G69"):
            problems.append(f"line {line_number}: unsupported coordinate/extrusion mode {opcode}")
        elif opcode in ("G0", "G1", "G2", "G3"):
            end = position.copy()
            for axis, key in enumerate("XYZ"):
                if key in parameters:
                    end[axis] = parameters[key] if absolute_xyz else end[axis] + parameters[key]
            new_e = parameters.get("E", extrusion if absolute_e else 0.)
            delta_e = new_e - extrusion if absolute_e else new_e
            if "E" in parameters:
                extrusion = new_e if absolute_e else extrusion + new_e
            if delta_e < 0:
                retraction_debt -= delta_e
            elif delta_e > 0:
                repaid = min(delta_e, retraction_debt)
                retraction_debt -= repaid
                delta_e -= repaid
            kind = 1 if feature in MODEL_FEATURES else 3 if feature == "Support interface" else 2 if feature == "Support" else 4 if feature in ("Brim", "Skirt") else 0
            if delta_e > 1e-10 and not kind and np.linalg.norm(end-position)>1e-10:
                problems.append(f"line {line_number}: unclassified positive deposition ({feature})")
            if delta_e > 1e-10 and kind:
                if not known_e_mode:
                    problems.append(f"line {line_number}: extrusion mode not declared")
                if not np.isfinite([*end, delta_e, layer_height]).all() or layer_height <= 0:
                    problems.append(f"line {line_number}: invalid deposition values")
                    position = end
                    continue
                points = np.array([position, end])
                if opcode in ("G2", "G3"):
                    if "R" in parameters or not ("I" in parameters or "J" in parameters):
                        problems.append(f"line {line_number}: unsupported arc definition")
                        position = end
                        continue
                    centre = position[:2] + [parameters.get("I", 0.), parameters.get("J", 0.)]
                    radius = float(np.linalg.norm(position[:2] - centre))
                    if radius <= 0 or abs(np.linalg.norm(end[:2] - centre) - radius) > .025:
                        problems.append(f"line {line_number}: inconsistent arc radius")
                        position = end
                        continue
                    begin = math.atan2(position[1] - centre[1], position[0] - centre[0])
                    finish = math.atan2(end[1] - centre[1], end[0] - centre[0])
                    angle = (finish - begin) % (2 * math.pi)
                    if opcode == "G2":
                        angle = -((begin - finish) % (2 * math.pi))
                    if abs(angle) < 1e-12:
                        angle = -2 * math.pi if opcode == "G2" else 2 * math.pi
                    step = max(.001, min(.25, 2 * math.acos(max(-1., 1 - .01 / radius))))
                    count = max(1, int(math.ceil(abs(angle) / step)))
                    angles = begin + np.linspace(0., angle, count + 1)
                    points = np.column_stack([centre[0] + radius * np.cos(angles),
                                              centre[1] + radius * np.sin(angles),
                                              np.linspace(position[2], end[2], count + 1)])
                    points[0], points[-1] = position, end
                lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
                total = float(lengths.sum())
                if total <= 1e-10:
                    stationary += 1
                else:
                    width = filament_area * delta_e / total / layer_height + layer_height * (1 - math.pi / 4)
                    for index, length in enumerate(lengths):
                        if length > 1e-10:
                            rows.extend([*points[index], *points[index + 1], delta_e * length / total,
                                         width, layer_height, float(kind)])
            position = end
    segments = np.frombuffer(rows, dtype=np.float64).reshape(-1, 10).copy()
    model = segments[:, 9] == 1 if len(segments) else np.zeros(0, bool)
    support = np.isin(segments[:, 9], [2, 3]) if len(segments) else np.zeros(0, bool)
    problems = list(dict.fromkeys(problems))
    summary = _result("REVIEW_REQUIRED" if problems else "PASS",
                      positive_model_segments=int(model.sum()), positive_support_segments=int(support.sum()),
                      interface_segments=int(np.count_nonzero(segments[:, 9] == 3)),
                      commanded_model_extrusion_mm=float(segments[model, 6].sum()),
                      model_path_length_mm=float(np.linalg.norm(segments[model, 3:6] - segments[model, :3], axis=1).sum()),
                      stationary_positive_extrusions=stationary, unresolved=problems,
                      arc_max_sagitta_mm=.01, width_source="nominal_from_commanded_E_1.75mm_filament",
                      config=config)
    return segments, summary


def audit_gcode(text):
    return _parse_gcode(text)[1]


def _xml(data):
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("DTD/entity declarations are not accepted")
    return ET.fromstring(data)


def _children(element, name):
    return [child for child in element if child.tag.rsplit("}", 1)[-1] == name]


def _transform(value):
    result = np.eye(4)
    if value:
        numbers = np.fromstring(value, sep=" ")
        if numbers.size != 12 or not np.isfinite(numbers).all():
            raise ValueError("invalid 3MF transform")
        result[:3] = numbers.reshape(4, 3).T
    return result


def _triangle_digest(mesh, tolerance=1e-5):
    triangles = np.rint(np.asarray(mesh.triangles) / tolerance).astype(np.int64)
    order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    triangles = np.take_along_axis(triangles, order[:, :, None], axis=1).reshape(-1, 9)
    order = np.lexsort(tuple(triangles[:, k] for k in range(8, -1, -1)))
    return hashlib.sha256(np.ascontiguousarray(triangles[order], dtype="<i8").tobytes()).hexdigest()


def _archive_model(archive, source_mesh):
    import trimesh
    roots = {}
    def root(path):
        if path not in roots:
            parsed = _xml(archive.read(path))
            if parsed.attrib.get("unit", "millimeter") != "millimeter":
                raise ValueError("only explicit millimetre 3MF models are supported")
            roots[path] = parsed
        return roots[path]
    def geometry(path, object_id, stack=()):
        key = (path, object_id)
        if key in stack or len(stack) > 16:
            raise ValueError("cyclic or overly nested 3MF components")
        resources = _children(root(path), "resources")[0]
        objects = [obj for obj in _children(resources, "object") if obj.attrib["id"] == object_id]
        if len(objects) != 1:
            raise ValueError("ambiguous 3MF object")
        obj = objects[0]
        mesh_nodes = _children(obj, "mesh")
        if mesh_nodes:
            vertices = [[float(v.attrib[a]) for a in "xyz"] for v in _children(_children(mesh_nodes[0], "vertices")[0], "vertex")]
            faces = [[int(t.attrib[a]) for a in ("v1", "v2", "v3")] for t in _children(_children(mesh_nodes[0], "triangles")[0], "triangle")]
            return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        parts = []
        for component in _children(_children(obj, "components")[0], "component"):
            linked = next((v for k, v in component.attrib.items() if k.rsplit("}", 1)[-1] == "path"), path)
            linked = linked.lstrip("/") if linked.startswith("/") else str(PurePosixPath(path).parent / linked) if linked != path else path
            if ".." in PurePosixPath(linked).parts:
                raise ValueError("3MF component path traversal")
            part = geometry(linked, component.attrib["objectid"], stack + (key,))
            part.apply_transform(_transform(component.attrib.get("transform")))
            parts.append(part)
        return trimesh.util.concatenate(parts)
    main = "3D/3dmodel.model"
    items = _children(_children(root(main), "build")[0], "item")
    if len(items) != 1 or items[0].attrib.get("printable", "1") != "1":
        raise ValueError("one printable source instance is required")
    base = geometry(main, items[0].attrib["objectid"])
    matrix = _transform(items[0].attrib.get("transform"))
    rigid = (np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
             and abs(np.linalg.det(matrix[:3, :3]) - 1) < 1e-6
             and np.allclose(matrix[:3, 2], [0., 0., 1.], atol=1e-6)
             and abs(matrix[2, 3]) < 1e-4)
    same = len(base.faces) == len(source_mesh.faces) and _triangle_digest(base) == _triangle_digest(source_mesh)
    result = _result("PASS" if rigid and same else "FAIL" if not rigid else "REVIEW_REQUIRED",
                     rigid_z_preserving_arrangement=bool(rigid), source_triangles_equal=bool(same),
                     triangle_coordinate_tolerance_mm=1e-5, build_transform=matrix.tolist(),
                     source_faces=int(len(source_mesh.faces)), archive_faces=int(len(base.faces)))
    base.apply_transform(matrix)
    return base, result


def _raster_section(mesh, z, low, shape, pitch):
    import cv2
    section = mesh.section(plane_origin=[0., 0., float(z)], plane_normal=[0., 0., 1.])
    mask = np.zeros(shape, np.uint8)
    if section is None:
        return mask.astype(bool)
    for contour in section.discrete:
        if not np.allclose(contour[0], contour[-1], atol=1e-5):
            raise ValueError("open mesh section")
        pixels = np.rint((contour[:, :2] - low) / pitch).astype(np.int32)
        ring = np.zeros(shape, np.uint8)
        cv2.fillPoly(ring, [pixels], 1)
        mask ^= ring
    return mask.astype(bool)


def _section_contours(mesh, z):
    section = mesh.section(plane_origin=[0., 0., float(z)], plane_normal=[0., 0., 1.])
    contours=[]
    if section is not None:
        for ring in section.discrete:
            if not np.isfinite(ring).all() or not np.allclose(ring[0],ring[-1],atol=1e-5,rtol=0):
                raise ValueError('open or nonfinite mesh section')
            contours.append(np.asarray(ring[:,:2],float))
    return contours


def _contour_mask(contours, low, shape, pitch):
    """Exact scan-line parity at grid centers, retaining all inner contours."""
    events=np.zeros((shape[0],shape[1]+1),bool)
    for contour in contours:
        for a,b in zip(contour,contour[1:]):
            if abs(a[1]-b[1])<1e-12:
                continue
            begin=max(0,int(math.ceil((min(a[1],b[1])-low[1])/pitch)))
            end=min(shape[0],int(math.ceil((max(a[1],b[1])-low[1])/pitch)))
            rows=np.arange(begin,end)
            x=a[0]+(low[1]+rows*pitch-a[1])*(b[0]-a[0])/(b[1]-a[1])
            cols=np.clip(np.ceil((x-low[0])/pitch).astype(int),0,shape[1])
            np.logical_xor.at(events,(rows,cols),True)
    return np.logical_xor.accumulate(events,axis=1)[:,:shape[1]]


def _bead_masks(layer, low, shape, pitch):
    """Point samples of exact nominal round-ended capsules, not cv2 line widths."""
    masks={k:np.zeros(shape,bool) for k in (1,2,3,4)}
    for segment in layer:
        a,b=segment[:2],segment[3:5]; radius=segment[7]/2
        start=np.maximum(0,np.ceil((np.minimum(a,b)-radius-low)/pitch).astype(int))
        stop=np.minimum(shape[::-1],np.floor((np.maximum(a,b)+radius-low)/pitch).astype(int)+1)
        if np.any(stop<=start):
            continue
        x=low[0]+np.arange(start[0],stop[0])*pitch
        y=low[1]+np.arange(start[1],stop[1])*pitch
        delta=b-a; norm=float(delta@delta)
        t=np.clip(((x[None,:]-a[0])*delta[0]+(y[:,None]-a[1])*delta[1])/max(norm,1e-30),0,1)
        distance2=(x[None,:]-a[0]-t*delta[0])**2+(y[:,None]-a[1]-t*delta[1])**2
        masks[int(segment[9])][start[1]:stop[1],start[0]:stop[0]] |= distance2<=radius**2+1e-12
    return masks


def _boundary_measure(contours, model_segments, pitch, tolerance=.15):
    """Weighted surface samples against exact variable-width path capsules."""
    from scipy.spatial import cKDTree
    total=missing=longest=0.
    for contour in contours:
        arrays=[]; weights=[]
        for a,b in zip(contour,contour[1:]):
            length=float(np.linalg.norm(b-a))
            if length<=1e-12:
                continue
            count=max(1,int(math.ceil(length/pitch)))
            arrays.append(a+((np.arange(count)+.5)/count)[:,None]*(b-a))
            weights.append(np.full(count,length/count))
        if not arrays:
            continue
        points=np.concatenate(arrays); lengths=np.concatenate(weights)
        if len(points)>1_000_000:
            raise ValueError('boundary sample capacity exceeded')
        covered=np.zeros(len(points),bool); tree=cKDTree(points)
        for segment in model_segments:
            a,b=segment[:2],segment[3:5]; delta=b-a; squared=float(delta@delta)
            radius=segment[7]/2+tolerance
            candidates=np.asarray(tree.query_ball_point((a+b)/2,np.sqrt(squared)/2+radius),int)
            candidates=candidates[~covered[candidates]]
            if not len(candidates):
                continue
            points_here=points[candidates]
            t=np.clip(((points_here-a)@delta)/max(squared,1e-30),0,1)
            inside=np.sum((points_here-a-t[:,None]*delta)**2,axis=1)<=radius**2
            covered[candidates[inside]]=True
        total+=float(lengths.sum()); missing+=float(lengths[~covered].sum())
        if not covered.any():
            longest=max(longest,float(lengths.sum()))
        elif not covered.all():
            # Rotate past a known covered point to join gaps across ring closure.
            start=int(np.flatnonzero(covered)[0])+1
            run=0.
            for good,length in zip(np.roll(covered,-start),np.roll(lengths,-start)):
                run=0. if good else run+float(length)
                longest=max(longest,run)
    return {'boundary_length_mm':total,'uncovered_boundary_mm':missing,'longest_boundary_gap_mm':longest,
            'uncovered_boundary_fraction':missing/max(total,1e-30)}


def _digital_layer_screen(mesh, segments, pitch, sections):
    endpoints=segments[:,[0,1,3,4]].reshape(-1,2)
    low=np.minimum(mesh.bounds[0,:2],endpoints.min(0))-1.
    high=np.maximum(mesh.bounds[1,:2],endpoints.max(0))+1.
    dimensions=np.ceil((high-low)/pitch).astype(int)+1
    if np.any(dimensions>4096):
        raise ValueError('RASTER_CAPACITY')
    shape=tuple(dimensions[::-1]); previous=np.zeros(shape,bool)
    actual=np.unique(np.round(segments[:,5],5))
    required=np.round(np.arange(.2,mesh.bounds[1,2]+.19999,.2),5)
    heights=np.unique(np.r_[actual,required]); rows=[]; failures=[]
    if len(heights)>1500:
        raise ValueError('LAYER_CAPACITY')
    for z in heights:
        layer=segments[np.isclose(segments[:,5],z,atol=1e-5,rtol=0)]
        model=layer[layer[:,9]==1]
        masks=_bead_masks(layer,low,shape,pitch)
        deposited=masks[1]; all_paths=np.logical_or.reduce(list(masks.values()))
        sample_heights=[float(z-.175),float(z-.1),float(z-.025)]
        contours=[]; source_masks=[]; measures=[]
        for sample_z in sample_heights:
            key=round(sample_z,8)
            if key not in sections:
                sections[key]=_section_contours(mesh,sample_z)
            ring=sections[key]; contours.append(ring)
            source_masks.append(_contour_mask(ring,low,shape,pitch))
            measures.append(_boundary_measure(ring,model,pitch))
        expected=source_masks[1]; envelope=np.logical_or.reduce(source_masks)
        labels,count=ndi.label(expected)
        hit_labels=np.unique(labels[deposited & expected])
        absent=count-int(np.count_nonzero(hit_labels))
        if absent:
            failures.append('LOST_MODEL_ISLANDS')
        if envelope.any() and not len(model):
            failures.append('MISSING_LAYER_GAP')
        boundary={name:max(m[name] for m in measures) for name in measures[0]}
        if boundary['longest_boundary_gap_mm']>.2+1e-9 or boundary['uncovered_boundary_mm']>.2+1e-9:
            failures.append('SOURCE_BOUNDARY_GAP')
        # EDM uses exact center samples; the two pitches must agree below.
        outside_distance=ndi.distance_transform_edt(~envelope)*pitch if envelope.any() else np.full(shape,np.inf)
        outside=float(np.count_nonzero(deposited & (outside_distance>.15+1e-9))*pitch**2)
        if outside>.02+1e-9:
            failures.append('MODEL_PATH_OUTSIDE_SOURCE')
        void_labels,_=ndi.label(~expected)
        exterior=np.unique(np.r_[void_labels[0],void_labels[-1],void_labels[:,0],void_labels[:,-1]])
        cavities=(~expected)&~np.isin(void_labels,exterior)
        support=masks[2]|masks[3]
        cavity_area=float(np.count_nonzero(support & cavities)*pitch**2)
        if cavity_area>.02+1e-9:
            failures.append('INTERNAL_CAVITY_SUPPORT')
        inside_distance=ndi.distance_transform_edt(expected)*pitch
        intrusion=float(np.count_nonzero(support & (inside_distance>.15+1e-9))*pitch**2)
        if intrusion>.02+1e-9:
            failures.append('SUPPORT_INTRUDES_SOURCE_MATERIAL')
        components,number=ndi.label(all_paths)
        anchors=ndi.distance_transform_edt(~previous)*pitch<=.25+1e-9 if previous.any() else np.zeros(shape,bool)
        anchored=np.unique(components[anchors & all_paths])
        floating=0 if abs(z-.2)<1e-5 else number-int(np.count_nonzero(anchored))
        if floating:
            failures.append('UNANCHORED_DEPOSITION_ISLANDS')
        previous=all_paths
        rows.append({'z_mm':float(z),'model_islands':int(count),'lost_model_islands':int(absent),
                     'unanchored_path_islands':int(floating),'outside_model_area_mm2':outside,
                     'cavity_support_area_mm2':cavity_area,'support_material_intrusion_mm2':intrusion,
                     'model_deposited_area_mm2':float(deposited.sum()*pitch**2),**boundary})
    return {'raster_pitch_mm':pitch,'failures':sorted(set(failures)),'layers':rows}


def audit_toolpath_correspondence(mesh_on_bed, gcode_text, settings, *, raster_pitch_mm=.1):
    """Converged DIGITAL surface/nominal-bead checks, not physical extrusion proof.

    Source boundaries at three heights per actual .2mm layer must be covered
    within .15mm; missing runs/lengths >.2mm fail. Extruded material outside the
    source envelope by >.15mm, and support in source cavities/material, fail
    above .02mm² per layer. Interior infill is intentionally not solid occupancy.
    Nominal deposition components must meet the bed or the preceding layer
    within .25mm; this is connectivity, not an unsupported-span strength model.
    .1/.05mm (or finer requested) sampling must converge before DIGITAL PASS.
    """
    segments,parsed=_parse_gcode(gcode_text); problems=list(parsed['unresolved'])
    pitch=float(raster_pitch_mm)
    if not math.isfinite(pitch) or pitch<=0:
        raise ValueError('raster pitch must be positive')
    if not np.any(segments[:,9]==1):
        return _result('FAIL',lost_model_islands=0,reason='NO_MODEL_DEPOSITION',parse=parsed)
    if not np.isfinite(segments).all():
        problems.append('NONFINITE_DEPOSITION')
    if np.any(np.abs(segments[:,2]-segments[:,5])>.001):
        problems.append('NONPLANAR_DEPOSITION')
    if np.any(np.abs(segments[:,8]-.2)>1e-6):
        problems.append('UNEXPECTED_DEPOSITION_LAYER_HEIGHT')
    if np.any((segments[:,7]<.05)|(segments[:,7]>2.)):
        problems.append('UNCERTAIN_COMMANDED_BEAD_WIDTH')
    if abs(mesh_on_bed.bounds[0,2])>1e-4:
        problems.append('SOURCE_NOT_ON_BUILD_PLATE')
    if not mesh_on_bed.is_watertight:
        problems.append('SOURCE_NOT_WATERTIGHT')
    if not audit_effective_settings(settings)['passed']:
        problems.append('UNVERIFIED_EFFECTIVE_PROFILE')
    # Do not rasterize invalid positions or unbounded widths.
    if not np.isfinite(segments).all() or np.any((segments[:,7]<.05)|(segments[:,7]>2.)):
        return _result('REVIEW_REQUIRED',unresolved=problems,lost_model_islands=0,parse=parsed)
    pitches=[min(.1,pitch),min(.1,pitch)/2]; sections={}
    try:
        screens=[_digital_layer_screen(mesh_on_bed,segments,p,sections) for p in pitches]
    except (ValueError,RuntimeError,IndexError) as error:
        return _result('REVIEW_REQUIRED',unresolved=problems+[str(error)],lost_model_islands=0,parse=parsed)
    coarse,fine=screens; metrics=('uncovered_boundary_mm','longest_boundary_gap_mm',
                                'outside_model_area_mm2','cavity_support_area_mm2','support_material_intrusion_mm2')
    deltas={key:max((abs(a[key]-b[key]) for a,b in zip(coarse['layers'],fine['layers'])),default=0.) for key in metrics}
    limits={'uncovered_boundary_mm':.15,'longest_boundary_gap_mm':.1,'outside_model_area_mm2':.05,
            'cavity_support_area_mm2':.05,'support_material_intrusion_mm2':.05}
    same_decision=coarse['failures']==fine['failures']
    converged=same_decision and all(deltas[k]<=limits[k]+1e-9 for k in metrics)
    definite=sorted(set(coarse['failures']) & set(fine['failures']))
    if not converged:
        problems.append('SAMPLING_NOT_CONVERGED')
    status='FAIL' if definite else 'REVIEW_REQUIRED' if problems else 'PASS'
    return _result(status,failures=definite,unresolved=list(dict.fromkeys(problems)),
                   capability_missing=[],physical_print_validated=False,
                   scope='digital_source_surface_and_nominal_swept_bead_correspondence_v1',
                   support_removal_scope='separate_artifact_bound_human_review_required',
                   interior_policy='configured_infill_not_solid_occupancy',
                   sampling_pitches_mm=pitches,raster_pitch_mm=pitches[-1],
                   section_offsets_below_layer_top_mm=[.175,.1,.025],
                   tolerances={'boundary_distance_mm':.15,'uncovered_boundary_limit_mm':.2,
                               'outside_distance_mm':.15,'outside_or_intrusion_area_limit_mm2':.02,
                               'anchoring_distance_mm':.25},
                   convergence={'passed':converged,'max_per_layer_deltas':deltas,'limits':limits,'same_gate_decisions':same_decision},
                   lost_model_islands=sum(row['lost_model_islands'] for row in fine['layers']),
                   lost_thin_regions=sum(row['longest_boundary_gap_mm']>.2 for row in fine['layers']),
                   unanchored_path_islands=sum(row['unanchored_path_islands'] for row in fine['layers']),
                   layers=fine['layers'],resolution_checks=screens,parse=parsed)


def audit_support_review(review, mesh_sha256, gcode_sha256, archive_sha256):
    if review is None:
        return _result("REVIEW_REQUIRED", reason="ARTIFACT_BOUND_HUMAN_SUPPORT_REVIEW_MISSING")
    try:
        data = json.loads(Path(review).read_text()) if isinstance(review, (str, Path)) else dict(review)
        hashes = (data.get("mesh_sha256"), data.get("gcode_sha256"), data.get("archive_sha256"))
        if hashes != (mesh_sha256, gcode_sha256, archive_sha256):
            return _result("FAIL", reason="SUPPORT_REVIEW_ARTIFACT_MISMATCH")
        if any(data.get(k) is False for k in ("supports_accessible", "supports_removable")):
            return _result("FAIL", reason="HUMAN_REVIEW_REJECTED_SUPPORT_REMOVAL", review=data)
        complete = (data.get("schema") == "shadowart_support_review_v1"
                    and data.get("decision") == "accessible_and_removable"
                    and all(data.get(k) is True for k in ("supports_accessible", "supports_removable", "all_contacts_reviewed"))
                    and isinstance(data.get("reviewer"), str) and bool(data["reviewer"].strip())
                    and isinstance(data.get("notes"), str) and bool(data["notes"].strip()))
        datetime.fromisoformat(data["reviewed_at"].replace("Z", "+00:00"))
        return _result("PASS" if complete else "REVIEW_REQUIRED", review=data,
                       scope="human_accessibility_and_removal_decision_only")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return _result("REVIEW_REQUIRED", reason=f"INVALID_SUPPORT_REVIEW: {error}")


def _warnings(result, archive, log_text):
    messages = []
    def visit(value, path="result"):
        if isinstance(value, dict):
            for key, item in value.items():
                if "warning" in key.lower() and item not in (None, "", [], {}):
                    messages.append((path + "." + key, str(item)))
                else:
                    visit(item, path + "." + key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
    visit(result)
    if "Metadata/slice_info.config" in archive.namelist():
        for element in _xml(archive.read("Metadata/slice_info.config")).iter():
            if element.tag.rsplit("}", 1)[-1] == "warning":
                messages.append(("archive.slice_info", element.attrib.get("msg", str(element.attrib))))
    messages += [("slicer.log", line.strip()) for line in log_text.splitlines()
                 if re.search(r"\b(warning|error|fatal)\b", line, re.I)]
    rows = []
    for source, message in messages:
        classification = ("ACKNOWLEDGED_NON_GEOMETRIC" if message == "not_support_traditional_timelapse"
                          else "FAIL" if re.search(r"floating|empty.?model|collision|non.?manifold|lost.?feature|fatal", message, re.I)
                          else "REVIEW_REQUIRED")
        rows.append({"source": source, "message": message, "classification": classification})
    status = "FAIL" if any(r["classification"] == "FAIL" for r in rows) else "REVIEW_REQUIRED" if any(r["classification"] == "REVIEW_REQUIRED" for r in rows) else "PASS"
    return _result(status, warnings=rows)


def audit_sliced_artifacts(print_mesh_path, archive_path, effective_settings_path, result_path,
                           *, log_path=None, support_review=None):
    """Read-only audit of actual or synthetic artifacts; no slicer launch."""
    import trimesh
    source_path, archive_path = Path(print_mesh_path), Path(archive_path)
    checks = {}; hashes = {"mesh_sha256": _sha(source_path), "archive_sha256": _sha(archive_path),
                           "effective_settings_sha256": _sha(effective_settings_path), "result_sha256": _sha(result_path)}
    effective = json.loads(Path(effective_settings_path).read_text())
    result = json.loads(Path(result_path).read_text())
    checks["effective_settings"] = audit_effective_settings(effective)
    checks["slicer_completion"] = _result("PASS" if result.get("return_code") == 0 and result.get("sliced_plates") else "FAIL")
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or sum(i.file_size for i in archive.infolist()) > 512 * 1024 ** 2:
            raise ValueError("duplicate or oversized archive members")
        if any(i.file_size > 256 * 1024 ** 2 for i in archive.infolist()):
            raise ValueError("oversized archive member")
        gcode_paths = [n for n in names if re.fullmatch(r"Metadata/plate_\d+\.gcode", n)]
        if len(gcode_paths) != 1:
            raise ValueError("exactly one plate G-code is required")
        data = archive.read(gcode_paths[0]); hashes["gcode_sha256"] = hashlib.sha256(data).hexdigest()
        declared_md5 = archive.read(gcode_paths[0] + ".md5").decode().strip().lower()
        crc_ok = archive.testzip() is None
        checks["archive_integrity"] = _result("PASS" if crc_ok and hashlib.md5(data).hexdigest() == declared_md5 else "FAIL",
                                               zip_crc_passed=crc_ok, members_sha256={n: hashlib.sha256(archive.read(n)).hexdigest() for n in names})
        text = data.decode("utf-8")
        parsed = audit_gcode(text)
        checks["gcode_parse"] = parsed
        checks["gcode_effective_settings"] = audit_effective_settings(parsed["config"])
        archive_settings = json.loads(archive.read("Metadata/project_settings.config"))
        checks["archive_effective_settings"] = audit_effective_settings(archive_settings)
        log_text = Path(log_path).read_text(errors="replace") if log_path and Path(log_path).is_file() else ""
        if log_path and Path(log_path).is_file():
            hashes["slicer_log_sha256"] = _sha(log_path)
        checks["warnings"] = _warnings(result, archive, log_text)
        source = trimesh.load(source_path, force="mesh", process=True)
        try:
            placed, checks["archive_model"] = _archive_model(archive, source)
            if checks["archive_model"]["passed"]:
                checks["toolpath_correspondence"] = audit_toolpath_correspondence(placed, text, effective)
            else:
                checks["toolpath_correspondence"] = _result("REVIEW_REQUIRED", reason="SOURCE_MODEL_CORRESPONDENCE_UNVERIFIED")
        except Exception as error:
            checks["archive_model"] = _result("REVIEW_REQUIRED", reason=str(error))
            checks["toolpath_correspondence"] = _result("REVIEW_REQUIRED", reason="SOURCE_MODEL_CORRESPONDENCE_UNVERIFIED")
    checks["support_removal"] = audit_support_review(support_review, hashes["mesh_sha256"], hashes["gcode_sha256"], hashes["archive_sha256"])
    support_pass = checks["support_removal"]["passed"]
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks.values()) else "PASS" if all(c["passed"] for c in checks.values()) else "REVIEW_REQUIRED"
    return _result(status, checks=checks, source_hashes=hashes,
                   source_paths={'mesh':str(Path(source_path).resolve()),
                                 'archive':str(Path(archive_path).resolve()),
                                 'effective_settings':str(Path(effective_settings_path).resolve()),
                                 'slicer_result':str(Path(result_path).resolve()),
                                 'slicer_log':str(Path(log_path).resolve()) if log_path else None,
                                 'gcode_member':gcode_paths[0]},
                   support_removal_pass=support_pass, physical_print_validated=False,
                   printer_started=False, structural_validation="NOT_REQUESTED")


def run_slicer(print_mesh_path, stage_dir, *, appimage=APPIMAGE, support_review=None):
    """Run only the existing local Bambu CLI wrapper, then apply stricter audits."""
    source, stage, executable = Path(print_mesh_path).resolve(), Path(stage_dir).resolve(), Path(appimage).resolve()
    if not source.is_file() or not executable.is_file():
        return _result("FAIL", checks={"inputs": _result("FAIL", reason="SOURCE_OR_SLICER_MISSING")}, support_removal_pass=False)
    if stage.exists() and (not stage.is_dir() or any(stage.iterdir())):
        raise FileExistsError("slicer stage must be new or empty")
    stage.mkdir(parents=True, exist_ok=True)
    source_hash, executable_hash = _sha(source), _sha(executable)
    try:
        import importlib.util
        from .provenance import capture_dependencies,verify_dependencies
        names=('run_shadowart_manufacturing_20260905','shadowart_manufacturing_20260905',
               'multiview_contract_delivery_20260902','multiview_export_adapter_20260830',
               'multiview_native_delivery_20260905')
        sources=[Path(importlib.util.find_spec('experiments.'+name).origin) for name in names]
        dependencies=capture_dependencies(sources,stage/'legacy_source_snapshot')
        from experiments.run_shadowart_manufacturing_20260905 import slice_supported
        legacy = slice_supported(executable, source, stage)
        report = audit_sliced_artifacts(source, stage / "slicer/shadowart_supported.gcode.3mf",
                                        stage / "slicer/effective_settings.json", stage / "slicer/result.json",
                                        log_path=stage / "slicer/slicer.log", support_review=support_review)
        report["legacy_cli_provenance"] = legacy
        unchanged = (_sha(source) == source_hash and _sha(executable) == executable_hash
                     and verify_dependencies(dependencies))
        report['external_dependency_sources']=dependencies
        report['external_dependency_execution']='legacy_sources_hash_checked_before_and_after'
        report["checks"]["immutable_source_and_executable"] = _result("PASS" if unchanged else "FAIL")
        report["slicer_executable_sha256"] = executable_hash
        if not unchanged:
            report["status"], report["passed"] = "FAIL", False
        return report
    except Exception as error:
        return _result("FAIL", checks={"slicer_execution": _result("FAIL", reason=str(error),
                        failure_code=getattr(error, "status", type(error).__name__))},
                       support_removal_pass=False, mesh_sha256=source_hash,
                       slicer_executable_sha256=executable_hash,
                       physical_print_validated=False, printer_started=False)
