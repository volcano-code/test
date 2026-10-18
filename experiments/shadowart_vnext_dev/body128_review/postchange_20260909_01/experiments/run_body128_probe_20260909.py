"""Bounded, frozen-soft regional A/B diagnostic; never publishes a delivery."""
import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--soft-artifact-dir', type=Path, required=True)
    p.add_argument('--silhouettes', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--regions', type=int, choices=range(1, 129), default=16)
    p.add_argument('--body-reinforcement', action='store_true')
    a = p.parse_args()
    source, output = a.soft_artifact_dir.resolve(), a.output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        p.error('immutable source and output must be separate')
    from experiments.shadowart_vnext.artifacts import create_run
    from experiments.shadowart_vnext.provenance import freeze_sources
    create_run(output)
    name, hashes = freeze_sources(output / 'source_snapshot', ROOT / 'experiments/shadowart_vnext', __file__)
    def module(s):
        return importlib.import_module(name + '.' + s)
    import numpy as np
    contracts, artifacts = module('contracts'), module('artifacts')
    write_json, file_hash = artifacts.write_json, contracts.file_hash
    contract = contracts.CameraContract.from_dict(json.loads((source / 'CAMERA_CONTRACT.json').read_text()))
    targets = contracts.TargetSet.from_paths(a.silhouettes)
    if targets.input_hashes != contract.input_hashes or tuple(contracts.mask_hash(m) for m in
            targets.render(contract.cameras, contract.image_size, contract.layout)) != contract.target_hashes:
        raise ValueError('FAILED_TARGET_HASH')
    if contract.volume_resolution != 128:
        raise ValueError('native 128 required for this controlled probe')
    source_hashes = {f.name:file_hash(f) for f in source.iterdir() if f.is_file()}
    fields = artifacts.load_soft(source, contract)
    closed, closure = module('ray_closure').shadow_closure(fields['occupancy'], contract.cameras,
        contract.extent, minimum_block_width=2)
    np.save(output / 'diagnostics/closure_xyz.npy', closed, allow_pickle=False)
    report = {'status':'RUNNING', 'canonical_delivery_published':False, 'structural_status':'NOT_REQUESTED',
        'paper_method_equivalence':False, 'native_resolution':128, 'soft_source':str(source),
        'input_sha256':source_hashes, 'source_sha256':hashes, 'camera_digest':contract.digest,
        'source_execution':'captured_source_snapshot', 'closure':closure, 'arguments':{
            'regions':a.regions, 'body_reinforcement':a.body_reinforcement, 'image_size':1024,
            'max_primitives':24, 'max_selected':4, 'mean_budget':.001, 'view_budget':.002,
            'time_limit':5., 'region_order':'largest'}}
    write_json(output / 'REPORT.json', report)
    masks = targets.render(contract.cameras, 1024, contract.layout)
    regional = module('regional_geometry')
    # Inspect first two islands independently before changing the working volume.
    proposal_rows = []
    candidates = module('regional_candidates')
    protected = module('geometry')._enclosed_voids(closed)
    for points in candidates.component_order(closed, 2, order='largest')[:2]:
        base, base_pred, primitives, projections, descriptions = candidates.propose(closed, fields['guidance'],
            points, masks, contract.cameras, protected, 2, contract.extent, 24,
            body_reinforcement=a.body_reinforcement)
        rows = []
        for indices, projection, description in zip(primitives, projections, descriptions):
            candidate = base.copy(); candidate.ravel()[indices] = True
            q = module('layout').iou_report(base_pred | projection, masks)
            rows.append({'description':description, 'mean_iou':q['mean_iou'], 'worst_iou':q['worst_iou'],
                'new_voxels':int((candidate & ~closed).sum()),
                'neck_guard':regional.thick_join_preserved(candidate, base, candidate & ~closed, 2)})
        proposal_rows.append({'component_voxels':len(points), 'candidates':rows})
        print(json.dumps({'stage':'proposal_diagnostic', **proposal_rows[-1]}), flush=True)
    write_json(output / 'diagnostics/first_two_pools.json', proposal_rows)
    def progress(row):
        write_json(output / 'diagnostics/latest.json', row)
        print(json.dumps({k:row[k] for k in ('region','accepted','primitive_count')}
            | {k:row[k] for k in ('quality','physical_after') if k in row}), flush=True)
    archive, trace, search = regional.repair_regional(closed, fields['guidance'], masks, contract.cameras,
        extent=contract.extent, max_regions=a.regions, max_primitives=24, region_order='largest',
        callback=progress, body_reinforcement=a.body_reinforcement)
    write_json(output / 'diagnostics/geometry_trace.json', trace)
    report['trace_summary'] = {k:v for k,v in trace.items() if k != 'history'}
    report['candidates'] = {}
    for label, volume in (('archive', archive), ('search', search)):
        np.save(output / f'diagnostics/{label}_xyz.npy', volume, allow_pickle=False)
        mesh = module('geometry').occupancy_to_mesh(volume, extent=contract.extent)
        qa, files = module('delivery').audit_serialized_mesh(mesh, contract, targets,
            output / f'diagnostics/{label}_serialized', geometry_settings={
                'method':trace['algorithm'], 'body_reinforcement':a.body_reinforcement,
                'diagnostic_only':True, 'acceptance_protocol':'continuous_mesh_v1',
                'longest_mm':100., 'min_wall_mm':.9, 'min_neck_mm':1.35})
        report['candidates'][label] = {'quality':qa['quality'], 'qa_passed':qa['passed']}
        print(json.dumps({'stage':label+'_serialized', **report['candidates'][label]}), flush=True)
        write_json(output / 'REPORT.json', report)
    if source_hashes != {f.name:file_hash(f) for f in source.iterdir() if f.is_file()}:
        raise RuntimeError('SOURCE_ARTIFACT_CHANGED')
    report.update(status='DIAGNOSTIC_COMPLETED_NOT_DELIVERED', source_hashes_unchanged=True)
    write_json(output / 'REPORT.json', report)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
