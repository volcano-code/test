"""Frozen-soft, two-island diagnostic of a complete thick replacement sequence."""
import argparse
import importlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--soft-artifact-dir', type=Path, required=True)
    p.add_argument('--silhouettes', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--block-width', type=int, choices=(2, 3, 4), default=2)
    p.add_argument('--whole-path', action='store_true', help='try bounded complete thick paths at greedy stalls')
    p.add_argument('--allow-route-dip', action='store_true', help='allow a temporary route objective dip; final gates unchanged')
    a = p.parse_args(argv)
    if a.allow_route_dip and not a.whole_path:
        p.error('allow-route-dip requires whole-path')
    if not 0 <= a.steps <= 1024:
        p.error('steps must be in [0,1024]')
    source, output = a.soft_artifact_dir.resolve(), a.output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        p.error('output must be separate from the immutable source tree')
    return a


def main(argv=None):
    a = parse_args(argv)
    from experiments.shadowart_vnext.artifacts import create_run, write_json
    from experiments.shadowart_vnext.provenance import freeze_sources
    source, output = a.soft_artifact_dir.resolve(), create_run(a.output.resolve())
    name, hashes = freeze_sources(output / 'source_snapshot', ROOT / 'experiments/shadowart_vnext', __file__)
    def module(s):
        return importlib.import_module(name + '.' + s)
    import numpy as np
    contracts, artifacts = module('contracts'), module('artifacts')
    file_hash, write_json = contracts.file_hash, artifacts.write_json
    report = {'status':'RUNNING', 'canonical_delivery_published':False, 'structural_status':'NOT_REQUESTED',
              'paper_method_equivalence':False, 'physical_print_validated':False,
              'source_sha256':hashes, 'source_execution':'captured_source_snapshot',
              'python_executable':sys.executable, 'python_version':platform.python_version(),
              'arguments':{'soft_artifact_dir':str(source), 'steps':a.steps, 'block_width':a.block_width,
                           'image_size':1024, 'islands':2, 'whole_path':a.whole_path,
                           'allow_route_dip':a.allow_route_dip,
                           'independent_transactions':True}, 'candidates':{}}
    write_json(output / 'REPORT.json', report)
    try:
        contract = contracts.CameraContract.from_dict(json.loads((source / 'CAMERA_CONTRACT.json').read_text()))
        targets = contracts.TargetSet.from_paths(a.silhouettes)
        if targets.input_hashes != contract.input_hashes or tuple(contracts.mask_hash(m) for m in
                targets.render(contract.cameras, contract.image_size, contract.layout)) != contract.target_hashes:
            raise ValueError('FAILED_TARGET_HASH')
        if contract.volume_resolution != 128:
            raise ValueError('native 128 required')
        fields = artifacts.load_soft(source, contract)
        source_hashes = {f.name:file_hash(f) for f in source.iterdir() if f.is_file()}
        closed, closure = module('ray_closure').shadow_closure(fields['occupancy'], contract.cameras,
            contract.extent, minimum_block_width=2)
        np.save(output / 'diagnostics/closure_xyz.npy', closed, allow_pickle=False)
        write_json(output / 'CAMERA_CONTRACT.json', contract.to_dict())
        masks = targets.render(contract.cameras, 1024, contract.layout)
        initial = module('layout').iou_report(module('rendering').render_binary(closed,
            contract.cameras, 1024, contract.extent), masks)
        report.update(native_resolution=128, camera_digest=contract.digest,
                      input_sha256=source_hashes, closure=closure, initial_hard_quality=initial)
        write_json(output / 'REPORT.json', report)
        islands = module('regional_candidates').component_order(closed, 2, order='largest')[:2]
        for index, points in enumerate(islands, 1):
            stage = output / f'diagnostics/island_{index}'
            stage.mkdir()
            selected = np.zeros_like(closed); selected[tuple(points.T)] = True
            np.save(stage / 'selected_xyz.npy', selected, allow_pickle=False)
            def progress(row):
                if row['step'] == 1 or row['step'] % 16 == 0:
                    write_json(stage / 'latest.json', row)
                    print(json.dumps({'island':index, 'step':row['step'], 'quality':row['quality']}), flush=True)
            result, trace, candidate = module('component_replacement').replace_component(closed,
                fields['guidance'], selected, masks, contract.cameras, extent=contract.extent,
                longest_mm=100., block_width=a.block_width, max_steps=a.steps, callback=progress,
                whole_path=a.whole_path, allow_route_dip=a.allow_route_dip)
            np.save(stage / 'result_xyz.npy', result, allow_pickle=False)
            np.save(stage / 'candidate_xyz.npy', candidate, allow_pickle=False)
            write_json(stage / 'TRACE.json', trace)
            row = {k:v for k,v in trace.items() if k != 'history'}
            report['candidates'][f'island_{index}'] = row
            write_json(output / 'REPORT.json', report)
            print(json.dumps({'island':index, 'accepted':trace['accepted'],
                'hard_quality':trace['candidate_quality'], 'reasons':trace['reasons']}), flush=True)
            mesh = module('geometry').occupancy_to_mesh(candidate, extent=contract.extent)
            qa, _ = module('delivery').audit_serialized_mesh(mesh, contract, targets, stage / 'serialized',
                geometry_settings={'method':trace['algorithm'], 'max_steps':a.steps,
                    'block_width':a.block_width, 'whole_path':a.whole_path,
                    'allow_route_dip':a.allow_route_dip,
                    'selected_component_sha256':file_hash(stage / 'selected_xyz.npy'),
                    'candidate_accepted_as_transaction':trace['accepted'], 'diagnostic_only':True,
                    'acceptance_protocol':'continuous_mesh_v1', 'longest_mm':100.,
                    'min_wall_mm':.9, 'min_neck_mm':1.35})
            row.update(mesh_quality=qa['quality'], mesh_qa_passed=qa['passed'])
            write_json(output / 'REPORT.json', report)
            print(json.dumps({'island':index, 'mesh_quality':qa['quality'], 'mesh_qa_passed':qa['passed']}), flush=True)
        if source_hashes != {f.name:file_hash(f) for f in source.iterdir() if f.is_file()}:
            raise RuntimeError('SOURCE_ARTIFACT_CHANGED')
        report.update(status='DIAGNOSTIC_COMPLETED_NOT_DELIVERED', source_hashes_unchanged=True)
    except Exception as error:
        report.update(status='ERROR', error=str(error))
        write_json(output / 'REPORT.json', report)
        raise
    write_json(output / 'REPORT.json', report)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
