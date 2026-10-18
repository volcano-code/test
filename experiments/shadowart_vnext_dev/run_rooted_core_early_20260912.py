"""Frozen three-hero native128 from-scratch early-core-loss development pair."""
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
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--camera-contract', type=Path, required=True)
    parser.add_argument('--silhouettes', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.camera_contract.resolve().parent
    output = args.output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        parser.error('output must not overlap the frozen protocol source tree')
    return args


def train_then_diagnose(train, diagnose):
    """Finish and persist both arms before any optional diagnostic can fail."""
    result = train()
    for name, arm in result['arms'].items():
        diagnose(name, arm['probability'])
    return result


def main(argv=None):
    args = parse_args(argv)
    import numpy as np
    import torch
    from experiments.shadowart_vnext.artifacts import create_run, write_json
    from experiments.shadowart_vnext.provenance import freeze_sources, capture_dependencies
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    output = create_run(args.output.resolve())
    package_dir = ROOT / 'experiments/shadowart_vnext'
    package, hashes = freeze_sources(output / 'source_snapshot', package_dir, __file__)
    native = capture_dependencies([package_dir / 'rooted_core_kernel.cpp'], output / 'source_snapshot/native')
    module = lambda name: importlib.import_module(package + '.' + name)
    early, core = module('rooted_core_early'), module('rooted_core')
    file_hash = module('contracts').file_hash
    report = {'status': 'RUNNING', 'scope': 'shadowart_geometry_only',
              'structural_status': 'NOT_REQUESTED', 'paper_method_equivalence': False,
              'canonical_delivery_published': False, 'digital_manufacturing_passed': False,
              'experiment_type': 'from_scratch_shared_prefix_early_auxiliary_development_pair',
              'source_execution': 'captured_sources', 'source_sha256': hashes,
              'native_dependencies': native, 'trained_field_initialization_used': False,
              'settings': {'resolution': 128, 'training_image_size': 512, 'steps': 2000,
                           'warmup': 300, 'ramp': 300, 'lr': .007, 'gradient_ratio': .1,
                           'width': 2, 'initial_logits': -4., 'view_update': 'fixed_joint',
                           'seed_label': 42, 'random_initialization_noise': 0.,
                           'device': 'cpu', 'threads': 4, 'geometry': 'shadow_closure_width2_padded_sdf'},
              'environment': {'python': sys.executable, 'python_version': platform.python_version(),
                              'torch': str(torch.__version__), 'deterministic_algorithms': True},
              'arms': {}}
    write_json(output / 'REPORT.json', report)
    try:
        contract, targets = early.load_protocol(args.camera_contract, args.silhouettes)
        tracked = {str(package_dir / name): digest for name, digest in hashes.items()
                   if name != Path(__file__).name}
        tracked[str(Path(__file__).resolve())] = hashes[Path(__file__).name]
        tracked.update({path: row['sha256'] for path, row in native.items()})
        tracked.update({str(p.resolve()): file_hash(p) for p in [args.camera_contract, *args.silhouettes]})
        if not all(file_hash(path) == digest for path, digest in tracked.items()):
            raise ValueError('FAILED_SOURCE_CAPTURE_IDENTITY')
        report.update(tracked_sha256=tracked, camera_digest=contract.digest)
        write_json(output / 'CAMERA_CONTRACT.json', contract.to_dict())
        backend = core.build_backend(output / 'native_build',
                                     source_path=output / 'source_snapshot/native/rooted_core_kernel.cpp')
        report['native_build'] = {'source_sha256': backend.source_sha256,
                                  'library_sha256': backend.library_sha256,
                                  'library_path': str(backend.library_path),
                                  'compile_command': backend.compile_command}
        truth = targets.render(contract.cameras, 512, contract.layout)
        root = None

        def progress(arm, row):
            if row['step'] == 1 or row['step'] % 50 == 0 or row['step'] == 301:
                write_json(output / 'latest.json', {'arm': arm, **row})
                print(json.dumps({'arm': arm, **row}), flush=True)

        def split(logits, common, calibration):
            nonlocal root
            root = tuple(common['root_anchor_xyz'])
            np.save(output / 'diagnostics/common_step300_logits.npy', logits, allow_pickle=False)
            report.update(common=common, calibration=calibration)
            write_json(output / 'REPORT.json', report)
            print(json.dumps({'phase': 'shared_prefix_complete', 'root': root,
                              'calibration': calibration}), flush=True)

        def finish_arm(name, probability, metadata):
            folder = output / 'diagnostics' / name
            folder.mkdir()
            np.save(folder / 'final_probability.npy', probability, allow_pickle=False)
            volume = probability >= .5
            np.save(folder / 'final_occupancy.npy', volume, allow_pickle=False)
            report['arms'][name] = {**metadata, 'geometry_status': 'PENDING'}
            write_json(output / 'REPORT.json', report)

        def diagnose_arm(name, probability):
            folder = output / 'diagnostics' / name
            volume = probability >= .5
            measure, core_probability = module('rooted_core_probe').measure(
                probability, contract, targets, width=2, root=root, backend=backend)
            measure['stage'] = 'from_scratch_early_core_soft'
            np.save(folder / 'final_core_probability.npy', core_probability, allow_pickle=False)
            report['arms'][name]['soft_metrics'] = measure
            write_json(output / 'REPORT.json', report)
            if not volume.any():
                report['arms'][name].update(geometry_status='FAILED_EMPTY_OCCUPANCY', mesh_qa_passed=False)
                write_json(output / 'REPORT.json', report)
                return
            closed, closure = module('ray_closure').shadow_closure(
                volume, contract.cameras, contract.extent, minimum_block_width=2)
            np.save(folder / 'closure_occupancy.npy', closed, allow_pickle=False)
            post = module('metrics').occupancy_metrics(closed, contract, targets, stage='post_shadow_closure')
            mesh = module('geometry').occupancy_to_mesh(closed, contract.extent)
            qa, _ = module('delivery').audit_serialized_mesh(
                mesh, contract, targets, folder / 'serialized', geometry_settings={
                    'method': 'matched_shadow_closure_width2_padded_sdf', 'diagnostic_only': True,
                    'acceptance_protocol': 'continuous_mesh_v1', 'longest_mm': 100.,
                    'min_wall_mm': .9, 'min_neck_mm': 1.35})
            report['arms'][name].update(geometry_status='DIAGNOSTIC_COMPLETE', closure=closure,
                                        post_geometry_metrics=post, mesh_quality=qa['quality'],
                                        mesh_qa_passed=qa['passed'], mesh_qa_file=str(folder / 'serialized/QA.json'))
            write_json(output / 'REPORT.json', report)
            print(json.dumps({'arm': name, 'phase': 'diagnostics_complete',
                              'mesh_quality': qa['quality'], 'mesh_qa_passed': qa['passed']}), flush=True)

        result = train_then_diagnose(lambda: early.train_pair(
                                  truth, contract.cameras, resolution=128, steps=2000,
                                  warmup=300, ramp=300, lr=.007, gradient_ratio=.1, width=2,
                                  extent=contract.extent, backend=backend, callback=progress,
                                  split_callback=split, arm_callback=finish_arm), diagnose_arm)
        report['common'] = result['common']
        report['calibration'] = result['calibration']
        report['shared_optimizer_verified'] = all(
            arm['start_state_sha256'] == result['common']['state_sha256'] for arm in result['arms'].values())
        report['source_unchanged'] = all(file_hash(path) == digest for path, digest in tracked.items())
        if not report['shared_optimizer_verified'] or not report['source_unchanged']:
            raise ValueError('FAILED_PAIR_OR_IMMUTABILITY')
        report['status'] = 'DIAGNOSTIC_COMPLETED_NOT_DELIVERED'
        write_json(output / 'REPORT.json', report)
        print(json.dumps({'status': report['status'], 'output': str(output)}), flush=True)
        return 2
    except Exception as error:
        report.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        write_json(output / 'REPORT.json', report)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
