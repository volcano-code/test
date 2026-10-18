"""Frozen native128 paired rooted-core loss pilot. No canonical mesh delivery."""
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
    p.add_argument('--steps', type=int, default=20)
    a = p.parse_args(argv)
    if not 1 <= a.steps <= 200:
        p.error('mechanism pilot permits 1..200 steps; not a full ablation')
    source, output = a.soft_artifact_dir.resolve(), a.output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        p.error('output must be separate from immutable source tree')
    return a


def main(argv=None):
    a = parse_args(argv)
    import numpy as np
    import torch
    from experiments.shadowart_vnext.artifacts import create_run, write_json
    from experiments.shadowart_vnext.provenance import freeze_sources, capture_dependencies
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    source = a.soft_artifact_dir.resolve()
    output = create_run(a.output.resolve())
    package_dir = ROOT / 'experiments/shadowart_vnext'
    package, hashes = freeze_sources(output / 'source_snapshot', package_dir, __file__)
    native = capture_dependencies([package_dir / 'rooted_core_kernel.cpp'],
                                  output / 'source_snapshot/native')
    module = lambda name: importlib.import_module(package + '.' + name)
    probe, core = module('rooted_core_probe'), module('rooted_core')
    file_hash = module('contracts').file_hash
    report = {'status': 'RUNNING', 'method': 'rooted_thick_core_projection_loss_pilot_v1',
              'scope': 'shadowart_geometry_only', 'structural_status': 'NOT_REQUESTED',
              'canonical_delivery_published': False, 'paper_method_equivalence': False,
              'digital_manufacturing_passed': False, 'mesh_qa_status': 'NOT_RUN',
              'experiment_type': 'paired_warm_start_reset_Adam_not_from_scratch_ablation',
              'source_execution': 'captured_sources', 'source_sha256': hashes,
              'native_dependencies': native, 'initial_artifact': str(source),
              'arguments': {'steps': a.steps, 'lr': .007, 'gradient_ratio': .1,
                            'width_cells': 2, 'root_policy': 'fixed_single_largest_thick_core_anchor',
                            'device': 'cpu', 'threads': 4, 'nominal_pitch_mm': 100. / 128.,
                            'training_image_size': 512},
              'environment': {'python': sys.executable, 'python_version': platform.python_version(),
                              'torch': str(torch.__version__), 'deterministic_algorithms': True},
              'arms': {}}
    write_json(output / 'REPORT.json', report)
    try:
        contract, targets, fields = probe.load_frozen(source, a.silhouettes)
        source_hashes = {str(p): file_hash(p) for p in source.iterdir() if p.is_file()}
        input_hashes = {str(p.resolve()): file_hash(p) for p in a.silhouettes}
        existing_sources = {str(package_dir / name): digest for name, digest in hashes.items()
                            if name != Path(__file__).name}
        existing_sources[str(Path(__file__).resolve())] = hashes[Path(__file__).name]
        existing_sources.update({path: row['sha256'] for path, row in native.items()})
        if not all(file_hash(path) == digest for path, digest in existing_sources.items()):
            raise ValueError('FAILED_SOURCE_CHANGED_AFTER_CAPTURE')
        report.update(camera_digest=contract.digest, input_sha256=input_hashes,
                      artifact_sha256=source_hashes, existing_python_sources=existing_sources)
        write_json(output / 'CAMERA_CONTRACT.json', contract.to_dict())
        backend = core.build_backend(output / 'native_build',
                                     source_path=output / 'source_snapshot/native/rooted_core_kernel.cpp')
        report['native_build'] = {name: str(getattr(backend, name)) for name in
                                  ('source_sha256', 'library_sha256', 'library_path')}
        report['native_build']['compile_command'] = backend.compile_command
        p = np.array(fields['guidance'], copy=True)
        root = core.choose_root(p, 2)
        report['root_anchor_xyz'] = list(root)
        np.save(output / 'diagnostics/initial_probability.npy', p, allow_pickle=False)
        initial_metrics, initial_core = probe.measure(p, contract, targets, width=2,
                                                      root=root, backend=backend)
        report['initial'] = initial_metrics
        np.save(output / 'diagnostics/initial_core_probability.npy', initial_core, allow_pickle=False)
        truth = targets.render(contract.cameras, 512, contract.layout)
        calibration = probe.calibrate(p, truth, contract.cameras, width=2, root=root,
                                      backend=backend, gradient_ratio=.1, extent=contract.extent)
        report['calibration'] = calibration
        write_json(output / 'REPORT.json', report)
        print(json.dumps({'phase': 'calibrated', **calibration}), flush=True)
        for arm, coefficient in [('baseline', 0.), ('rooted_core', calibration['coefficient'])]:
            folder = output / 'diagnostics' / arm
            folder.mkdir()
            def progress(row):
                write_json(folder / 'latest.json', row)
                print(json.dumps({'arm': arm, **row}), flush=True)
            result = probe.run_arm(p, truth, contract.cameras, width=2, root=root, backend=backend,
                                   coefficient=coefficient, steps=a.steps, lr=.007,
                                   extent=contract.extent, callback=progress)
            final_p = result.pop('probability')
            np.save(folder / 'final_probability.npy', final_p, allow_pickle=False)
            np.save(folder / 'final_occupancy.npy', final_p >= .5, allow_pickle=False)
            measurements, final_core = probe.measure(final_p, contract, targets, width=2,
                                                     root=root, backend=backend)
            np.save(folder / 'final_core_probability.npy', final_core, allow_pickle=False)
            report['arms'][arm] = {**result, 'metrics': measurements}
            write_json(output / 'REPORT.json', report)
        report['paired_initialization_verified'] = (
            report['arms']['baseline']['initial_logits_sha256'] ==
            report['arms']['rooted_core']['initial_logits_sha256'])
        tracked = {**source_hashes, **input_hashes, **existing_sources}
        report['source_unchanged'] = all(file_hash(path) == digest for path, digest in tracked.items())
        if not report['paired_initialization_verified'] or not report['source_unchanged']:
            raise ValueError('FAILED_PAIRED_IDENTITY_OR_IMMUTABILITY')
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
