"""Run the diagnostic entrypoint against a real immutable tiny soft artifact."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image


def test_fidelity_probe_parser_defaults_to_thick_closure_and_rejects_other_widths(tmp_path):
    """Catches a silent return to historical one-cell closure as the default."""
    probe=importlib.import_module('experiments.run_shadowart_fidelity_probe')
    required=['--silhouettes','target.png','--soft-artifact-dir','soft',
              '--output',str(tmp_path/'probe')]
    assert probe.parse_args(required).closure_width==4
    with pytest.raises(SystemExit):
        probe.parse_args(required+['--closure-width','3'])
    source=tmp_path/'soft'
    nested_output=source/'nested'
    with pytest.raises(SystemExit):
        probe.parse_args(['--silhouettes','target.png','--soft-artifact-dir',str(source),
                          '--output',str(nested_output)])
    with pytest.raises(SystemExit):
        probe.parse_args(['--silhouettes','target.png','--soft-artifact-dir',str(nested_output),
                          '--output',str(source)])
    with pytest.raises(SystemExit):
        probe.main(['--silhouettes','target.png','--soft-artifact-dir',str(source),
                    '--output',str(nested_output)])
    assert not nested_output.exists()


@pytest.mark.parametrize('closure_width', [1, 2])
def test_fidelity_probe_freezes_inputs_and_never_publishes_diagnostics(tmp_path,capsys,closure_width):
    entry=Path(__file__).parents[1]/'experiments/run_shadowart_fidelity_probe.py'
    assert entry.is_file(), 'fidelity probe entrypoint missing'
    from experiments.run_shadowart_vnext import main as train
    mask=np.zeros((8,8),np.uint8); mask[2:6,2:6]=255
    source=tmp_path/'target.png'; Image.fromarray(mask).save(source)
    soft=tmp_path/'soft'
    assert train(['--silhouettes',str(source),'--resolution','8','--image-size','8',
                  '--device','cpu','--output',str(soft),'--stage','soft','--steps','1',
                  '--layout-candidates','1']) == 0
    manifest=(soft/'soft_training/SOFT_MANIFEST.json').read_bytes()
    output=tmp_path/'probe'
    main=importlib.import_module('experiments.run_shadowart_fidelity_probe').main
    assert main(['--silhouettes',str(source),'--soft-artifact-dir',str(soft/'soft_training'),
                 '--output',str(output),'--restore-steps','1',
                 '--working-image-size','512','--closure-width',str(closure_width),
                 *(['--probability-surface'] if closure_width==2 else [])]) == 2
    assert 'serialized_reload' not in capsys.readouterr().out
    report=json.loads((output/'REPORT.json').read_text())
    assert report['canonical_delivery_published'] is False
    assert report['structural_status']=='NOT_REQUESTED'
    expected={'restore_steps':1,'working_image_size':512,
              'closure_width':closure_width,'effective_closure_width':closure_width}
    if closure_width==2:
        expected['probability_surface']=True
        assert 'source_probability' in report['candidates']
        assert report['candidates']['source_probability']['occupancy_diagnostic_scope']=='source_threshold_field_not_mesh_voxelization'
        physical=json.loads((output/'diagnostics/source_probability_manufacturing.json').read_text())
        assert physical['occupancy_diagnostic_scope']=='source_threshold_field_not_mesh_voxelization'
        assert physical['mesh_manufacturing_certified'] is False
        assert (output/'source_snapshot/continuous_surface.py').exists()
    else:
        assert 'source_probability' not in report['candidates']
    assert report['arguments']==expected
    assert [r['image_size'] for r in report['stages']['source_hard']]==[256,512,1024]
    assert 'source_cell_boundary' in report['stages']
    assert 'source_sdf' in report['stages']
    assert 'restored_sdf' in report['stages']
    # Delivery QA may omit 1024 for a stable pair; this diagnostic must retain
    # the unconditional, exact serialized-reload comparison at every size.
    serialized=report['candidates']['source']['serialized_reload']
    assert set(serialized)=={'obj','stl'}
    for row in serialized.values():
        assert [metric['image_size'] for metric in row['evaluations']]==[256,512,1024]
        assert all('masks_identical' in metric and 'pixels_disagree' in metric
                   for metric in row['evaluations'])
    assert (output/'source_snapshot/cell_boundary.py').exists()
    assert not (output/'delivery').exists()
    assert (soft/'soft_training/SOFT_MANIFEST.json').read_bytes()==manifest
