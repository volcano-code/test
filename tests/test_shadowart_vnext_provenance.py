import importlib
import json
from pathlib import Path

import pytest


def test_frozen_package_executes_snapshot_even_after_original_changes(tmp_path):
    from experiments.shadowart_vnext.provenance import freeze_sources
    package=tmp_path/'package'
    package.mkdir()
    (package/'__init__.py').write_text('')
    (package/'later.py').write_text('VALUE=17\n')
    entry=tmp_path/'entry.py'
    entry.write_text('# entry\n')
    name,manifest=freeze_sources(tmp_path/'snapshot',package,entry)
    (package/'later.py').write_text('VALUE=99\n')
    assert importlib.import_module(name+'.later').VALUE==17
    assert set(manifest)=={'__init__.py','later.py','entry.py'}
    import hashlib
    assert manifest['later.py']==hashlib.sha256(b'VALUE=17\n').hexdigest()


@pytest.mark.parametrize('option,value',[('--lr','nan'),('--lr','0'),('--aux-weight','inf'),
                                       ('--geometry-rounds','65'),('--seed','-1')])
def test_invalid_cli_numeric_settings_fail_before_run(option,value,tmp_path):
    from experiments.run_shadowart_vnext import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--silhouettes','x.png','--output',str(tmp_path/'run'),option,value])
    assert not (tmp_path/'run').exists()


def test_frozen_layout_and_exact_loss_deletion_config_cli(tmp_path):
    from experiments.run_shadowart_vnext import parse_args
    args=parse_args(['--silhouettes','x.png','--output',str(tmp_path/'run'),
                     '--layout-contract','CAMERA_CONTRACT.json','--loss-config','loss.json',
                     '--calibration-steps','500','1000','1500'])
    assert args.layout_contract==Path('CAMERA_CONTRACT.json')
    assert args.loss_config==Path('loss.json')
    assert args.calibration_steps==[500,1000,1500]


def test_loss_config_cannot_be_silently_overridden(tmp_path):
    from experiments.run_shadowart_vnext import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--silhouettes','x.png','--output',str(tmp_path/'run'),
                    '--loss-config','loss.json','--loss-core','l1'])


def test_declared_external_dependency_change_is_detected(tmp_path):
    from experiments.shadowart_vnext.provenance import capture_dependencies,verify_dependencies
    source=tmp_path/'helper.py'
    source.write_text('VALUE=1\n')
    record=capture_dependencies([source],tmp_path/'captured')
    assert verify_dependencies(record)
    source.write_text('VALUE=2\n')
    assert not verify_dependencies(record)
    assert (tmp_path/'captured'/'helper.py').read_text()=='VALUE=1\n'


def test_geometry_ablation_cli_records_exact_disabled_stages(tmp_path):
    from experiments.run_shadowart_vnext import parse_args
    args=parse_args(['--silhouettes','x.png','--output',str(tmp_path/'run'),
                     '--disable-geometry','reinforcement','surface'])
    assert args.disable_geometry==['reinforcement','surface']


def test_finalize_existing_cli_needs_no_new_targets(tmp_path):
    from experiments.run_shadowart_vnext import parse_args
    args=parse_args(['--finalize-run','oldrun','--support-review','review.json','--output',str(tmp_path/'new')])
    assert args.finalize_run==Path('oldrun')
    assert args.silhouettes is None


@pytest.mark.parametrize('flag,value',[('--loss-core','l1'),('--seed','17'),('--steps','10')])
def test_artifact_replay_rejects_ignored_training_overrides(flag,value,tmp_path):
    from experiments.run_shadowart_vnext import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--silhouettes','x.png','--soft-artifact-dir','soft','--output',str(tmp_path/'new'),flag,value])


def test_finalizer_cannot_create_output_inside_immutable_source(tmp_path):
    from experiments.run_shadowart_vnext import parse_args
    from experiments.shadowart_vnext.provenance import execute_frozen
    source=tmp_path/'original';source.mkdir()
    args=parse_args(['--finalize-run',str(source),'--support-review','review.json',
                     '--output',str(source/'changed')])
    with pytest.raises(ValueError,match='immutable source'):
        execute_frozen(args)
    assert list(source.iterdir())==[]
