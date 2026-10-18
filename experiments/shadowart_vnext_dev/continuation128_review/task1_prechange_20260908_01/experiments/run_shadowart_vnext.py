#!/usr/bin/env python3
"""Isolated physical ShadowArt pipeline; never starts a printer or validates PINO."""
import argparse
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--silhouettes',nargs='+',type=Path)
    p.add_argument('--finalize-run',type=Path,help='re-audit existing mesh and slicing bytes without reslicing')
    p.add_argument('--tsf',type=int,choices=(0,1))
    p.add_argument('--mirror',choices=('auto','0','1'))
    p.add_argument('--camera-config',type=Path)
    p.add_argument('--soft-artifact-dir',type=Path)
    p.add_argument('--layout-contract',type=Path,help='reuse a frozen vNext layout for matched loss arms')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('preflight','soft','delivery'),default='delivery')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--resolution',type=int,default=256,help='non256 is engineering-only, cannot publish')
    p.add_argument('--image-size',type=int,default=256)
    p.add_argument('--layout-candidates',type=int,default=512)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--lr',type=float,default=.007)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--projection-mode',choices=('area_soft','dense_soft','dense_hard_st','dense_entity_max'),default='area_soft',
                   help='isolated dense supervision / binary-forward feasibility probe')
    p.add_argument('--loss-core',choices=('l1','l2','l2_l1'),default='l2_l1')
    p.add_argument('--auxiliary',choices=('none','lovasz','ray_bce'),default='none')
    p.add_argument('--aux-weight',type=float,default=0.)
    p.add_argument('--loss-config',type=Path,help='exact serialized LossConfig, including deletion coefficients')
    p.add_argument('--calibration-steps',nargs='+',type=int,default=[])
    p.add_argument('--geometry-rounds',type=int,default=32)
    p.add_argument('--geometry-method',choices=('local','joint','regional'),default='local')
    p.add_argument('--joint-steps',type=int,default=256,help='bounded solid-block steps inside a joint transaction')
    p.add_argument('--regional-regions',type=int,default=16,help='bounded component-local constrained transactions')
    p.add_argument('--regional-primitives',type=int,default=24,help='maximum full-path/patch alternatives per region')
    p.add_argument('--regional-order',choices=('nearest','largest'),default='nearest')
    p.add_argument('--disable-geometry',nargs='+',choices=('reinforcement','fin_removal','surface'),default=[])
    p.add_argument('--support-review',type=Path)
    p.add_argument('--bambu-appimage',type=Path,
                   default=Path('/home/lin/Applications/BambuStudio_ubuntu24.04-v02.07.01.62.AppImage'))
    a=p.parse_args(argv)
    supplied={word.split('=')[0] for word in (sys.argv[1:] if argv is None else argv)}
    options={word for word in supplied if word.startswith('--')}
    if a.finalize_run:
        if options-{'--finalize-run','--output','--support-review'} or not a.support_review:
            p.error('finalization accepts only --finalize-run --output --support-review')
    elif not a.silhouettes:
        p.error('--silhouettes is required unless finalizing an existing run')
    if a.loss_config and supplied.intersection({'--loss-core','--auxiliary','--aux-weight'}):
        p.error('--loss-config cannot mix with individual loss switches')
    if a.soft_artifact_dir and (a.layout_contract or a.calibration_steps):
        p.error('artifact replay cannot retrain or replace the frozen layout')
    if a.soft_artifact_dir and options.intersection({'--loss-core','--loss-config','--auxiliary','--aux-weight',
                                                    '--steps','--lr','--seed','--calibration-steps','--layout-candidates',
                                                    '--projection-mode'}):
        p.error('artifact replay cannot override training settings or rerun layout search')
    if a.projection_mode!='area_soft' and a.calibration_steps:
        p.error('dense projection probe cannot use historical optical calibration')
    if a.camera_config and (a.tsf is not None or a.mirror is not None):
        p.error('explicit --camera-config cannot mix with --tsf/--mirror')
    if not 1<=a.layout_candidates<=512 or a.resolution<2 or a.image_size<2 or a.steps<1:
        p.error('invalid resolution, steps or layout budget')
    if not 0<=a.geometry_rounds<=64 or not 0<=a.seed<2**63:
        p.error('geometry rounds must be in [0,64]; seed must be in [0,2**63)')
    if not math.isfinite(a.lr) or a.lr<=0 or not math.isfinite(a.aux_weight) or a.aux_weight<0:
        p.error('learning rate must be positive and finite; auxiliary weight nonnegative and finite')
    if len(set(a.calibration_steps))!=len(a.calibration_steps) or any(not 1<=s<=a.steps for s in a.calibration_steps):
        p.error('calibration steps must be distinct and within the training budget')
    if len(set(a.disable_geometry))!=len(a.disable_geometry):
        p.error('geometry deletions must be distinct')
    if not 0<=a.joint_steps<=1024:
        p.error('joint steps must be in [0,1024]')
    if a.geometry_method=='joint' and (a.disable_geometry or '--geometry-rounds' in options):
        p.error('joint geometry does not use local rounds or local deletion switches')
    if a.geometry_method!='joint' and '--joint-steps' in options:
        p.error('--joint-steps requires --geometry-method joint')
    if not 0<=a.regional_regions<=128 or not 1<=a.regional_primitives<=256:
        p.error('regional regions must be in [0,128], primitives in [1,256]')
    if a.geometry_method=='regional' and (a.disable_geometry or '--geometry-rounds' in options):
        p.error('regional geometry does not use local rounds or deletion switches')
    if a.geometry_method!='regional' and options.intersection({'--regional-regions','--regional-primitives','--regional-order'}):
        p.error('regional budgets require --geometry-method regional')
    return a


def main(argv=None):
    from experiments.shadowart_vnext.provenance import execute_frozen
    return execute_frozen(parse_args(argv))


if __name__=='__main__':
    raise SystemExit(main())
