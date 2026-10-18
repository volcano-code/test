#!/usr/bin/env python3
"""Frozen-soft graphics diagnosis; diagnostic meshes are never print deliveries."""
import argparse
import importlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--silhouettes',nargs='+',type=Path,required=True)
    parser.add_argument('--soft-artifact-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--restore-steps',type=int,default=0,help='opt-in attached ray-restoration transactions, up to 64')
    parser.add_argument('--working-image-size',type=int,choices=(512,1024),default=1024,
                        help='fixed target raster size for opt-in restoration')
    parser.add_argument('--closure-width',type=int,choices=(1,2,4),default=4,
                        help='minimum solid block width for diagnostic shadow closure; use 2 for native 128 at 100 mm')
    parser.add_argument('--probability-surface',action='store_true',
                        help='also audit fixed .5 native probability isosurface; no level search or upsampling')
    parser.add_argument('--delivery-protocol',choices=('strict_voxel_v1','continuous_mesh_v1'),
                        default='strict_voxel_v1',
                        help='label QA with the same acceptance protocol; probe remains diagnostic-only')
    args=parser.parse_args(argv)
    if not 0<=args.restore_steps<=64:
        parser.error('--restore-steps must be in [0,64]')
    source,args_output=args.soft_artifact_dir.resolve(),args.output.resolve()
    if args_output.is_relative_to(source) or source.is_relative_to(args_output):
        parser.error('output and immutable soft source must be separate directories')
    return args


def main(argv=None):
    args=parse_args(argv)
    from experiments.shadowart_vnext.artifacts import create_run,write_json
    from experiments.shadowart_vnext.provenance import freeze_sources
    source=args.soft_artifact_dir.resolve(); output=args.output.resolve()
    output=create_run(output)
    name,hashes=freeze_sources(output/'source_snapshot',ROOT/'experiments/shadowart_vnext',Path(__file__))
    try:
        return importlib.import_module(name+'.fidelity_probe').execute(args,output,hashes)
    except Exception as error:
        write_json(output/'FAILED.json',{'status':'ERROR','error':str(error),
                   'canonical_delivery_published':False,'structural_status':'NOT_REQUESTED'})
        print(str(error),file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
