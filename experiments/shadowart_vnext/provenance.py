"""Execute each run from its captured source bundle, never delayed live imports."""
import hashlib
import importlib
import importlib.util
import importlib.metadata
import platform
from pathlib import Path
import sys

from .artifacts import create_run,write_json


def capture_dependencies(paths, snapshot):
    """Declare legacy helper sources separately; they are not vNext modules."""
    snapshot=Path(snapshot)
    snapshot.mkdir(parents=True,exist_ok=False)
    records={}
    for source in paths:
        source=Path(source).resolve()
        raw=source.read_bytes()
        with (snapshot/source.name).open('xb') as stream:
            stream.write(raw)
        records[str(source)]={'sha256':hashlib.sha256(raw).hexdigest(),
                             'snapshot':str(snapshot/source.name)}
    return records


def verify_dependencies(records):
    return all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==row['sha256']
               for p,row in records.items())


def freeze_sources(snapshot, package, entry):
    snapshot,package,entry=Path(snapshot),Path(package),Path(entry)
    snapshot.mkdir(parents=True,exist_ok=False)
    sources={p.name:p.read_bytes() for p in sorted(package.glob('*.py'))}
    sources[entry.name]=entry.read_bytes()
    hashes={name:hashlib.sha256(raw).hexdigest() for name,raw in sources.items()}
    # Bytes are read once: the snapshot and its identity cannot disagree during
    # a concurrent edit to the development copy. Later imports use this copy.
    for name,raw in sources.items():
        with (snapshot/name).open('xb') as stream:
            stream.write(raw)
    name='_shadowart_frozen_'+hashlib.sha256(str(snapshot.resolve()).encode()).hexdigest()
    spec=importlib.util.spec_from_file_location(name,snapshot/'__init__.py',
                                               submodule_search_locations=[str(snapshot)])
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return name,hashes


def execute_frozen(args):
    source=args.finalize_run or args.soft_artifact_dir
    if source and Path(args.output).resolve().is_relative_to(Path(source).resolve()):
        raise ValueError('output must not modify the immutable source directory')
    output=create_run(args.output)
    versions={d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()
              if d.metadata.get('Name')}
    write_json(output/'ENVIRONMENT.json',{'python_executable':sys.executable,
               'python_version':platform.python_version(),'platform':platform.platform(),
               'distribution_versions':dict(sorted(versions.items()))})
    package=Path(__file__).parent
    name,hashes=freeze_sources(output/'source_snapshot',package,
                               package.parent/'run_shadowart_vnext.py')
    if args.finalize_run:
        finalizer=importlib.import_module(name+'.finalize')
        return finalizer.finalize_existing(args,prepared_output=output,source_hashes=hashes)
    pipeline=importlib.import_module(name+'.pipeline')
    return pipeline.execute(args,prepared_output=output,source_hashes=hashes)
