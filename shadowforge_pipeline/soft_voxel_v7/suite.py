"""Input-only v7 suite selection, binding, and strict authentication."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import fcntl
import hashlib
import importlib
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from types import MappingProxyType

import cv2
import numpy as np
from scipy import ndimage
import torch

import experiments.soft_loss_v3_registry as registry_module
import shadowforge_pipeline.soft_voxel_v3.provenance as registry_provenance_module
from shadowforge_pipeline.soft_voxel_v3.data import normalize_target
import shadowforge_pipeline.soft_voxel_v3.data as data_module
from . import core as core_module
from .core import build_registered_target_stack, xyz_to_dhw
from . import provenance as provenance_module
from .provenance import (
    TASK1_SOURCE_SHA256,
    AuthenticatedV7Root,
    AuthenticatedFileSnapshot,
    _canonical_existing_directory,
    _deep_read_only,
    _path_argument,
    bootstrap_v7_root,
    canonical_json_bytes,
    canonical_sha256,
    decode_canonical_json,
    fsync_directory,
    publish_bytes_once,
    revalidate_snapshot,
    snapshot_regular_file,
    validate_v7_root,
)


SUITE_SCHEMA = "pcraft_soft_loss_v7_suite_v1"
SELECTOR_NAME = "pcraft_soft_loss_v7_suite_v1"
DEVELOPMENT_IDS = ("arcane_casters", "blade_heroes", "stage_dancers")
FORMAL_IDS = tuple(f"v7_formal_{index:02d}" for index in range(1, 11))
AXES = ("x", "y", "z")
SELECTOR_SEED = 20260829
CANDIDATE_ROOTS = (
    "data/silhouettes/bench3",
    "data/silhouettes/complex_v2_filled",
    "data/silhouettes/original_anime_30_20260704",
    "data/silhouettes/anime_sheet_100_20260704",
)
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
EXCLUDED_BASENAME = "CONTACT_SHEET.png"
EXPECTED_ELIGIBLE_COUNT = 103
EXPECTED_TRIPLE_COUNT = 77862
EXPECTED_HULL_CHECKS = 164
EXPECTED_STREAM_INDICES = (8, 37, 45, 108, 113, 121, 161, 170, 239, 269)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PUBLICATION_HOOK = lambda stage: None
APPROVED_DEVELOPMENT_SUITE_PATH = Path(
    "/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel/"
    "runs/soft_loss_v3_20260826/manifests/development_suite.json"
)
APPROVED_DEVELOPMENT_SUITE_SHA256 = "15da2ecc37424bb1513f922f5aac6b4217b547744e53099ecaaafc8243c08fe0"
APPROVED_DEVELOPMENT_SUITE_SEMANTIC_SHA256 = "43c44793492084b0eb4b61d4e582bff63819af6a6d32ffc3bef072cfb987abc4"
REGISTERED_MANIFEST_NAMES = frozenset(
    {"development_suite.json", "formal_suite.json", "campaign.json", "formal_matrix.json"}
)
EXECUTED_SOURCE_RELATIVES = (
    "experiments/bind_soft_loss_v7_suites.py",
    "experiments/soft_loss_v3_registry.py",
    "shadowforge_pipeline/soft_voxel_v3/data.py",
    "shadowforge_pipeline/soft_voxel_v3/provenance.py",
    "shadowforge_pipeline/soft_voxel_v7/core.py",
    "shadowforge_pipeline/soft_voxel_v7/provenance.py",
    "shadowforge_pipeline/soft_voxel_v7/suite.py",
)

FORMAL_ANCHOR_PATHS = tuple(
    f"results/fixed3_mr1_connected10_pending_20260722/inputs/connected_{task:02d}_view_{axis}.png"
    for task in range(1, 11) for axis in AXES
)
DEVELOPMENT_ANCHOR_PATHS = tuple(
    str(
        provenance_module.PROJECT_ROOT
        / f"results/table2_anime_fixed3_inputs_20260722/{task}/view_{view}.png"
    )
    for task in DEVELOPMENT_IDS for view in range(3)
)
FORMAL_ANCHOR_SHA256 = (
    "075ea08eb090ccd5fc2350b4e7393524cb7c154da5fd17e4656b1fb8d367146c", "ed684af3ebc6c054bfe272d9b9be2e6df0617231b9691873572517b7f2afa547", "5de721e47808bfd7785eb94e40f804f9659718282a135b2b27cf6d99d6ab6f1b",
    "80f21ad6dea0788f5db9ac2a178bc36358341a8198f786aa72c8406d7a4820f6", "054101ae816c840ccc7ac4a1c76e944c5ebef85ee1a00c01f8539c8e237f8e26", "8eb27cf3e4b8036bb757f737b81c44cc93a930cbb089d72cc8053528e70f3a29",
    "df5a512e697e539286c8a38b4c9dadf1b8f80554f41d026b0b6ff2f2d06dbaa0", "6546aac52ae974a5cab2d38fe0dcf689795b2d4013eb940c9df5cb39c46b854f", "078b717e50981d0f4992d7d7838e734653c780bc915dc1a5a56c49296998e2ef",
    "8b569f92474e94fd68c2f4314603aee607c329a2e31694bdaddcf304d70dd687", "fee6d5ca6056b6307fd65e09903da2db16b1b16718ff0fb03fbefb182f990e01", "54a2b64b26404e228acbcdbe9bf52b704e9aaf3340e5dbe7074972ae87c9310b",
    "3149c1f78115e1574cfa8ab61344a3402f76012839d5193727f6ae24eead7659", "ad2186ee8be6067167b880df207dd2ad5d3eb9ed6639f8a85732890820e7ca2b", "bb45fedb9c4216ff7761b236db822dfb787252290920e165d6ea12aae8194bff",
    "d5e8e58ae1a0a6172735a2906e04af45805ad440a6e984178961f4db5ff6c656", "ed64299b700dc0d2547f02b0d0b15b94a4751d61a07d2a4796931a445f0e3fd0", "8542e7cdb126ff952343eafd585b1893877299cc589265df57c425714bb35ef4",
    "ee13ec568621c9cfb1ad65d6c8469ae137e9353779ca5d14a1c7c60d286f8562", "37887af0fea53b2fa5fc03b886250549a525cb2028c5d60045fa918e58d191bf", "cf72b7d013419e2f68b9cecb2aedd78f0218e5b62f65978d1f908e9205712b17",
    "ddcc150eb1410f7df82db8d360c55a0adbec71b1efd395abbddc6c3cdf874236", "f5fe8520d5cd72122ab63b1f249c8153a35992abf475af4dbf71638edad3d7d2", "3309b1e5c3df3bd77fd2c894da8456753d6636b679d75937c2666c6f4b454586",
    "358682ddb647c0d23b836c64a0a3261831309e2b8c2150c46a631f36e5563dd1", "91e134653b2877cfb251bd3a235c64e33c7ef592c7f6f97840ce85aed73e4a4b", "6549faf1c6231e403eee2334577464c7603cd4dbe31e6cd2117f26a022844319",
    "8f9072dab70d1226e49bbdae5757de9070a242c891499966b686fc05e9d2c1f3", "64c89430125b41b3f9e04041e6d62d17a884e006595011432a9acf90f033077a", "5110720b0c142d86cb94f3412f9109800ab5257e06a13c4bd9ede2f5b5d88220",
)
DEVELOPMENT_ANCHOR_SHA256 = (
    "a29f29b9f135946b9e50ecfad7ff62c9739ad4dec596e8a31982289a15788893", "fd50a22dd5753757fa052b96e37451911f3b4e457f9c6de68d6cfa412f8c00a3", "66ff6f3c87f6d37da3ba93e43ffa07cf251d3fd390c84618dc153bf21332b247",
    "aee7a981e22b5e43707748dbce75a4f541c06ecf383499b68d0d3c00fb51b5db", "4d30af8e343eaaa8806bd72ebe37ca80450f4ff04798826ac71e359a6debf201", "1e23da2746048ce28f52bb57c1f5df93b738a7ea6966181476e0e4e823a12d51",
    "b6b372ad673fb01863c1544b863e562e1e9e1f0319bfd19bd03689b5137be750", "486b1bf5f667f466ec294cc7b2ba1d7e255cb269aab5b6bab528fa2b1527aa97", "2035d46d00af3ef3f2bfc7b0e08e9f7cf624fc54664ee29612a1cb24d9cd9339",
)
RAW_ANCHOR_ROWS = tuple(
    zip(
        (*FORMAL_ANCHOR_PATHS, *DEVELOPMENT_ANCHOR_PATHS),
        (*FORMAL_ANCHOR_SHA256, *DEVELOPMENT_ANCHOR_SHA256),
    )
)

_FROZEN_ROWS = (
    (("bench3/reptiles_turtle.png", "bench3/sea_shrimp.png", "anime_sheet_100_20260704/anime_sheet_007.png"), ("b328cac726d5c262433af8b2ff6f2a7dd4e3e1ea71d7eff677fb8c312195ef9c", "5d8aef88002f5da86758e9416e3bc4dc2e8689ecd1ebe247107bc776f7a74eb3", "c8b037e6ee04b3ec5284ee0c0b3da36b02ab49048127b0e7caeee11476098af1"), 1185084),
    (("bench3/birds_owl.png", "bench3/sea_squid.png", "anime_sheet_100_20260704/anime_sheet_066.png"), ("67ed0a0a7cea26e14b543fc7409915ff1f1bdb95a6c67599747c4b0965665e8f", "a6c15f044352e771158b0d6d3e8d52b5b7d280b415538e9a5a6bf9a3b55f293c", "21d6d17d764414f6623a656d51f660b3cb4c2023c5c863ce043c96cd663028ac"), 1291038),
    (("anime_sheet_100_20260704/anime_sheet_074.png", "bench3/x_peacock.png", "bench3/x_scorpion.png"), ("6db8a3a06b520f52f37a65031247acdaf586807d939ceacf595ffa0476582575", "3c5430b4b0fe3ebf0365a8b7d1e6514d0eacbe1bd43b977152d1b980b0cf3f76", "8c47927bfa5cafb56f430fea067fd2bfa10a6bac3454038e7852336febb02a8c"), 1274666),
    (("anime_sheet_100_20260704/anime_sheet_059.png", "bench3/x_bat.png", "bench3/x_lobster.png"), ("13afbd87ed3fd5cb7e46afe14849210625133f5523319d795491f700203a106f", "ea2218c626a2699e76529158027073dd19faeea6025131f5fff8797f49062af4", "4a3712d8ac0ba1ea1eed432d4b24768764b9929dea02009b983903a67aff9756"), 1251147),
    (("anime_sheet_100_20260704/anime_sheet_027.png", "anime_sheet_100_20260704/anime_sheet_019.png", "original_anime_30_20260704/anime30_18.png"), ("c551a2b46084cf018b62cad2d4ebee7284055fba9f479ad8f9087ff0cb93ce65", "13558ecbc215fff7d0dd7e3cc1eef182e20821f79b339e93163ea53f8c285377", "0ebe5a15482079c70aa4e530889d573d4438b49f7350c82abedb1ce9a4e12cbb"), 1333735),
    (("bench3/bugs_ant.png", "complex_v2_filled/complex_castle.png", "anime_sheet_100_20260704/anime_sheet_073.png"), ("6937ea284d3c3599d79a42bb92b697e7ce1255260ab1960f0fb31f357971605f", "ea381f2ef00a0180d6c173902bed46e797e26b399bbb3a16d632552568d836ea", "b5de60200438caee086d84e369861fb752bede190516ac94abb91287e68d0043"), 1485865),
    (("bench3/x_spider.png", "bench3/x_ladybug.png", "anime_sheet_100_20260704/anime_sheet_035.png"), ("aea42b3f055baaa7ddcd9bacf0aaed53bba30c893581856231fda03ae8d0eece", "dcce83f8b80b405479c29a26f4022383d191fa7f43f9ac4b1f10df4bfb2fc76e", "2a811b73b8500607a82c27890a52bb1161ab24935da0a72be5e5c7800cf3227a"), 1267519),
    (("bench3/x_trex.png", "anime_sheet_100_20260704/anime_sheet_061.png", "bench3/x_bee.png"), ("b100748f1087db2edc649b35aa96cc2771b88e92eb584d9c1999c196de9b10ea", "dd9d5704efe492f6cf3ce35066c2275b032ae9b54eaea3e000eaba3a330879bf", "341bc0d673d02be438ab38d799c4e034a932a663a3ba5ad21070fb2927287faa"), 1275076),
    (("bench3/bugs_cricket.png", "anime_sheet_100_20260704/anime_sheet_067.png", "anime_sheet_100_20260704/anime_sheet_087.png"), ("0dc59bd5f18b2125ed114b7ccedc621376b690f24387e7873d68be44dac161d9", "5327e386aeeb77197021899be46acb13cea14bfafd981aa24c74e00786a2c971", "029eb1bc8595ff65be4e2b86265b97dd73854689388d6261c648f61318897c32"), 1077860),
    (("bench3/bugs_beetle.png", "anime_sheet_100_20260704/anime_sheet_029.png", "anime_sheet_100_20260704/anime_sheet_077.png"), ("f0a14029219338ac0a8db85bae74f422d05252ab10cf8b0642dcc959fc039c0c", "69d889a732005734b11dca1fa02f784784b0f1ff6f7f405ac58f279a48070598", "de10c305767eb2752c46915d2f5f39067d41a4421ffe08aa5f5bb7c55aacdb72"), 1472795),
)


@dataclass(frozen=True)
class V7BoundInput:
    axis: str
    source_path: str
    resolved_path: str
    raw_sha256: str
    normalized_sha256: str
    foreground_count: int
    component_count: int
    foreground_ratio: float
    complexity: float
    foreground_tertile: int
    complexity_tertile: int


@dataclass(frozen=True)
class V7SuiteTask:
    case_id: str
    inputs: tuple[V7BoundInput, V7BoundInput, V7BoundInput]
    registered_target_sha256: str
    hull_xyz_sha256: str
    hull_dhw_sha256: str
    hull_occupied_count: int
    hull_component_count: int
    accepted_stream_index: int | None
    candidate_row_index: int | None


@dataclass(frozen=True)
class AuthenticatedV7Suite:
    path: str
    sha256: str
    semantic_sha256: str
    kind: str
    tasks: tuple[V7SuiteTask, ...]
    payload: Mapping[str, object]


@dataclass
class _Candidate:
    source_path: str
    resolved_path: str
    raw_sha256: str
    normalized_sha256: str
    foreground_count: int
    component_count: int
    foreground_ratio: float
    complexity: float
    mask: np.ndarray
    snapshot: AuthenticatedFileSnapshot
    foreground_tertile: int = -1
    complexity_tertile: int = -1


def _digest(value: object, *, label: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_mapping(value: object, keys: set[str], *, label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{label} has the wrong exact schema")
    return value


def _component_count(mask: np.ndarray, *, connectivity: int) -> int:
    if type(mask) is not np.ndarray or mask.dtype != np.bool_:
        raise ValueError("component mask must be an exact boolean ndarray")
    if connectivity == 8 and mask.ndim == 2:
        structure = np.ones((3, 3), dtype=np.uint8)
    elif connectivity == 6 and mask.ndim == 3:
        structure = np.zeros((3, 3, 3), dtype=np.uint8)
        structure[1, 1, :] = 1
        structure[1, :, 1] = 1
        structure[:, 1, 1] = 1
    else:
        raise ValueError("unsupported component connectivity")
    return int(ndimage.label(mask, structure=structure)[1])


def _decode_normalized(snapshot: AuthenticatedFileSnapshot) -> np.ndarray:
    encoded = np.frombuffer(snapshot.payload, dtype=np.uint8)
    gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"candidate image cannot be decoded: {snapshot.path}")
    normalized = normalize_target(gray, 256, False)
    if type(normalized) is not np.ndarray or normalized.shape != (256, 256):
        raise ValueError("production normalizer returned the wrong shape")
    return np.ascontiguousarray(normalized > 0, dtype=np.bool_)


def _candidate_from_snapshot(
    snapshot: AuthenticatedFileSnapshot, *, source_path: str
) -> _Candidate:
    mask = _decode_normalized(snapshot)
    area = int(mask.sum())
    components = 0 if area == 0 else _component_count(mask, connectivity=8)
    horizontal = int((mask[:, :-1] & mask[:, 1:]).sum())
    vertical = int((mask[:-1, :] & mask[1:, :]).sum())
    complexity = 0.0 if area == 0 else (4 * area - 2 * (horizontal + vertical)) / math.sqrt(area)
    return _Candidate(
        source_path=source_path,
        resolved_path=snapshot.path,
        raw_sha256=snapshot.sha256,
        normalized_sha256=hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
        foreground_count=area,
        component_count=components,
        foreground_ratio=area / 65536.0,
        complexity=float(complexity),
        mask=mask,
        snapshot=snapshot,
    )


def _bound_input(candidate: _Candidate, axis: str) -> V7BoundInput:
    return V7BoundInput(
        axis, candidate.source_path, candidate.resolved_path, candidate.raw_sha256,
        candidate.normalized_sha256, candidate.foreground_count,
        candidate.component_count, candidate.foreground_ratio, candidate.complexity,
        candidate.foreground_tertile, candidate.complexity_tertile,
    )


def _task_from_candidates(
    case_id: str,
    candidates: tuple[_Candidate, _Candidate, _Candidate],
    *,
    accepted_stream_index: int | None,
    candidate_row_index: int | None,
) -> V7SuiteTask:
    masks = tuple(candidate.mask for candidate in candidates)
    hull_xyz = np.ascontiguousarray(
        masks[0][None, :, :] & masks[1][:, None, :] & masks[2][:, :, None],
        dtype=np.bool_,
    )
    hull_components = 0 if not hull_xyz.any() else _component_count(hull_xyz, connectivity=6)
    hull_tensor = torch.from_numpy(hull_xyz.copy())
    hull_dhw = xyz_to_dhw(hull_tensor).numpy()
    registered = build_registered_target_stack(
        tuple((axis, torch.from_numpy(mask.copy())) for axis, mask in zip(AXES, masks))
    ).numpy()
    return V7SuiteTask(
        case_id=case_id,
        inputs=tuple(_bound_input(candidate, axis) for axis, candidate in zip(AXES, candidates)),
        registered_target_sha256=hashlib.sha256(registered.tobytes(order="C")).hexdigest(),
        hull_xyz_sha256=hashlib.sha256(hull_xyz.tobytes(order="C")).hexdigest(),
        hull_dhw_sha256=hashlib.sha256(hull_dhw.tobytes(order="C")).hexdigest(),
        hull_occupied_count=int(hull_xyz.sum()),
        hull_component_count=hull_components,
        accepted_stream_index=accepted_stream_index,
        candidate_row_index=candidate_row_index,
    )


def _validate_loaded_execution(project_root: Path) -> None:
    binder_module = importlib.import_module("experiments.bind_soft_loss_v7_suites")
    expected_modules = {
        sys.modules[__name__]: project_root / "shadowforge_pipeline/soft_voxel_v7/suite.py",
        provenance_module: project_root / "shadowforge_pipeline/soft_voxel_v7/provenance.py",
        core_module: project_root / "shadowforge_pipeline/soft_voxel_v7/core.py",
        data_module: project_root / "shadowforge_pipeline/soft_voxel_v3/data.py",
        registry_module: project_root / "experiments/soft_loss_v3_registry.py",
        registry_provenance_module: (
            project_root / "shadowforge_pipeline/soft_voxel_v3/provenance.py"
        ),
        binder_module: project_root / "experiments/bind_soft_loss_v7_suites.py",
    }
    for module, expected in expected_modules.items():
        loaded = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
        if loaded != expected:
            raise ValueError(f"loaded execution module has a foreign source path: {loaded}")
    expected_functions = (
        (normalize_target, data_module.normalize_target, expected_modules[data_module]),
        (
            build_registered_target_stack,
            core_module.build_registered_target_stack,
            expected_modules[core_module],
        ),
        (xyz_to_dhw, core_module.xyz_to_dhw, expected_modules[core_module]),
        (
            canonical_json_bytes,
            provenance_module.canonical_json_bytes,
            expected_modules[provenance_module],
        ),
        (
            registry_module.load_bound_suite_identity,
            registry_module.load_bound_suite_identity,
            expected_modules[registry_module],
        ),
        (
            registry_module.canonical_json_bytes,
            registry_provenance_module.canonical_json_bytes,
            expected_modules[registry_provenance_module],
        ),
        (
            binder_module.binding_main,
            binding_main,
            expected_modules[sys.modules[__name__]],
        ),
        (
            binder_module.write_v7_suite_bundle,
            write_v7_suite_bundle,
            expected_modules[sys.modules[__name__]],
        ),
    )
    for loaded, authoritative, expected_source in expected_functions:
        source = inspect.getsourcefile(loaded)
        if loaded is not authoritative or source is None or Path(source).resolve(strict=True) != expected_source:
            raise ValueError("loaded execution function has foreign code or source")


def _snapshot_executed_sources(
    project_root: Path,
) -> tuple[dict[str, str], tuple[AuthenticatedFileSnapshot, ...]]:
    execution_root = provenance_module.PROJECT_ROOT
    _validate_loaded_execution(execution_root)
    snapshots = tuple(
        snapshot_regular_file(
            execution_root / relative, root=execution_root, exclusive=False
        )
        for relative in EXECUTED_SOURCE_RELATIVES
    )
    evidence = {
        relative: snapshot.sha256
        for relative, snapshot in zip(EXECUTED_SOURCE_RELATIVES, snapshots, strict=True)
    }
    for snapshot in snapshots:
        revalidate_snapshot(snapshot, root=execution_root)
    return evidence, snapshots


def _anchor_rows() -> tuple[tuple[str, ...], tuple[str, ...]]:
    if type(RAW_ANCHOR_ROWS) is not tuple or len(RAW_ANCHOR_ROWS) != 39:
        raise ValueError("frozen raw anchor set must contain exactly 39 rows")
    labels: list[str] = []
    hashes: list[str] = []
    for row in RAW_ANCHOR_ROWS:
        if type(row) is not tuple or len(row) != 2 or type(row[0]) is not str:
            raise ValueError("frozen raw anchor row has the wrong exact schema")
        labels.append(row[0])
        hashes.append(_digest(row[1], label="raw anchor sha256"))
    if len(set(labels)) != 39 or len(set(hashes)) != 39:
        raise ValueError("frozen raw anchor set must contain 39 unique identities")
    return tuple(labels), tuple(hashes)


def _enumerate_candidates(
    project_root: Path, anchors: set[str]
) -> tuple[list[_Candidate], tuple[AuthenticatedFileSnapshot, ...]]:
    base = project_root / "data/silhouettes"
    candidates: list[_Candidate] = []
    snapshots: list[AuthenticatedFileSnapshot] = []
    seen_paths: set[str] = set()
    for relative_root in CANDIDATE_ROOTS:
        candidate_root = _canonical_existing_directory(project_root / relative_root, label="candidate root")
        paths = sorted(
            (
                path for path in candidate_root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                and path.name != EXCLUDED_BASENAME
            ),
            key=lambda path: path.relative_to(candidate_root).as_posix(),
        )
        for path in paths:
            source_path = path.relative_to(base).as_posix()
            if source_path in seen_paths:
                raise ValueError("duplicate formal candidate path")
            seen_paths.add(source_path)
            snapshot = snapshot_regular_file(path, root=project_root, exclusive=False)
            snapshots.append(snapshot)
            if snapshot.sha256 in anchors:
                continue
            candidate = _candidate_from_snapshot(snapshot, source_path=source_path)
            if candidate.foreground_count == 0 or candidate.component_count != 1:
                continue
            candidates.append(candidate)
    if len(candidates) != EXPECTED_ELIGIBLE_COUNT:
        raise ValueError(
            f"frozen eligible universe changed: {len(candidates)} != {EXPECTED_ELIGIBLE_COUNT}"
        )
    for metric, field in (("foreground_ratio", "foreground_tertile"), ("complexity", "complexity_tertile")):
        ordered = sorted(candidates, key=lambda item: (getattr(item, metric), item.raw_sha256, item.source_path))
        for index, candidate in enumerate(ordered):
            setattr(candidate, field, 3 * index // len(candidates))
    return candidates, tuple(snapshots)


def _candidate_record(candidate: _Candidate) -> dict[str, object]:
    return {
        "source_path": candidate.source_path,
        "resolved_path": candidate.resolved_path,
        "raw_sha256": candidate.raw_sha256,
        "normalized_sha256": candidate.normalized_sha256,
        "foreground_count": candidate.foreground_count,
        "component_count": candidate.component_count,
        "foreground_ratio": candidate.foreground_ratio,
        "complexity": candidate.complexity,
        "foreground_tertile": candidate.foreground_tertile,
        "complexity_tertile": candidate.complexity_tertile,
    }


def _selection(
    project_root: Path,
) -> tuple[
    tuple[V7SuiteTask, ...], dict[str, object], tuple[AuthenticatedFileSnapshot, ...]
]:
    project = _canonical_existing_directory(_path_argument(project_root, label="project root"), label="project root")
    executed_sources, execution_snapshots = _snapshot_executed_sources(project)
    anchor_labels, anchor_hashes = _anchor_rows()
    candidates, candidate_snapshots = _enumerate_candidates(project, set(anchor_hashes))
    permutations = tuple(itertools.permutations((0, 1, 2)))
    rows: list[tuple[_Candidate, _Candidate, _Candidate]] = []
    for foreground_bins in permutations:
        for complexity_bins in permutations:
            buckets = tuple(
                sorted(
                    (
                        candidate for candidate in candidates
                        if candidate.foreground_tertile == foreground_bins[axis]
                        and candidate.complexity_tertile == complexity_bins[axis]
                    ),
                    key=lambda item: (item.raw_sha256, item.source_path),
                )
                for axis in range(3)
            )
            rows.extend(itertools.product(*buckets))
    if len(rows) != EXPECTED_TRIPLE_COUNT:
        raise ValueError("frozen stratified candidate triple count changed")
    order = np.random.Generator(np.random.PCG64(SELECTOR_SEED)).permutation(len(rows))
    selected: list[V7SuiteTask] = []
    used: set[str] = set()
    hull_checks = 0
    for zero_based_stream, row_index_value in enumerate(order):
        row_index = int(row_index_value)
        triple = rows[row_index]
        if any(candidate.raw_sha256 in used for candidate in triple):
            continue
        hull_xyz = np.ascontiguousarray(
            triple[0].mask[None, :, :] & triple[1].mask[:, None, :] & triple[2].mask[:, :, None],
            dtype=np.bool_,
        )
        hull_checks += 1
        if not hull_xyz.any() or _component_count(hull_xyz, connectivity=6) != 1:
            continue
        task = _task_from_candidates(
            FORMAL_IDS[len(selected)], triple,
            accepted_stream_index=zero_based_stream + 1,
            candidate_row_index=row_index,
        )
        expected_paths, expected_hashes, expected_occupied = _FROZEN_ROWS[len(selected)]
        if (
            tuple(item.source_path for item in task.inputs) != expected_paths
            or tuple(item.raw_sha256 for item in task.inputs) != expected_hashes
            or task.hull_occupied_count != expected_occupied
            or task.accepted_stream_index != EXPECTED_STREAM_INDICES[len(selected)]
        ):
            raise ValueError("formal selection differs from the frozen ordered suite")
        selected.append(task)
        used.update(candidate.raw_sha256 for candidate in triple)
        if len(selected) == 10:
            break
    if len(selected) != 10 or hull_checks != EXPECTED_HULL_CHECKS:
        raise ValueError("formal selector acceptance/check count changed")
    if len(used) != 30 or used.intersection(anchor_hashes):
        raise ValueError("formal selection violates raw-SHA disjointness")
    selector_snapshot = next(
        snapshot for relative, snapshot in zip(
            EXECUTED_SOURCE_RELATIVES, execution_snapshots, strict=True
        ) if relative == "shadowforge_pipeline/soft_voxel_v7/suite.py"
    )
    ordered_universe = sorted(
        (_candidate_record(candidate) for candidate in candidates),
        key=lambda row: (row["raw_sha256"], row["source_path"]),
    )
    evidence: dict[str, object] = {
        "selector": SELECTOR_NAME,
        "selector_seed": SELECTOR_SEED,
        "candidate_roots": list(CANDIDATE_ROOTS),
        "image_suffixes": sorted(IMAGE_SUFFIXES),
        "excluded_basename": EXCLUDED_BASENAME,
        "normalization": "normalize_target(gray,256,False)>0",
        "connectivity_2d": 8,
        "connectivity_3d": 6,
        "eligible_count": len(candidates),
        "eligible_universe_sha256": canonical_sha256(ordered_universe),
        "candidate_triple_count": len(rows),
        "hull_checks": hull_checks,
        "accepted_stream_indices": list(EXPECTED_STREAM_INDICES),
        "anchor_paths": list(anchor_labels),
        "anchor_sha256": list(anchor_hashes),
        "executed_sources": executed_sources,
        "selector_source": {"path": selector_snapshot.path, "sha256": selector_snapshot.sha256},
    }
    dependencies = (*execution_snapshots, *candidate_snapshots)
    for snapshot in dependencies:
        path = Path(snapshot.path)
        revalidate_snapshot(
            snapshot,
            root=(provenance_module.PROJECT_ROOT if path.is_relative_to(provenance_module.PROJECT_ROOT) else project),
        )
    return tuple(selected), evidence, dependencies


def _select_v7_formal_tasks_with_evidence(
    project_root: Path,
) -> tuple[tuple[V7SuiteTask, ...], dict[str, object]]:
    tasks, evidence, _ = _selection(project_root)
    return tasks, evidence


def select_v7_formal_tasks(project_root: Path) -> tuple[V7SuiteTask, ...]:
    tasks, _, _ = _selection(project_root)
    return tasks


def _normalization_identity(
    project_root: Path,
) -> tuple[dict[str, object], AuthenticatedFileSnapshot]:
    source = snapshot_regular_file(
        project_root / "shadowforge_pipeline/soft_voxel_v3/data.py",
        root=project_root, exclusive=False,
    )
    return {
        "producer": "shadowforge_pipeline.soft_voxel_v3.data.normalize_target",
        "source_path": source.path,
        "source_sha256": source.sha256,
        "size": 256,
        "preserve_aspect": False,
        "foreground_rule": "normalized > 0",
        "mask_bytes": "numpy.bool C-order",
    }, source


def _registration_identity() -> dict[str, object]:
    return {
        "task1_sources": dict(TASK1_SOURCE_SHA256),
        "registered_target_stack": "[D4_3(Bz),Bx,D4_7(By),D4_7(Bz),D4_4(Bx),D4_3(By)]",
        "xyz_to_dhw": "transpose(2,1,0),flip(0,1,2)",
    }


def _development_tasks(
    source_path: Path, project_root: Path
) -> tuple[
    tuple[V7SuiteTask, ...], dict[str, object], tuple[AuthenticatedFileSnapshot, ...]
]:
    source_argument = _path_argument(source_path, label="development source")
    if source_argument != APPROVED_DEVELOPMENT_SUITE_PATH:
        raise ValueError("development source path is not the uniquely approved suite")
    source = registry_module.load_bound_suite_identity(source_argument, "development")
    source_snapshot = snapshot_regular_file(
        Path(source.path), root=source_argument.parent, exclusive=False
    )
    if (
        source_snapshot.sha256 != source.sha256
        or source_snapshot.sha256 != APPROVED_DEVELOPMENT_SUITE_SHA256
        or source.semantic_sha256 != APPROVED_DEVELOPMENT_SUITE_SEMANTIC_SHA256
    ):
        raise ValueError("development source suite identity is not approved")
    if tuple(task.case_id for task in source.tasks) != DEVELOPMENT_IDS:
        raise ValueError("development source IDs/order changed")
    all_snapshots: list[AuthenticatedFileSnapshot] = [source_snapshot]
    seen_hashes: set[str] = set()
    tasks: list[V7SuiteTask] = []
    for source_task in source.tasks:
        if tuple(item.axis for item in source_task.inputs) != AXES:
            raise ValueError("development inputs must be exact ordered x/y/z")
        candidates: list[_Candidate] = []
        for source_input in source_task.inputs:
            resolved = Path(source_input.resolved_path)
            snapshot = snapshot_regular_file(resolved, exclusive=False)
            if snapshot.sha256 != source_input.sha256 or snapshot.path != str(resolved):
                raise ValueError("development input identity changed")
            if snapshot.sha256 in seen_hashes:
                raise ValueError("development input SHA values must be globally unique")
            seen_hashes.add(snapshot.sha256)
            candidate = _candidate_from_snapshot(snapshot, source_path=str(resolved))
            if candidate.foreground_count == 0:
                raise ValueError("development mask must be nonempty")
            candidates.append(candidate)
            all_snapshots.append(snapshot)
        task = _task_from_candidates(
            source_task.case_id, (candidates[0], candidates[1], candidates[2]),
            accepted_stream_index=None, candidate_row_index=None,
        )
        tasks.append(task)
    source_record = {
        "path": source_snapshot.path,
        "sha256": source_snapshot.sha256,
        "semantic_sha256": source.semantic_sha256,
        "schema": "pcraft_soft_loss_v3_suite_v1",
    }
    authoritative_rows = tuple(
        (str(item.resolved_path), item.sha256)
        for task in source.tasks for item in task.inputs
    )
    if authoritative_rows != tuple(zip(DEVELOPMENT_ANCHOR_PATHS, DEVELOPMENT_ANCHOR_SHA256, strict=True)):
        raise ValueError("development source raw path/SHA identities are not authoritative")
    for snapshot in all_snapshots:
        revalidate_snapshot(snapshot)
    return tuple(tasks), source_record, tuple(all_snapshots)


def _input_payload(item: V7BoundInput) -> dict[str, object]:
    return {
        "axis": item.axis,
        "source_path": item.source_path,
        "resolved_path": item.resolved_path,
        "raw_sha256": item.raw_sha256,
        "normalized_sha256": item.normalized_sha256,
        "foreground_count": item.foreground_count,
        "component_count": item.component_count,
        "foreground_ratio": item.foreground_ratio,
        "complexity": item.complexity,
        "foreground_tertile": item.foreground_tertile,
        "complexity_tertile": item.complexity_tertile,
    }


def _task_payload(task: V7SuiteTask) -> dict[str, object]:
    return {
        "case_id": task.case_id,
        "inputs": [_input_payload(item) for item in task.inputs],
        "registered_target_sha256": task.registered_target_sha256,
        "hull_xyz_sha256": task.hull_xyz_sha256,
        "hull_dhw_sha256": task.hull_dhw_sha256,
        "hull_occupied_count": task.hull_occupied_count,
        "hull_component_count": task.hull_component_count,
        "accepted_stream_index": task.accepted_stream_index,
        "candidate_row_index": task.candidate_row_index,
    }


def _root_identity(root: AuthenticatedV7Root) -> dict[str, object]:
    return {
        "path": root.path,
        "marker_path": root.marker_path,
        "marker_sha256": root.marker_sha256,
        "semantic_sha256": root.semantic_sha256,
    }


def _suite_payload(
    kind: str,
    root: AuthenticatedV7Root,
    project_root: Path,
    tasks: tuple[V7SuiteTask, ...],
    normalization: dict[str, object],
    *,
    source_suite: dict[str, object] | None = None,
    selector: dict[str, object] | None = None,
    development_suite: dict[str, object] | None = None,
) -> dict[str, object]:
    if kind not in ("development", "formal"):
        raise ValueError("suite payload kind/dependency mismatch")
    if kind == "development":
        if source_suite is None or selector is not None or development_suite is not None:
            raise ValueError("suite payload kind/dependency mismatch")
    elif selector is None or development_suite is None or source_suite is not None:
        raise ValueError("suite payload kind/dependency mismatch")
    payload: dict[str, object] = {
        "schema": SUITE_SCHEMA,
        "status": "REGISTERED",
        "kind": kind,
        "root_identity": _root_identity(root),
        "project_root": str(project_root),
        "task1_registration_identity": _registration_identity(),
        "normalization": normalization,
        "tasks": [_task_payload(task) for task in tasks],
        "task_count": len(tasks),
        "input_count": 3 * len(tasks),
        "input_order": list(AXES),
    }
    if kind == "development":
        payload["source_suite"] = source_suite
    else:
        payload["selector"] = selector
        payload["development_suite"] = development_suite
    payload["semantic_sha256"] = canonical_sha256(payload)
    return payload


def _identity_from_payload(path: Path, payload: dict[str, object], tasks: tuple[V7SuiteTask, ...]) -> AuthenticatedV7Suite:
    raw = canonical_json_bytes(payload)
    return AuthenticatedV7Suite(
        str(path), hashlib.sha256(raw).hexdigest(), str(payload["semantic_sha256"]),
        str(payload["kind"]), tasks, _deep_read_only(payload),
    )


def _revalidate_all(
    snapshots: Sequence[AuthenticatedFileSnapshot], project_root: Path
) -> None:
    for snapshot in snapshots:
        path = Path(snapshot.path)
        if path.is_relative_to(provenance_module.PROJECT_ROOT):
            containment: Path | None = provenance_module.PROJECT_ROOT
        elif path.is_relative_to(project_root):
            containment = project_root
        else:
            containment = None
        revalidate_snapshot(snapshot, root=containment)


def _validate_manifest_namespace(manifests: Path) -> None:
    directory = _canonical_existing_directory(manifests, label="manifests directory")
    names = {entry.name for entry in os.scandir(directory)}
    if not names.issubset(REGISTERED_MANIFEST_NAMES):
        raise ValueError("v7 manifests contains an unexpected or unregistered namespace entry")
    if "formal_suite.json" in names and "development_suite.json" not in names:
        raise ValueError("formal commit marker precedes development suite")


def _validate_manifest_namespace_bound(root: object) -> None:
    descriptor = provenance_module._open_relative_directory_fd(
        getattr(root, "descriptor"), Path("manifests"), label="manifests directory"
    )
    try:
        names = set(os.listdir(descriptor))
    finally:
        os.close(descriptor)
    if not names.issubset(REGISTERED_MANIFEST_NAMES):
        raise ValueError("v7 manifests contains an unexpected or unregistered namespace entry")
    if "formal_suite.json" in names and "development_suite.json" not in names:
        raise ValueError("formal commit marker precedes development suite")


def _ensure_manifests_directory_bound(root: object) -> None:
    descriptor = getattr(root, "descriptor")
    try:
        metadata = os.stat("manifests", dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        os.mkdir("manifests", mode=0o755, dir_fd=descriptor)
        os.fsync(descriptor)
    else:
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise ValueError("manifests entry must be an ordinary directory")
    provenance_module._verify_bound_directory(root, label="v7 root")


def _development_manifest_identity(suite: AuthenticatedV7Suite) -> dict[str, object]:
    return {
        "path": suite.path,
        "sha256": suite.sha256,
        "semantic_sha256": suite.semantic_sha256,
    }


def write_v7_suite_bundle(
    output_root: Path, *, development_source: Path, project_root: Path
) -> tuple[AuthenticatedV7Suite, AuthenticatedV7Suite]:
    """Publish root, development snapshot, then formal suite commit marker."""

    project = _canonical_existing_directory(_path_argument(project_root, label="project root"), label="project root")
    if project != provenance_module.PROJECT_ROOT:
        raise ValueError("published v7 suites must bind the current canonical project root")
    source_argument = _path_argument(development_source, label="development source")
    development_tasks, source_record, development_dependencies = _development_tasks(source_argument, project)
    formal_tasks, selector, formal_dependencies = _selection(project)
    normalization, normalization_snapshot = _normalization_identity(project)
    dependencies = (*development_dependencies, *formal_dependencies, normalization_snapshot)
    _revalidate_all(dependencies, project)
    root = bootstrap_v7_root(_path_argument(output_root, label="output root"))
    root_path = Path(root.path)
    root_bound = provenance_module._bind_directory(root_path, label="v7 root")
    descriptor = root_bound.descriptor
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        root = provenance_module._validate_v7_root_bound(root_bound)
        manifests = root_path / "manifests"
        _ensure_manifests_directory_bound(root_bound)
        _validate_manifest_namespace_bound(root_bound)
        development_payload = _suite_payload(
            "development", root, project, development_tasks, normalization,
            source_suite=source_record,
        )
        development_path = manifests / "development_suite.json"
        formal_path = manifests / "formal_suite.json"
        _revalidate_all(dependencies, project)
        provenance_module._publish_bytes_once_bound(
            development_path, canonical_json_bytes(development_payload), root=root_bound
        )
        _PUBLICATION_HOOK("after_development_publish")
        provenance_module._verify_bound_directory(root_bound, label="v7 root")
        _revalidate_all(dependencies, project)
        provenance_module._validate_v7_root_bound(root_bound)
        _validate_manifest_namespace_bound(root_bound)
        authenticated_development = _validate_v7_suite_bound(
            development_path, expected_kind="development", root_bound=root_bound
        )
        formal_payload = _suite_payload(
            "formal", root, project, formal_tasks, normalization,
            selector=selector,
            development_suite=_development_manifest_identity(authenticated_development),
        )
        _revalidate_all(dependencies, project)
        authenticated_development = _validate_v7_suite_bound(
            development_path, expected_kind="development", root_bound=root_bound
        )
        if formal_payload["development_suite"] != _development_manifest_identity(
            authenticated_development
        ):
            raise ValueError("development suite prefix changed before formal publication")
        provenance_module._publish_bytes_once_bound(
            formal_path, canonical_json_bytes(formal_payload), root=root_bound
        )
        _PUBLICATION_HOOK("after_formal_publish")
        provenance_module._verify_bound_directory(root_bound, label="v7 root")
        _revalidate_all(dependencies, project)
        provenance_module._validate_v7_root_bound(root_bound)
        development_snapshot = provenance_module._snapshot_regular_file_bound(
            development_path, root=root_bound, exclusive=True
        )
        formal_snapshot = provenance_module._snapshot_regular_file_bound(
            formal_path, root=root_bound, exclusive=True
        )
        if development_snapshot.payload != canonical_json_bytes(development_payload):
            raise FileExistsError("development suite immutable bytes conflict")
        if formal_snapshot.payload != canonical_json_bytes(formal_payload):
            raise FileExistsError("formal suite immutable bytes conflict")
        return (
            _identity_from_payload(development_path, development_payload, development_tasks),
            _identity_from_payload(formal_path, formal_payload, formal_tasks),
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(root_bound.descriptor)


def _input_from_payload(value: object) -> V7BoundInput:
    row = _exact_mapping(value, {
        "axis", "source_path", "resolved_path", "raw_sha256", "normalized_sha256",
        "foreground_count", "component_count", "foreground_ratio", "complexity",
        "foreground_tertile", "complexity_tertile",
    }, label="suite input")
    if type(row["axis"]) is not str or type(row["source_path"]) is not str or type(row["resolved_path"]) is not str:
        raise ValueError("suite input paths/axis have wrong exact types")
    for key in ("raw_sha256", "normalized_sha256"):
        _digest(row[key], label=key)
    for key in ("foreground_count", "component_count", "foreground_tertile", "complexity_tertile"):
        if type(row[key]) is not int:
            raise ValueError(f"{key} must be an exact integer")
    for key in ("foreground_ratio", "complexity"):
        if type(row[key]) is not float or not math.isfinite(row[key]):
            raise ValueError(f"{key} must be an exact finite float")
    return V7BoundInput(**row)


def _task_from_payload(value: object) -> V7SuiteTask:
    row = _exact_mapping(value, {
        "case_id", "inputs", "registered_target_sha256", "hull_xyz_sha256",
        "hull_dhw_sha256", "hull_occupied_count", "hull_component_count",
        "accepted_stream_index", "candidate_row_index",
    }, label="suite task")
    if type(row["case_id"]) is not str or type(row["inputs"]) is not list or len(row["inputs"]) != 3:
        raise ValueError("suite task ID/inputs have wrong exact types")
    inputs = tuple(_input_from_payload(item) for item in row["inputs"])
    if tuple(item.axis for item in inputs) != AXES:
        raise ValueError("suite inputs are not exact ordered x/y/z")
    for key in ("registered_target_sha256", "hull_xyz_sha256", "hull_dhw_sha256"):
        _digest(row[key], label=key)
    for key in ("hull_occupied_count", "hull_component_count"):
        if type(row[key]) is not int:
            raise ValueError(f"{key} must be an exact integer")
    for key in ("accepted_stream_index", "candidate_row_index"):
        if row[key] is not None and type(row[key]) is not int:
            raise ValueError(f"{key} must be an exact integer or null")
    return V7SuiteTask(
        row["case_id"], inputs, row["registered_target_sha256"], row["hull_xyz_sha256"],
        row["hull_dhw_sha256"], row["hull_occupied_count"], row["hull_component_count"],
        row["accepted_stream_index"], row["candidate_row_index"],
    )


def _validate_v7_suite_bound(
    path: Path, *, expected_kind: str, root_bound: object,
) -> AuthenticatedV7Suite:
    if type(expected_kind) is not str or expected_kind not in ("development", "formal"):
        raise ValueError("expected_kind must be development or formal")
    absolute = _path_argument(path, label="suite path")
    root_path = getattr(root_bound, "path")
    if absolute != root_path / "manifests" / f"{expected_kind}_suite.json":
        raise ValueError("suite path is not the canonical same-root child")
    root = provenance_module._validate_v7_root_bound(root_bound)
    _validate_manifest_namespace_bound(root_bound)
    snapshot = provenance_module._snapshot_regular_file_bound(
        absolute, root=root_bound, exclusive=True
    )
    payload = decode_canonical_json(snapshot.payload, label=f"v7 {expected_kind} suite")
    if snapshot.payload != canonical_json_bytes(payload):
        raise ValueError("v7 suite is not canonical JSON")
    common = {
        "schema", "status", "kind", "root_identity", "project_root",
        "task1_registration_identity", "normalization", "tasks", "task_count",
        "input_count", "input_order", "semantic_sha256",
    }
    dependency_keys = (
        {"source_suite"}
        if expected_kind == "development"
        else {"selector", "development_suite"}
    )
    _exact_mapping(payload, common | dependency_keys, label="v7 suite")
    if payload["schema"] != SUITE_SCHEMA or payload["status"] != "REGISTERED" or payload["kind"] != expected_kind:
        raise ValueError("v7 suite schema/status/kind mismatch")
    semantic = _digest(payload["semantic_sha256"], label="suite semantic_sha256")
    unsigned = dict(payload)
    del unsigned["semantic_sha256"]
    if canonical_sha256(unsigned) != semantic:
        raise ValueError("v7 suite semantic SHA mismatch")
    if payload["root_identity"] != _root_identity(root):
        raise ValueError("v7 suite has a foreign root identity")
    project = _canonical_existing_directory(Path(root.payload["project_root"]), label="project root")
    if payload["project_root"] != str(project):
        raise ValueError("v7 suite project root mismatch")
    normalization, normalization_snapshot = _normalization_identity(project)
    if payload["normalization"] != normalization or payload["task1_registration_identity"] != _registration_identity():
        raise ValueError("v7 suite normalization/Task1 identity changed")
    expected_ids = DEVELOPMENT_IDS if expected_kind == "development" else FORMAL_IDS
    if type(payload["tasks"]) is not list or type(payload["task_count"]) is not int or type(payload["input_count"]) is not int:
        raise ValueError("v7 suite counts/tasks have wrong exact types")
    tasks = tuple(_task_from_payload(item) for item in payload["tasks"])
    if tuple(task.case_id for task in tasks) != expected_ids or payload["task_count"] != len(expected_ids) or payload["input_count"] != 3 * len(expected_ids):
        raise ValueError("v7 suite task order/count changed")
    if payload["input_order"] != list(AXES):
        raise ValueError("v7 suite input order changed")
    raw_hashes = [item.raw_sha256 for task in tasks for item in task.inputs]
    if len(raw_hashes) != len(set(raw_hashes)):
        raise ValueError("v7 suite contains duplicate input SHA values")
    if expected_kind == "development":
        source_record = _exact_mapping(payload["source_suite"], {"path", "sha256", "semantic_sha256", "schema"}, label="source suite")
        if type(source_record["path"]) is not str or type(source_record["schema"]) is not str:
            raise ValueError("source suite path/schema have wrong exact types")
        _digest(source_record["sha256"], label="source suite sha256")
        _digest(source_record["semantic_sha256"], label="source suite semantic_sha256")
        source_path = Path(source_record["path"])
        expected_tasks, expected_source, dependencies = _development_tasks(source_path, project)
        expected = _suite_payload(
            "development", root, project, expected_tasks, normalization,
            source_suite=expected_source,
        )
    else:
        development_record = _exact_mapping(
            payload["development_suite"],
            {"path", "sha256", "semantic_sha256"},
            label="development suite prefix",
        )
        if type(development_record["path"]) is not str:
            raise ValueError("development suite prefix path has wrong exact type")
        _digest(development_record["sha256"], label="development suite prefix sha256")
        _digest(
            development_record["semantic_sha256"],
            label="development suite prefix semantic_sha256",
        )
        development_path = root_path / "manifests" / "development_suite.json"
        if development_record["path"] != str(development_path):
            raise ValueError("formal suite binds a foreign development prefix")
        authenticated_development = _validate_v7_suite_bound(
            development_path, expected_kind="development", root_bound=root_bound
        )
        if development_record != _development_manifest_identity(authenticated_development):
            raise ValueError("formal suite development prefix identity changed")
        expected_tasks, selector, dependencies = _selection(project)
        development_hashes = {
            item.raw_sha256
            for task in authenticated_development.tasks for item in task.inputs
        }
        development_paths = {
            item.resolved_path
            for task in authenticated_development.tasks for item in task.inputs
        }
        formal_hashes = {
            item.raw_sha256 for task in expected_tasks for item in task.inputs
        }
        formal_paths = {
            item.resolved_path for task in expected_tasks for item in task.inputs
        }
        if development_hashes.intersection(formal_hashes) or development_paths.intersection(formal_paths):
            raise ValueError("development/formal suites overlap by path or raw SHA")
        expected = _suite_payload(
            "formal", root, project, expected_tasks, normalization,
            selector=selector,
            development_suite=_development_manifest_identity(authenticated_development),
        )
    if payload != expected:
        raise ValueError(f"v7 {expected_kind} suite differs from independent recomputation")
    revalidate_snapshot(normalization_snapshot, root=project)
    _revalidate_all(dependencies, project)
    current_snapshot = provenance_module._snapshot_regular_file_bound(
        absolute, root=root_bound, exclusive=True
    )
    if current_snapshot != snapshot:
        raise ValueError("v7 suite identity or bytes changed")
    if expected_kind == "formal":
        current_development = _validate_v7_suite_bound(
            development_path, expected_kind="development", root_bound=root_bound
        )
        if current_development != authenticated_development:
            raise ValueError("development suite prefix identity changed after selector replay")
    else:
        provenance_module._validate_v7_root_bound(root_bound)
    provenance_module._verify_bound_directory(root_bound, label="v7 root")
    return AuthenticatedV7Suite(
        snapshot.path, snapshot.sha256, semantic, expected_kind, tasks,
        _deep_read_only(payload),
    )


def validate_v7_suite(path: Path, *, expected_kind: str) -> AuthenticatedV7Suite:
    absolute = _path_argument(path, label="suite path")
    root_path = absolute.parent.parent
    root_bound = provenance_module._bind_directory(root_path, label="v7 root")
    try:
        return _validate_v7_suite_bound(
            absolute, expected_kind=expected_kind, root_bound=root_bound
        )
    finally:
        os.close(root_bound.descriptor)


def build_binding_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bind immutable development and formal soft-loss-v7 suites",
        allow_abbrev=False,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser


def binding_main(argv: Sequence[str] | None = None) -> int:
    args = build_binding_parser().parse_args(argv)
    write_v7_suite_bundle(
        args.output_root,
        development_source=args.development_source,
        project_root=args.project_root,
    )
    return 0


__all__ = [
    "AuthenticatedV7Suite",
    "V7BoundInput",
    "V7SuiteTask",
    "binding_main",
    "build_binding_parser",
    "select_v7_formal_tasks",
    "validate_v7_suite",
    "write_v7_suite_bundle",
]
