"""Exact development/formal suite selection and publication contracts."""

from __future__ import annotations

import ast
from collections.abc import Iterator
import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

import experiments.soft_loss_v3_registry as v3_registry
import shadowforge_pipeline.soft_voxel_v7.suite as suite_module
from shadowforge_pipeline.soft_voxel_v7.provenance import canonical_json_bytes, canonical_sha256
from shadowforge_pipeline.soft_voxel_v7.suite import (
    AuthenticatedV7Suite,
    V7BoundInput,
    V7SuiteTask,
    select_v7_formal_tasks,
    validate_v7_suite,
    write_v7_suite_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = (
    (("bench3/reptiles_turtle.png", "bench3/sea_shrimp.png", "anime_sheet_100_20260704/anime_sheet_007.png"), ("b328cac726d5c262433af8b2ff6f2a7dd4e3e1ea71d7eff677fb8c312195ef9c", "5d8aef88002f5da86758e9416e3bc4dc2e8689ecd1ebe247107bc776f7a74eb3", "c8b037e6ee04b3ec5284ee0c0b3da36b02ab49048127b0e7caeee11476098af1"), 1185084, "a8b67e47ce4e66475a86d456f68347014c0ce4524ea2501418c8f35f1aa9bf25"),
    (("bench3/birds_owl.png", "bench3/sea_squid.png", "anime_sheet_100_20260704/anime_sheet_066.png"), ("67ed0a0a7cea26e14b543fc7409915ff1f1bdb95a6c67599747c4b0965665e8f", "a6c15f044352e771158b0d6d3e8d52b5b7d280b415538e9a5a6bf9a3b55f293c", "21d6d17d764414f6623a656d51f660b3cb4c2023c5c863ce043c96cd663028ac"), 1291038, "cc34b02e447bace02e0d7811ab94653ed472738f14c2b4312917115e73d7161d"),
    (("anime_sheet_100_20260704/anime_sheet_074.png", "bench3/x_peacock.png", "bench3/x_scorpion.png"), ("6db8a3a06b520f52f37a65031247acdaf586807d939ceacf595ffa0476582575", "3c5430b4b0fe3ebf0365a8b7d1e6514d0eacbe1bd43b977152d1b980b0cf3f76", "8c47927bfa5cafb56f430fea067fd2bfa10a6bac3454038e7852336febb02a8c"), 1274666, "ec810e898e4be308dc15963e64e9ef9375bac70ff4c2a4ccb760e46c8a51e9b0"),
    (("anime_sheet_100_20260704/anime_sheet_059.png", "bench3/x_bat.png", "bench3/x_lobster.png"), ("13afbd87ed3fd5cb7e46afe14849210625133f5523319d795491f700203a106f", "ea2218c626a2699e76529158027073dd19faeea6025131f5fff8797f49062af4", "4a3712d8ac0ba1ea1eed432d4b24768764b9929dea02009b983903a67aff9756"), 1251147, "7efb9bd35b69e767b078006173e7a9da491230b220965cac0b65c0947d28dd46"),
    (("anime_sheet_100_20260704/anime_sheet_027.png", "anime_sheet_100_20260704/anime_sheet_019.png", "original_anime_30_20260704/anime30_18.png"), ("c551a2b46084cf018b62cad2d4ebee7284055fba9f479ad8f9087ff0cb93ce65", "13558ecbc215fff7d0dd7e3cc1eef182e20821f79b339e93163ea53f8c285377", "0ebe5a15482079c70aa4e530889d573d4438b49f7350c82abedb1ce9a4e12cbb"), 1333735, "a1d19563f88baca2d515015ff807ff1c44f14209bfdea7ef34a907577b1a5ce9"),
    (("bench3/bugs_ant.png", "complex_v2_filled/complex_castle.png", "anime_sheet_100_20260704/anime_sheet_073.png"), ("6937ea284d3c3599d79a42bb92b697e7ce1255260ab1960f0fb31f357971605f", "ea381f2ef00a0180d6c173902bed46e797e26b399bbb3a16d632552568d836ea", "b5de60200438caee086d84e369861fb752bede190516ac94abb91287e68d0043"), 1485865, "92fcb9bc54a83aaedffae4696b70614273e3e2c001784d4b5afb0e62092cd34c"),
    (("bench3/x_spider.png", "bench3/x_ladybug.png", "anime_sheet_100_20260704/anime_sheet_035.png"), ("aea42b3f055baaa7ddcd9bacf0aaed53bba30c893581856231fda03ae8d0eece", "dcce83f8b80b405479c29a26f4022383d191fa7f43f9ac4b1f10df4bfb2fc76e", "2a811b73b8500607a82c27890a52bb1161ab24935da0a72be5e5c7800cf3227a"), 1267519, "32ce453a79fe8c031e6dde5800aaff38d134ca09e2c99d3f969f93301b2dfe3b"),
    (("bench3/x_trex.png", "anime_sheet_100_20260704/anime_sheet_061.png", "bench3/x_bee.png"), ("b100748f1087db2edc649b35aa96cc2771b88e92eb584d9c1999c196de9b10ea", "dd9d5704efe492f6cf3ce35066c2275b032ae9b54eaea3e000eaba3a330879bf", "341bc0d673d02be438ab38d799c4e034a932a663a3ba5ad21070fb2927287faa"), 1275076, "75a8416f130afd070b5b9520f6fa4ea538fc8baec48b1b8a9cce7c9bd435442d"),
    (("bench3/bugs_cricket.png", "anime_sheet_100_20260704/anime_sheet_067.png", "anime_sheet_100_20260704/anime_sheet_087.png"), ("0dc59bd5f18b2125ed114b7ccedc621376b690f24387e7873d68be44dac161d9", "5327e386aeeb77197021899be46acb13cea14bfafd981aa24c74e00786a2c971", "029eb1bc8595ff65be4e2b86265b97dd73854689388d6261c648f61318897c32"), 1077860, "582ab0dc0d50d6ec214252587976f54d31889be15f9bc7d93b40fe265b5e1a86"),
    (("bench3/bugs_beetle.png", "anime_sheet_100_20260704/anime_sheet_029.png", "anime_sheet_100_20260704/anime_sheet_077.png"), ("f0a14029219338ac0a8db85bae74f422d05252ab10cf8b0642dcc959fc039c0c", "69d889a732005734b11dca1fa02f784784b0f1ff6f7f405ac58f279a48070598", "de10c305767eb2752c46915d2f5f39067d41a4421ffe08aa5f5bb7c55aacdb72"), 1472795, "995f649dd4c49dcc5bd649f15bbe83fe98037bd58f7a4cf8e98770f4d40d2862"),
)
STREAM_INDICES = (8, 37, 45, 108, 113, 121, 161, 170, 239, 269)
FORMAL_ANCHOR_PATHS = tuple(
    f"results/fixed3_mr1_connected10_pending_20260722/inputs/connected_{task:02d}_view_{axis}.png"
    for task in range(1, 11) for axis in ("x", "y", "z")
)
DEVELOPMENT_ANCHOR_PATHS = tuple(
    str(PROJECT_ROOT / f"results/table2_anime_fixed3_inputs_20260722/{task}/view_{view}.png")
    for task in ("arcane_casters", "blade_heroes", "stage_dancers") for view in range(3)
)
FORMAL_ANCHOR_SHA256 = (
    "075ea08eb090ccd5fc2350b4e7393524cb7c154da5fd17e4656b1fb8d367146c",
    "ed684af3ebc6c054bfe272d9b9be2e6df0617231b9691873572517b7f2afa547",
    "5de721e47808bfd7785eb94e40f804f9659718282a135b2b27cf6d99d6ab6f1b",
    "80f21ad6dea0788f5db9ac2a178bc36358341a8198f786aa72c8406d7a4820f6",
    "054101ae816c840ccc7ac4a1c76e944c5ebef85ee1a00c01f8539c8e237f8e26",
    "8eb27cf3e4b8036bb757f737b81c44cc93a930cbb089d72cc8053528e70f3a29",
    "df5a512e697e539286c8a38b4c9dadf1b8f80554f41d026b0b6ff2f2d06dbaa0",
    "6546aac52ae974a5cab2d38fe0dcf689795b2d4013eb940c9df5cb39c46b854f",
    "078b717e50981d0f4992d7d7838e734653c780bc915dc1a5a56c49296998e2ef",
    "8b569f92474e94fd68c2f4314603aee607c329a2e31694bdaddcf304d70dd687",
    "fee6d5ca6056b6307fd65e09903da2db16b1b16718ff0fb03fbefb182f990e01",
    "54a2b64b26404e228acbcdbe9bf52b704e9aaf3340e5dbe7074972ae87c9310b",
    "3149c1f78115e1574cfa8ab61344a3402f76012839d5193727f6ae24eead7659",
    "ad2186ee8be6067167b880df207dd2ad5d3eb9ed6639f8a85732890820e7ca2b",
    "bb45fedb9c4216ff7761b236db822dfb787252290920e165d6ea12aae8194bff",
    "d5e8e58ae1a0a6172735a2906e04af45805ad440a6e984178961f4db5ff6c656",
    "ed64299b700dc0d2547f02b0d0b15b94a4751d61a07d2a4796931a445f0e3fd0",
    "8542e7cdb126ff952343eafd585b1893877299cc589265df57c425714bb35ef4",
    "ee13ec568621c9cfb1ad65d6c8469ae137e9353779ca5d14a1c7c60d286f8562",
    "37887af0fea53b2fa5fc03b886250549a525cb2028c5d60045fa918e58d191bf",
    "cf72b7d013419e2f68b9cecb2aedd78f0218e5b62f65978d1f908e9205712b17",
    "ddcc150eb1410f7df82db8d360c55a0adbec71b1efd395abbddc6c3cdf874236",
    "f5fe8520d5cd72122ab63b1f249c8153a35992abf475af4dbf71638edad3d7d2",
    "3309b1e5c3df3bd77fd2c894da8456753d6636b679d75937c2666c6f4b454586",
    "358682ddb647c0d23b836c64a0a3261831309e2b8c2150c46a631f36e5563dd1",
    "91e134653b2877cfb251bd3a235c64e33c7ef592c7f6f97840ce85aed73e4a4b",
    "6549faf1c6231e403eee2334577464c7603cd4dbe31e6cd2117f26a022844319",
    "8f9072dab70d1226e49bbdae5757de9070a242c891499966b686fc05e9d2c1f3",
    "64c89430125b41b3f9e04041e6d62d17a884e006595011432a9acf90f033077a",
    "5110720b0c142d86cb94f3412f9109800ab5257e06a13c4bd9ede2f5b5d88220",
)
DEVELOPMENT_ANCHOR_SHA256 = (
    "a29f29b9f135946b9e50ecfad7ff62c9739ad4dec596e8a31982289a15788893",
    "fd50a22dd5753757fa052b96e37451911f3b4e457f9c6de68d6cfa412f8c00a3",
    "66ff6f3c87f6d37da3ba93e43ffa07cf251d3fd390c84618dc153bf21332b247",
    "aee7a981e22b5e43707748dbce75a4f541c06ecf383499b68d0d3c00fb51b5db",
    "4d30af8e343eaaa8806bd72ebe37ca80450f4ff04798826ac71e359a6debf201",
    "1e23da2746048ce28f52bb57c1f5df93b738a7ea6966181476e0e4e823a12d51",
    "b6b372ad673fb01863c1544b863e562e1e9e1f0319bfd19bd03689b5137be750",
    "486b1bf5f667f466ec294cc7b2ba1d7e255cb269aab5b6bab528fa2b1527aa97",
    "2035d46d00af3ef3f2bfc7b0e08e9f7cf624fc54664ee29612a1cb24d9cd9339",
)
APPROVED_DEVELOPMENT_SUITE_PATH = Path(
    "/data/xvxinyao/xvxinyao/xvxinyao1/xvxinyao/XUXINYAO/shadow-project2/mesh/voxel/"
    "runs/soft_loss_v3_20260826/manifests/development_suite.json"
)
APPROVED_DEVELOPMENT_SUITE_SHA256 = "15da2ecc37424bb1513f922f5aac6b4217b547744e53099ecaaafc8243c08fe0"
APPROVED_DEVELOPMENT_SUITE_SEMANTIC_SHA256 = "43c44793492084b0eb4b61d4e582bff63819af6a6d32ffc3bef072cfb987abc4"
EXECUTED_SOURCE_RELATIVES = {
    "shadowforge_pipeline/soft_voxel_v7/suite.py",
    "shadowforge_pipeline/soft_voxel_v7/provenance.py",
    "shadowforge_pipeline/soft_voxel_v7/core.py",
    "shadowforge_pipeline/soft_voxel_v3/data.py",
    "shadowforge_pipeline/soft_voxel_v3/provenance.py",
    "experiments/soft_loss_v3_registry.py",
    "experiments/bind_soft_loss_v7_suites.py",
}


def _development_source(tmp_path: Path) -> Path:
    payload = v3_registry.bind_development_suite(
        PROJECT_ROOT / "results/table2_anime_fixed3_inputs_20260722"
    )
    path = tmp_path / "source" / "development_suite.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(v3_registry.canonical_json_bytes(dict(payload)))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == APPROVED_DEVELOPMENT_SUITE_SHA256
    assert payload["semantic_sha256"] == APPROVED_DEVELOPMENT_SUITE_SEMANTIC_SHA256
    return path


@pytest.fixture(scope="module")
def bound_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Path, Path, Path, Path]]:
    temporary = tmp_path_factory.mktemp("v7-suite")
    source = _development_source(temporary)
    patcher = pytest.MonkeyPatch()
    patcher.setattr(suite_module, "APPROVED_DEVELOPMENT_SUITE_PATH", source, raising=False)
    patcher.setattr(
        suite_module.provenance_module, "EXPECTED_SPECIFICATION_SHA256",
        "d6f3219080f61acc9a3d6c2b435dfd0fd12eac06b68c2f4a301f0b5fedfa74c1",
    )
    root = temporary / "output"
    try:
        development, formal = write_v7_suite_bundle(
            root, development_source=source, project_root=PROJECT_ROOT
        )
        assert isinstance(development, AuthenticatedV7Suite)
        assert isinstance(formal, AuthenticatedV7Suite)
        yield root, source, Path(development.path), Path(formal.path)
    finally:
        patcher.undo()


def test_public_records_are_frozen_exact_and_selector_reproduces_frozen_rows() -> None:
    """Break caught: selector order, PCG stream, registration, or hull identity drifts."""

    item = V7BoundInput("x", "a", "/a", "b" * 64, "c" * 64, 1, 1, 0.1, 2.0, 0, 1)
    with pytest.raises(FrozenInstanceError):
        item.axis = "y"  # type: ignore[misc]
    assert tuple(item.__dataclass_fields__) == (
        "axis", "source_path", "resolved_path", "raw_sha256", "normalized_sha256",
        "foreground_count", "component_count", "foreground_ratio", "complexity",
        "foreground_tertile", "complexity_tertile",
    )

    tasks, evidence = suite_module._select_v7_formal_tasks_with_evidence(PROJECT_ROOT)
    assert select_v7_formal_tasks(PROJECT_ROOT) == tasks
    assert evidence["selector"] == "pcraft_soft_loss_v7_suite_v1"
    assert evidence["eligible_count"] == 103
    assert evidence["candidate_triple_count"] == 77862
    assert evidence["hull_checks"] == 164
    assert evidence["accepted_stream_indices"] == list(STREAM_INDICES)
    assert len(evidence["anchor_sha256"]) == len(set(evidence["anchor_sha256"])) == 39
    assert tuple(evidence["anchor_paths"]) == (*FORMAL_ANCHOR_PATHS, *DEVELOPMENT_ANCHOR_PATHS)
    assert tuple(evidence["anchor_sha256"]) == (*FORMAL_ANCHOR_SHA256, *DEVELOPMENT_ANCHOR_SHA256)
    assert set(evidence["executed_sources"]) == EXECUTED_SOURCE_RELATIVES
    assert evidence["executed_sources"] == {
        relative: hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest()
        for relative in sorted(EXECUTED_SOURCE_RELATIVES)
    }
    assert [task.case_id for task in tasks] == [f"v7_formal_{index:02d}" for index in range(1, 11)]
    for task, expected, stream_index in zip(tasks, EXPECTED_ROWS, STREAM_INDICES):
        paths, hashes, occupied, hull_sha = expected
        assert tuple(item.source_path for item in task.inputs) == paths
        assert tuple(item.raw_sha256 for item in task.inputs) == hashes
        assert tuple(item.axis for item in task.inputs) == ("x", "y", "z")
        assert all(item.component_count == 1 for item in task.inputs)
        assert {item.foreground_tertile for item in task.inputs} == {0, 1, 2}
        assert {item.complexity_tertile for item in task.inputs} == {0, 1, 2}
        assert task.hull_occupied_count == occupied
        assert task.hull_component_count == 1
        assert task.hull_xyz_sha256 == hull_sha
        assert task.accepted_stream_index == stream_index
    assert len({item.raw_sha256 for task in tasks for item in task.inputs}) == 30
    selected_paths = {item.resolved_path for task in tasks for item in task.inputs}
    selected_hashes = {item.raw_sha256 for task in tasks for item in task.inputs}
    assert not selected_paths.intersection(DEVELOPMENT_ANCHOR_PATHS)
    assert not selected_hashes.intersection(DEVELOPMENT_ANCHOR_SHA256)
    assert not selected_hashes.intersection(evidence["anchor_sha256"])
    assert torch.cuda.is_initialized() is False


def test_selector_has_no_renderer_training_metric_result_or_cuda_dependency() -> None:
    """Break caught: a pre-import guard observes forbidden modules, CUDA, or legacy inputs6."""

    source = (PROJECT_ROOT / "shadowforge_pipeline/soft_voxel_v7/suite.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imports.add(node.module)
            imports.update(alias.name for alias in node.names)
    forbidden = ("renderer", "trainer", "metric", "checkpoint", "result", "pytorch3d", "cuda")
    assert not any(any(part in name.lower() for part in forbidden) for name in imports)
    assert "/data/xvxinyao/pcraft_soft_loss_v7_hit_barrier_256_20260829" not in source
    assert "objectlike_anime_fixed_10cm_256_20260708" not in source

    guard = f'''
import importlib.abc
import os
from pathlib import Path
import sys
import torch

project = Path({str(PROJECT_ROOT)!r})
# The approved Task 1 package initializer owns its renderer-facing protocol.
# Load that immutable package boundary before guarding the first Task 2 import.
import shadowforge_pipeline.soft_voxel_v7
assert "shadowforge_pipeline.soft_voxel_v7.suite" not in sys.modules
forbidden_modules = ("renderer", "trainer", "metrics", "checkpoint", "analyze", "result")
class PoisonFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(part in fullname.lower() for part in forbidden_modules):
            raise RuntimeError("forbidden module access: " + fullname)
        return None
sys.meta_path.insert(0, PoisonFinder())
torch.cuda._lazy_init = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("CUDA access"))
real_open = os.open
def guarded_open(path, *args, **kwargs):
    text = os.fsdecode(path)
    if "objectlike_anime_fixed_10cm_256_20260708" in text or "/checkpoints/" in text:
        raise RuntimeError("forbidden path access: " + text)
    return real_open(path, *args, **kwargs)
os.open = guarded_open
from shadowforge_pipeline.soft_voxel_v7.suite import select_v7_formal_tasks
select_v7_formal_tasks(project)
assert torch.cuda.is_initialized() is False
'''
    result = subprocess.run(
        [sys.executable, "-c", guard], cwd=PROJECT_ROOT, text=True,
        capture_output=True, check=False,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(PROJECT_ROOT), CUDA_VISIBLE_DEVICES=""),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "" and result.stderr == ""


def test_bundle_exact_schema_development_and_formal_recompute(bound_bundle: tuple[Path, Path, Path, Path]) -> None:
    """Break caught: manifests trust copied fields instead of source bytes and selector replay."""

    root, source, development_path, formal_path = bound_bundle
    development = validate_v7_suite(development_path, expected_kind="development")
    formal = validate_v7_suite(formal_path, expected_kind="formal")
    assert [task.case_id for task in development.tasks] == ["arcane_casters", "blade_heroes", "stage_dancers"]
    assert [task.case_id for task in formal.tasks] == [f"v7_formal_{index:02d}" for index in range(1, 11)]
    dev_payload = json.loads(development_path.read_bytes())
    formal_payload = json.loads(formal_path.read_bytes())
    assert set(dev_payload) == {
        "schema", "status", "kind", "root_identity", "project_root", "source_suite",
        "task1_registration_identity", "normalization", "tasks", "task_count",
        "input_count", "input_order", "semantic_sha256",
    }
    assert set(formal_payload) == {
        "schema", "status", "kind", "root_identity", "project_root", "selector",
        "development_suite",
        "task1_registration_identity", "normalization", "tasks", "task_count",
        "input_count", "input_order", "semantic_sha256",
    }
    assert dev_payload["source_suite"]["path"] == str(source)
    assert dev_payload["task_count"] == 3 and dev_payload["input_count"] == 9
    assert formal_payload["task_count"] == 10 and formal_payload["input_count"] == 30
    assert formal_payload["selector"]["eligible_count"] == 103
    assert formal_payload["selector"]["candidate_triple_count"] == 77862
    assert formal_payload["selector"]["hull_checks"] == 164
    assert formal_payload["selector"]["accepted_stream_indices"] == list(STREAM_INDICES)
    assert formal_payload["development_suite"] == {
        "path": str(development_path),
        "sha256": hashlib.sha256(development_path.read_bytes()).hexdigest(),
        "semantic_sha256": dev_payload["semantic_sha256"],
    }
    development_hashes = {
        item["raw_sha256"] for task in dev_payload["tasks"] for item in task["inputs"]
    }
    formal_hashes = {
        item["raw_sha256"] for task in formal_payload["tasks"] for item in task["inputs"]
    }
    development_paths = {
        item["resolved_path"] for task in dev_payload["tasks"] for item in task["inputs"]
    }
    formal_paths = {
        item["resolved_path"] for task in formal_payload["tasks"] for item in task["inputs"]
    }
    assert development_hashes == set(DEVELOPMENT_ANCHOR_SHA256)
    assert development_paths == set(DEVELOPMENT_ANCHOR_PATHS)
    assert development_hashes.isdisjoint(formal_hashes)
    assert development_paths.isdisjoint(formal_paths)
    assert "sadekar_tfs_targets" not in json.dumps(dev_payload)
    assert development.path == str(root / "manifests/development_suite.json")
    assert formal.path == str(root / "manifests/formal_suite.json")


def test_only_the_approved_development_suite_path_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a byte-identical mirror substitutes for the uniquely approved source."""

    approved = _development_source(tmp_path / "approved")
    mirror = tmp_path / "mirror" / "development_suite.json"
    mirror.parent.mkdir()
    mirror.write_bytes(approved.read_bytes())
    monkeypatch.setattr(
        suite_module, "APPROVED_DEVELOPMENT_SUITE_PATH", approved, raising=False
    )
    with pytest.raises(ValueError, match="approved|source|path|identity"):
        suite_module._development_tasks(mirror, PROJECT_ROOT)


def test_formal_validation_independently_authenticates_development_prefix(
    bound_bundle: tuple[Path, Path, Path, Path]
) -> None:
    """Break caught: a valid formal marker remains accepted over an opaque dev prefix."""

    _, _, development_path, formal_path = bound_bundle
    original = development_path.read_bytes()
    try:
        development_path.write_bytes(b"opaque-development-prefix")
        with pytest.raises(ValueError, match="development|canonical|JSON|prefix|suite"):
            validate_v7_suite(formal_path, expected_kind="formal")
    finally:
        development_path.write_bytes(original)


def test_formal_publication_reauthenticates_development_before_commit_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: formal publication occurs after the development prefix is replaced."""

    source = _development_source(tmp_path)
    root = tmp_path / "output"
    monkeypatch.setattr(
        suite_module, "APPROVED_DEVELOPMENT_SUITE_PATH", source, raising=False
    )
    monkeypatch.setattr(
        suite_module.provenance_module, "EXPECTED_SPECIFICATION_SHA256",
        "d6f3219080f61acc9a3d6c2b435dfd0fd12eac06b68c2f4a301f0b5fedfa74c1",
    )

    def replace_prefix(stage: str) -> None:
        if stage == "after_development_publish":
            (root / "manifests/development_suite.json").write_bytes(b"opaque")

    monkeypatch.setattr(suite_module, "_PUBLICATION_HOOK", replace_prefix)
    with pytest.raises(ValueError, match="development|canonical|JSON|prefix|suite|changed"):
        write_v7_suite_bundle(root, development_source=source, project_root=PROJECT_ROOT)
    assert not (root / "manifests/formal_suite.json").exists()


def test_suite_validation_allows_only_registered_future_manifest_names(
    bound_bundle: tuple[Path, Path, Path, Path]
) -> None:
    """Break caught: downstream campaign files break suites, or arbitrary names are allowed."""

    root, _, development, formal = bound_bundle
    manifests = root / "manifests"
    campaign = manifests / "campaign.json"
    matrix = manifests / "formal_matrix.json"
    campaign.write_bytes(b"opaque-task3-owned")
    matrix.write_bytes(b"opaque-task3-owned")
    assert validate_v7_suite(development, expected_kind="development").kind == "development"
    assert validate_v7_suite(formal, expected_kind="formal").kind == "formal"
    unexpected = manifests / "arbitrary.json"
    unexpected.write_bytes(b"not-registered")
    try:
        with pytest.raises(ValueError, match="unexpected|namespace|registered"):
            validate_v7_suite(formal, expected_kind="formal")
    finally:
        unexpected.unlink()
        matrix.unlink()
        campaign.unlink()


def test_public_root_transaction_creates_downstream_directories_without_invalidating_suites(
    bound_bundle: tuple[Path, Path, Path, Path],
) -> None:
    """Break caught: safe downstream lifecycle handoff invalidates committed suite manifests."""

    root, _, development, formal = bound_bundle
    with suite_module.provenance_module.open_v7_root_transaction(root) as transaction:
        for name in ("authorizations", "runs", "analysis", "postprocess"):
            transaction.ensure_registered_directory(name)
    assert validate_v7_suite(development, expected_kind="development").kind == "development"
    assert validate_v7_suite(formal, expected_kind="formal").kind == "formal"


def test_selector_rejects_foreign_delegating_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: output-equivalent foreign code replaces the frozen normalizer closure."""

    authoritative = suite_module.normalize_target

    def foreign_normalize_target(*args: object, **kwargs: object) -> object:
        return authoritative(*args, **kwargs)

    monkeypatch.setattr(suite_module, "normalize_target", foreign_normalize_target)
    with pytest.raises(ValueError, match="normalizer|execution|closure|source|loaded"):
        select_v7_formal_tasks(PROJECT_ROOT)


def test_selector_rejects_foreign_loaded_binder_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: binder bytes are recorded without authenticating loaded code."""

    foreign = ModuleType("experiments.bind_soft_loss_v7_suites")
    foreign.__file__ = str(PROJECT_ROOT / "tests/test_soft_loss_v7_suite.py")
    foreign.binding_main = suite_module.binding_main  # type: ignore[attr-defined]
    foreign.write_v7_suite_bundle = suite_module.write_v7_suite_bundle  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    with pytest.raises(ValueError, match="binder|execution|closure|source|loaded"):
        select_v7_formal_tasks(PROJECT_ROOT)


def test_publication_boundary_revalidates_executed_source_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a closure drift after dev publication still commits formal."""

    source = _development_source(tmp_path)
    root = tmp_path / "output"
    monkeypatch.setattr(
        suite_module, "APPROVED_DEVELOPMENT_SUITE_PATH", source, raising=False
    )
    monkeypatch.setattr(
        suite_module.provenance_module, "EXPECTED_SPECIFICATION_SHA256",
        "d6f3219080f61acc9a3d6c2b435dfd0fd12eac06b68c2f4a301f0b5fedfa74c1",
    )
    real_revalidate = suite_module.provenance_module._revalidate_snapshot
    after_development = False

    def stage(stage_name: str) -> None:
        nonlocal after_development
        if stage_name == "after_development_publish":
            after_development = True

    def reject_drifted_closure(snapshot: object, *, root: Path | None = None) -> None:
        if after_development and str(getattr(snapshot, "path", "")).endswith(
            "experiments/bind_soft_loss_v7_suites.py"
        ):
            raise ValueError("executed source closure changed")
        real_revalidate(snapshot, root=root)  # type: ignore[arg-type]

    monkeypatch.setattr(suite_module, "_PUBLICATION_HOOK", stage)
    monkeypatch.setattr(
        suite_module.provenance_module, "revalidate_snapshot",
        reject_drifted_closure, raising=False,
    )
    monkeypatch.setattr(
        suite_module, "revalidate_snapshot", reject_drifted_closure, raising=False
    )
    with pytest.raises(ValueError, match="executed source closure changed"):
        write_v7_suite_bundle(root, development_source=source, project_root=PROJECT_ROOT)
    assert not (root / "manifests/formal_suite.json").exists()


def test_suite_publication_retains_original_root_inode_across_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: suite lock survives but writes continue in a same-name root replacement."""

    source = _development_source(tmp_path)
    root = tmp_path / "output"
    displaced = tmp_path / "output-original-inode"
    monkeypatch.setattr(
        suite_module, "APPROVED_DEVELOPMENT_SUITE_PATH", source, raising=False
    )
    monkeypatch.setattr(
        suite_module.provenance_module, "EXPECTED_SPECIFICATION_SHA256",
        "d6f3219080f61acc9a3d6c2b435dfd0fd12eac06b68c2f4a301f0b5fedfa74c1",
    )
    swapped = False

    def replace_root(stage: str) -> None:
        nonlocal swapped
        if stage == "after_development_publish" and not swapped:
            root.rename(displaced)
            root.mkdir()
            (root / "manifests").mkdir()
            (root / "V7_ROOT.json").write_bytes(
                (displaced / "V7_ROOT.json").read_bytes()
            )
            (root / "manifests/development_suite.json").write_bytes(
                (displaced / "manifests/development_suite.json").read_bytes()
            )
            swapped = True

    monkeypatch.setattr(suite_module, "_PUBLICATION_HOOK", replace_root)
    with pytest.raises(ValueError, match="root|directory|identity|changed|binding"):
        write_v7_suite_bundle(root, development_source=source, project_root=PROJECT_ROOT)
    assert swapped
    assert not (root / "manifests/formal_suite.json").exists()


def test_formal_validator_reauthenticates_development_after_selector_replay(
    bound_bundle: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Break caught: development prefix mutates during replay after its only authentication."""

    _, _, development_path, formal_path = bound_bundle
    original_development = development_path.read_bytes()
    original_selection = suite_module._selection
    mutated = False

    def mutate_after_replay(project_root: Path) -> object:
        nonlocal mutated
        result = original_selection(project_root)
        development_path.write_bytes(b"opaque-development-after-replay")
        mutated = True
        return result

    monkeypatch.setattr(suite_module, "_selection", mutate_after_replay)
    try:
        with pytest.raises(ValueError, match="development|prefix|suite|identity|changed"):
            validate_v7_suite(formal_path, expected_kind="formal")
        assert mutated
    finally:
        development_path.write_bytes(original_development)


@pytest.mark.parametrize(
    "mutation",
    (
        "extra", "missing", "reorder", "duplicate_sha", "path", "anchor",
        "development_overlap", "foreign_root", "bool_count",
    ),
)
def test_formal_validator_rejects_schema_identity_and_selection_mutations(
    bound_bundle: tuple[Path, Path, Path, Path], mutation: str
) -> None:
    """Break caught: a self-rehashed forged formal bundle bypasses independent replay."""

    _, _, _, formal_path = bound_bundle
    original = formal_path.read_bytes()
    payload = json.loads(original)
    try:
        if mutation == "extra":
            payload["extra"] = True
        elif mutation == "missing":
            del payload["input_order"]
        elif mutation == "reorder":
            payload["tasks"] = list(reversed(payload["tasks"]))
        elif mutation == "duplicate_sha":
            payload["tasks"][0]["inputs"][0]["raw_sha256"] = payload["tasks"][0]["inputs"][1]["raw_sha256"]
        elif mutation == "path":
            payload["tasks"][0]["inputs"][0]["source_path"] = "bench3/other.png"
        elif mutation == "anchor":
            payload["selector"]["anchor_sha256"][0] = payload["tasks"][0]["inputs"][0]["raw_sha256"]
        elif mutation == "development_overlap":
            payload["tasks"][0]["inputs"][0]["raw_sha256"] = DEVELOPMENT_ANCHOR_SHA256[0]
            payload["tasks"][0]["inputs"][0]["resolved_path"] = DEVELOPMENT_ANCHOR_PATHS[0]
        elif mutation == "foreign_root":
            payload["root_identity"]["path"] = str(formal_path.parent.parent.parent)
        else:
            payload["task_count"] = True
        payload.pop("semantic_sha256", None)
        payload["semantic_sha256"] = canonical_sha256(payload)
        formal_path.write_bytes(canonical_json_bytes(payload))
        with pytest.raises(ValueError):
            validate_v7_suite(formal_path, expected_kind="formal")
    finally:
        formal_path.write_bytes(original)


def test_validators_are_read_only_zero_write(bound_bundle: tuple[Path, Path, Path, Path]) -> None:
    """Break caught: authentication refreshes manifests or creates caches/artifacts."""

    root, _, development, formal = bound_bundle
    files = sorted(path for path in root.rglob("*") if path.is_file())
    before = {path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) for path in files}
    validate_v7_suite(development, expected_kind="development")
    validate_v7_suite(formal, expected_kind="formal")
    after_files = sorted(path for path in root.rglob("*") if path.is_file())
    assert after_files == files
    assert {path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) for path in files} == before


def test_publication_recovers_valid_prefix_revalidates_toctou_and_rejects_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a crash/source swap yields a mixed or overwritten bundle."""

    source = _development_source(tmp_path)
    root = tmp_path / "output"
    calls: list[str] = []
    monkeypatch.setattr(
        suite_module, "APPROVED_DEVELOPMENT_SUITE_PATH", source, raising=False
    )

    def crash(stage: str) -> None:
        calls.append(stage)
        if stage == "after_development_publish":
            raise RuntimeError("synthetic crash")

    monkeypatch.setattr(suite_module, "_PUBLICATION_HOOK", crash)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        write_v7_suite_bundle(root, development_source=source, project_root=PROJECT_ROOT)
    assert (root / "V7_ROOT.json").is_file()
    assert (root / "manifests/development_suite.json").is_file()
    assert not (root / "manifests/formal_suite.json").exists()
    monkeypatch.setattr(suite_module, "_PUBLICATION_HOOK", lambda stage: None)
    write_v7_suite_bundle(root, development_source=source, project_root=PROJECT_ROOT)
    formal = root / "manifests/formal_suite.json"
    first = formal.read_bytes()
    write_v7_suite_bundle(root, development_source=source, project_root=PROJECT_ROOT)
    assert formal.read_bytes() == first
    formal.write_bytes(b"foreign")
    with pytest.raises((FileExistsError, ValueError)):
        write_v7_suite_bundle(root, development_source=source, project_root=PROJECT_ROOT)
    assert formal.read_bytes() == b"foreign"

    root2 = tmp_path / "toctou"
    source2 = _development_source(tmp_path / "second")
    def drift(stage: str) -> None:
        if stage == "after_development_publish":
            source2.write_bytes(source2.read_bytes() + b"\n")
    monkeypatch.setattr(suite_module, "_PUBLICATION_HOOK", drift)
    with pytest.raises(ValueError, match="source|dependency|changed|canonical"):
        write_v7_suite_bundle(root2, development_source=source2, project_root=PROJECT_ROOT)
    assert not (root2 / "manifests/formal_suite.json").exists()


def test_concurrent_identical_publishers_converge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: concurrent writers publish different bytes or mixed ordering."""

    source = _development_source(tmp_path)
    root = tmp_path / "fresh-concurrent-root"
    monkeypatch.setattr(
        suite_module, "APPROVED_DEVELOPMENT_SUITE_PATH", source, raising=False
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: write_v7_suite_bundle(root, development_source=source, project_root=PROJECT_ROOT), range(2)))
    assert len(results) == 2
    development = root / "manifests/development_suite.json"
    formal = root / "manifests/formal_suite.json"
    assert results[0] == results[1]
    assert hashlib.sha256(development.read_bytes()).hexdigest() == results[0][0].sha256
    assert hashlib.sha256(formal.read_bytes()).hexdigest() == results[0][1].sha256


def _copy_selector_inputs(destination: Path) -> Path:
    for relative in (
        "data/silhouettes/bench3", "data/silhouettes/complex_v2_filled",
        "data/silhouettes/original_anime_30_20260704", "data/silhouettes/anime_sheet_100_20260704",
    ):
        shutil.copytree(PROJECT_ROOT / relative, destination / relative)
    return destination


@pytest.mark.parametrize("attack", ("byte_drift", "disconnected"))
def test_selector_rejects_candidate_drift_anchor_collision_and_disconnection(
    tmp_path: Path, attack: str
) -> None:
    """Break caught: selection silently accepts changed bytes or invalid components."""

    project = _copy_selector_inputs(tmp_path / "project")
    target = project / "data/silhouettes/bench3/reptiles_turtle.png"
    if attack == "byte_drift":
        target.write_bytes(target.read_bytes() + b"x")
    else:
        import cv2
        image = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
        image[:] = 0
        image[10:20, 10:20] = 255
        image[200:210, 200:210] = 255
        assert cv2.imwrite(str(target), image)
    with pytest.raises(ValueError, match="frozen|eligible|selection|hash|component|universe"):
        select_v7_formal_tasks(project)


def test_selector_excludes_a_raw_sha_collision_with_frozen_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: formal candidates matching a frozen raw SHA remain eligible."""

    project = _copy_selector_inputs(tmp_path / "project")
    target = project / "data/silhouettes/bench3/reptiles_turtle.png"
    target_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    frozen = tuple(zip(
        (*FORMAL_ANCHOR_PATHS, *DEVELOPMENT_ANCHOR_PATHS),
        (*FORMAL_ANCHOR_SHA256, *DEVELOPMENT_ANCHOR_SHA256),
    ))
    monkeypatch.setattr(
        suite_module,
        "RAW_ANCHOR_ROWS",
        ((frozen[0][0], target_sha), *frozen[1:]),
        raising=False,
    )
    with pytest.raises(ValueError, match="eligible|frozen|selection|universe"):
        select_v7_formal_tasks(project)


def test_selector_revalidates_source_mutation_during_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Break caught: a candidate changes after its snapshot but before selection returns."""

    project = _copy_selector_inputs(tmp_path / "project")
    original = suite_module._candidate_from_snapshot
    changed = False

    def mutate_after_snapshot(snapshot: object, *, source_path: str) -> object:
        nonlocal changed
        candidate = original(snapshot, source_path=source_path)
        if not changed:
            path = Path(getattr(snapshot, "path"))
            path.write_bytes(path.read_bytes() + b"drift")
            changed = True
        return candidate

    monkeypatch.setattr(suite_module, "_candidate_from_snapshot", mutate_after_snapshot)
    with pytest.raises(ValueError, match="dependency|identity|changed|frozen"):
        select_v7_formal_tasks(project)
    assert changed


def test_exact_connectivity_rejects_disconnected_masks_and_hulls() -> None:
    """Break caught: diagonal-only 2D/3D regions are treated as valid connected hulls."""

    disconnected_mask = np.zeros((4, 4), dtype=np.bool_)
    disconnected_mask[0, 0] = True
    disconnected_mask[3, 3] = True
    disconnected_hull = np.zeros((4, 4, 4), dtype=np.bool_)
    disconnected_hull[0, 0, 0] = True
    disconnected_hull[3, 3, 3] = True
    assert suite_module._component_count(disconnected_mask, connectivity=8) == 2
    assert suite_module._component_count(disconnected_hull, connectivity=6) == 2


def test_cli_is_exact_warning_clean_and_delegates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Break caught: launcher accepts aliases/extra knobs or owns a second implementation."""

    parser = suite_module.build_binding_parser()
    actions = {action.dest: action for action in parser._actions if action.dest != "help"}
    assert set(actions) == {"output_root", "development_source", "project_root"}
    assert all(action.required for action in actions.values())
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "/tmp/x"])
    result = subprocess.run(
        [sys.executable, "-W", "error", str(PROJECT_ROOT / "experiments/bind_soft_loss_v7_suites.py"), "--help"],
        cwd=PROJECT_ROOT, text=True, capture_output=True, check=False,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(PROJECT_ROOT), CUDA_VISIBLE_DEVICES=""),
    )
    assert result.returncode == 0 and result.stderr == ""
    assert all(option in result.stdout for option in ("--output-root", "--development-source", "--project-root"))
    launcher = (PROJECT_ROOT / "experiments/bind_soft_loss_v7_suites.py").read_text(encoding="utf-8")
    assert "write_v7_suite_bundle" in launcher
    assert "/data" not in launcher
