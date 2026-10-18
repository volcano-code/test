"""Real-CPU regressions for controlled soft-voxel initialization."""
import copy
import dataclasses
import hashlib
import json

import numpy as np
import pytest

from experiments.shadowart_vnext import training
from experiments.shadowart_vnext.contracts import build_contract, file_hash, mask_hash


def _contract_and_targets():
    targets = np.zeros((1, 4, 4), dtype=bool)
    targets[:, 1:3, 1:3] = True
    contract = build_contract(1, image_size=4, resolution=4)
    contract = dataclasses.replace(
        contract,
        input_hashes=(file_hash(__file__),),
        target_hashes=(mask_hash(targets[0]),),
    )
    return contract, targets


def _train(initialization_mode=None):
    contract, targets = _contract_and_targets()
    kwargs = dict(
        resolution=4,
        image_size=4,
        steps=1,
        seed=17,
        device="cpu",
    )
    if initialization_mode is not None:
        kwargs["initialization_mode"] = initialization_mode
    return contract, training.train_soft(targets, contract.cameras, **kwargs)


def test_full_minus4_initial_hash_comes_from_exact_literal_float32_cube():
    _, fields = _train("full_minus4")
    literal = np.full((4, 4, 4), -4.0, dtype=np.float32)
    _, targets = _contract_and_targets()
    hull_targets = np.repeat(np.repeat(targets, 2, axis=1), 2, axis=2)
    expected = hashlib.sha256(
        str((literal.shape, str(literal.dtype))).encode() + literal.tobytes()
    ).hexdigest()
    assert fields["metadata"]["initial_logits_sha256"] == expected
    assert fields["metadata"]["initialization"] == {
        "method": "full_minus4",
        "logit": -4.0,
        "noise_std": 0.0,
        "hull_image_size": 8,
        "target_source": "nearest_nominal_binary",
        "post_init_hull_clipping": False,
        "target_hashes": [mask_hash(hull_targets[0])],
    }


def test_default_and_explicit_visual_hull_are_bit_identical():
    _, default = _train()
    _, explicit = _train("visual_hull")
    for key in (
        "occupancy",
        "density",
        "probabilities",
        "guidance",
        "soft_projections",
        "hard_projections",
        "hard_projections_512",
    ):
        np.testing.assert_array_equal(default[key], explicit[key])
    assert default["history"] == explicit["history"]
    assert default["metadata"] == explicit["metadata"]


def test_full_minus4_and_visual_hull_differ_with_the_same_seed():
    _, visual_hull = _train("visual_hull")
    _, full_minus4 = _train("full_minus4")
    assert (
        visual_hull["metadata"]["initial_logits_sha256"]
        != full_minus4["metadata"]["initial_logits_sha256"]
    )
    assert not np.array_equal(visual_hull["density"], full_minus4["density"])


def test_invalid_initialization_mode_is_rejected_by_api_and_cli(tmp_path):
    contract, targets = _contract_and_targets()
    with pytest.raises(ValueError, match="initialization_mode"):
        training.train_soft(
            targets,
            contract.cameras,
            resolution=4,
            image_size=4,
            steps=1,
            initialization_mode="invented",
        )

    from experiments.run_shadowart_vnext import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--silhouettes",
                "x.png",
                "--output",
                str(tmp_path / "run"),
                "--initialization-mode",
                "invented",
            ]
        )


def test_cli_defaults_mode_and_replay_rejects_explicit_override(tmp_path):
    from experiments.run_shadowart_vnext import parse_args

    common = ["--silhouettes", "x.png", "--output", str(tmp_path / "run")]
    assert parse_args(common).initialization_mode == "visual_hull"
    with pytest.raises(SystemExit):
        parse_args(
            common
            + [
                "--soft-artifact-dir",
                "soft",
                "--initialization-mode",
                "visual_hull",
            ]
        )


def test_real_full_minus4_artifact_round_trip(tmp_path):
    from experiments.shadowart_vnext.artifacts import load_soft, save_soft

    contract, fields = _train("full_minus4")
    directory = tmp_path / "soft"
    save_soft(directory, fields, contract)
    loaded = load_soft(directory, contract)
    np.testing.assert_array_equal(loaded["density"], fields["density"])
    assert loaded["metadata"]["initialization"] == fields["metadata"]["initialization"]


@pytest.mark.parametrize(
    "mutation",
    [
        "method",
        "logit",
        "noise_std",
        "inside_label",
        "float_hull_image_size",
        "integer_hull_clipping",
    ],
)
def test_artifact_rejects_inconsistent_full_minus4_initialization(
    tmp_path, mutation
):
    from experiments.shadowart_vnext.artifacts import save_soft

    contract, original = _train("full_minus4")
    fields = copy.deepcopy(original)
    initialization = fields["metadata"]["initialization"]
    if mutation == "method":
        initialization["method"] = "canonical_visual_hull"
    elif mutation == "logit":
        initialization["logit"] = -3.0
    elif mutation == "noise_std":
        initialization["noise_std"] = 0.01
    elif mutation == "inside_label":
        initialization["inside_logit"] = -4.0
    elif mutation == "float_hull_image_size":
        initialization["hull_image_size"] = 8.0
    else:
        initialization["post_init_hull_clipping"] = 0
    with pytest.raises(ValueError, match="INITIALIZATION"):
        save_soft(tmp_path / "bad", fields, contract)


def test_pipeline_persists_requested_full_minus4_mode(tmp_path):
    from PIL import Image

    from experiments.run_shadowart_vnext import main
    from experiments.shadowart_vnext.artifacts import load_soft
    from experiments.shadowart_vnext.contracts import CameraContract

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    source = tmp_path / "square.png"
    Image.fromarray(mask).save(source)
    output = tmp_path / "run"
    assert main(
        [
            "--silhouettes",
            str(source),
            "--output",
            str(output),
            "--stage",
            "soft",
            "--resolution",
            "4",
            "--image-size",
            "4",
            "--steps",
            "1",
            "--device",
            "cpu",
            "--layout-candidates",
            "1",
            "--initialization-mode",
            "full_minus4",
        ]
    ) == 0
    contract = CameraContract.from_dict(
        json.loads((output / "CAMERA_CONTRACT.json").read_text())
    )
    loaded = load_soft(output / "soft_training", contract)
    assert loaded["metadata"]["initialization"]["method"] == "full_minus4"
