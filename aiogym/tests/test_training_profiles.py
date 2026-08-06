from __future__ import annotations

from importlib.resources import files

import pytest

from aiogym.rl.profiles import build_training_config, list_training_profiles


@pytest.mark.parametrize("profile", list_training_profiles())
def test_quick_training_profile_comes_from_packaged_resource(profile):
    prefix = "package:aiogym/"
    assert profile.source.startswith(prefix)
    resource = files("aiogym").joinpath(profile.source.removeprefix(prefix))
    assert resource.is_file()
    assert profile.config_template["track_id"] == profile.track_id
    assert profile.config_template["algorithm_id"] == profile.algorithm_id


@pytest.mark.parametrize(
    "target",
    ("quadruple", "cascade", "cascade-recirculating"),
)
def test_quick_sac_profile_builds_canonical_config(target, tmp_path):
    config, profile = build_training_config(
        target,
        "sac",
        "quick",
        training_seed=3,
        device="cpu",
        output_directory=str(tmp_path),
        dataset_path=None,
        resume_checkpoint=None,
    )
    assert config.track_id.endswith("-regulation-generalist-v2")
    assert config.algorithm_id == "sac"
    assert config.budget_unit == "environment_transitions"
    assert config.budget_value == 1000
    assert config.output["name"] == f"{target}-sac-quick-seed3"
    assert profile.id == "quick-v1"


@pytest.mark.parametrize("algorithm", ("bc", "rlpd"))
def test_offline_profiles_require_dataset(algorithm):
    with pytest.raises(ValueError, match="requires --dataset"):
        build_training_config(
            "quadruple",
            algorithm,
            "quick",
            training_seed=0,
            device=None,
            output_directory=None,
            dataset_path=None,
            resume_checkpoint=None,
        )


def test_online_profile_rejects_dataset():
    with pytest.raises(ValueError, match="does not accept --dataset"):
        build_training_config(
            "quadruple",
            "sac",
            "quick",
            training_seed=0,
            device=None,
            output_directory=None,
            dataset_path="dataset",
            resume_checkpoint=None,
        )


def test_unregistered_profile_lists_available_combinations():
    with pytest.raises(ValueError, match="available combinations"):
        build_training_config(
            "quadruple",
            "ppo",
            "quick",
            training_seed=0,
            device=None,
            output_directory=None,
            dataset_path=None,
            resume_checkpoint=None,
        )
