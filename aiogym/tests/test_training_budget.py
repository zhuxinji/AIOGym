from __future__ import annotations

import pytest

from aiogym.rl.config import RLTrainingConfig


TRACK_ID = "quadruple-regulation-generalist-v1"


def _v3(algorithm_id: str, unit: str):
    return {
        "schema_version": "aiogym.rl_training_config.v3",
        "track_id": TRACK_ID,
        "algorithm_id": algorithm_id,
        "training_seed": 0,
        "budget": {"unit": unit, "value": 10},
        "n_envs": 1,
    }


def test_bc_budget_is_optimizer_updates():
    config = RLTrainingConfig.from_mapping(
        _v3("bc", "optimizer_updates")
    )
    assert config.budget_unit == "optimizer_updates"
    assert config.budget_value == 10
    assert config.as_dict()["budget"] == {
        "unit": "optimizer_updates",
        "value": 10,
    }


@pytest.mark.parametrize("algorithm_id", ["sac", "td3", "ppo", "rlpd"])
def test_online_algorithms_require_environment_transition_budget(
    algorithm_id,
):
    with pytest.raises(ValueError, match="environment_transitions"):
        RLTrainingConfig.from_mapping(
            _v3(algorithm_id, "optimizer_updates")
        )


def test_v2_config_migrates_with_explicit_unit_and_metadata():
    with pytest.warns(DeprecationWarning, match="explicit v3 budget"):
        config = RLTrainingConfig.from_mapping(
            {
                "schema_version": "aiogym.rl_training_config.v2",
                "track_id": TRACK_ID,
                "algorithm_id": "bc",
                "training_seed": 0,
                "total_transitions": 7,
                "n_envs": 1,
            }
        )
    assert config.schema_version == "aiogym.rl_training_config.v3"
    assert config.budget_unit == "optimizer_updates"
    assert "total_transitions" not in config.as_dict()
    assert config.migration_metadata["resolved_budget_unit"] == (
        "optimizer_updates"
    )
