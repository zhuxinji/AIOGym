from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import aiogym
from aiogym.models.cases.schema import case_profile_hash


GOLDEN = Path(__file__).with_name("golden")


@pytest.mark.parametrize(
    "filename",
    ("quadruple-env-trace-v1.json", "cascade-env-trace-v1.json"),
)
def test_direct_and_config_environment_paths_match_golden_trace(filename):
    fixture = json.loads((GOLDEN / filename).read_text(encoding="utf-8"))
    declaration = {
        "scenario": fixture["scenario"],
        "case": fixture["case"],
        "reward_spec": fixture["reward_spec"],
        "environment": {},
    }
    direct = aiogym.make_env(
        fixture["scenario"],
        case=fixture["case"],
        reward_spec=fixture["reward_spec"],
    )
    configured = aiogym.make_env(config=declaration)
    try:
        assert direct.env_spec.spec_hash == fixture["env_spec_hash"]
        assert configured.env_spec.spec_hash == fixture["env_spec_hash"]
        assert case_profile_hash(direct.case_profile) == fixture["case_profile_hash"]
        assert direct.reward_spec.version == fixture["reward_spec_version"]
        assert list(direct.action_space.shape) == fixture["action_space"]["shape"]
        assert list(direct.observation_space.shape) == fixture["observation_space"]["shape"]
        np.testing.assert_array_equal(
            direct.action_space.low,
            fixture["action_space"]["low"],
        )
        np.testing.assert_array_equal(
            direct.action_space.high,
            fixture["action_space"]["high"],
        )
        assert np.all(np.isneginf(direct.observation_space.low))
        assert np.all(np.isposinf(direct.observation_space.high))

        for env in (direct, configured):
            observation, _ = env.reset(seed=fixture["seed"])
            np.testing.assert_allclose(observation, fixture["observations"][0])
            for index, action in enumerate(fixture["actions"]):
                observation, reward, terminated, truncated, info = env.step(
                    np.asarray(action, dtype=np.float32)
                )
                np.testing.assert_allclose(
                    observation,
                    fixture["observations"][index + 1],
                )
                assert reward == pytest.approx(fixture["rewards"][index])
                assert info["costs"] == fixture["costs"][index]
                assert terminated is fixture["terminated"][index]
                assert truncated is fixture["truncated"][index]
                assert (
                    info.get("termination_reason")
                    == fixture["termination_reason"][index]
                )
    finally:
        direct.close()
        configured.close()
