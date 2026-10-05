from __future__ import annotations

import numpy as np
import pytest

from aiogym.core.env import make_env
from aiogym.core.rollout import rollout
from tests.core.test_env import make_toy_scenario


class ConstantPolicy:
    env = None

    def __init__(self, value: float):
        self.value = value
        self.seed = None

    def reset(self, seed=None):
        self.seed = seed

    def act(self, observation, context):
        del observation, context
        return np.asarray([self.value], dtype=np.float32)

    def metadata(self):
        return {"id": "constant", "value": self.value, "seed": self.seed}


def test_rollout_is_the_complete_episode_loop():
    scenario = make_toy_scenario()
    result = rollout(make_env(scenario), ConstantPolicy(0.5), seed=11)
    assert len(result.transitions) == 3
    assert result.transitions[-1].truncated
    assert result.policy_metadata["seed"] == 11
    assert result.episode_return == pytest.approx(
        sum(row.reward for row in result.transitions)
    )
    assert [row.step_index for row in result.transitions] == [0, 1, 2]
    assert [row.physical_time for row in result.transitions] == [0.5, 1.0, 1.5]


def test_rollout_requires_direct_environment_actions():
    scenario = make_toy_scenario()
    with pytest.raises(ValueError, match="directly"):
        rollout(
            make_env(scenario),
            ConstantPolicy(-0.1),
            seed=0,
            max_steps=1,
        )
