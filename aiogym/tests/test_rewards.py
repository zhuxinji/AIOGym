"""Canonical RewardSpec and reward-engine invariants."""
from __future__ import annotations

import numpy as np
import pytest

from aiogym import make_env as public_make_env
from aiogym.tests._env import make_test_env as make_env
from aiogym.rewards import RewardScaleWrapper, get_reward_spec, list_reward_specs


def _env(spec: str):
    return make_env(
        "cstr",
        reward_spec=spec,
        episode_steps=4,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
        randomize_plant=False,
        plant_drift=False,
    )


def test_registry_contains_only_canonical_specs():
    assert list_reward_specs() == ("economic-v1", "regulation-v1")
    assert get_reward_spec("economic-v1").goal == "economic"
    assert get_reward_spec("regulation-v1").goal == "regulation"


def test_reward_spec_does_not_change_state_trajectory():
    envs = [_env(spec) for spec in list_reward_specs()]
    try:
        observations = [env.reset(seed=41)[0] for env in envs]
        np.testing.assert_array_equal(observations[0], observations[1])
        for action in (
            np.array([0.2, 0.8], dtype=np.float32),
            np.array([0.8, 0.2], dtype=np.float32),
        ):
            transitions = [env.step(action) for env in envs]
            np.testing.assert_array_equal(
                transitions[0][0],
                transitions[1][0],
            )
            np.testing.assert_array_equal(envs[0].integ.x, envs[1].integ.x)
            assert transitions[0][2:4] == transitions[1][2:4]
    finally:
        for env in envs:
            env.close()


@pytest.mark.parametrize("spec", list_reward_specs())
def test_reward_decomposition_is_exact(spec):
    env = _env(spec)
    try:
        env.reset(seed=0)
        state = list(env.integ.x)
        result = env.evaluate_transition(
            state,
            np.array([0.3, 0.7], dtype=np.float32),
            state,
        )
    finally:
        env.close()
    assert result.goal_reward == pytest.approx(sum(result.reward_terms.values()))
    assert result.scalar_reward == pytest.approx(
        result.goal_reward
        - sum(result.applied_cost_penalties.values())
        - result.terminal_failure_cost
    )
    assert result.info["reward_spec_id"] == spec


def test_reward_scale_wrapper_changes_only_returned_scalar():
    base = _env("regulation-v1")
    wrapped_base = _env("regulation-v1")
    wrapped = RewardScaleWrapper(wrapped_base, 0.1)
    try:
        base_obs, _ = base.reset(seed=12)
        wrapped_obs, _ = wrapped.reset(seed=12)
        np.testing.assert_array_equal(base_obs, wrapped_obs)
        action = np.array([0.3, 0.7], dtype=np.float32)
        base_transition = base.step(action)
        wrapped_transition = wrapped.step(action)
    finally:
        base.close()
        wrapped.close()
    np.testing.assert_array_equal(base_transition[0], wrapped_transition[0])
    assert wrapped_transition[1] == pytest.approx(0.1 * base_transition[1])
    assert wrapped_transition[2:4] == base_transition[2:4]
    assert wrapped_transition[4]["unscaled_reward"] == base_transition[1]


def test_custom_stage_reward_is_not_in_the_stable_environment_api():
    with pytest.raises(TypeError, match="custom_stage_reward"):
        public_make_env(
            "cstr",
            custom_stage_reward=lambda *_: 3.5,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reward_mode": "tracking"},
        {"reward_scale": 0.03},
        {"tracking_q_y": [1.0]},
        {"tracking_r_move": 0.1},
    ],
)
def test_retired_reward_options_are_rejected(kwargs):
    with pytest.raises(TypeError):
        public_make_env("cstr", **kwargs)
