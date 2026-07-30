"""Canonical reward surface after deleting legacy engines and scorers."""
from __future__ import annotations

import importlib.util

import pytest

import aiogym
from aiogym.rewards import get_reward_spec, list_reward_specs


def test_only_canonical_reward_specs_exist():
    assert list_reward_specs() == ("economic-v1", "regulation-v1")
    for retired in (
        "legacy-kpi-v1",
        "legacy-economic-v1",
        "legacy-tracking-v1",
    ):
        with pytest.raises(ValueError, match="reward_spec must be one of"):
            get_reward_spec(retired)


@pytest.mark.parametrize(
    "module_name",
    (
        "aiogym.rewards.legacy",
        "aiogym.evaluation.metrics.kpi",
        "aiogym.evaluation.objectives",
    ),
)
def test_legacy_reward_modules_are_deleted(module_name):
    assert importlib.util.find_spec(module_name) is None


@pytest.mark.parametrize(
    "retired_option",
    (
        {"reward_mode": "tracking"},
        {"reward_scale": 0.1},
        {"tracking_q_y": 2.0},
        {"tracking_r_move": 0.1},
        {"w_prod": 2.0},
    ),
)
def test_environment_rejects_retired_reward_options(retired_option):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        aiogym.make_env("cstr", **retired_option)


def test_reward_result_has_one_canonical_scalar_field():
    env = aiogym.make_env(
        config={
            "scenario": "cstr",
            "environment": {"episode_steps": 1},
        }
    )
    try:
        env.reset(seed=4)
        result = env.evaluate_transition(
            env.integ.x,
            env.model.default_action(),
            env.integ.x,
        )
    finally:
        env.close()

    assert isinstance(result.scalar_reward, float)
    assert not hasattr(result, "reward")
    assert not hasattr(result, "kpi")
    assert not hasattr(env, "scorer")
