"""Goal-only evaluation after removing Objective/Protocol/Suite runtime."""
from __future__ import annotations

import importlib.util

import pytest

import aiogym
from aiogym.tests._env import make_test_env
from aiogym.controllers import make_controller
from aiogym.evaluation import evaluate_controller, rollout_controller


@pytest.mark.parametrize(
    "module_name",
    (
        "aiogym.evaluation.cases",
        "aiogym.evaluation.objective_specs",
        "aiogym.evaluation.protocols",
        "aiogym.evaluation.suite",
        "aiogym.evaluation.execution.benchmark",
        "aiogym.evaluation.execution.runner",
    ),
)
def test_retired_runtime_modules_are_deleted(module_name):
    assert importlib.util.find_spec(module_name) is None


def test_evaluation_uses_goal_and_reward_spec_only():
    env = make_test_env(
        "cstr",
        reward_spec="regulation-v1",
        episode_steps=2,
        auto_events=False,
        randomize=False,
    )
    controller = make_controller("pid", scenario="cstr")
    try:
        result = evaluate_controller(
            controller,
            env,
            seed_list=[11],
            goal_specification="regulation",
        )
    finally:
        env.close()

    assert result["goal"] == "regulation"
    assert result["reward_spec_id"] == "regulation-v1"
    assert result["metric"] == "regulation_cost"
    assert "objective" not in result
    assert "objective_spec" not in result
    assert "protocol" not in result


def test_goal_must_match_environment_reward_spec():
    env = make_test_env(
        "cstr",
        reward_spec="regulation-v1",
        episode_steps=1,
    )
    try:
        with pytest.raises(ValueError, match="does not match"):
            evaluate_controller(
                make_controller("pid", scenario="cstr"),
                env,
                goal_specification="economic",
            )
    finally:
        env.close()


def test_rollout_metadata_has_no_objective_or_protocol():
    env = make_test_env(
        "cstr",
        reward_spec="regulation-v1",
        episode_steps=1,
    )
    try:
        rollout = rollout_controller(
            make_controller("pid", scenario="cstr"),
            env,
            seed=3,
        )
    finally:
        env.close()

    assert rollout["goal"] == "regulation"
    assert rollout["reward_spec_id"] == "regulation-v1"
    assert "objective" not in rollout
    assert "protocol" not in rollout
