"""Canonical Case behavior for the open cascade process."""
from __future__ import annotations

import numpy as np
import pytest

import aiogym


CASE_NAMES = {
    "commissioning",
    "continuous-benchmark",
    "disturbance-rejection",
    "safety-recovery",
    "temperature-step",
}


def test_open_cascade_cases_are_discoverable():
    assert {f"cascade/{name}" for name in CASE_NAMES} <= set(
        aiogym.list_cases("cascade")
    )
    for case_id in aiogym.list_cases("cascade"):
        case = aiogym.load_case(case_id)
        assert case["scenario"] == "cascade"
        assert "goal" not in case
        assert "reward_spec" not in case


def test_commissioning_case_starts_cold_with_declared_schedule():
    case = aiogym.load_case("cascade/commissioning")
    env = aiogym.AIOGymEnv(
        "cascade",
        case=case,
        reward_spec="regulation-v1",
        episode_steps=2,
    )
    try:
        env.reset(seed=0)
        assert list(env.integ.x) == pytest.approx(case["initialization"]["state"])
        assert list(env.y_sp) == pytest.approx(case["setpoints"]["initial"])
        assert case["setpoints"]["schedule"]
    finally:
        env.close()


def test_temperature_step_case_targets_are_physically_bounded():
    case = aiogym.load_case("cascade/temperature-step")
    model = aiogym.make_model("cascade")
    low = np.asarray([bounds[0] for bounds in model.output_bounds.values()])
    high = np.asarray([bounds[1] for bounds in model.output_bounds.values()])
    targets = [
        case["setpoints"]["initial"],
        *(event["values"] for event in case["setpoints"]["schedule"]),
    ]
    assert all(np.all(np.asarray(target) >= low) for target in targets)
    assert all(np.all(np.asarray(target) <= high) for target in targets)


def test_disturbance_case_applies_feed_pump_loss():
    env = aiogym.AIOGymEnv(
        "cascade",
        case="disturbance-rejection",
        reward_spec="regulation-v1",
    )
    try:
        env.reset(seed=0)
        for _ in range(201):
            _, _, _, _, info = env.step(env.model.default_action())
        assert env._env()["pump_flow_factor"] < 1.0
    finally:
        env.close()


def test_safety_recovery_case_starts_inside_hard_limits():
    env = aiogym.AIOGymEnv(
        "cascade",
        case="safety-recovery",
        reward_spec="regulation-v1",
        episode_steps=2,
    )
    try:
        env.reset(seed=0)
        _, _, terminated, _, info = env.step(env.model.default_action())
        assert not terminated
        assert "termination_reason" not in info
    finally:
        env.close()


@pytest.mark.parametrize("controller", ["pid", "mpc"])
def test_controllers_complete_short_commissioning_case(controller):
    agent = aiogym.make_controller(controller, scenario="cascade")
    env = aiogym.AIOGymEnv(
        "cascade",
        case="commissioning",
        reward_spec="regulation-v1",
        episode_steps=2,
    )
    try:
        result = aiogym.evaluate_controller(agent, env, seed=0)
    finally:
        env.close()
    assert result["execution_status"] == "passed"
    assert result["controller_status"] == "ok"


def test_continuous_case_can_use_economic_goal():
    env = aiogym.AIOGymEnv(
        "cascade",
        case="continuous-benchmark",
        reward_spec="economic-v1",
        episode_steps=2,
    )
    try:
        result = aiogym.evaluate_controller(
            aiogym.make_controller(
                "oracle",
                scenario="cascade",
                config={"goal": "economic", "reward_spec": "economic-v1"},
            ),
            env,
            seed=0,
        )
    finally:
        env.close()
    assert result["goal"] == "economic"
    assert result["metric"] == "profit"
