"""Smoke tests for the crystallization backend scenario."""
from __future__ import annotations

import numpy as np
import pytest

from aiogym.controllers import make_controller
from aiogym.tests._env import make_test_env as make_env
from aiogym.evaluation.execution import evaluate_controller, rollout_controller
from aiogym import list_scenarios
from aiogym.models import make_model


def _make_env(**kwargs):
    return make_env(
        "crystallization",
        action_mode="actuator",
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
        **kwargs,
    )


def test_registered_model_contract():
    assert "crystallization" in list_scenarios()
    model = make_model("crystallization")
    assert model.scenario == "crystallization"
    assert model.action_dim() == 1
    assert len(model.initial_state()) == 5


def test_reset_contract():
    env = _make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (10,)
    assert env.observation_space.shape == (10,)
    assert env.action_space.shape == (1,)
    assert np.all(np.isfinite(obs))
    assert np.allclose(obs[-3:], [1.0, 1.0, 0.0])


def test_action_to_temperature_mapping():
    model = make_model("crystallization")
    for aT, Tc in ((0.0, 30.0), (0.5, 35.0), (1.0, 40.0)):
        assert np.isclose(model.action_to_tc([aT]), Tc)


def test_unified_reward_modes_are_finite():
    model = make_model("crystallization")
    assert not callable(getattr(model, "reward_terms", None))
    for reward_spec in ("regulation-v1", "economic-v1"):
        env = _make_env(reward_spec=reward_spec, crystal_ln_sp=10.5, crystal_cv_sp=0.85, episode_steps=3)
        obs, _ = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert not terminated
        assert not truncated
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert np.isfinite(info["track"])
        assert "goal_reward" in info


@pytest.mark.oracle
def test_tracking_controllers_build():
    pid = make_controller("pid", scenario="crystallization")
    mpc = make_controller("mpc", scenario="crystallization")
    oracle = make_controller("oracle", scenario="crystallization")
    assert pid.metadata()["scenario"] == "crystallization"
    assert pid.metadata()["loops"][0]["y_index"] == 1
    assert pid.metadata()["loops"][0]["reverse"] is True
    assert mpc.metadata()["scenario"] == "crystallization"
    assert mpc.metadata()["horizon"] == 2
    assert oracle.metadata()["scenario"] == "crystallization"
    assert oracle.metadata()["horizon"] == 4
    assert oracle.metadata()["goal"] == "economic"
    assert oracle.metadata()["reward_spec_id"] == "economic-v1"
    assert "mode" not in oracle.metadata()


def test_crystallization_mpc_uses_affine_output_linearization():
    env = _make_env(
        reward_spec="regulation-v1",
        episode_steps=3,
        control_dt=0.5,
    )
    try:
        rollout = rollout_controller(
            make_controller("mpc", scenario="crystallization"),
            env,
            seed=0,
        )
    finally:
        env.close()
    actions = [row["action"][0] for row in rollout["rollout"]]
    assert actions[0] != 0.0
    assert actions != [0.0] * len(actions)


def test_pid_tracking_rollout():
    env = _make_env(
        reward_spec="regulation-v1",
        episode_steps=3,
        control_dt=1.0,
        randomize_plant=False,
        plant_drift=False,
    )
    try:
        result = evaluate_controller(
            make_controller("pid", scenario="crystallization"),
            env,
            episodes=1,
            seed=0,
        )
    finally:
        env.close()
    assert result["controller_name"] == "PID"
    assert result["metric"] == "regulation_cost"
    assert np.isfinite(result["tracking_cost"])
    assert np.isfinite(result["tracking_mse"])
    assert np.isfinite(result["tracking_iae"])
    assert result["controller"]["scenario"] == "crystallization"


def test_nominal_rollout_no_nan():
    env = _make_env(episode_steps=60)
    obs, _ = env.reset(seed=0)
    info = {}
    for _ in range(60):
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert not terminated
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        assert np.isfinite(info["Ln"])
        assert np.isfinite(info["CV"])
        assert np.isfinite(info["Tc"])
        assert 30.0 <= info["Tc"] <= 40.0
    assert truncated
    for key in ("Ln", "CV", "Tc", "c", "S", "Ceq", "B0", "Ginf"):
        assert key in info


def test_custom_setpoint_observation():
    env = _make_env(crystal_ln_sp=10.5, crystal_cv_sp=0.85)
    obs, _ = env.reset(seed=0)
    assert np.isclose(obs[5], 0.85)
    assert np.isclose(obs[6], 10.5)


def test_random_target_observation():
    env = _make_env(
        crystal_random_targets=True,
        crystal_ln_range=(10.25, 10.25),
        crystal_cv_range=(0.82, 0.82),
    )
    obs, _ = env.reset(seed=0)
    assert np.isclose(obs[5], 0.82)
    assert np.isclose(obs[6], 10.25)
