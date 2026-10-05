from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

import aiogym
from aiogym.workflows._metadata import environment_metadata
from aiogym.scenarios.cascade import (
    as_hybrid_physical_policy,
    three_tank_pid_heater_control,
    three_tank_pid_temperature_control,
)


class _ConstantHeaterPolicy:
    def __init__(self, action):
        self.env = None
        self.action = np.asarray(action, dtype=np.float32)
        self.observations = []

    def reset(self, seed=None):
        del seed
        self.observations = []

    def act(self, observation, context):
        del context
        self.observations.append(np.asarray(observation).copy())
        return self.action.copy()

    def metadata(self) -> Mapping:
        return {"id": "constant_heaters", "kind": "test_policy"}


def test_hybrid_environment_exposes_heaters_and_applies_three_tank_pid():
    env = three_tank_pid_heater_control(aiogym.make_env("cascade"))
    try:
        observation, info = env.reset(seed=4)
        assert observation.shape == (20,)
        assert env.action_space.shape == (3,)
        assert env.observation_space.contains(observation)
        assert observation[-4:] == pytest.approx(info["hydraulic_action"])
        interface = environment_metadata(env)["policy_interface"]
        assert len(interface["observation"]) == 20
        assert [row["name"] for row in interface["action"]] == [
            row["name"] for row in env.unwrapped.model.action_schema()[-3:]
        ]
        assert [row["index"] for row in interface["action"]] == [0, 1, 2]

        heaters = np.asarray([0.2, 0.3, 0.4], dtype=np.float32)
        next_observation, reward, terminated, truncated, step_info = env.step(
            heaters
        )
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert step_info["policy_action"] == pytest.approx(heaters)
        assert step_info["commanded_action"][:4] == pytest.approx(
            step_info["hydraulic_action"]
        )
        assert step_info["commanded_action"][4:] == pytest.approx(heaters)
        assert step_info["applied_action"][4:] == pytest.approx([0.2, 0.0, 0.0])
        assert next_observation[-4:] == pytest.approx(
            step_info["next_hydraulic_action"]
        )
    finally:
        env.close()


def test_temperature_training_adds_errors_and_replaces_only_tracking_reward():
    env = three_tank_pid_temperature_control(aiogym.make_env("cascade"))
    try:
        observation, info = env.reset(seed=4)
        expected_error = (
            np.asarray(info["reference"])[3:] - np.asarray(info["y"])[3:]
        ) / 5.0
        assert observation.shape == (23,)
        interface = environment_metadata(env)["policy_interface"]
        assert len(interface["observation"]) == 23
        assert [row["kind"] for row in interface["observation"][-3:]] == ["error"] * 3
        assert observation[-3:] == pytest.approx(expected_error)
        assert env.observation_space.contains(observation)

        next_observation, reward, terminated, truncated, step_info = env.step(
            np.asarray([0.2, 0.3, 0.4], dtype=np.float32)
        )
        transition_error = (
            np.asarray(step_info["y"])[3:]
            - np.asarray(step_info["transition_reference"])[3:]
        ) / 5.0
        assert reward == pytest.approx(
            -env.unwrapped.control_dt * np.mean(transition_error**2)
        )
        assert step_info["experimental_reward"] == "temperature_tracking"
        assert step_info["physical_reward"] != pytest.approx(reward)
        assert next_observation[-3:] == pytest.approx(
            (np.asarray(step_info["reference"])[3:] - np.asarray(step_info["y"])[3:])
            / 5.0,
            abs=1e-6,
        )
        assert not terminated
        assert not truncated
    finally:
        env.close()


def test_hybrid_policy_adapter_returns_direct_physical_action():
    env = aiogym.make_env("cascade")
    heater_policy = _ConstantHeaterPolicy([0.2, 0.3, 0.4])
    policy = as_hybrid_physical_policy(heater_policy, env=env)
    try:
        observation, info = env.reset(seed=2)
        policy.reset(seed=2)
        action = policy.act(
            observation,
            {"reference": info["reference"]},
        )
        assert env.action_space.contains(action)
        assert action[4:] == pytest.approx([0.2, 0.3, 0.4])
        assert heater_policy.observations[-1].shape == (20,)
        assert heater_policy.observations[-1][-4:] == pytest.approx(action[:4])
        assert policy.metadata()["hydraulic_controller"]["kind"] == "matrix_pid"
    finally:
        policy.close()
        env.close()


def test_hybrid_policy_adapter_can_append_temperature_errors():
    env = aiogym.make_env("cascade")
    heater_policy = _ConstantHeaterPolicy([0.2, 0.3, 0.4])
    policy = as_hybrid_physical_policy(
        heater_policy,
        env=env,
        temperature_error_observation=True,
    )
    try:
        observation, info = env.reset(seed=2)
        policy.reset(seed=2)
        policy.act(
            observation,
            {"reference": info["reference"]},
        )
        augmented = heater_policy.observations[-1]
        assert augmented.shape == (23,)
        assert augmented[-3:] == pytest.approx(
            (np.asarray(info["reference"])[3:] - np.asarray(info["y"])[3:])
            / 5.0
        )
        assert policy.metadata()["temperature_error_observation"] is True
    finally:
        policy.close()
        env.close()


def test_hybrid_control_rejects_other_or_already_wrapped_environments():
    non_cascade = aiogym.make_env("three_tank")
    try:
        with pytest.raises(ValueError, match="cascade scenario"):
            three_tank_pid_heater_control(non_cascade)
    finally:
        non_cascade.close()

    hybrid = three_tank_pid_heater_control(aiogym.make_env("cascade"))
    try:
        with pytest.raises(ValueError, match="direct 16-observation"):
            three_tank_pid_heater_control(hybrid)
    finally:
        hybrid.close()
