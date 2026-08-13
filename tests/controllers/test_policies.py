from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import aiogym.scenarios  # noqa: F401
from aiogym.controllers.base import make_controller
from aiogym.controllers.policies import SB3CheckpointPolicy
from aiogym.core.env import make_env
from aiogym.core.rollout import rollout


@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_stable_scenario_pid_and_mpc_emit_environment_actions(controller_id):
    env = make_env("quadruple", benchmark="tracking")
    try:
        policy = make_controller(controller_id, env=env)
        result = rollout(env, policy, seed=0, max_steps=2)
        assert len(result.transitions) == 2
        assert all(env.action_space.contains(row.action) for row in result.transitions)
        assert np.isfinite(result.episode_return)
    finally:
        env.close()


@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_fixed_three_tank_pid_and_mpc_contracts(controller_id):
    env = make_env("three_tank", reward="regulation")
    try:
        result = rollout(
            env,
            make_controller(controller_id, env=env),
            seed=7,
            max_steps=2,
        )
        assert len(result.transitions) == 2
        assert all(env.action_space.contains(row.action) for row in result.transitions)
        assert np.isfinite(result.episode_return)
        if controller_id == "mpc":
            assert result.policy_metadata["initialization_status"] in {
                "default_action",
                "tracking_steady_state_action",
            }
    finally:
        env.close()


@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_tank3_tracking_pid_and_mpc_use_the_physical_interface(controller_id):
    env = make_env("three_tank", benchmark="tracking")
    try:
        result = rollout(
            env,
            make_controller(controller_id, env=env),
            seed=2,
            max_steps=120,
        )
        assert len(result.transitions) == 120
        assert not result.transitions[-1].terminated
        assert result.transitions[0].action.shape == (5,)
        assert result.transitions[0].info["applied_action"].shape == (5,)
    finally:
        env.close()


def test_three_tank_controller_config_does_not_route_on_benchmark():
    metadata = []
    for benchmark in ("tracking", "disturbance-rejection", "boundary-safety"):
        env = make_env("three_tank", benchmark=benchmark)
        try:
            metadata.append(
                (
                    make_controller("pid", env=env).metadata(),
                    make_controller("mpc", env=env).metadata(),
                )
            )
        finally:
            env.close()
    assert metadata[0] == metadata[1] == metadata[2]
    pid, mpc = metadata[0]
    assert max(abs(value) for row in pid["kp"] for value in row) == 32.0
    assert max(abs(value) for row in pid["ki"][:4] for value in row) == pytest.approx(
        0.002
    )
    assert pid["feedforward"] == "tracking_steady_state_action"
    assert mpc["feedforward_reseed"] == "setpoint_or_feedforward_change"
    assert mpc["horizon"] == 120
    assert mpc["steady_input_weight"][-1] == pytest.approx(0.015)


def test_name_bound_pid_rejects_unknown_actuator_before_rollout():
    env = make_env("three_tank", reward="regulation")
    try:
        with pytest.raises(ValueError, match="unknown PID actuator"):
            make_controller(
                "pid",
                env=env,
                config={
                    "loops": [
                        {
                            "actuator": "not-installed",
                            "output": "tank_1_level",
                            "pid": [1.0, 0.0, 0.0],
                        }
                    ]
                },
            )
    finally:
        env.close()


def test_mpc_can_reseed_when_disturbance_changes_steady_feedforward():
    env = make_env("three_tank", reward="regulation")
    try:
        policy = make_controller(
            "mpc",
            env=env,
            config={
                "P": 5,
                "move_supp": 1.0,
                "steady_input_weight": 1.0,
                "reseed_on_feedforward_change": True,
            },
        )
        observation, info = env.reset(seed=0)
        first = policy.act(observation, {"info": info})
        env.set_disturbances({"pump_flow_factor": 0.85})
        changed_info = {
            **info,
            "disturbance": {
                **info["disturbance"],
                "pump_flow_factor": 0.85,
            },
        }
        second = policy.act(observation, {"info": changed_info})
        assert second[0] > first[0]
    finally:
        env.close()


def test_pid_can_update_bias_from_steady_feedforward():
    env = make_env("three_tank", reward="regulation")
    try:
        policy = make_controller(
            "pid",
            env=env,
            config={"feedforward": "tracking_steady_state_action"},
        )
        observation, info = env.reset(seed=0)
        first = policy.act(observation, {"env": env, "info": info})
        env.set_disturbances({"pump_flow_factor": 0.85})
        second = policy.act(observation, {"env": env, "info": info})
        assert second[0] > first[0]
        assert policy.metadata()["feedforward"] == "tracking_steady_state_action"
    finally:
        env.close()


def test_mpc_accepts_per_actuator_regularization_weights():
    env = make_env("three_tank", reward="regulation")
    try:
        policy = make_controller(
            "mpc",
            env=env,
            config={
                "move_supp": [1.0, 1.0, 1.0, 1.0, 0.01],
                "steady_input_weight": [0.1, 0.1, 0.1, 0.1, 0.0],
            },
        )
        result = rollout(env, policy, seed=0, max_steps=2)
        assert len(result.transitions) == 2
        assert result.policy_metadata["move_supp"][-1] == pytest.approx(0.01)
    finally:
        env.close()


def test_hold_and_random_are_seeded_environment_action_policies():
    env = make_env("three_tank", reward="regulation")
    try:
        held = rollout(env, make_controller("hold", env=env), seed=2, max_steps=2)
        first = rollout(env, make_controller("random", env=env), seed=3, max_steps=2)
        second = rollout(env, make_controller("random", env=env), seed=3, max_steps=2)
        assert np.array_equal(held.transitions[0].action, held.transitions[1].action)
        assert all(
            np.array_equal(left.action, right.action)
            for left, right in zip(first.transitions, second.transitions)
        )
    finally:
        env.close()


def test_pid_acts_on_seeded_noisy_benchmark_observation():
    env = make_env("quadruple", benchmark="tracking")
    try:
        first = rollout(
            env,
            make_controller("pid", env=env),
            seed=3,
            max_steps=1,
        )
        second = rollout(
            env,
            make_controller("pid", env=env),
            seed=4,
            max_steps=1,
        )
        assert not np.array_equal(
            first.transitions[0].action,
            second.transitions[0].action,
        )
    finally:
        env.close()


def test_sb3_adapter_uses_predict_without_importing_sb3_at_module_import():
    code = (
        "import sys; import aiogym.controllers.policies; "
        "assert 'stable_baselines3' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

    class FakeModel:
        def predict(self, observation, deterministic=True):
            assert deterministic
            return np.asarray([0.25, 0.75], dtype=np.float32), None

    policy = SB3CheckpointPolicy(
        FakeModel(), algorithm="sac", checkpoint="fake.zip"
    )
    action = policy.act(np.zeros(3), {})
    assert action.tolist() == pytest.approx([0.25, 0.75])
    assert policy.metadata()["algorithm"] == "sac"
