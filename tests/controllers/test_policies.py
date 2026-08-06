from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import aiogym.scenarios  # noqa: F401
from aiogym.controllers.base import make_controller
from aiogym.controllers.policies import SB3CheckpointPolicy
from aiogym.core import make_env, rollout


@pytest.mark.parametrize(
    ("task", "preset"),
    (
        ("quadruple/regulation", "minimum-phase"),
        ("cascade/regulation", "continuous-benchmark"),
        ("cascade_recirculating/regulation", "commissioning"),
    ),
)
@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_stable_scenario_pid_and_mpc_share_policy_contract(
    task, preset, controller_id
):
    env = make_env(task, preset=preset)
    try:
        policy = make_controller(controller_id, env=env)
        result = rollout(env, policy, seed=0, max_steps=2)
        assert len(result.transitions) == 2
        assert all(env.action_space.contains(row.action) for row in result.transitions)
        assert result.policy_metadata["action_contract"] == "env.action_space"
        assert np.isfinite(result.episode_return)
    finally:
        env.close()


@pytest.mark.parametrize(
    "plant_id",
    ("open-cascade-v1", "recirculating-h1-v1", "lab-three-tank-v1"),
)
@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_unified_three_tank_pid_and_mpc_contracts(plant_id, controller_id):
    env = make_env("three_tank/regulation", plant=plant_id)
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
        assert result.policy_metadata["interface_hash"] == env.identity.interface_hash
        if controller_id == "mpc":
            assert result.policy_metadata["initialization_status"] in {
                "default_action",
                "tracking_steady_state_action",
            }
    finally:
        env.close()


def test_name_bound_pid_rejects_unknown_actuator_before_rollout():
    env = make_env("three_tank/regulation", plant="recirculating-h1-v1")
    try:
        with pytest.raises(ValueError, match="unknown PID actuator"):
            make_controller(
                "pid",
                env=env,
                config={
                    "loops": [
                        {
                            "actuator": "not-installed",
                            "output": "level_1",
                            "pid": [1.0, 0.0, 0.0],
                        }
                    ]
                },
            )
    finally:
        env.close()


def test_hold_and_random_are_seeded_environment_action_policies():
    env = make_env("three_tank/regulation")
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
