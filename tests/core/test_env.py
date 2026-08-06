from __future__ import annotations

import numpy as np
import pytest

from aiogym.core import (
    PlantConfig,
    ProcessControlEnv,
    ResolvedPlant,
    ScenarioPlugin,
    TaskSpec,
    make_env,
    register_scenario,
    unregister_scenario,
)


class ToyModel:
    scenario = "core-toy"

    def initial_state(self):
        return [0.0]

    def dynamics(self, state, action, disturbances=None):
        del disturbances
        return [float(action[0]) - float(state[0])]

    def outputs(self, state):
        return list(state)

    def action_schema(self):
        return [{"name": "u", "low": 0.0, "high": 1.0}]

    def state_schema(self):
        return [{"name": "x", "low": -10.0, "high": 10.0}]

    def default_action(self):
        return [0.0]

    def default_setpoint_vector(self):
        return [1.0]


def _reward(state, action, next_state, context):
    del state, action
    error = float(context["reference"][0]) - float(next_state[0])
    return -(error**2), {"tracking": -(error**2)}


def register_toy() -> ScenarioPlugin:
    unregister_scenario("core-toy")
    task = TaskSpec(
        id="core-toy/regulation",
        scenario="core-toy",
        objective="regulation",
        reward=_reward,
        metrics=("return", "iae"),
        primary_metric="iae",
        metric_direction="minimize",
    )
    plugin = ScenarioPlugin(
        id="core-toy",
        make_model=lambda plant: ToyModel(),
        default_plant=lambda: {
            "schema_version": "aiogym.plant.v2",
            "id": "core-toy-default",
            "scenario": "core-toy",
            "plant": {"gain": 1.0},
            "conditions": {
                "default": {
                    "id": "default",
                    "initial_state": [0.0],
                    "reference": [1.0],
                    "control_dt": 0.5,
                    "horizon": 3,
                },
                "short": {
                    "id": "short",
                    "initial_state": [0.0],
                    "reference": [1.0],
                    "control_dt": 0.5,
                    "horizon": 2,
                },
            },
            "default_condition": "default",
        },
        resolve_plant=lambda config: ResolvedPlant(
            config=config,
            parameters=config.plant,
            provenance={"source": "test"},
        ),
        tasks={"regulation": task},
    )
    register_scenario(plugin)
    return plugin


def test_make_env_uses_one_resolver_and_concrete_environment():
    register_toy()
    try:
        env = make_env("core-toy/regulation", preset="short")
        assert isinstance(env, ProcessControlEnv)
        observation, info = env.reset(seed=7)
        assert env.action_space.shape == (1,)
        assert env.observation_space.contains(observation)
        assert info["task_id"] == "core-toy/regulation"
        assert len(info["task_hash"]) == 64
        assert len(info["plant_hash"]) == 64

        next_observation, reward, terminated, truncated, step_info = env.step(
            np.asarray([0.5], dtype=np.float32)
        )
        assert np.allclose(next_observation, [0.19661458])
        assert reward == pytest.approx(-(1.0 - float(next_observation[0])) ** 2)
        assert not terminated
        assert not truncated
        assert step_info["commanded_action"].tolist() == [0.5]
        env.step(np.asarray([0.5], dtype=np.float32))
        assert env._step_index == 2
    finally:
        unregister_scenario("core-toy")


def test_env_rejects_cross_scenario_plant_and_non_space_action():
    register_toy()
    try:
        wrong = PlantConfig(
            id="wrong",
            scenario="different",
            plant={},
        )
        with pytest.raises(ValueError, match="does not match"):
            make_env("core-toy/regulation", plant=wrong)
        env = make_env("core-toy/regulation")
        env.reset(seed=0)
        with pytest.raises(ValueError, match="action_space"):
            env.step(np.asarray([1.5], dtype=np.float32))
    finally:
        unregister_scenario("core-toy")
