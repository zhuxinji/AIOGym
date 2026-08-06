from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import aiogym.scenarios  # noqa: F401
from aiogym.core import PlantConfig, get_task, list_scenarios, list_tasks, make_env
from aiogym.scenarios.three_tank import design_v1_to_plant


GOLDEN = json.loads(
    (Path(__file__).with_name("golden") / "core-scenario-migration-v1.json").read_text(
        encoding="utf-8"
    )
)
DESIGN = (
    Path(__file__).resolve().parents[2]
    / "aiogym/scenarios/three_tank/default-design-v1.json"
)


@pytest.mark.parametrize(
    ("scenario", "preset"),
    (
        ("quadruple", "minimum-phase"),
        ("cascade", "continuous-benchmark"),
        ("cascade_recirculating", "commissioning"),
    ),
)
def test_stable_scenario_plugins_match_phase0_golden(scenario, preset):
    expected = GOLDEN[scenario]
    env = make_env(f"{scenario}/regulation", preset=preset)
    try:
        observation, info = env.reset(seed=1729)
        assert np.allclose(observation, expected["initial_observation"], atol=1e-8)
        assert env.observation_space.contains(observation)
        assert info["task_id"] == f"{scenario}/regulation"
        assert info["condition_id"] == preset
        rewards = []
        for _ in range(5):
            observation, reward, terminated, truncated, info = env.step(
                np.asarray(expected["action"], dtype=np.float32)
            )
            rewards.append(reward)
            assert not terminated
            assert not truncated
        assert np.allclose(rewards, expected["rewards"], atol=1e-12)
        assert np.allclose(observation, expected["final_observation"], atol=1e-8)
        assert len(info["task_hash"]) == 64
        assert len(info["plant_hash"]) == 64
    finally:
        env.close()


def test_cascade_tasks_share_one_physical_model_with_distinct_objectives():
    regulation = make_env("cascade/regulation", preset="continuous-benchmark")
    economic = make_env("cascade/economic", preset="continuous-benchmark")
    try:
        assert regulation.model._model.p == economic.model._model.p
        assert regulation.task.objective == "regulation"
        assert economic.task.objective == "economic"
        assert regulation.task.task_hash != economic.task.task_hash
    finally:
        regulation.close()
        economic.close()


def test_three_tank_is_a_registered_six_action_parameterized_scenario():
    legacy = json.loads(DESIGN.read_text(encoding="utf-8"))
    plant = PlantConfig.from_mapping(design_v1_to_plant(legacy))
    env = make_env("three_tank/regulation", plant=plant, preset="commissioning")
    try:
        observation, info = env.reset(seed=7)
        assert env.action_space.shape == (6,)
        assert env.observation_space.contains(observation)
        assert env.model.heater_mask == (True, True, True)
        assert info["plant_hash"] == plant.plant_hash
        action = np.asarray(env.model.default_action(), dtype=np.float32)
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        assert env.action_space.contains(action)
        assert env.observation_space.contains(next_observation)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        assert next_info["plant_hash"] == plant.plant_hash
    finally:
        env.close()


def test_registered_scenarios_are_vertical_core_scenarios():
    expected = {
        "cascade",
        "cascade_recirculating",
        "quadruple",
        "three_tank",
    }
    assert set(list_scenarios()) == expected
    assert "three_tank/regulation" in list_tasks()
    assert get_task("cascade/economic").primary_metric == "economic_objective"
