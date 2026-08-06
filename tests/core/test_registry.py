from __future__ import annotations

import json

import pytest

from aiogym.core import (
    PlantConfig,
    PresetSpec,
    ResolvedPlant,
    ScenarioPlugin,
    TaskSpec,
    get_scenario,
    get_task,
    list_scenarios,
    list_tasks,
    register_scenario,
    stable_hash,
    unregister_scenario,
    write_json,
)


def _reward(state, action, next_state, context):
    del state, action, context
    return -abs(float(next_state[0]))


def _plugin():
    task = TaskSpec(
        id="registry-toy/regulation",
        scenario="registry-toy",
        objective="regulation",
        reward=_reward,
        metrics=("return",),
        primary_metric="return",
        metric_direction="maximize",
        horizon=2,
        control_dt=1.0,
        presets={"default": PresetSpec("default")},
    )
    return ScenarioPlugin(
        id="registry-toy",
        make_model=lambda plant: object(),
        default_plant=lambda: {
            "schema_version": "aiogym.plant.v1",
            "id": "registry-default",
            "scenario": "registry-toy",
            "plant": {},
        },
        resolve_plant=lambda config: ResolvedPlant(config, config.plant),
        tasks={"regulation": task},
    )


def test_registry_is_the_single_scenario_and_task_index():
    unregister_scenario("registry-toy")
    plugin = _plugin()
    register_scenario(plugin)
    try:
        assert get_scenario("registry-toy") is plugin
        assert get_task("registry-toy/regulation") is plugin.tasks["regulation"]
        assert "registry-toy" in list_scenarios()
        assert list_tasks(scenario="registry-toy") == ("registry-toy/regulation",)
        with pytest.raises(ValueError, match="already registered"):
            register_scenario(plugin)
    finally:
        unregister_scenario("registry-toy")


def test_plant_hash_and_atomic_json_are_canonical(tmp_path):
    first = PlantConfig.from_mapping(
        {
            "schema_version": "aiogym.plant.v1",
            "id": "plant-a",
            "scenario": "registry-toy",
            "plant": {"b": 2, "a": 1},
        }
    )
    second = PlantConfig.from_mapping(
        {
            "scenario": "registry-toy",
            "id": "plant-a",
            "plant": {"a": 1, "b": 2},
        }
    )
    assert first.plant_hash == second.plant_hash
    assert stable_hash(
        {"scenario": first.scenario, "plant": dict(first.plant)}
    ) == first.plant_hash
    assert stable_hash(first.as_dict(include_hash=False)) == first.config_hash
    target = write_json(tmp_path / "plant.json", first.as_dict())
    assert json.loads(target.read_text(encoding="utf-8"))["plant_hash"] == first.plant_hash
    with pytest.raises(FileExistsError):
        write_json(target, first.as_dict())
    with pytest.raises(ValueError, match="NaN or Infinity"):
        write_json(tmp_path / "invalid.json", {"value": float("nan")})
