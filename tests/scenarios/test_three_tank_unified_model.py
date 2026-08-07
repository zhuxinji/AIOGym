from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aiogym.core import PlantConfig, ProcessControlEnv, ResolvedPlant, TaskSpec
from aiogym.scenarios._shared import regulation_reward
from aiogym.scenarios.three_tank.equipment import ThreeTankDesignModel
from aiogym.scenarios.three_tank.model import ThreeTankModel
from aiogym.scenarios.three_tank.physics import ThreeTankPhysicsKernel
from aiogym.scenarios.three_tank.topologies import (
    OpenCascadeTopology,
    RecirculatingTopology,
)


ROOT = Path(__file__).resolve().parents[2]
PLANTS = ROOT / "aiogym/scenarios/three_tank/plants"
GOLDEN = json.loads(
    (Path(__file__).with_name("golden") / "three-tank-unification-phase0-v1.json")
    .read_text(encoding="utf-8")
)


def _environment(plant_id):
    config = PlantConfig.from_mapping(
        json.loads((PLANTS / f"{plant_id}.json").read_text(encoding="utf-8"))
    )
    plant = ResolvedPlant(config=config, parameters={}, provenance={"source": "test"})
    condition = config.conditions[config.default_condition]
    task = TaskSpec(
        id="three_tank/regulation",
        scenario="three_tank",
        objective="regulation",
        reward=regulation_reward,
        metrics=("return", "tracking_iae"),
        primary_metric="tracking_iae",
        metric_direction="minimize",
        reward_id="normalized-tracking-mse-v1",
        metric_suite_id="regulation-core-v1",
    )
    return ProcessControlEnv(ThreeTankModel(plant), task, plant, condition)


@pytest.mark.parametrize("plant_id", tuple(GOLDEN["cases"]))
def test_unified_model_preserves_phase0_five_step_numerics(plant_id):
    expected = GOLDEN["cases"][plant_id]
    env = _environment(plant_id)
    try:
        observation, _ = env.reset(seed=GOLDEN["seed"])
        assert observation == pytest.approx(expected["initial_observation"], abs=1e-8)
        action = np.asarray(expected["fixed_action"], dtype=np.float32)
        for row in expected["steps"]:
            observation, reward, terminated, truncated, info = env.step(action)
            state = np.asarray(info["true_state"], dtype=float)
            energy_kw = env.model.action_energy_kw(
                action, state, info["disturbance"]
            )
            assert state == pytest.approx(row["state"], abs=1e-12)
            assert observation == pytest.approx(row["observation"], abs=1e-8)
            assert reward == pytest.approx(row["reward"], abs=1e-12)
            assert energy_kw == pytest.approx(row["energy_kw"], abs=1e-12)
            assert info["constraint_costs"] == row["constraints"]
            assert list(info.get("protection_events", ())) == row[
                "protection_events"
            ]
            assert not terminated
            assert not truncated
    finally:
        env.close()


def test_unified_model_uses_two_topologies_and_variable_action_schemas():
    expected = {
        "open-cascade-v1": ("open_cascade", 7),
        "recirculating-h1-v1": ("recirculating_loop", 4),
        "lab-three-tank-v1": ("recirculating_loop", 6),
    }
    for plant_id, (topology, action_dim) in expected.items():
        env = _environment(plant_id)
        try:
            assert env.model.topology == topology
            assert env.action_space.shape == (action_dim,)
            assert [row["name"] for row in env.model.state_schema()] == [
                "h1", "T1", "h2", "T2", "h3", "T3"
            ]
        finally:
            env.close()


def test_topologies_share_one_physics_kernel_without_per_plant_model_copies():
    assert issubclass(OpenCascadeTopology, ThreeTankPhysicsKernel)
    assert issubclass(RecirculatingTopology, ThreeTankPhysicsKernel)
    assert issubclass(ThreeTankDesignModel, RecirculatingTopology)
    shared_methods = {
        "_assemble_dynamics",
        "_effective_action",
        "controlled_output",
        "display_outputs",
        "disturbance_vector",
        "integral_observation_limits",
        "mpc_init",
    }
    for topology in (
        OpenCascadeTopology,
        RecirculatingTopology,
        ThreeTankDesignModel,
    ):
        assert shared_methods.isdisjoint(topology.__dict__)
