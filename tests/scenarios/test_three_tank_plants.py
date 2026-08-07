from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiogym.core import PlantConfig, make_env
PLANTS = Path(__file__).resolve().parents[2] / "aiogym/scenarios/three_tank/plants"


@pytest.mark.parametrize(
    ("plant_id", "topology", "action_dim", "condition_id"),
    (
        ("open-cascade-v1", "open_cascade", 7, "continuous-benchmark"),
        ("recirculating-h1-v1", "recirculating_loop", 4, "commissioning"),
        ("lab-three-tank-v1", "recirculating_loop", 6, "commissioning"),
    ),
)
def test_builtin_plant_v2_declarations_are_complete(
    plant_id, topology, action_dim, condition_id
):
    raw = json.loads((PLANTS / f"{plant_id}.json").read_text(encoding="utf-8"))
    config = PlantConfig.from_mapping(raw)
    condition = config.conditions[condition_id]
    assert config.schema_version == "aiogym.plant.v2"
    assert config.id == plant_id
    assert config.scenario == "three_tank"
    assert config.plant["topology"] == topology
    actuator_count = (
        len(config.plant["actuators"])
        if "actuators" in config.plant
        else 3 + len(config.plant.get("heaters", ()))
    )
    assert actuator_count == action_dim
    assert config.default_condition == condition_id
    assert len(condition.initial_state) == 6
    assert len(condition.reference) == 6
    assert condition.horizon > 0
    assert condition.control_dt > 0
    assert len(config.plant_hash) == 64
    assert len(config.study_hash) == 64
    assert len(config.config_hash) == 64


@pytest.mark.parametrize(
    ("plant_id", "action_dim"),
    (
        ("open-cascade-v1", 7),
        ("recirculating-h1-v1", 4),
        ("lab-three-tank-v1", 6),
    ),
)
def test_builtin_plant_and_condition_resolver_builds_identity(plant_id, action_dim):
    env = make_env("three_tank/regulation", plant=plant_id)
    try:
        _, info = env.reset(seed=3)
        assert env.action_space.shape == (action_dim,)
        assert info["condition_id"] == env.plant.config.default_condition
        assert info["condition_hash"] == env.condition.condition_hash
        assert info["interface_hash"] == env.identity.interface_hash
        assert info["env_hash"] == env.identity.env_hash
    finally:
        env.close()


def test_condition_validation_and_capability_fail_before_rollout():
    with pytest.raises(ValueError, match="requires capability product_flow"):
        make_env("three_tank/economic", plant="recirculating-h1-v1")
    with pytest.raises(KeyError, match="available: commissioning"):
        make_env(
            "three_tank/regulation",
            plant="recirculating-h1-v1",
            condition="unknown",
        )
    with pytest.raises(ValueError, match="initial_state must contain 6"):
        make_env(
            "three_tank/regulation",
            plant="recirculating-h1-v1",
            condition={
                "id": "bad",
                "initial_state": [0.2],
                "reference": [0.2] * 6,
                "control_dt": 1.0,
                "horizon": 2,
            },
        )
    with pytest.raises(ValueError, match="unknown disturbances"):
        make_env(
            "three_tank/regulation",
            plant="recirculating-h1-v1",
            condition={
                "id": "bad-disturbance",
                "initial_state": [0.2, 20.0] * 3,
                "reference": [0.2] * 3 + [20.0] * 3,
                "control_dt": 1.0,
                "horizon": 2,
                "disturbances": {"imaginary": 1.0},
            },
        )
