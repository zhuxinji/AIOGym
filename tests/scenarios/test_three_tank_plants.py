from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiogym.core import PlantConfig
from aiogym.scenarios.three_tank.migration import (
    design_v1_to_plant_v2,
    plant_v1_to_v2,
)


PLANTS = Path(__file__).resolve().parents[2] / "aiogym/scenarios/three_tank/plants"
DESIGN_V1 = PLANTS.parent / "default-design-v1.json"


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


def test_design_v1_conversion_matches_packaged_lab_plant():
    source = json.loads(DESIGN_V1.read_text(encoding="utf-8"))
    expected = json.loads(
        (PLANTS / "lab-three-tank-v1.json").read_text(encoding="utf-8")
    )
    assert design_v1_to_plant_v2(source) == expected


def test_plant_v1_reader_converts_identity_without_hashing_notes_as_physics():
    converted = plant_v1_to_v2(
        {
            "schema_version": "aiogym.plant.v1",
            "id": "old",
            "scenario": "cascade",
            "description": "note",
            "plant": {"topology": "open_cascade", "gain": 2.0},
            "references": ["source"],
        }
    )
    config = PlantConfig.from_mapping(converted)
    changed = PlantConfig.from_mapping({**converted, "description": "new note"})
    assert config.scenario == "three_tank"
    assert config.plant_hash == changed.plant_hash
    assert config.config_hash != changed.config_hash
