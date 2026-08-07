from __future__ import annotations

import copy
from types import MappingProxyType

import pytest

from aiogym.core import PlantConfig, ResolvedPlant


def _raw_config() -> dict:
    return {
        "schema_version": "aiogym.plant.v2",
        "id": "immutable-test",
        "scenario": "three_tank",
        "description": "not part of plant identity",
        "plant": {
            "pump": {"max_flow_m3s": 0.01},
            "heaters": [{"tank": 1, "power_w": 1000.0}],
        },
        "conditions": {
            "nominal": {
                "id": "nominal",
                "initial_state": [0.1, 20.0],
                "reference": [0.2, 25.0],
                "control_dt": 1.0,
                "horizon": 10,
                "disturbances": {"ambient": 18.0},
                "disturbance_schedule": {"2": {"ambient": 17.0}},
            }
        },
        "default_condition": "nominal",
        "study": {"requirements": {"maximum_heatup_time_s": 100.0}},
        "references": [{"kind": "bom", "metadata": {"revision": 1}}],
    }


def test_inputs_and_nested_paths_are_defensively_immutable():
    raw = _raw_config()
    config = PlantConfig.from_mapping(raw)
    original_hash = config.config_hash

    raw["plant"]["pump"]["max_flow_m3s"] = 99.0
    raw["study"]["requirements"]["maximum_heatup_time_s"] = 999.0
    raw["references"][0]["metadata"]["revision"] = 2
    raw["conditions"]["nominal"]["disturbance_schedule"]["2"]["ambient"] = 99.0

    assert config.plant["pump"]["max_flow_m3s"] == 0.01
    assert config.study["requirements"]["maximum_heatup_time_s"] == 100.0
    assert config.references[0]["metadata"]["revision"] == 1
    assert config.conditions["nominal"].disturbance_schedule[2]["ambient"] == 17.0
    assert config.config_hash == original_hash

    with pytest.raises(TypeError):
        config.plant["pump"]["max_flow_m3s"] = 0.02
    with pytest.raises(TypeError):
        config.references[0]["metadata"]["revision"] = 3
    assert isinstance(config.plant["pump"], MappingProxyType)


def test_as_dict_is_a_fully_independent_mutable_copy():
    config = PlantConfig.from_mapping(_raw_config())
    payload = config.as_dict()
    payload["plant"]["pump"]["max_flow_m3s"] = 0.5
    payload["study"]["requirements"]["maximum_heatup_time_s"] = 1.0
    payload["references"][0]["metadata"]["revision"] = 4
    payload["conditions"]["nominal"]["disturbance_schedule"]["2"]["ambient"] = 5.0

    assert config.plant["pump"]["max_flow_m3s"] == 0.01
    assert config.study["requirements"]["maximum_heatup_time_s"] == 100.0
    assert config.references[0]["metadata"]["revision"] == 1
    assert config.conditions["nominal"].disturbance_schedule[2]["ambient"] == 17.0


def test_resolved_plant_parameters_and_provenance_are_deeply_immutable():
    parameters = {"pump": {"max_flow_m3s": 0.01}}
    provenance = {"source": {"kind": "test"}}
    resolved = ResolvedPlant(
        config=PlantConfig.from_mapping(_raw_config()),
        parameters=parameters,
        provenance=provenance,
    )
    parameters["pump"]["max_flow_m3s"] = 9.0
    provenance["source"]["kind"] = "changed"

    assert resolved.parameters["pump"]["max_flow_m3s"] == 0.01
    assert resolved.provenance["source"]["kind"] == "test"
    with pytest.raises(TypeError):
        resolved.parameters["pump"]["max_flow_m3s"] = 2.0


def test_hash_boundaries_and_equivalent_inputs():
    raw = _raw_config()
    base = PlantConfig.from_mapping(raw)
    equivalent = PlantConfig.from_mapping(copy.deepcopy(raw))
    assert equivalent.as_dict() == base.as_dict()

    physical_raw = copy.deepcopy(raw)
    physical_raw["plant"]["pump"]["max_flow_m3s"] = 0.02
    physical = PlantConfig.from_mapping(physical_raw)
    assert physical.plant_hash != base.plant_hash

    condition_raw = copy.deepcopy(raw)
    condition_raw["conditions"]["nominal"]["reference"] = [0.3, 25.0]
    condition = PlantConfig.from_mapping(condition_raw)
    assert condition.plant_hash == base.plant_hash
    assert condition.condition_hashes != base.condition_hashes
    assert condition.config_hash != base.config_hash

    study_raw = copy.deepcopy(raw)
    study_raw["study"]["requirements"]["maximum_heatup_time_s"] = 200.0
    study = PlantConfig.from_mapping(study_raw)
    assert study.plant_hash == base.plant_hash
    assert study.study_hash != base.study_hash
    assert study.config_hash != base.config_hash


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["condition_hashes"].update(extra="0" * 64),
        lambda payload: payload["condition_hashes"].update(nominal="0" * 64),
    ],
)
def test_declared_condition_hashes_are_strictly_validated(mutate):
    payload = PlantConfig.from_mapping(_raw_config()).as_dict()
    mutate(payload)
    with pytest.raises(ValueError, match="condition_hashes"):
        PlantConfig.from_mapping(payload)


def test_json_mapping_round_trip_preserves_all_hashes():
    config = PlantConfig.from_mapping(_raw_config())
    restored = PlantConfig.from_mapping(copy.deepcopy(config.as_dict()))
    assert restored.plant_hash == config.plant_hash
    assert restored.condition_hashes == config.condition_hashes
    assert restored.study_hash == config.study_hash
    assert restored.config_hash == config.config_hash
