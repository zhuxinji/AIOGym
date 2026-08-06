from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from aiogym.design import load_design_spec, validate_design_spec


EXAMPLE = (
    Path(__file__).parents[2]
    / "configs"
    / "design"
    / "cascade-recirculating-example-v1.json"
)


def _raw_example():
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_design_spec_is_normalized_and_hash_stable():
    first = load_design_spec(EXAMPLE)
    second = load_design_spec(_raw_example())

    assert first == second
    assert first["schema_version"] == "aiogym.design_spec.v1"
    assert len(first["design_hash"]) == 64
    assert [heater["tank"] for heater in first["heaters"]] == [1, 2, 3]
    assert load_design_spec(first) == first


def test_design_spec_rejects_invalid_hydraulic_and_safety_ordering():
    invalid_pump = _raw_example()
    invalid_pump["pump"]["shutoff_head_m"] = 1.0
    with pytest.raises(ValueError, match="shutoff_head_m"):
        validate_design_spec(invalid_pump)

    invalid_tank = _raw_example()
    invalid_tank["tanks"][0]["overflow_level_m"] = 0.3
    with pytest.raises(ValueError, match="high trip < overflow"):
        validate_design_spec(invalid_tank)


def test_design_spec_rejects_duplicate_heater_slots_and_unknown_fields():
    duplicate = _raw_example()
    duplicate["heaters"][1]["tank"] = 1
    with pytest.raises(ValueError, match="one heater slot"):
        validate_design_spec(duplicate)

    unknown = copy.deepcopy(_raw_example())
    unknown["magic"] = 1
    with pytest.raises(ValueError, match="unknown design fields"):
        validate_design_spec(unknown)
