"""One-release converters for three-tank PlantConfig v1 declarations."""
from __future__ import annotations

import copy
from collections.abc import Mapping


def plant_v1_to_v2(
    source: Mapping,
    *,
    conditions: Mapping[str, Mapping] | None = None,
    default_condition: str | None = None,
) -> dict:
    raw = copy.deepcopy(dict(source))
    if raw.get("schema_version") == "aiogym.plant.v2":
        return raw
    if raw.get("schema_version", "aiogym.plant.v1") != "aiogym.plant.v1":
        raise ValueError("unsupported legacy PlantConfig schema")
    resolved_conditions = copy.deepcopy(dict(conditions or {}))
    if not resolved_conditions and _is_condition(raw.get("operating_point", {})):
        condition = copy.deepcopy(dict(raw["operating_point"]))
        condition.setdefault("id", "default")
        resolved_conditions[condition["id"]] = condition
    resolved_default = default_condition or (
        next(iter(resolved_conditions)) if resolved_conditions else ""
    )
    return {
        "schema_version": "aiogym.plant.v2",
        "id": raw["id"],
        "scenario": "three_tank" if raw.get("scenario") in {
            "cascade", "cascade_recirculating", "three_tank"
        } else raw["scenario"],
        "description": raw.get("description", ""),
        "plant": copy.deepcopy(dict(raw.get("plant", {}))),
        "conditions": resolved_conditions,
        "default_condition": resolved_default,
        "study": copy.deepcopy(dict(raw.get("study", {}))),
        "references": copy.deepcopy(list(raw.get("references", []))),
    }


def design_v1_to_plant_v2(source: Mapping) -> dict:
    raw = copy.deepcopy(dict(source))
    operation = raw["operation"]
    hydraulics = copy.deepcopy(raw["hydraulics"])
    hydraulics["nominal_circulation_flow_m3s"] = operation[
        "circulation_flow_m3s"
    ]
    condition = {
        "id": "commissioning",
        "initial_state": _interleave(
            operation["initial_levels_m"], operation["initial_temperatures_degC"]
        ),
        "reference": [
            *operation["target_levels_m"],
            *operation["target_temperatures_degC"],
        ],
        "control_dt": operation["control_dt_s"],
        "horizon": max(
            1, round(operation["duration_s"] / operation["control_dt_s"])
        ),
        "disturbances": {
            "t_amb": operation["ambient_temperature_degC"],
            "pump_flow_factor": 1.0,
            "heater_efficiency": 1.0,
            "heat_loss_factor": 1.0,
        },
        "reference_schedule": {},
        "disturbance_schedule": {},
        "observation": "controlled-output",
    }
    return {
        "schema_version": "aiogym.plant.v2",
        "id": "lab-three-tank-v1",
        "scenario": "three_tank",
        "description": raw.get("description", ""),
        "plant": {
            "topology": "recirculating_loop",
            "tanks": raw["tanks"],
            "hydraulics": hydraulics,
            "pump": raw["pump"],
            "heaters": raw["heaters"],
            "safety": {
                "temperature_trip_degC": raw["requirements"][
                    "temperature_trip_degC"
                ],
                "temperature_hard_limit_degC": raw["requirements"][
                    "temperature_hard_limit_degC"
                ],
            },
        },
        "conditions": {"commissioning": condition},
        "default_condition": "commissioning",
        "study": {
            "requirements": raw["requirements"],
            "uncertainty": raw.get("uncertainties", {}),
        },
        "references": raw.get("references", []),
    }


def legacy_scenario_to_plant_v2(
    *,
    plant_id: str,
    topology: str,
    parameters: Mapping,
    actuators: list[str],
    condition_id: str,
    condition: Mapping,
) -> dict:
    resolved_condition = copy.deepcopy(dict(condition))
    resolved_condition["id"] = condition_id
    return {
        "schema_version": "aiogym.plant.v2",
        "id": plant_id,
        "scenario": "three_tank",
        "description": f"Built-in {topology} three-tank plant",
        "plant": {
            "topology": topology,
            "parameters": copy.deepcopy(dict(parameters)),
            "actuators": list(actuators),
        },
        "conditions": {condition_id: resolved_condition},
        "default_condition": condition_id,
        "study": {},
        "references": [],
    }


def _is_condition(value: Mapping) -> bool:
    return {
        "initial_state", "reference", "control_dt", "horizon"
    }.issubset(value)


def _interleave(levels, temperatures):
    return [
        value
        for pair in zip(levels, temperatures)
        for value in (float(pair[0]), float(pair[1]))
    ]


__all__ = [
    "design_v1_to_plant_v2",
    "legacy_scenario_to_plant_v2",
    "plant_v1_to_v2",
]
