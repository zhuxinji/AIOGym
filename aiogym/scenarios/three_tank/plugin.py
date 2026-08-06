"""PlantConfig validation and Task definitions for three_tank."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from aiogym.core import PlantConfig, PresetSpec, ResolvedPlant, ScenarioPlugin, TaskSpec
from aiogym.scenarios._legacy import regulation_reward

from .model import ThreeTankModel, _legacy_design_spec


_EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "configs/design/cascade-recirculating-example-v1.json"
)


def design_v1_to_plant(raw):
    return {
        "schema_version": "aiogym.plant.v1",
        "id": raw["id"],
        "scenario": "three_tank",
        "description": raw.get("description", ""),
        "plant": {
            "tanks": raw["tanks"],
            "heaters": raw["heaters"],
            "pump": raw["pump"],
            "hydraulics": raw["hydraulics"],
        },
        "operating_point": raw["operation"],
        "study": {
            "requirements": raw["requirements"],
            "uncertainty": raw.get("uncertainties", {}),
        },
        "references": raw.get("references", []),
    }


def default_plant():
    return design_v1_to_plant(json.loads(_EXAMPLE.read_text(encoding="utf-8")))


def resolve_plant(config: PlantConfig) -> ResolvedPlant:
    required = {"tanks", "heaters", "pump", "hydraulics"}
    unknown = set(config.plant) - required
    missing = required - set(config.plant)
    if missing or unknown:
        raise ValueError(
            f"three_tank plant fields missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    requirements = config.study.get("requirements", {})
    uncertainty = config.study.get("uncertainty", {})
    parameters = {
        **copy.deepcopy(dict(config.plant)),
        "operation": copy.deepcopy(dict(config.operating_point)),
        "requirements": copy.deepcopy(dict(requirements)),
        "uncertainties": copy.deepcopy(dict(uncertainty)),
    }
    resolved = ResolvedPlant(
        config=config,
        parameters=parameters,
        provenance={"source": "PlantConfig", "schema": config.schema_version},
    )
    from aiogym.design.spec import load_design_spec

    load_design_spec(_legacy_design_spec(resolved))
    return resolved


def _task():
    default = PlantConfig.from_mapping(default_plant())
    operation = default.operating_point
    horizon = max(1, round(operation["duration_s"] / operation["control_dt_s"]))
    return TaskSpec(
        id="three_tank/regulation",
        scenario="three_tank",
        objective="regulation",
        reward=regulation_reward,
        metrics=(
            "return",
            "tracking_iae",
            "tracking_ise",
            "settling_time",
            "overshoot",
            "constraint_violations",
            "energy",
        ),
        primary_metric="tracking_iae",
        metric_direction="minimize",
        horizon=horizon,
        control_dt=float(operation["control_dt_s"]),
        presets={
            "commissioning": PresetSpec("commissioning"),
            "regulation": PresetSpec("regulation"),
        },
        default_preset="commissioning",
        reference=tuple(operation["target_levels_m"])
        + tuple(operation["target_temperatures_degC"]),
    )


PLUGIN = ScenarioPlugin(
    id="three_tank",
    make_model=ThreeTankModel,
    default_plant=default_plant,
    resolve_plant=resolve_plant,
    tasks={"regulation": _task()},
    controller_defaults={
        "pid": {"commissioning": {"profile": "commissioning"}},
        "mpc": {},
    },
)

__all__ = ["PLUGIN", "default_plant", "design_v1_to_plant", "resolve_plant"]
