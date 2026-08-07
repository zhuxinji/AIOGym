"""The single public ScenarioPlugin for all supported three-tank plants."""
from __future__ import annotations

import json
from pathlib import Path

from aiogym.core import PlantConfig, ResolvedPlant, ScenarioPlugin, TaskSpec
from aiogym.scenarios._shared import economic_reward, regulation_reward

from .model import ThreeTankModel
from .controllers import resolve_controller_profile
from .study import ThreeTankStudyProvider


_PLANTS = Path(__file__).with_name("plants")


def _load_builtin(plant_id: str):
    return json.loads((_PLANTS / f"{plant_id}.json").read_text(encoding="utf-8"))


BUILT_IN_PLANTS = {
    plant_id: (lambda plant_id=plant_id: _load_builtin(plant_id))
    for plant_id in (
        "open-cascade-v1",
        "recirculating-h1-v1",
        "lab-three-tank-v1",
    )
}


def default_plant():
    return BUILT_IN_PLANTS["lab-three-tank-v1"]()


def resolve_plant(config: PlantConfig) -> ResolvedPlant:
    topology = config.plant.get("topology")
    if topology not in {"open_cascade", "recirculating_loop"}:
        raise ValueError(
            "three_tank plant topology must be open_cascade or recirculating_loop"
        )
    if "parameters" in config.plant:
        allowed = {"topology", "parameters", "actuators"}
        required = allowed
    else:
        allowed = {"topology", "tanks", "hydraulics", "pump", "heaters", "safety"}
        required = allowed
    missing = required - set(config.plant)
    unknown = set(config.plant) - allowed
    if missing or unknown:
        raise ValueError(
            f"three_tank plant fields missing={sorted(missing)} "
            f"unknown={sorted(unknown)}"
        )
    resolved = ResolvedPlant(
        config=config,
        parameters=dict(config.plant),
        provenance={"source": "PlantConfig", "schema": config.schema_version},
    )
    # Compilation validates the v2 equipment declaration before an environment
    # is returned.
    ThreeTankModel(resolved)
    return resolved


def _regulation_task():
    return TaskSpec(
        id="three_tank/regulation",
        scenario="three_tank",
        objective="regulation",
        revision=1,
        reward_id="normalized-tracking-mse-v1",
        metric_suite_id="regulation-core-v1",
        reward=regulation_reward,
        reward_term_names=("tracking_error", "slew", "effort"),
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
        required_capabilities=("tracking",),
    )


def _economic_task():
    return TaskSpec(
        id="three_tank/economic",
        scenario="three_tank",
        objective="economic",
        revision=1,
        reward_id="production-minus-energy-v1",
        metric_suite_id="economic-core-v1",
        reward=economic_reward,
        reward_term_names=("product_value", "energy_cost"),
        metrics=("return", "economic_objective", "energy", "constraint_violations"),
        primary_metric="economic_objective",
        metric_direction="maximize",
        required_capabilities=("product_flow", "energy"),
    )


PLUGIN = ScenarioPlugin(
    id="three_tank",
    make_model=ThreeTankModel,
    default_plant="lab-three-tank-v1",
    resolve_plant=resolve_plant,
    built_in_plants=BUILT_IN_PLANTS,
    tasks={"regulation": _regulation_task(), "economic": _economic_task()},
    controller_defaults={},
    resolve_controller_profile=resolve_controller_profile,
    study_provider=ThreeTankStudyProvider(),
)


__all__ = ["BUILT_IN_PLANTS", "PLUGIN", "default_plant", "resolve_plant"]
