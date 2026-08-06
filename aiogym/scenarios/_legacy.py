"""Temporary behavior-preserving adapter while numerical models move vertically."""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from aiogym.core import PlantConfig, PresetSpec, ResolvedPlant, ScenarioPlugin, TaskSpec


LEGACY_IDS = {"cascade_recirculating": "cascade-recirculating"}


def legacy_id(scenario: str) -> str:
    return LEGACY_IDS.get(scenario, scenario)


def _schema(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        bounds = row.pop("bounds", (None, None))
        low, high = bounds if bounds is not None else (None, None)
        row.setdefault("name", f"value_{index}")
        row["low"] = -np.inf if low is None else float(low)
        row["high"] = np.inf if high is None else float(high)
        converted.append(row)
    return converted


class LegacyProcessModel:
    """Expose a legacy numerical model through the new ProcessModel contract."""

    def __init__(self, scenario: str, plant: ResolvedPlant) -> None:
        from aiogym.models.registry import apply_model_params, make_model

        self.scenario = scenario
        self.legacy_scenario = legacy_id(scenario)
        self.plant = plant
        self._model = apply_model_params(
            make_model(self.legacy_scenario), dict(plant.parameters)
        )

    def initial_state(self):
        return self._model.initial_state()

    def dynamics(self, state, action, disturbances=None):
        return self._model.dynamics(state, action, disturbances or {})

    def outputs(self, state):
        return self._model.controlled_output(state)

    def action_schema(self):
        return _schema(self._model.action_schema())

    def state_schema(self):
        return _schema(self._model.state_schema())

    def default_action(self):
        return self._model.default_action()

    def default_setpoint_vector(self):
        return self._model.default_setpoint_vector()

    def build_env(self, *, task, preset, plant):
        from aiogym._environment.factory import make_env

        case = preset.config.get("legacy_case")
        declaration = {
            "scenario": self.legacy_scenario,
            "case": case,
            "reward_spec": f"{task.objective}-v1",
            "info_level": "full",
            "environment": {"model_params": dict(plant.parameters)},
        }
        return make_env(config=declaration)

    def __getattr__(self, name: str):
        return getattr(self._model, name)


def regulation_reward(state, action, next_state, context):
    del state, action
    reference = np.asarray(context["reference"], dtype=float)
    output = np.asarray(next_state, dtype=float)[: reference.size]
    error = output - reference
    iae = float(np.sum(np.abs(error)))
    ise = float(np.dot(error, error))
    return -ise, {"tracking_iae": iae, "tracking_ise": ise}


def economic_reward(state, action, next_state, context):
    del state, next_state, context
    energy = float(np.sum(np.asarray(action, dtype=float) ** 2))
    return -energy, {"energy": energy, "economic_objective": -energy}


def default_plant(scenario: str) -> dict[str, Any]:
    from aiogym.models.registry import make_model

    model = make_model(legacy_id(scenario))
    return {
        "schema_version": "aiogym.plant.v1",
        "id": f"{scenario}-default-v1",
        "scenario": scenario,
        "description": f"Built-in default parameters for {scenario}",
        "plant": {"parameters": copy.deepcopy(model.p)},
        "operating_point": {},
        "study": {},
        "references": [],
    }


def resolve_plant(config: PlantConfig) -> ResolvedPlant:
    unknown = set(config.plant) - {"parameters"}
    if unknown:
        raise ValueError(f"unknown {config.scenario} plant fields: {sorted(unknown)}")
    parameters = dict(config.plant.get("parameters", {}))
    from aiogym.models.registry import apply_model_params, make_model

    validated = apply_model_params(make_model(legacy_id(config.scenario)), parameters)
    return ResolvedPlant(
        config=config,
        parameters=copy.deepcopy(validated.p),
        provenance={"source": "builtin-model-parameters"},
    )


def build_plugin(
    scenario: str,
    *,
    model_factory=None,
    presets: Sequence[str] = (),
    default_preset: str = "default",
    economic: bool = False,
    horizon: int = 600,
    control_dt: float = 1.0,
    controller_defaults: Mapping[str, Mapping[str, Any]] | None = None,
) -> ScenarioPlugin:
    preset_specs = {
        name: PresetSpec(name, {"legacy_case": name}) for name in presets
    }
    if not preset_specs:
        preset_specs = {"default": PresetSpec("default")}
    regulation = TaskSpec(
        id=f"{scenario}/regulation",
        scenario=scenario,
        objective="regulation",
        reward=regulation_reward,
        metrics=(
            "return",
            "tracking_iae",
            "tracking_ise",
            "constraint_violations",
            "energy",
        ),
        primary_metric="tracking_iae",
        metric_direction="minimize",
        horizon=horizon,
        control_dt=control_dt,
        presets=preset_specs,
        default_preset=default_preset,
    )
    tasks = {"regulation": regulation}
    if economic:
        tasks["economic"] = TaskSpec(
            id=f"{scenario}/economic",
            scenario=scenario,
            objective="economic",
            reward=economic_reward,
            metrics=(
                "return",
                "economic_objective",
                "energy",
                "constraint_violations",
            ),
            primary_metric="economic_objective",
            metric_direction="maximize",
            horizon=horizon,
            control_dt=control_dt,
            presets=preset_specs,
            default_preset=default_preset,
        )
    return ScenarioPlugin(
        id=scenario,
        make_model=model_factory or (lambda plant: LegacyProcessModel(scenario, plant)),
        default_plant=lambda: default_plant(scenario),
        resolve_plant=resolve_plant,
        tasks=tasks,
        controller_defaults=dict(controller_defaults or {}),
    )


__all__ = ["LegacyProcessModel", "build_plugin", "legacy_id"]
