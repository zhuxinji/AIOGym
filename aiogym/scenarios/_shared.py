"""Shared adapters for scenario-owned numerical process models."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import numpy as np

from aiogym.core import ResolvedPlant, ScenarioPlugin, TaskSpec


def _schema(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        low, high = row.pop("bounds", (None, None))
        row.setdefault("name", f"value_{index}")
        row["low"] = -np.inf if low is None else float(low)
        row["high"] = np.inf if high is None else float(high)
        converted.append(row)
    return converted


def _parameter(value, current, bounds, name):
    if isinstance(current, (list, tuple)):
        if not isinstance(value, (list, tuple)) or len(value) != len(current):
            raise ValueError(f"plant parameter {name!r} must contain {len(current)} values")
        return [_scalar(item, bounds, f"{name}[{index}]") for index, item in enumerate(value)]
    return _scalar(value, bounds, name)


def _scalar(value, bounds, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"plant parameter {name!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"plant parameter {name!r} must be finite")
    if bounds:
        low, high = bounds
        if low is not None and number < float(low):
            raise ValueError(f"plant parameter {name!r} must be >= {low}")
        if high is not None and number > float(high):
            raise ValueError(f"plant parameter {name!r} must be <= {high}")
    return copy.deepcopy(value)


def apply_parameters(model, parameters):
    unknown = set(parameters) - set(model.p)
    if unknown:
        raise ValueError(f"unknown {model.scenario} plant parameters: {sorted(unknown)}")
    for name, value in parameters.items():
        model.p[name] = _parameter(
            value,
            model.p[name],
            getattr(model, "param_bounds", {}).get(name),
            name,
        )
    return model


class NumericProcessModel:
    """Expose a scenario-local numerical model through the core contract."""

    numerical_type = None

    def __init__(self, plant: ResolvedPlant):
        self.plant = plant
        self._model = apply_parameters(self.numerical_type(), dict(plant.parameters))
        self.scenario = plant.scenario
        self.dt_micro = float(getattr(self._model, "dt_micro", 0.02))

    def initial_state(self):
        return self._model.initial_state()

    def sample_initial_state(self, rng, preset):
        del rng
        condition = _condition_mapping(preset)
        return list(condition.get("initial_state", self.initial_state()))

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

    def controlled_output_scales(self):
        return self._model.controlled_output_scales()

    def default_disturbances(self):
        return dict(self._model.disturbance_defaults())

    def observation_schema(self, preset):
        state = _schema(self._model.state_schema())
        reference = _schema(self._model.setpoint_schema())
        mode = _condition_mapping(preset).get(
            "observation", "state-reference-disturbance"
        )
        if mode == "normalized-state-error-action":
            return [
                *({**row, "low": 0.0, "high": 1.0} for row in state),
                *({**row, "name": f"{row['name']}_error", "low": -1.0, "high": 1.0} for row in reference),
                *self.action_schema(),
            ]
        names = set(self._model.dynamics_disturbance_names())
        disturbance = _schema(
            row for row in self._model.disturbance_schema() if row["name"] in names
        )
        return [*state, *reference, *disturbance]

    def observation(self, state, reference, previous_action, disturbances, preset):
        mode = _condition_mapping(preset).get(
            "observation", "state-reference-disturbance"
        )
        if mode == "normalized-state-error-action":
            state_rows = self._model.state_schema()
            normalized_state = [
                (float(value) - float(row["bounds"][0]))
                / (float(row["bounds"][1]) - float(row["bounds"][0]))
                for value, row in zip(state, state_rows)
            ]
            output = np.asarray(self._model.controlled_output(state), dtype=float)
            scale = np.asarray(self._model.controlled_output_scales(), dtype=float)
            error = (np.asarray(reference, dtype=float) - output) / scale
            return [*normalized_state, *error.tolist(), *list(previous_action)]
        values = [float(value) for value in state]
        values.extend(float(value) for value in reference)
        values.extend(
            float(disturbances[name]) for name in self._model.dynamics_disturbance_names()
        )
        return values

    def clamp_state(self, state):
        clamp = getattr(self._model, "clamp_state", None)
        return list(state) if clamp is None else clamp(state)

    def constraint_costs(self, state, disturbances=None):
        output = self._model.outputs(state)
        reasons = self._model.hard_termination_reasons(
            state,
            output.get("levels", ()),
            output.get("temps", ()),
            self._model.runtime_env(disturbances or {}),
        )
        return {str(reason): 1.0 for reason in reasons}

    def step_info(self, state, action, disturbances=None):
        environment = self._model.runtime_env(disturbances or {})
        measurement = self._model.measurement(state, environment)
        applied = self.default_action() if action is None else action
        info = self._model.process_info(
            state,
            measurement.get("levels", ()),
            measurement.get("temps", ()),
            environment,
        )
        info["y"] = list(self._model.controlled_output(state))
        info["energy_kw"] = self._model.action_energy_kw(applied, state, environment)
        return info

    def action_energy_kw(self, action, state=None, disturbances=None):
        return self._model.action_energy_kw(action, state, disturbances or {})

    def capabilities(self):
        values = {"tracking", "energy"}
        if callable(getattr(self._model, "production", None)):
            values.add("product_flow")
        if getattr(self._model, "safety_constraints", ()):
            values.add("hardware_interlocks")
        return frozenset(values)

    def __getattr__(self, name):
        return getattr(self._model, name)


def regulation_reward(state, action, next_state, context):
    del state, action
    model = context["model"]
    output = np.asarray(model.outputs(next_state), dtype=float)
    reference = np.asarray(context["reference"], dtype=float)
    scale = np.asarray(model.controlled_output_scales(), dtype=float)
    rate = float(np.mean(((output - reference) / scale) ** 2))
    cost = float(context["control_dt"]) * rate
    return -cost, {"tracking_error": -cost, "slew": 0.0, "effort": 0.0}


def _condition_mapping(value):
    if hasattr(value, "as_dict"):
        return value.as_dict(include_hash=False)
    return getattr(value, "config", value)


def economic_reward(state, action, next_state, context):
    del state
    model = context["model"]
    energy = float(model.action_energy_kw(action, next_state, context["disturbances"]))
    production = getattr(model, "production", lambda *_: 0.0)(
        next_state, action, context["disturbances"]
    )
    profit = float(production) - energy
    return profit, {"product_value": float(production), "energy_cost": -energy}


def build_plugin(
    scenario,
    *,
    model_factory,
    presets,
    default_preset,
    economic=False,
    horizon=600,
    control_dt=1.0,
    controller_defaults=None,
    preset_configs=None,
):
    preset_configs = dict(preset_configs or {})
    numerical = model_factory.numerical_type()
    conditions = {}
    for name in presets:
        overrides = dict(preset_configs.get(name, {}))
        conditions[name] = {
            "id": name,
            "initial_state": overrides.pop(
                "initial_state", list(numerical.initial_state())
            ),
            "reference": overrides.pop(
                "reference", list(numerical.default_setpoint_vector())
            ),
            "control_dt": overrides.pop("control_dt", control_dt),
            "horizon": overrides.pop("horizon", horizon),
            "disturbances": overrides.pop("disturbances", {}),
            "reference_schedule": overrides.pop("reference_schedule", {}),
            "disturbance_schedule": overrides.pop("disturbance_schedule", {}),
            "observation": overrides.pop(
                "observation", "state-reference-disturbance"
            ),
        }
        if overrides:
            raise ValueError(
                f"unknown condition fields for {scenario}/{name}: {sorted(overrides)}"
            )

    def default_plant():
        model = model_factory.numerical_type()
        return {
            "schema_version": "aiogym.plant.v2",
            "id": f"{scenario}-default-v1",
            "scenario": scenario,
            "description": f"Built-in default parameters for {scenario}",
            "plant": {"parameters": copy.deepcopy(model.p)},
            "conditions": copy.deepcopy(conditions),
            "default_condition": default_preset,
            "study": {},
            "references": [],
        }

    def resolve_plant(config):
        unknown = set(config.plant) - {"parameters"}
        if unknown:
            raise ValueError(f"unknown {scenario} plant fields: {sorted(unknown)}")
        model = apply_parameters(
            model_factory.numerical_type(), dict(config.plant.get("parameters", {}))
        )
        return ResolvedPlant(
            config=config,
            parameters=copy.deepcopy(model.p),
            provenance={"source": "scenario-defaults-with-PlantConfig-overrides"},
        )

    def task(objective, reward, primary, direction):
        metrics = (
            ("return", "economic_objective", "energy", "constraint_violations")
            if objective == "economic"
            else ("return", "tracking_iae", "tracking_ise", "constraint_violations", "energy")
        )
        return TaskSpec(
            id=f"{scenario}/{objective}",
            scenario=scenario,
            objective=objective,
            reward=reward,
            metrics=metrics,
            primary_metric=primary,
            metric_direction=direction,
            revision=1,
            reward_id=(
                "economic-profit-v1"
                if objective == "economic"
                else "normalized-tracking-mse-v1"
            ),
            metric_suite_id=(
                "economic-core-v1"
                if objective == "economic"
                else "regulation-core-v1"
            ),
            reward_term_names=(
                ("product_value", "energy_cost")
                if objective == "economic"
                else ("tracking_error", "slew", "effort")
            ),
            required_capabilities=(
                ("product_flow", "energy")
                if objective == "economic"
                else ("tracking",)
            ),
        )

    tasks = {
        "regulation": task("regulation", regulation_reward, "tracking_iae", "minimize")
    }
    if economic:
        tasks["economic"] = task(
            "economic", economic_reward, "economic_objective", "maximize"
        )
    return ScenarioPlugin(
        id=scenario,
        make_model=model_factory,
        default_plant=default_plant,
        resolve_plant=resolve_plant,
        tasks=tasks,
        built_in_plants={f"{scenario}-default-v1": default_plant},
        controller_defaults=dict(controller_defaults or {}),
    )


__all__ = ["NumericProcessModel", "build_plugin"]
