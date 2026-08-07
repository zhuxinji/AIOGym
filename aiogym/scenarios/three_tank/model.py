"""Six-actuator parameterized three-tank model."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _schema(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for raw in rows:
        row = dict(raw)
        low, high = row.pop("bounds", (None, None))
        row["low"] = -np.inf if low is None else float(low)
        row["high"] = np.inf if high is None else float(high)
        result.append(row)
    return result


def _condition_mapping(value):
    if hasattr(value, "as_dict"):
        return value.as_dict(include_hash=False)
    return getattr(value, "config", value)


class ThreeTankModel:
    scenario = "three_tank"

    def __init__(self, plant):
        self.plant = plant
        topology = str(plant.config.plant.get("topology", "recirculating_loop"))
        self.topology = topology
        parameters = plant.config.plant.get("parameters")
        if parameters is not None:
            from aiogym.scenarios._shared import apply_parameters

            if topology == "open_cascade":
                from .topologies import OpenCascadeTopology

                self._model = apply_parameters(
                    OpenCascadeTopology(), dict(parameters)
                )
            elif topology == "recirculating_loop":
                from .topologies import RecirculatingTopology

                self._model = apply_parameters(
                    RecirculatingTopology(), dict(parameters)
                )
            else:
                raise ValueError(f"unsupported three_tank topology {topology!r}")
        else:
            from .equipment import ThreeTankDesignModel

            self._model = ThreeTankDesignModel(plant)
        self._model.scenario = "three_tank"
        self.dt_micro = float(self._model.dt_micro)

    def initial_state(self):
        return self._model.initial_state()

    def sample_initial_state(self, rng, condition):
        del rng
        config = _condition_mapping(condition)
        return list(config.get("initial_state", self.initial_state()))

    def dynamics(self, state, action, disturbances=None):
        return self._model.dynamics(state, action, disturbances or {})

    def outputs(self, state):
        return self._model.controlled_output(state)

    def action_schema(self):
        rows = _schema(self._model.action_schema())
        if self.topology == "open_cascade":
            names = (
                "feed_pump",
                "outlet_valve_1",
                "outlet_valve_2",
                "outlet_valve_3",
                "heater_H1",
                "heater_H2",
                "heater_H3",
            )
            for row, name in zip(rows, names):
                row["name"] = name
        return rows

    def state_schema(self):
        rows = _schema(self._model.state_schema())
        for row, name in zip(rows, ("h1", "T1", "h2", "T2", "h3", "T3")):
            row["name"] = name
        return rows

    def output_schema(self):
        raw_heights = self._model.p["height_max"]
        heights = (
            [float(raw_heights)] * 3
            if isinstance(raw_heights, (int, float))
            else [float(value) for value in raw_heights]
        )
        temperature_limit = float(self._model.p["temperature_hard_limit"])
        return [
            *(
                {"name": f"level_{index + 1}", "low": 0.0, "high": height}
                for index, height in enumerate(heights)
            ),
            *(
                {
                    "name": f"temperature_{index + 1}",
                    "low": 0.0,
                    "high": temperature_limit,
                }
                for index in range(3)
            ),
        ]

    def observation_schema(self, condition):
        config = _condition_mapping(condition)
        mode = config.get(
            "observation",
            "state-reference-disturbance"
            if "parameters" in self.plant.config.plant
            else "controlled-output",
        )
        if mode == "controlled-output":
            return self.output_schema()
        if mode != "state-reference-disturbance":
            raise ValueError(f"unknown observation mode {mode!r}")
        disturbance_names = set(self._model.dynamics_disturbance_names())
        disturbance_rows = _schema(
            row
            for row in self._model.disturbance_schema()
            if row["name"] in disturbance_names
        )
        return [*self.state_schema(), *self.output_schema(), *disturbance_rows]

    def observation(
        self, state, reference, previous_action, disturbances, condition
    ):
        del previous_action
        config = _condition_mapping(condition)
        mode = config.get(
            "observation",
            "state-reference-disturbance"
            if "parameters" in self.plant.config.plant
            else "controlled-output",
        )
        if mode == "controlled-output":
            return list(self.outputs(state))
        names = self._model.dynamics_disturbance_names()
        return [
            *list(state),
            *list(reference),
            *(float(disturbances[name]) for name in names),
        ]

    def default_action(self):
        return self._model.default_action()

    def default_setpoint_vector(self):
        return self._model.default_setpoint_vector()

    def default_disturbances(self):
        return dict(self._model.disturbance_defaults())

    def controlled_output_scales(self):
        return self._model.controlled_output_scales()

    def capabilities(self):
        values = {"tracking", "energy", "hardware_interlocks"}
        if self.topology == "open_cascade":
            values.add("product_flow")
        return frozenset(values)

    def clamp_state(self, state):
        clamp = getattr(self._model, "clamp_state", None)
        return list(state) if clamp is None else clamp(state)

    def constraint_costs(self, state, disturbances=None):
        values = list(state)
        levels = values[0::2]
        temperatures = values[1::2]
        context = self._model._resolved_env(disturbances)
        reasons = self._model.hard_termination_reasons(
            values, levels, temperatures, context
        )
        return {str(reason): 1.0 for reason in reasons}

    def step_info(self, state, action, disturbances=None):
        values = list(state)
        levels = values[0::2]
        temperatures = values[1::2]
        context = self._model._resolved_env(disturbances)
        applied = self.default_action() if action is None else list(action)
        info = self._model.process_info(
            values, levels, temperatures, context, applied
        )
        info["y"] = list(self.outputs(values))
        info["energy_kw"] = self.action_energy_kw(applied, values, context)
        return info

    def action_energy_kw(self, action, state, disturbances=None):
        context = self._model._resolved_env(disturbances)
        return self._model.action_energy_kw(action, state, context)

    def __getattr__(self, name: str):
        return getattr(self._model, name)


__all__ = ["ThreeTankModel"]
