"""Six-actuator parameterized three-tank model."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _legacy_design_spec(plant) -> dict[str, Any]:
    return {
        "schema_version": "aiogym.design_spec.v1",
        "id": plant.id,
        "description": plant.config.description,
        "topology": "three-tank-recirculating",
        "tanks": list(plant.parameters["tanks"]),
        "heaters": list(plant.parameters["heaters"]),
        "pump": dict(plant.parameters["pump"]),
        "hydraulics": dict(plant.parameters["hydraulics"]),
        "operation": dict(plant.parameters["operation"]),
        "requirements": dict(plant.parameters["requirements"]),
        "uncertainties": dict(plant.parameters.get("uncertainties", {})),
        "references": list(plant.config.references),
    }


def _schema(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for raw in rows:
        row = dict(raw)
        low, high = row.pop("bounds", (None, None))
        row["low"] = -np.inf if low is None else float(low)
        row["high"] = np.inf if high is None else float(high)
        result.append(row)
    return result


class ThreeTankModel:
    scenario = "three_tank"

    def __init__(self, plant):
        from .equipment import ThreeTankDesignModel

        self.plant = plant
        self._model = ThreeTankDesignModel(_legacy_design_spec(plant))

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

    def output_schema(self):
        heights = [float(value) for value in self._model.p["height_max"]]
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

    def default_action(self):
        return self._model.default_action()

    def default_setpoint_vector(self):
        return self._model.default_setpoint_vector()

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
        return self._model.process_info(
            values, levels, temperatures, context, applied
        )

    def action_energy_kw(self, action, state, disturbances=None):
        context = self._model._resolved_env(disturbances)
        return self._model.action_energy_kw(action, state, context)

    def __getattr__(self, name: str):
        return getattr(self._model, name)


__all__ = ["ThreeTankModel"]
