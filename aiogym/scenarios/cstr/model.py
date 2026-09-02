"""Two-input exothermic CSTR process model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np

from aiogym.core.model import PhysicsModelBase


class CSTRModel(PhysicsModelBase):
    """Two-state reactor with independently commanded feed and cooling."""

    scenario = "cstr"
    dt_micro = 0.01
    state_names = ("reactant_concentration", "reactor_temperature")
    state_units = {
        "reactant_concentration": "mol/L",
        "reactor_temperature": "degC",
    }
    state_bounds = {
        "reactant_concentration": (0.0, 1.5),
        "reactor_temperature": (0.0, 200.0),
    }
    action_names = ("feed_pump", "cooling")
    action_units = {
        "feed_pump": "normalized_flow",
        "cooling": "normalized_cooling",
    }
    action_bounds = {name: (0.0, 1.0) for name in action_names}
    action_kinds = {"feed_pump": "pump", "cooling": "cooler"}
    output_names = state_names
    output_units = state_units
    output_bounds = {
        "reactant_concentration": (0.02, 0.20),
        "reactor_temperature": (45.0, 90.0),
    }
    input_disturbances = (
        {
            "name": "feed_temperature",
            "event": "feed_temperature_step",
            "unit": "degC",
            "bounds": (0.0, 60.0),
            "default": 20.0,
            "description": "reactor feed temperature",
        },
        {
            "name": "feed_concentration",
            "event": "feed_concentration_step",
            "unit": "mol/L",
            "bounds": (0.2, 2.0),
            "default": 1.0,
            "description": "reactant concentration in the feed",
        },
        {
            "name": "coolant_temperature",
            "event": "coolant_temperature_step",
            "unit": "degC",
            "bounds": (-5.0, 35.0),
            "default": 10.0,
            "description": "cooling-jacket inlet temperature",
        },
    )

    parameter_units = MappingProxyType(
        {
            "maximum_dilution_rate": "1/s",
            "feed_concentration": "mol/L",
            "pre_exponential_factor": "1/s",
            "activation_temperature": "K",
            "reaction_temperature_gain": "degC/(mol/L)",
            "cooling_coefficient": "1/s",
            "coolant_temperature": "degC",
            "feed_temperature": "degC",
            "nominal_feed_action": "fraction",
            "maximum_cooling_power": "W",
            "maximum_feed_pump_power": "W",
            "temperature_trip": "degC",
        }
    )

    def __init__(self, parameters: Mapping[str, Any] | None = None):
        defaults = {
            "maximum_dilution_rate": 0.02,
            "feed_concentration": 1.0,
            "pre_exponential_factor": 1.0e8,
            "activation_temperature": 7000.0,
            "reaction_temperature_gain": 120.0,
            "cooling_coefficient": 0.05,
            "coolant_temperature": 10.0,
            "feed_temperature": 20.0,
            "nominal_feed_action": 0.5,
            "maximum_cooling_power": 80000.0,
            "maximum_feed_pump_power": 1200.0,
            "temperature_trip": 92.0,
        }
        self.p = _resolved_parameters(defaults, parameters)
        self._resolved_parameters = MappingProxyType(dict(self.p))

    @property
    def resolved_parameters(self) -> Mapping[str, Any]:
        return self._resolved_parameters

    def parameter(self, name: str) -> float:
        try:
            return float(self.p[str(name)])
        except KeyError as error:
            raise KeyError(f"unknown cstr parameter {name!r}") from error

    def default_disturbances(self):
        defaults = dict(super().default_disturbances())
        defaults.update(
            {
                "feed_temperature": float(self.p["feed_temperature"]),
                "feed_concentration": float(self.p["feed_concentration"]),
                "coolant_temperature": float(self.p["coolant_temperature"]),
            }
        )
        return defaults

    def action_slew_limits(self):
        return None

    def default_setpoint_vector(self):
        temperature = 60.0
        dilution_rate = float(self.p["nominal_feed_action"]) * float(
            self.p["maximum_dilution_rate"]
        )
        coefficient = self._rate_coefficient(temperature)
        concentration = (
            dilution_rate
            * float(self.p["feed_concentration"])
            / (dilution_rate + coefficient)
        )
        return [concentration, temperature]

    def initial_state(self):
        state = self.tracking_steady_state_state(
            self.default_setpoint_vector(),
            self.default_disturbances(),
        )
        if state is None:
            raise ValueError("default CSTR operating point is infeasible")
        return state

    def default_action(self):
        action = self.tracking_steady_state_action(
            self.default_setpoint_vector(),
            self.default_disturbances(),
        )
        if action is None:
            raise ValueError("default CSTR operating point is infeasible")
        return action

    def _rate_coefficient(self, temperature):
        absolute_temperature = float(temperature) + 273.15
        if absolute_temperature <= 0.0:
            raise ValueError("CSTR absolute temperature must be positive")
        return float(self.p["pre_exponential_factor"]) * math.exp(
            -float(self.p["activation_temperature"]) / absolute_temperature
        )

    def _dynamics(self, state, action, disturbances):
        concentration = float(state[0])
        temperature = float(state[1])
        dilution_rate = float(action[0]) * float(self.p["maximum_dilution_rate"])
        reaction_rate = self._rate_coefficient(temperature) * max(concentration, 0.0)
        feed_concentration = float(disturbances["feed_concentration"])
        feed_temperature = float(disturbances["feed_temperature"])
        coolant_temperature = float(disturbances["coolant_temperature"])
        cooling = float(action[1])
        return [
            dilution_rate * (feed_concentration - concentration) - reaction_rate,
            dilution_rate * (feed_temperature - temperature)
            + float(self.p["reaction_temperature_gain"]) * reaction_rate
            - float(self.p["cooling_coefficient"])
            * cooling
            * (temperature - coolant_temperature),
        ]

    def outputs(self, state):
        return [float(state[0]), float(state[1])]

    def display_outputs(self, state):
        return {"levels": [], "temps": [float(state[1])]}

    def _steady_operating_point(self, reference, disturbances=None):
        target = np.asarray(reference, dtype=float).reshape(-1)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError(
                "CSTR reference must contain finite concentration and temperature"
            )
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        concentration, temperature = (float(value) for value in target)
        coefficient = self._rate_coefficient(temperature)
        feed_concentration = float(context["feed_concentration"])
        concentration_driving_force = feed_concentration - concentration
        if concentration <= 0.0 or concentration_driving_force <= 0.0:
            return None
        dilution_rate = coefficient * concentration / concentration_driving_force
        feed_action = dilution_rate / float(self.p["maximum_dilution_rate"])
        reaction_rate = coefficient * concentration
        coolant_temperature = float(context["coolant_temperature"])
        cooling_denominator = float(self.p["cooling_coefficient"]) * (
            temperature - coolant_temperature
        )
        if cooling_denominator <= 0.0:
            return None
        cooling_action = (
            dilution_rate * (float(context["feed_temperature"]) - temperature)
            + float(self.p["reaction_temperature_gain"]) * reaction_rate
        ) / cooling_denominator
        action = np.asarray([feed_action, cooling_action], dtype=float)
        state = np.asarray([concentration, temperature], dtype=float)
        if (
            not np.isfinite(action).all()
            or not np.isfinite(state).all()
            or np.any(action < 0.0)
            or np.any(action > 1.0)
            or concentration < 0.0
            or concentration > 1.5
            or temperature < 0.0
            or temperature > 200.0
        ):
            return None
        return {"state": state.tolist(), "action": action.tolist()}

    def tracking_steady_state_action(self, reference, disturbances=None):
        operating_point = self._steady_operating_point(reference, disturbances)
        return None if operating_point is None else operating_point["action"]

    def tracking_steady_state_state(self, reference, disturbances=None):
        operating_point = self._steady_operating_point(reference, disturbances)
        return None if operating_point is None else operating_point["state"]

    def measurement(self, state, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        concentration = float(state[0])
        temperature = float(state[1])
        return {
            "x": [concentration, temperature],
            "y": [concentration, temperature],
            "levels": [],
            "temps": [temperature],
            "conc": [max(concentration, 0.0)],
            **context,
        }

    def clamp_state(self, state):
        return [float(value) for value in state]

    def constraint_costs(self, state, disturbances=None):
        del disturbances
        concentration = float(state[0])
        temperature = float(state[1])
        return {
            "reactant_concentration_lower": max(0.0, -concentration),
            "reactant_concentration_upper": max(0.0, concentration - 1.5),
            "reactor_temperature_lower": max(0.0, -temperature),
            "reactor_temperature_trip": max(
                0.0, temperature - float(self.p["temperature_trip"])
            ),
        }

    def safety_margins(self, state, disturbances=None):
        del disturbances
        concentration = float(state[0])
        temperature = float(state[1])
        trip = float(self.p["temperature_trip"])
        return {
            "reactant_concentration_lower": concentration / 1.5,
            "reactant_concentration_upper": (1.5 - concentration) / 1.5,
            "reactor_temperature_lower": temperature / trip,
            "reactor_temperature_trip": (trip - temperature) / trip,
        }

    def energy_kw(self, action):
        return (
            float(action[0]) * float(self.p["maximum_feed_pump_power"])
            + float(action[1]) * float(self.p["maximum_cooling_power"])
        ) / 1000.0

    def conversion(self, state, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        feed = float(context["feed_concentration"])
        if feed <= 0.0:
            raise ValueError("CSTR feed concentration must be positive")
        return max(0.0, min(1.0, (feed - max(float(state[0]), 0.0)) / feed))

    def step_info(self, state, action, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        applied = self.default_action() if action is None else action
        return {
            "y": self.outputs(state),
            "energy_kw": self.action_energy_kw(applied, state, context),
            "conversion": self.conversion(state, context),
            "feed_temperature": float(context["feed_temperature"]),
            "feed_concentration": float(context["feed_concentration"]),
            "coolant_temperature": float(context["coolant_temperature"]),
        }


def _resolved_parameters(defaults, overrides):
    if overrides is None:
        supplied = {}
    elif not isinstance(overrides, Mapping):
        raise TypeError("parameters must be a mapping or None")
    else:
        supplied = dict(overrides)
    if any(not isinstance(name, str) for name in supplied):
        raise TypeError("parameter keys must be strings")
    unknown = sorted(set(supplied) - set(defaults))
    if unknown:
        raise ValueError(
            f"unknown cstr parameters: {unknown}; available: {sorted(defaults)}"
        )
    resolved = deepcopy(defaults)
    for name, value in supplied.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"CSTR parameter {name!r} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"CSTR parameter {name!r} must be finite")
        resolved[name] = number
    for name in (
        "maximum_dilution_rate",
        "feed_concentration",
        "pre_exponential_factor",
        "activation_temperature",
        "reaction_temperature_gain",
        "cooling_coefficient",
        "maximum_cooling_power",
        "maximum_feed_pump_power",
        "temperature_trip",
    ):
        if resolved[name] <= 0.0:
            raise ValueError(f"CSTR parameter {name!r} must be positive")
    if not 0.0 < resolved["nominal_feed_action"] <= 1.0:
        raise ValueError("CSTR nominal_feed_action must stay within (0, 1]")
    if not -5.0 <= resolved["coolant_temperature"] <= 35.0:
        raise ValueError(
            "CSTR parameter 'coolant_temperature' must stay within [-5, 35]"
        )
    if not 0.0 <= resolved["feed_temperature"] <= 60.0:
        raise ValueError("CSTR parameter 'feed_temperature' must stay within [0, 60]")
    if not 45.0 < resolved["temperature_trip"] <= 200.0:
        raise ValueError("CSTR temperature_trip must stay within (45, 200]")
    return resolved


__all__ = ["CSTRModel"]
