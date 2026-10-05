"""Two-zone HVAC process model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np

from aiogym.core.information import state_limit_rules

from aiogym.core.model import PhysicsModelBase


class HVACModel(PhysicsModelBase):
    """Two coupled thermal zones with bidirectional heat-pump actuation."""

    scenario = "hvac"
    n = 2
    dt_micro = 0.25
    state_names = ("zone_0_temperature", "zone_1_temperature")
    state_units = {name: "degC" for name in state_names}
    state_bounds = {name: (-20.0, 60.0) for name in state_names}
    action_names = ("hvac_zone_0", "hvac_zone_1")
    action_units = {name: "normalized_power" for name in action_names}
    action_bounds = {name: (0.0, 1.0) for name in action_names}
    action_kinds = {name: "heater" for name in action_names}
    output_names = state_names
    output_units = state_units
    output_bounds = {name: (18.0, 26.0) for name in output_names}
    input_disturbances = (
        {
            "name": "outdoor_temperature",
            "event": "outdoor_temperature_step",
            "unit": "degC",
            "bounds": (-30.0, 50.0),
            "default": 5.0,
            "description": "outdoor air temperature",
        },
        {
            "name": "internal_heat_load_zone_0",
            "event": "zone_0_internal_heat_load_step",
            "unit": "W",
            "bounds": (-1000.0, 2000.0),
            "default": 0.0,
            "description": "net internal heat load in zone 0",
        },
        {
            "name": "internal_heat_load_zone_1",
            "event": "zone_1_internal_heat_load_step",
            "unit": "W",
            "bounds": (-1000.0, 2000.0),
            "default": 0.0,
            "description": "net internal heat load in zone 1",
        },
        {
            "name": "hvac_efficiency",
            "event": "hvac_efficiency_shift",
            "unit": "fraction",
            "bounds": (0.4, 1.3),
            "default": 1.0,
            "description": "delivered HVAC power multiplier",
        },
    )

    parameter_units = MappingProxyType(
        {
            "zone_thermal_capacity": "J/K",
            "maximum_zone_power": "W",
            "interzone_conductance": "W/K",
            "outdoor_conductance": "W/K",
            "outdoor_temperature": "degC",
        }
    )

    parameter_metadata = {
        'zone_thermal_capacity': ('Thermal capacity of each zone', 'Finite number > 0'),
        'maximum_zone_power': ('Maximum heating or cooling power magnitude per zone', 'Finite number > 0'),
        'interzone_conductance': ('Thermal conductance between the two zones', 'Finite number >= 0'),
        'outdoor_conductance': ('Thermal conductance from each zone to outdoor air', 'Finite number >= 0'),
        'outdoor_temperature': ('Default outdoor air temperature', '[-30, 50]'),
    }
    variable_descriptions = {
        'zone_0_temperature': 'Air temperature in zone 0',
        'zone_1_temperature': 'Air temperature in zone 1',
        'hvac_zone_0': 'Heating and cooling command for zone 0',
        'hvac_zone_1': 'Heating and cooling command for zone 1',
    }

    def action_metadata(self):
        return {name: {
            "interpretation": f"0 = maximum cooling; 0.5 = off; 1 = maximum heating; power = (2 * action - 1) * {self.p['maximum_zone_power']:g} W * hvac_efficiency",
        } for name in self.action_names}

    def safety_metadata(self):
        return state_limit_rules(self.state_bounds, self.state_units)

    def __init__(self, parameters: Mapping[str, Any] | None = None):
        defaults = {
            "zone_thermal_capacity": 6000.0,
            "maximum_zone_power": 1800.0,
            "interzone_conductance": 35.0,
            "outdoor_conductance": 45.0,
            "outdoor_temperature": 5.0,
        }
        self._parameter_defaults = deepcopy(defaults)
        self.p = _resolved_parameters(defaults, parameters)
        self._resolved_parameters = MappingProxyType(dict(self.p))

    @property
    def resolved_parameters(self) -> Mapping[str, Any]:
        return self._resolved_parameters

    def parameter(self, name: str) -> float:
        try:
            return float(self.p[str(name)])
        except KeyError as error:
            raise KeyError(f"unknown hvac parameter {name!r}") from error

    def default_disturbances(self):
        defaults = dict(super().default_disturbances())
        defaults["outdoor_temperature"] = float(self.p["outdoor_temperature"])
        return defaults

    def action_slew_limits(self):
        return None

    def initial_state(self):
        return list(self.default_setpoint_vector())

    def default_setpoint_vector(self):
        return [22.0, 22.0]

    def default_action(self):
        action = self.tracking_steady_state_action(
            self.default_setpoint_vector(),
            self.default_disturbances(),
        )
        if action is None:
            raise ValueError("default HVAC operating point is infeasible")
        return action

    def _dynamics(self, state, action, disturbances):
        capacity = float(self.p["zone_thermal_capacity"])
        maximum_power = float(self.p["maximum_zone_power"])
        interzone = float(self.p["interzone_conductance"])
        outdoor = float(self.p["outdoor_conductance"])
        outdoor_temperature = float(disturbances["outdoor_temperature"])
        efficiency = float(disturbances["hvac_efficiency"])
        loads = (
            float(disturbances["internal_heat_load_zone_0"]),
            float(disturbances["internal_heat_load_zone_1"]),
        )
        temperatures = (float(state[0]), float(state[1]))
        powers = tuple(
            (float(value) - 0.5) * 2.0 * maximum_power * efficiency for value in action
        )
        return [
            (
                powers[0]
                + loads[0]
                + interzone * (temperatures[1] - temperatures[0])
                + outdoor * (outdoor_temperature - temperatures[0])
            )
            / capacity,
            (
                powers[1]
                + loads[1]
                + interzone * (temperatures[0] - temperatures[1])
                + outdoor * (outdoor_temperature - temperatures[1])
            )
            / capacity,
        ]

    def outputs(self, state):
        return [float(state[0]), float(state[1])]

    def tracking_steady_state_action(self, reference, disturbances=None):
        target = np.asarray(reference, dtype=float).reshape(-1)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError("HVAC reference must contain two finite temperatures")
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        efficiency = float(context["hvac_efficiency"])
        if not math.isfinite(efficiency) or efficiency <= 0.0:
            raise ValueError("HVAC efficiency must be finite and positive")
        maximum_power = float(self.p["maximum_zone_power"])
        interzone = float(self.p["interzone_conductance"])
        outdoor = float(self.p["outdoor_conductance"])
        outside = float(context["outdoor_temperature"])
        loads = np.asarray(
            [
                context["internal_heat_load_zone_0"],
                context["internal_heat_load_zone_1"],
            ],
            dtype=float,
        )
        required_power = (
            np.asarray(
                [
                    outdoor * (target[0] - outside)
                    + interzone * (target[0] - target[1]),
                    outdoor * (target[1] - outside)
                    + interzone * (target[1] - target[0]),
                ]
            )
            - loads
        )
        action = 0.5 + required_power / (2.0 * maximum_power * efficiency)
        if (
            not np.isfinite(action).all()
            or np.any(action < 0.0)
            or np.any(action > 1.0)
        ):
            return None
        return action.tolist()

    def tracking_steady_state_state(self, reference, disturbances=None):
        if self.tracking_steady_state_action(reference, disturbances) is None:
            return None
        return [float(value) for value in reference]

    def measurement(self, state, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        temperatures = self.outputs(state)
        return {
            "x": list(temperatures),
            "y": list(temperatures),
            "levels": [],
            "temps": list(temperatures),
            **context,
        }

    def clamp_state(self, state):
        return [float(value) for value in state]

    def constraint_costs(self, state, disturbances=None):
        del disturbances
        low, high = self.state_bounds[self.state_names[0]]
        values = [float(value) for value in state]
        return {
            **{
                f"zone_{index}_temperature_lower": max(0.0, low - value)
                for index, value in enumerate(values)
            },
            **{
                f"zone_{index}_temperature_upper": max(0.0, value - high)
                for index, value in enumerate(values)
            },
        }

    def safety_margins(self, state, disturbances=None):
        del disturbances
        low, high = self.state_bounds[self.state_names[0]]
        span = high - low
        values = [float(value) for value in state]
        return {
            **{
                f"zone_{index}_temperature_lower": (value - low) / span
                for index, value in enumerate(values)
            },
            **{
                f"zone_{index}_temperature_upper": (high - value) / span
                for index, value in enumerate(values)
            },
        }

    def energy_kw(self, action):
        maximum_power = float(self.p["maximum_zone_power"])
        return (
            sum(abs((float(value) - 0.5) * 2.0 * maximum_power) for value in action)
            / 1000.0
        )

    def step_info(self, state, action, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        applied = self.default_action() if action is None else action
        return {
            "y": self.outputs(state),
            "energy_kw": self.action_energy_kw(applied, state, context),
            "outdoor_temperature": float(context["outdoor_temperature"]),
            "internal_heat_load_zone_0": float(context["internal_heat_load_zone_0"]),
            "internal_heat_load_zone_1": float(context["internal_heat_load_zone_1"]),
            "hvac_efficiency": float(context["hvac_efficiency"]),
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
            f"unknown hvac parameters: {unknown}; available: {sorted(defaults)}"
        )
    resolved = deepcopy(defaults)
    for name, value in supplied.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"HVAC parameter {name!r} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"HVAC parameter {name!r} must be finite")
        resolved[name] = number
    for name in ("zone_thermal_capacity", "maximum_zone_power"):
        if resolved[name] <= 0.0:
            raise ValueError(f"HVAC parameter {name!r} must be positive")
    for name in ("interzone_conductance", "outdoor_conductance"):
        if resolved[name] < 0.0:
            raise ValueError(f"HVAC parameter {name!r} must be non-negative")
    if not -30.0 <= resolved["outdoor_temperature"] <= 50.0:
        raise ValueError(
            "HVAC parameter 'outdoor_temperature' must stay within [-30, 50]"
        )
    return resolved


__all__ = ["HVACModel"]
