"""Lumped fired-heater process model."""

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


class HeaterModel(PhysicsModelBase):
    """Three-state heater with independent combustion-air and fuel commands."""

    scenario = "heater"
    dt_micro = 0.05
    state_names = (
        "firebox_temperature",
        "outlet_temperature",
        "flue_oxygen",
    )
    state_units = {
        "firebox_temperature": "degC",
        "outlet_temperature": "degC",
        "flue_oxygen": "%",
    }
    state_bounds = {
        "firebox_temperature": (20.0, 1400.0),
        "outlet_temperature": (20.0, 650.0),
        "flue_oxygen": (0.0, 20.9),
    }
    action_names = ("air_damper", "fuel_valve")
    action_units = {name: "fraction" for name in action_names}
    action_bounds = {name: (0.0, 1.0) for name in action_names}
    action_kinds = {"air_damper": "valve", "fuel_valve": "heater"}
    output_names = ("flue_oxygen", "outlet_temperature")
    output_units = {"flue_oxygen": "%", "outlet_temperature": "degC"}
    output_bounds = {
        "flue_oxygen": (1.8, 5.0),
        "outlet_temperature": (364.0, 372.0),
    }
    input_disturbances = (
        {
            "name": "feed_temperature",
            "event": "feed_temperature_step",
            "unit": "degC",
            "bounds": (240.0, 330.0),
            "default": 280.0,
            "description": "process-feed inlet temperature",
        },
        {
            "name": "ambient_temperature",
            "event": "ambient_temperature_step",
            "unit": "degC",
            "bounds": (-10.0, 45.0),
            "default": 20.0,
            "description": "ambient temperature around the firebox",
        },
        {
            "name": "feed_flow",
            "event": "feed_flow_step",
            "unit": "kg/s",
            "bounds": (60.0, 120.0),
            "default": 88.0,
            "description": "process-feed mass flow",
        },
        {
            "name": "fuel_heating_value_factor",
            "event": "fuel_heating_value_shift",
            "unit": "fraction",
            "bounds": (0.7, 1.3),
            "default": 1.0,
            "description": "fuel lower-heating-value multiplier",
        },
    )

    parameter_units = MappingProxyType(
        {
            "maximum_fuel_flow": "kg/s",
            "fuel_lower_heating_value": "J/kg",
            "stoichiometric_air_fuel_ratio": "kg_air/kg_fuel",
            "maximum_air_flow": "kg/s",
            "flue_gas_heat_capacity": "J/(kg*K)",
            "heat_transfer_coefficient": "W/K",
            "firebox_heat_capacity": "J/K",
            "process_heat_capacity": "J/K",
            "nominal_feed_flow": "kg/s",
            "feed_specific_heat_capacity": "J/(kg*K)",
            "oxygen_time_constant": "s",
            "feed_temperature": "degC",
            "ambient_temperature": "degC",
            "outlet_temperature_trip": "degC",
            "minimum_safe_oxygen": "%",
        }
    )

    parameter_metadata = {
        'maximum_fuel_flow': ('Fuel mass flow at fully open fuel valve', 'Finite number > 0'),
        'fuel_lower_heating_value': ('Fuel lower heating value', 'Finite number > 0'),
        'stoichiometric_air_fuel_ratio': ('Air mass required for complete combustion per unit fuel mass', 'Finite number > 0'),
        'maximum_air_flow': ('Air mass flow at fully open damper', 'Finite number > 0'),
        'flue_gas_heat_capacity': ('Flue-gas specific heat capacity', 'Finite number > 0'),
        'heat_transfer_coefficient': ('Firebox-to-process thermal conductance', 'Finite number > 0'),
        'firebox_heat_capacity': ('Effective firebox thermal capacity', 'Finite number > 0'),
        'process_heat_capacity': ('Effective process-side thermal capacity', 'Finite number > 0'),
        'nominal_feed_flow': ('Default process-feed mass flow', 'Finite number > 0'),
        'feed_specific_heat_capacity': ('Process-feed specific heat capacity', 'Finite number > 0'),
        'oxygen_time_constant': ('Flue-oxygen first-order response time constant', 'Finite number > 0'),
        'feed_temperature': ('Default process-feed inlet temperature', '[240, 330]'),
        'ambient_temperature': ('Default ambient temperature around the firebox', '[-10, 45]'),
        'outlet_temperature_trip': ('Process outlet temperature threshold for episode termination', '(20, 650]'),
        'minimum_safe_oxygen': ('Flue-oxygen lower threshold for episode termination', '[0, 20.9)'),
    }
    variable_descriptions = {
        'firebox_temperature': 'Firebox temperature',
        'outlet_temperature': 'Process-fluid outlet temperature',
        'flue_oxygen': 'Flue-gas oxygen percentage',
        'air_damper': 'Combustion-air damper opening',
        'fuel_valve': 'Fuel-valve opening',
    }

    def action_metadata(self):
        return {
            "air_damper": {"interpretation": f"Air flow = action * {self.p['maximum_air_flow']:g} kg/s"},
            "fuel_valve": {"interpretation": f"Fuel flow = action * {self.p['maximum_fuel_flow']:g} kg/s"},
        }

    def safety_metadata(self):
        return state_limit_rules({
            "firebox_temperature": self.state_bounds["firebox_temperature"],
            "outlet_temperature": (20.0, self.p["outlet_temperature_trip"]),
            "flue_oxygen": (self.p["minimum_safe_oxygen"], 20.9),
        }, self.state_units)

    def __init__(self, parameters: Mapping[str, Any] | None = None):
        defaults = {
            "maximum_fuel_flow": 1.0,
            "fuel_lower_heating_value": 46.0e6,
            "stoichiometric_air_fuel_ratio": 17.2,
            "maximum_air_flow": 40.0,
            "flue_gas_heat_capacity": 1400.0,
            "heat_transfer_coefficient": 42.0e3,
            "firebox_heat_capacity": 3.5e6,
            "process_heat_capacity": 7.0e6,
            "nominal_feed_flow": 88.0,
            "feed_specific_heat_capacity": 2300.0,
            "oxygen_time_constant": 20.0,
            "feed_temperature": 280.0,
            "ambient_temperature": 20.0,
            "outlet_temperature_trip": 415.0,
            "minimum_safe_oxygen": 1.2,
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
            raise KeyError(f"unknown heater parameter {name!r}") from error

    def default_disturbances(self):
        defaults = dict(super().default_disturbances())
        defaults.update(
            {
                "feed_temperature": float(self.p["feed_temperature"]),
                "ambient_temperature": float(self.p["ambient_temperature"]),
                "feed_flow": float(self.p["nominal_feed_flow"]),
            }
        )
        return defaults

    def action_slew_limits(self):
        return None

    def default_setpoint_vector(self):
        return [3.0, 370.0]

    def initial_state(self):
        state = self.tracking_steady_state_state(
            self.default_setpoint_vector(),
            self.default_disturbances(),
        )
        if state is None:
            raise ValueError("default heater operating point is infeasible")
        return state

    def default_action(self):
        action = self.tracking_steady_state_action(
            self.default_setpoint_vector(),
            self.default_disturbances(),
        )
        if action is None:
            raise ValueError("default heater operating point is infeasible")
        return action

    def combustion(self, action, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        return self._combustion(self.action_vector(action), context)

    def _combustion(self, values, context):
        fuel_flow = values[1] * float(self.p["maximum_fuel_flow"])
        air_flow = values[0] * float(self.p["maximum_air_flow"])
        stoichiometric_air = float(self.p["stoichiometric_air_fuel_ratio"]) * fuel_flow
        if fuel_flow <= 0.0:
            completeness = 0.0
            reacted_fuel = 0.0
            flue_oxygen = 20.9
        else:
            completeness = min(
                1.0,
                max(0.0, air_flow / stoichiometric_air),
            )
            reacted_fuel = fuel_flow * completeness
            flue_mass = air_flow + reacted_fuel
            excess_air = max(
                0.0,
                air_flow
                - float(self.p["stoichiometric_air_fuel_ratio"]) * reacted_fuel,
            )
            flue_oxygen = 0.0 if flue_mass <= 0.0 else 20.9 * excess_air / flue_mass
        duty = (
            reacted_fuel
            * float(self.p["fuel_lower_heating_value"])
            * float(context["fuel_heating_value_factor"])
        )
        return {
            "fuel_flow": fuel_flow,
            "air_flow": air_flow,
            "combustion_completeness": completeness,
            "duty": duty,
            "flue_oxygen_equilibrium": flue_oxygen,
            "flue_mass_flow": air_flow + reacted_fuel,
        }

    def _dynamics(self, state, action, disturbances):
        context = self._resolve_disturbances(disturbances)
        firebox_temperature = float(state[0])
        outlet_temperature = float(state[1])
        flue_oxygen = float(state[2])
        combustion = self._combustion(action, context)
        feed_temperature = float(context["feed_temperature"])
        ambient_temperature = float(context["ambient_temperature"])
        feed_flow = float(context["feed_flow"])
        mean_process_temperature = 0.5 * (feed_temperature + outlet_temperature)
        transferred_heat = float(self.p["heat_transfer_coefficient"]) * (
            firebox_temperature - mean_process_temperature
        )
        stack_loss = (
            float(combustion["flue_mass_flow"])
            * float(self.p["flue_gas_heat_capacity"])
            * (firebox_temperature - ambient_temperature)
        )
        process_heat = (
            feed_flow
            * float(self.p["feed_specific_heat_capacity"])
            * (feed_temperature - outlet_temperature)
        )
        return [
            (float(combustion["duty"]) - transferred_heat - stack_loss)
            / float(self.p["firebox_heat_capacity"]),
            (process_heat + transferred_heat) / float(self.p["process_heat_capacity"]),
            (float(combustion["flue_oxygen_equilibrium"]) - flue_oxygen)
            / float(self.p["oxygen_time_constant"]),
        ]

    def outputs(self, state):
        return [float(state[2]), float(state[1])]

    def _steady_operating_point(self, reference, disturbances=None):
        target = np.asarray(reference, dtype=float).reshape(-1)
        if target.shape != (2,) or not np.isfinite(target).all():
            raise ValueError(
                "heater reference must contain finite oxygen and temperature"
            )
        oxygen = float(target[0])
        outlet_temperature = float(target[1])
        if not 0.0 < oxygen < 20.9:
            return None
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        feed_temperature = float(context["feed_temperature"])
        ambient_temperature = float(context["ambient_temperature"])
        feed_flow = float(context["feed_flow"])
        heating_value_factor = float(context["fuel_heating_value_factor"])
        if feed_flow <= 0.0 or heating_value_factor <= 0.0:
            return None
        process_duty = (
            feed_flow
            * float(self.p["feed_specific_heat_capacity"])
            * (outlet_temperature - feed_temperature)
        )
        if process_duty <= 0.0:
            return None
        mean_process_temperature = 0.5 * (feed_temperature + outlet_temperature)
        firebox_temperature = mean_process_temperature + process_duty / float(
            self.p["heat_transfer_coefficient"]
        )
        oxygen_fraction = oxygen / 20.9
        air_fuel_ratio = (
            float(self.p["stoichiometric_air_fuel_ratio"]) + oxygen_fraction
        ) / (1.0 - oxygen_fraction)
        net_heating_value = float(
            self.p["fuel_lower_heating_value"]
        ) * heating_value_factor - (air_fuel_ratio + 1.0) * float(
            self.p["flue_gas_heat_capacity"]
        ) * (firebox_temperature - ambient_temperature)
        if net_heating_value <= 0.0:
            return None
        fuel_flow = process_duty / net_heating_value
        air_flow = air_fuel_ratio * fuel_flow
        action = np.asarray(
            [
                air_flow / float(self.p["maximum_air_flow"]),
                fuel_flow / float(self.p["maximum_fuel_flow"]),
            ],
            dtype=float,
        )
        state = np.asarray(
            [firebox_temperature, outlet_temperature, oxygen],
            dtype=float,
        )
        state_low = np.asarray(
            [self.state_bounds[name][0] for name in self.state_names],
            dtype=float,
        )
        state_high = np.asarray(
            [self.state_bounds[name][1] for name in self.state_names],
            dtype=float,
        )
        if (
            not np.isfinite(action).all()
            or not np.isfinite(state).all()
            or np.any(action < 0.0)
            or np.any(action > 1.0)
            or np.any(state < state_low)
            or np.any(state > state_high)
        ):
            return None
        derivative = np.asarray(self.dynamics(state, action, context), dtype=float)
        if not np.allclose(derivative, 0.0, rtol=0.0, atol=1e-10):
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
        return {
            "x": [float(value) for value in state],
            "y": self.outputs(state),
            "levels": [],
            "temps": [float(state[0]), float(state[1])],
            **context,
        }

    def clamp_state(self, state):
        return [float(value) for value in state]

    def constraint_costs(self, state, disturbances=None):
        del disturbances
        firebox_temperature = float(state[0])
        outlet_temperature = float(state[1])
        oxygen = float(state[2])
        return {
            "firebox_temperature_lower": max(0.0, 20.0 - firebox_temperature),
            "firebox_temperature_upper": max(0.0, firebox_temperature - 1400.0),
            "outlet_temperature_lower": max(0.0, 20.0 - outlet_temperature),
            "outlet_temperature_trip": max(
                0.0,
                outlet_temperature - float(self.p["outlet_temperature_trip"]),
            ),
            "flue_oxygen_trip": max(0.0, float(self.p["minimum_safe_oxygen"]) - oxygen),
            "flue_oxygen_upper": max(0.0, oxygen - 20.9),
        }

    def safety_margins(self, state, disturbances=None):
        del disturbances
        firebox_temperature = float(state[0])
        outlet_temperature = float(state[1])
        oxygen = float(state[2])
        outlet_trip = float(self.p["outlet_temperature_trip"])
        oxygen_trip = float(self.p["minimum_safe_oxygen"])
        return {
            "firebox_temperature_lower": (firebox_temperature - 20.0) / 1380.0,
            "firebox_temperature_upper": (1400.0 - firebox_temperature) / 1380.0,
            "outlet_temperature_lower": (outlet_temperature - 20.0)
            / (outlet_trip - 20.0),
            "outlet_temperature_trip": (outlet_trip - outlet_temperature)
            / (outlet_trip - 20.0),
            "flue_oxygen_trip": (oxygen - oxygen_trip) / (20.9 - oxygen_trip),
            "flue_oxygen_upper": (20.9 - oxygen) / 20.9,
        }

    def energy_kw(self, action):
        fuel_flow = float(action[1]) * float(self.p["maximum_fuel_flow"])
        return fuel_flow * float(self.p["fuel_lower_heating_value"]) / 1000.0

    def step_info(self, state, action, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        applied = self.default_action() if action is None else action
        combustion = self.combustion(applied, context)
        return {
            "y": self.outputs(state),
            "energy_kw": self.action_energy_kw(applied, state, context),
            "air_flow": float(combustion["air_flow"]),
            "fuel_flow": float(combustion["fuel_flow"]),
            "combustion_completeness": float(combustion["combustion_completeness"]),
            "combustion_duty_kw": float(combustion["duty"]) / 1000.0,
            "feed_temperature": float(context["feed_temperature"]),
            "ambient_temperature": float(context["ambient_temperature"]),
            "feed_flow": float(context["feed_flow"]),
            "fuel_heating_value_factor": float(context["fuel_heating_value_factor"]),
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
            f"unknown heater parameters: {unknown}; available: {sorted(defaults)}"
        )
    resolved = deepcopy(defaults)
    for name, value in supplied.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"heater parameter {name!r} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"heater parameter {name!r} must be finite")
        resolved[name] = number
    for name in (
        "maximum_fuel_flow",
        "fuel_lower_heating_value",
        "stoichiometric_air_fuel_ratio",
        "maximum_air_flow",
        "flue_gas_heat_capacity",
        "heat_transfer_coefficient",
        "firebox_heat_capacity",
        "process_heat_capacity",
        "nominal_feed_flow",
        "feed_specific_heat_capacity",
        "oxygen_time_constant",
    ):
        if resolved[name] <= 0.0:
            raise ValueError(f"heater parameter {name!r} must be positive")
    if not 240.0 <= resolved["feed_temperature"] <= 330.0:
        raise ValueError(
            "heater parameter 'feed_temperature' must stay within [240, 330]"
        )
    if not -10.0 <= resolved["ambient_temperature"] <= 45.0:
        raise ValueError(
            "heater parameter 'ambient_temperature' must stay within [-10, 45]"
        )
    if not 20.0 < resolved["outlet_temperature_trip"] <= 650.0:
        raise ValueError("heater outlet_temperature_trip must stay within (20, 650]")
    if not 0.0 <= resolved["minimum_safe_oxygen"] < 20.9:
        raise ValueError("heater minimum_safe_oxygen must stay within [0, 20.9)")
    return resolved


__all__ = ["HeaterModel"]
