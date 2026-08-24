"""Five-stage counter-current liquid-gas extraction model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np

from aiogym.core.model import PhysicsModelBase


class ExtractionModel(PhysicsModelBase):
    """Five idealized stages with liquid and gas feed-flow actuation."""

    scenario = "extraction"
    time_unit = "h"
    dt_micro = 0.01
    state_names = tuple(
        name
        for stage in range(1, 6)
        for name in (
            f"stage_{stage}_liquid_concentration",
            f"stage_{stage}_gas_concentration",
        )
    )
    state_units = {name: "fraction" for name in state_names}
    state_bounds = {name: (0.0, 1.0) for name in state_names}
    action_names = ("liquid_feed_flow", "gas_feed_flow")
    action_units = {name: "normalized_flow" for name in action_names}
    action_bounds = {name: (0.0, 1.0) for name in action_names}
    action_kinds = {name: "pump" for name in action_names}
    output_names = ("stage_5_liquid_concentration",)
    output_units = {"stage_5_liquid_concentration": "fraction"}
    output_bounds = {"stage_5_liquid_concentration": (0.05, 0.50)}
    input_disturbances = (
        {
            "name": "liquid_feed_concentration",
            "event": "liquid_feed_concentration_step",
            "unit": "fraction",
            "bounds": (0.3, 0.9),
            "default": 0.6,
            "description": "solute concentration in the liquid feed",
        },
        {
            "name": "gas_feed_concentration",
            "event": "gas_feed_concentration_step",
            "unit": "fraction",
            "bounds": (0.0, 0.12),
            "default": 0.05,
            "description": "solute concentration in the gas feed",
        },
        {
            "name": "mass_transfer_coefficient",
            "event": "mass_transfer_shift",
            "unit": "1/h",
            "bounds": (2.0, 8.0),
            "default": 5.0,
            "description": "volumetric liquid-gas mass-transfer coefficient",
        },
    )

    parameter_units = MappingProxyType(
        {
            "liquid_stage_volume": "volume",
            "gas_stage_volume": "volume",
            "equilibrium_constant": "dimensionless",
            "mass_transfer_coefficient": "1/h",
            "equilibrium_exponent": "dimensionless",
            "liquid_feed_concentration": "fraction",
            "gas_feed_concentration": "fraction",
            "minimum_liquid_flow": "volume/h",
            "maximum_liquid_flow": "volume/h",
            "minimum_gas_flow": "volume/h",
            "maximum_gas_flow": "volume/h",
            "nominal_gas_action": "fraction",
            "maximum_liquid_pump_power": "W",
            "maximum_gas_pump_power": "W",
        }
    )

    def __init__(self, parameters: Mapping[str, Any] | None = None):
        defaults = {
            "liquid_stage_volume": 5.0,
            "gas_stage_volume": 5.0,
            "equilibrium_constant": 1.0,
            "mass_transfer_coefficient": 5.0,
            "equilibrium_exponent": 2.0,
            "liquid_feed_concentration": 0.6,
            "gas_feed_concentration": 0.05,
            "minimum_liquid_flow": 5.0,
            "maximum_liquid_flow": 500.0,
            "minimum_gas_flow": 10.0,
            "maximum_gas_flow": 1000.0,
            "nominal_gas_action": 0.5,
            "maximum_liquid_pump_power": 1000.0,
            "maximum_gas_pump_power": 1000.0,
        }
        self.p = _resolved_parameters(defaults, parameters)
        self._resolved_parameters = MappingProxyType(dict(self.p))
        self._steady_cache = {}

    @property
    def resolved_parameters(self) -> Mapping[str, Any]:
        return self._resolved_parameters

    def parameter(self, name: str) -> float:
        try:
            return float(self.p[str(name)])
        except KeyError as error:
            raise KeyError(f"unknown extraction parameter {name!r}") from error

    def default_disturbances(self):
        defaults = dict(super().default_disturbances())
        defaults.update(
            {
                "liquid_feed_concentration": float(self.p["liquid_feed_concentration"]),
                "gas_feed_concentration": float(self.p["gas_feed_concentration"]),
                "mass_transfer_coefficient": float(self.p["mass_transfer_coefficient"]),
            }
        )
        return defaults

    def action_slew_limits(self):
        return None

    def default_setpoint_vector(self):
        return [0.30]

    def initial_state(self):
        state = self.tracking_steady_state_state(
            self.default_setpoint_vector(),
            self.default_disturbances(),
        )
        if state is None:
            raise ValueError("default extraction operating point is infeasible")
        return state

    def default_action(self):
        action = self.tracking_steady_state_action(
            self.default_setpoint_vector(),
            self.default_disturbances(),
        )
        if action is None:
            raise ValueError("default extraction operating point is infeasible")
        return action

    def physical_flows(self, action):
        values = self.action_vector(action)
        liquid = float(self.p["minimum_liquid_flow"]) + values[0] * (
            float(self.p["maximum_liquid_flow"]) - float(self.p["minimum_liquid_flow"])
        )
        gas = float(self.p["minimum_gas_flow"]) + values[1] * (
            float(self.p["maximum_gas_flow"]) - float(self.p["minimum_gas_flow"])
        )
        return liquid, gas

    def _dynamics(self, state, action, disturbances):
        context = self._resolve_disturbances(disturbances)
        liquid_flow, gas_flow = self.physical_flows(action)
        liquid_volume = float(self.p["liquid_stage_volume"])
        gas_volume = float(self.p["gas_stage_volume"])
        equilibrium_constant = float(self.p["equilibrium_constant"])
        exponent = float(self.p["equilibrium_exponent"])
        coefficient = float(context["mass_transfer_coefficient"])
        liquid_feed = float(context["liquid_feed_concentration"])
        gas_feed = float(context["gas_feed_concentration"])
        derivative = []
        for stage in range(5):
            liquid = float(state[2 * stage])
            gas = float(state[2 * stage + 1])
            previous_liquid = (
                liquid_feed if stage == 0 else float(state[2 * (stage - 1)])
            )
            next_gas = gas_feed if stage == 4 else float(state[2 * (stage + 1) + 1])
            equilibrium_liquid = max(gas, 0.0) ** exponent / equilibrium_constant
            transfer = coefficient * (liquid - equilibrium_liquid) * liquid_volume
            derivative.extend(
                [
                    (liquid_flow * (previous_liquid - liquid) - transfer)
                    / liquid_volume,
                    (gas_flow * (next_gas - gas) + transfer) / gas_volume,
                ]
            )
        return derivative

    def outputs(self, state):
        return [float(state[8])]

    def display_outputs(self, state):
        return {"levels": [], "temps": []}

    def liquid_concentrations(self, state):
        return [float(state[2 * stage]) for stage in range(5)]

    def gas_concentrations(self, state):
        return [float(state[2 * stage + 1]) for stage in range(5)]

    def _equilibrium_state(self, action, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        liquid_flow, gas_flow = self.physical_flows(action)

        def residual(first_gas):
            _state, terminal_gas = self._propagate_equilibrium(
                first_gas,
                liquid_flow,
                gas_flow,
                context,
            )
            return terminal_gas - float(context["gas_feed_concentration"])

        lower = 0.0
        upper = 1.0
        lower_residual = residual(lower)
        upper_residual = residual(upper)
        if lower_residual == 0.0:
            root = lower
        elif upper_residual == 0.0:
            root = upper
        elif lower_residual * upper_residual > 0.0:
            return None
        else:
            for _iteration in range(60):
                midpoint = 0.5 * (lower + upper)
                midpoint_residual = residual(midpoint)
                if lower_residual * midpoint_residual <= 0.0:
                    upper = midpoint
                else:
                    lower = midpoint
                    lower_residual = midpoint_residual
            root = 0.5 * (lower + upper)
        state, _terminal_gas = self._propagate_equilibrium(
            root,
            liquid_flow,
            gas_flow,
            context,
        )
        values = np.asarray(state, dtype=float)
        if (
            not np.isfinite(values).all()
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            return None
        return values.tolist()

    def _propagate_equilibrium(
        self,
        first_gas,
        liquid_flow,
        gas_flow,
        disturbances,
    ):
        liquid_volume = float(self.p["liquid_stage_volume"])
        equilibrium_constant = float(self.p["equilibrium_constant"])
        exponent = float(self.p["equilibrium_exponent"])
        coefficient = float(disturbances["mass_transfer_coefficient"])
        previous_liquid = float(disturbances["liquid_feed_concentration"])
        gas = float(first_gas)
        state = []
        capacity = coefficient * liquid_volume
        for _stage in range(5):
            transfer = (
                capacity
                * (previous_liquid - gas**exponent / equilibrium_constant)
                / (1.0 + capacity / liquid_flow)
            )
            liquid = previous_liquid - transfer / liquid_flow
            next_gas = gas - transfer / gas_flow
            state.extend([liquid, gas])
            previous_liquid = liquid
            gas = next_gas
        return state, gas

    def _steady_operating_point(self, reference, disturbances=None):
        target = np.asarray(reference, dtype=float).reshape(-1)
        if target.shape != (1,) or not np.isfinite(target).all():
            raise ValueError(
                "extraction reference must contain one finite concentration"
            )
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        cache_key = (
            float(target[0]),
            float(context["liquid_feed_concentration"]),
            float(context["gas_feed_concentration"]),
            float(context["mass_transfer_coefficient"]),
        )
        if cache_key in self._steady_cache:
            cached = self._steady_cache[cache_key]
            return {
                "state": list(cached["state"]),
                "action": list(cached["action"]),
            }
        gas_action = float(self.p["nominal_gas_action"])

        def state_and_error(liquid_action):
            action = [float(liquid_action), gas_action]
            state = self._equilibrium_state(action, context)
            if state is None:
                return None, None
            return state, float(state[8]) - float(target[0])

        lower = 0.0
        upper = 1.0
        lower_state, lower_error = state_and_error(lower)
        upper_state, upper_error = state_and_error(upper)
        if lower_state is None or upper_state is None:
            return None
        if lower_error == 0.0:
            liquid_action = lower
            state = lower_state
        elif upper_error == 0.0:
            liquid_action = upper
            state = upper_state
        elif lower_error * upper_error > 0.0:
            return None
        else:
            state = None
            for _iteration in range(60):
                midpoint = 0.5 * (lower + upper)
                midpoint_state, midpoint_error = state_and_error(midpoint)
                if midpoint_state is None:
                    return None
                if lower_error * midpoint_error <= 0.0:
                    upper = midpoint
                else:
                    lower = midpoint
                    lower_error = midpoint_error
                state = midpoint_state
            liquid_action = 0.5 * (lower + upper)
            state = self._equilibrium_state(
                [liquid_action, gas_action],
                context,
            )
        if state is None:
            return None
        action = [liquid_action, gas_action]
        derivative = np.asarray(self.dynamics(state, action, context), dtype=float)
        if not np.allclose(derivative, 0.0, rtol=0.0, atol=1e-10):
            return None
        result = {
            "state": tuple(float(value) for value in state),
            "action": tuple(float(value) for value in action),
        }
        self._steady_cache[cache_key] = result
        return {
            "state": list(result["state"]),
            "action": list(result["action"]),
        }

    def tracking_steady_state_action(self, reference, disturbances=None):
        operating_point = self._steady_operating_point(reference, disturbances)
        return None if operating_point is None else operating_point["action"]

    def tracking_steady_state_state(self, reference, disturbances=None):
        operating_point = self._steady_operating_point(reference, disturbances)
        return None if operating_point is None else operating_point["state"]

    def observation_schema(self):
        return [
            *(
                {**row, "kind": "measurement", "low": 0.0, "high": 1.0}
                for row in self.state_schema()
            ),
            *(
                {
                    **row,
                    "name": f"{row['name']}_setpoint",
                    "kind": "reference",
                    "low": 0.0,
                    "high": 1.0,
                }
                for row in self.output_schema()
            ),
        ]

    def observation(self, state, reference, previous_action, disturbances):
        del previous_action, disturbances
        return [
            *_normalized(state, self.state_schema()),
            *_normalized(reference, self.output_schema()),
        ]

    def measurement(self, state, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        liquids = self.liquid_concentrations(state)
        gases = self.gas_concentrations(state)
        return {
            "x": [float(value) for value in state],
            "y": [liquids[-1]],
            "levels": [],
            "temps": [],
            "conc": liquids,
            "liquid_concentrations": liquids,
            "gas_concentrations": gases,
            **context,
        }

    def measurement_from_observation(self, observation, disturbances=None):
        values = np.asarray(observation, dtype=float).reshape(-1)
        expected = len(self.observation_schema())
        if values.shape != (expected,) or not np.isfinite(values).all():
            raise ValueError(
                "extraction policy observation must match observation_schema"
            )
        state_rows = self.state_schema()
        state_dimension = len(state_rows)
        low = np.asarray([row["low"] for row in state_rows], dtype=float)
        high = np.asarray([row["high"] for row in state_rows], dtype=float)
        state = low + values[:state_dimension] * (high - low)
        return self.measurement(state, disturbances)

    def clamp_state(self, state):
        return [float(value) for value in state]

    def constraint_costs(self, state, disturbances=None):
        del disturbances
        costs = {}
        for name, value in zip(self.state_names, state):
            costs[f"{name}_lower"] = max(0.0, -float(value))
            costs[f"{name}_upper"] = max(0.0, float(value) - 1.0)
        return costs

    def safety_margins(self, state, disturbances=None):
        del disturbances
        margins = {}
        for name, value in zip(self.state_names, state):
            margins[f"{name}_lower"] = float(value)
            margins[f"{name}_upper"] = 1.0 - float(value)
        return margins

    def energy_kw(self, action):
        return (
            float(action[0]) * float(self.p["maximum_liquid_pump_power"])
            + float(action[1]) * float(self.p["maximum_gas_pump_power"])
        ) / 1000.0

    def step_info(self, state, action, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        applied = self.default_action() if action is None else action
        liquid_flow, gas_flow = self.physical_flows(applied)
        feed = float(context["liquid_feed_concentration"])
        outlet = float(state[8])
        removal = 0.0 if feed <= 0.0 else (feed - outlet) / feed
        return {
            "y": self.outputs(state),
            "energy_kw": self.action_energy_kw(applied, state, context),
            "liquid_feed_flow": liquid_flow,
            "gas_feed_flow": gas_flow,
            "stage_5_liquid_concentration": outlet,
            "removal_fraction": removal,
        }


def _normalized(values, rows):
    return [
        (float(value) - float(row["low"])) / (float(row["high"]) - float(row["low"]))
        for value, row in zip(values, rows)
    ]


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
            f"unknown extraction parameters: {unknown}; available: {sorted(defaults)}"
        )
    resolved = deepcopy(defaults)
    for name, value in supplied.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"extraction parameter {name!r} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"extraction parameter {name!r} must be finite")
        resolved[name] = number
    for name in (
        "liquid_stage_volume",
        "gas_stage_volume",
        "equilibrium_constant",
        "mass_transfer_coefficient",
        "equilibrium_exponent",
        "maximum_liquid_pump_power",
        "maximum_gas_pump_power",
    ):
        if resolved[name] <= 0.0:
            raise ValueError(f"extraction parameter {name!r} must be positive")
    for name in (
        "liquid_feed_concentration",
        "gas_feed_concentration",
        "nominal_gas_action",
    ):
        if not 0.0 <= resolved[name] <= 1.0:
            raise ValueError(f"extraction parameter {name!r} must be within [0, 1]")
    for minimum, maximum in (
        ("minimum_liquid_flow", "maximum_liquid_flow"),
        ("minimum_gas_flow", "maximum_gas_flow"),
    ):
        if resolved[minimum] < 0.0 or resolved[maximum] <= resolved[minimum]:
            raise ValueError(f"extraction requires 0 <= {minimum} < {maximum}")
    return resolved


__all__ = ["ExtractionModel"]
