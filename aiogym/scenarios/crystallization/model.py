"""Method-of-moments batch-crystallization model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from types import MappingProxyType
from typing import Any

import numpy as np

from aiogym.core.information import state_limit_rules

from aiogym.core.model import PhysicsModelBase, integrate_process_state


class CrystallizationModel(PhysicsModelBase):
    """Five-state batch model controlled by cooling-medium temperature."""

    scenario = "crystallization"
    dt_micro = 0.02
    reference_observation_suffix = "target"
    state_names = (
        "zeroth_crystal_moment",
        "first_crystal_moment",
        "second_crystal_moment",
        "third_crystal_moment",
        "solute_concentration",
    )
    state_units = {
        "zeroth_crystal_moment": "moment",
        "first_crystal_moment": "moment",
        "second_crystal_moment": "moment",
        "third_crystal_moment": "moment",
        "solute_concentration": "kg/L",
    }
    state_bounds = {
        "zeroth_crystal_moment": (0.0, 1.0e4),
        "first_crystal_moment": (0.0, 1.0e6),
        "second_crystal_moment": (0.0, 1.0e8),
        "third_crystal_moment": (0.0, 1.0e10),
        "solute_concentration": (0.0, 2.0),
    }
    action_names = ("cooling_temperature_fraction",)
    action_units = {"cooling_temperature_fraction": "fraction"}
    action_bounds = {"cooling_temperature_fraction": (0.0, 1.0)}
    action_kinds = {"cooling_temperature_fraction": "cooler"}
    output_names = ("coefficient_of_variation", "mean_crystal_size")
    output_units = {
        "coefficient_of_variation": "dimensionless",
        "mean_crystal_size": "um",
    }
    output_bounds = {
        "coefficient_of_variation": (0.7, 1.2),
        "mean_crystal_size": (8.0, 11.5),
    }
    input_disturbances = (
        {
            "name": "growth_rate_factor",
            "event": "growth_rate_factor_step",
            "unit": "fraction",
            "bounds": (0.7, 1.3),
            "default": 1.0,
            "description": "multiplicative crystal-growth-rate factor",
        },
        {
            "name": "nucleation_rate_factor",
            "event": "nucleation_rate_factor_step",
            "unit": "fraction",
            "bounds": (0.7, 1.3),
            "default": 1.0,
            "description": "multiplicative nucleation-rate factor",
        },
        {
            "name": "solubility_concentration_bias",
            "event": "solubility_concentration_bias_step",
            "unit": "g/L",
            "bounds": (-10.0, 10.0),
            "default": 0.0,
            "description": "additive solubility-curve offset",
        },
    )

    parameter_units = MappingProxyType(
        {
            "nucleation_rate_coefficient": "1/s",
            "nucleation_activation_temperature": "K",
            "nucleation_supersaturation_exponent": "dimensionless",
            "nucleation_third_moment_exponent": "dimensionless",
            "growth_rate_coefficient": "1/s",
            "growth_activation_temperature": "K",
            "growth_supersaturation_exponent": "dimensionless",
            "size_independent_growth_weight": "dimensionless",
            "size_dependent_growth_weight": "dimensionless",
            "crystal_shape_factor": "dimensionless",
            "crystal_density": "kg/L",
            "minimum_cooling_temperature": "degC",
            "maximum_cooling_temperature": "degC",
            "nominal_cooling_temperature": "degC",
            "numerical_epsilon": "dimensionless",
            "maximum_nucleation_rate": "1/s",
            "growth_rate_scale": "dimensionless",
            "maximum_growth_rate": "um/s",
        }
    )

    parameter_metadata = {
        'nucleation_rate_coefficient': ('Pre-exponential coefficient in the nucleation-rate model', 'Finite number > 0'),
        'nucleation_supersaturation_exponent': ('Supersaturation exponent in the nucleation-rate model', 'Finite number > 0'),
        'nucleation_third_moment_exponent': ('Third-moment exponent in the nucleation-rate model', 'Finite number > 0'),
        'growth_rate_coefficient': ('Pre-exponential coefficient in the crystal-growth model', 'Finite number > 0'),
        'growth_supersaturation_exponent': ('Supersaturation exponent in the crystal-growth model', 'Finite number > 0'),
        'size_independent_growth_weight': ('Weight of size-independent crystal growth', 'Finite number > 0'),
        'size_dependent_growth_weight': ('Weight of size-dependent crystal growth', 'Finite number > 0'),
        'crystal_shape_factor': ('Factor converting the third moment to crystal volume', 'Finite number > 0'),
        'crystal_density': ('Crystal material density', 'Finite number > 0'),
        'numerical_epsilon': ('Positive floor used in moment ratios and kinetic calculations', 'Finite number > 0'),
        'maximum_nucleation_rate': ('Upper cap on the nucleation rate including disturbance scaling', 'Finite number > 0'),
        'growth_rate_scale': ('Scale factor applied to the crystal-growth rate', 'Finite number > 0'),
        'maximum_growth_rate': ('Upper cap on the growth rate including disturbance scaling', 'Finite number > 0'),
        'nucleation_activation_temperature': ('Signed activation-temperature coefficient in the nucleation exponential', 'Finite number < 0'),
        'growth_activation_temperature': ('Signed activation-temperature coefficient in the growth exponential', 'Finite number < 0'),
        'minimum_cooling_temperature': ('Cooling-medium temperature at zero command', '0 <= minimum_cooling_temperature < maximum_cooling_temperature <= 100'),
        'maximum_cooling_temperature': ('Cooling-medium temperature at full command', '0 <= minimum_cooling_temperature < maximum_cooling_temperature <= 100'),
        'nominal_cooling_temperature': ('Cooling-medium temperature used for the default action', '[minimum_cooling_temperature, maximum_cooling_temperature]'),
    }
    variable_descriptions = {
        'zeroth_crystal_moment': 'Zeroth crystal-size-distribution moment (number-related)',
        'first_crystal_moment': 'First crystal-size-distribution moment (length-related)',
        'second_crystal_moment': 'Second crystal-size-distribution moment (area-related)',
        'third_crystal_moment': 'Third crystal-size-distribution moment (volume-related)',
        'solute_concentration': 'Dissolved solute concentration',
        'coefficient_of_variation': 'Crystal-size standard deviation divided by mean size',
        'mean_crystal_size': 'Mean crystal size derived from the moments',
        'cooling_temperature_fraction': 'Normalized cooling-medium temperature command',
    }

    def action_metadata(self):
        return {"cooling_temperature_fraction": {
            "interpretation": f"Temperature = {self.p['minimum_cooling_temperature']:g} + action * {self.p['maximum_cooling_temperature'] - self.p['minimum_cooling_temperature']:g} degC; larger commands mean warmer cooling medium",
        }}

    def observation_metadata(self):
        return {**super().observation_metadata(), "remaining_batch_time": {
            "description": "Normalized remaining planned batch time",
            "source": "remaining_time",
            "normalization": "t_remaining / (100 s + t_remaining)",
        }}

    def safety_metadata(self):
        return state_limit_rules(self.state_bounds, self.state_units)

    def __init__(self, parameters: Mapping[str, Any] | None = None):
        defaults = {
            "nucleation_rate_coefficient": 0.92,
            "nucleation_activation_temperature": -6800.0,
            "nucleation_supersaturation_exponent": 0.92,
            "nucleation_third_moment_exponent": 1.3,
            "growth_rate_coefficient": 48.0,
            "growth_activation_temperature": -4900.0,
            "growth_supersaturation_exponent": 1.9,
            "size_independent_growth_weight": 0.51,
            "size_dependent_growth_weight": 7.3,
            "crystal_shape_factor": 7.5,
            "crystal_density": 2.7,
            "minimum_cooling_temperature": 30.0,
            "maximum_cooling_temperature": 40.0,
            "nominal_cooling_temperature": 35.0,
            "numerical_epsilon": 1.0e-9,
            "maximum_nucleation_rate": 0.05,
            "growth_rate_scale": 2.0e-6,
            "maximum_growth_rate": 2.0e-4,
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
            raise KeyError(f"unknown crystallization parameter {name!r}") from error

    def default_disturbances(self):
        return dict(super().default_disturbances())

    def action_slew_limits(self):
        return None

    def initial_state(self):
        return [1.0, 15.0, 250.0, 4500.0, 0.90]

    def default_setpoint_vector(self):
        return [0.8735039653128486, 10.057345731688848]

    def default_action(self):
        minimum = float(self.p["minimum_cooling_temperature"])
        maximum = float(self.p["maximum_cooling_temperature"])
        nominal = float(self.p["nominal_cooling_temperature"])
        return [(nominal - minimum) / (maximum - minimum)]

    def observation_schema(self):
        return [*super().observation_schema(), {
            "name": "remaining_batch_time", "kind": "time",
            "unit": "t_remaining / (100 s + t_remaining)",
            "low": 0.0, "high": 1.0,
        }]

    def observation(self, state, reference, previous_action, disturbances, *, remaining_time):
        return [
            *super().observation(state, reference, previous_action, disturbances),
            remaining_time / (100.0 + remaining_time),
        ]

    def cooling_temperature(self, action):
        return self._cooling_temperature(self.action_vector(action))

    def _cooling_temperature(self, action):
        fraction = float(action[0])
        return float(self.p["minimum_cooling_temperature"]) + fraction * (
            float(self.p["maximum_cooling_temperature"])
            - float(self.p["minimum_cooling_temperature"])
        )

    def kinetics(self, state, action, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        return self._kinetics(self.state_vector(state), self.action_vector(action), context)

    def _kinetics(self, values, action, context):
        third_moment = float(values[3])
        concentration = float(values[4])
        cooling_temperature = self._cooling_temperature(action)
        absolute_temperature = cooling_temperature + 273.15
        equilibrium_concentration = (
            -686.2686
            + 3.579165 * absolute_temperature
            - 0.00292874 * absolute_temperature**2
            + float(context["solubility_concentration_bias"])
        )
        supersaturation = max(
            0.0,
            concentration * 1000.0 - equilibrium_concentration,
        )
        nucleation_rate = min(
            float(self.p["maximum_nucleation_rate"]),
            max(
                0.0,
                float(self.p["nucleation_rate_coefficient"])
                * math.exp(
                    float(self.p["nucleation_activation_temperature"])
                    / absolute_temperature
                )
                * supersaturation
                ** float(self.p["nucleation_supersaturation_exponent"])
                * max(abs(third_moment), float(self.p["numerical_epsilon"]))
                ** float(self.p["nucleation_third_moment_exponent"])
                * float(context["nucleation_rate_factor"]),
            ),
        )
        growth_rate = min(
            float(self.p["maximum_growth_rate"]),
            max(
                0.0,
                float(self.p["growth_rate_coefficient"])
                * math.exp(
                    float(self.p["growth_activation_temperature"])
                    / absolute_temperature
                )
                * supersaturation ** float(self.p["growth_supersaturation_exponent"])
                * float(self.p["growth_rate_scale"])
                * float(context["growth_rate_factor"]),
            ),
        )
        return {
            "cooling_temperature": cooling_temperature,
            "equilibrium_concentration": equilibrium_concentration,
            "supersaturation": supersaturation,
            "nucleation_rate": nucleation_rate,
            "growth_rate": growth_rate,
        }

    def _dynamics(self, state, action, disturbances):
        zeroth, first, second, third, _concentration = (float(value) for value in state)
        rates = self._kinetics(state, action, self._resolve_disturbances(disturbances))
        nucleation = float(rates["nucleation_rate"])
        growth = float(rates["growth_rate"])
        independent = float(self.p["size_independent_growth_weight"])
        dependent = float(self.p["size_dependent_growth_weight"])
        growth_second = independent * second * 1.0e-8 + dependent * third * 1.0e-12
        return [
            nucleation,
            growth * (independent * zeroth + dependent * first * 1.0e-4) * 1.0e4,
            2.0
            * growth
            * (independent * first * 1.0e-4 + dependent * second * 1.0e-8)
            * 1.0e8,
            3.0 * growth * growth_second * 1.0e12,
            -0.5
            * float(self.p["crystal_density"])
            * float(self.p["crystal_shape_factor"])
            * growth
            * growth_second,
        ]

    def crystal_quality(self, state):
        zeroth, first, second, _third, _concentration = (
            float(value) for value in state
        )
        epsilon = float(self.p["numerical_epsilon"])
        coefficient_of_variation = math.sqrt(
            max(second * zeroth / (first * first + epsilon) - 1.0, 0.0)
        )
        mean_crystal_size = first / (zeroth + epsilon)
        return [coefficient_of_variation, mean_crystal_size]

    def outputs(self, state):
        return self.crystal_quality(state)

    def batch_endpoint(
        self,
        action,
        *,
        horizon_steps=100,
        control_dt=1.0,
        initial_state=None,
        disturbances=None,
    ):
        if (
            isinstance(horizon_steps, bool)
            or int(horizon_steps) != horizon_steps
            or int(horizon_steps) <= 0
        ):
            raise ValueError("horizon_steps must be a positive integer")
        action_values = self.action_vector(action)
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        state = self.initial_state() if initial_state is None else initial_state
        for _ in range(int(horizon_steps)):
            state = integrate_process_state(
                self, state, action_values, context, duration=control_dt,
            )
        return {
            "state": state.tolist(),
            "output": self.outputs(state),
        }

    def tracking_steady_state_action(self, reference, disturbances=None):
        del disturbances
        _validated_reference(reference)
        return None

    def tracking_steady_state_state(self, reference, disturbances=None):
        del disturbances
        _validated_reference(reference)
        return None

    def measurement(self, state, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        values = [float(value) for value in state]
        return {
            "x": values,
            "y": self.outputs(values),
            "levels": [],
            "temps": [],
            "conc": [values[4]],
            **context,
        }

    def clamp_state(self, state):
        return [float(value) for value in state]

    def constraint_costs(self, state, disturbances=None):
        del disturbances
        costs = {}
        for name, value in zip(self.state_names, state):
            low, high = self.state_bounds[name]
            costs[f"{name}_lower"] = max(0.0, low - float(value))
            costs[f"{name}_upper"] = max(0.0, float(value) - high)
        return costs

    def safety_margins(self, state, disturbances=None):
        del disturbances
        margins = {}
        for name, value in zip(self.state_names, state):
            low, high = self.state_bounds[name]
            span = high - low
            margins[f"{name}_lower"] = (float(value) - low) / span
            margins[f"{name}_upper"] = (high - float(value)) / span
        return margins

    def energy_kw(self, action):
        del action
        return 0.0

    def step_info(self, state, action, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        applied = self.default_action() if action is None else action
        quality = self.crystal_quality(state)
        rates = self.kinetics(state, applied, context)
        return {
            "y": quality,
            "energy_kw": 0.0,
            "coefficient_of_variation": float(quality[0]),
            "mean_crystal_size": float(quality[1]),
            "solute_concentration": float(state[4]),
            **{name: float(value) for name, value in rates.items()},
        }


def _validated_reference(reference):
    values = np.asarray(reference, dtype=float).reshape(-1)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError(
            "crystallization reference must contain finite CV and mean size"
        )
    return values


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
            "unknown crystallization parameters: "
            f"{unknown}; available: {sorted(defaults)}"
        )
    resolved = deepcopy(defaults)
    for name, value in supplied.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"crystallization parameter {name!r} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"crystallization parameter {name!r} must be finite")
        resolved[name] = number
    for name in (
        "nucleation_rate_coefficient",
        "nucleation_supersaturation_exponent",
        "nucleation_third_moment_exponent",
        "growth_rate_coefficient",
        "growth_supersaturation_exponent",
        "size_independent_growth_weight",
        "size_dependent_growth_weight",
        "crystal_shape_factor",
        "crystal_density",
        "numerical_epsilon",
        "maximum_nucleation_rate",
        "growth_rate_scale",
        "maximum_growth_rate",
    ):
        if resolved[name] <= 0.0:
            raise ValueError(f"crystallization parameter {name!r} must be positive")
    for name in (
        "nucleation_activation_temperature",
        "growth_activation_temperature",
    ):
        if resolved[name] >= 0.0:
            raise ValueError(f"crystallization parameter {name!r} must be negative")
    minimum = resolved["minimum_cooling_temperature"]
    maximum = resolved["maximum_cooling_temperature"]
    nominal = resolved["nominal_cooling_temperature"]
    if not 0.0 <= minimum < maximum <= 100.0:
        raise ValueError(
            "crystallization cooling temperatures require 0 <= minimum < maximum <= 100"
        )
    if not minimum <= nominal <= maximum:
        raise ValueError(
            "crystallization nominal_cooling_temperature must stay within "
            "the cooling-temperature bounds"
        )
    return resolved


__all__ = ["CrystallizationModel"]
