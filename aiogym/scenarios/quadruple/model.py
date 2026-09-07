"""Public quadruple-tank process model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import numpy as np

from .physics import _QuadruplePhysicsKernel


class QuadrupleModel(_QuadruplePhysicsKernel):
    """Johansson quadruple-tank model with validated parameter overrides."""

    parameter_units = MappingProxyType(
        {
            "tank_area": "cm^2",
            "outlet_area": "cm^2",
            "pump_gain": "cm^3/(s*V)",
            "gamma": "fraction",
            "gravity": "cm/s^2",
            "max_voltage": "V",
            "max_level": "cm",
            "nominal_voltage": "V",
        }
    )

    def __init__(self, parameters: Mapping[str, Any] | None = None):
        super().__init__()
        self.p = _resolved_parameters(self.p, parameters)
        self._resolved_parameters = MappingProxyType(
            {
                name: tuple(value) if isinstance(value, list) else value
                for name, value in self.p.items()
            }
        )

    @property
    def resolved_parameters(self) -> Mapping[str, Any]:
        return self._resolved_parameters

    def parameter(self, name):
        try:
            return deepcopy(self.p[str(name)])
        except KeyError as error:
            raise KeyError(f"unknown quadruple parameter {name!r}") from error

    def default_disturbances(self):
        return dict(super().default_disturbances())

    def action_slew_limits(self):
        return None

    def observation_schema(self):
        state = self.state_schema()
        reference = self.output_schema()
        return [
            *({**row, "kind": "measurement", "low": 0.0, "high": 1.0} for row in state),
            *(
                {
                    **row,
                    "name": f"{row['name']}_setpoint",
                    "kind": "reference",
                    "low": 0.0,
                    "high": 1.0,
                }
                for row in reference
            ),
            *(
                {
                    "name": f"{row['name']}_tracking_error",
                    "kind": "derived",
                    "unit": "normalized_error",
                    "low": -1.0,
                    "high": 1.0,
                }
                for row in reference
            ),
        ]

    def observation(self, state, reference, previous_action, disturbances):
        del previous_action, disturbances
        state_rows = self.state_schema()
        normalized_state = [
            (float(value) - float(row["low"]))
            / (float(row["high"]) - float(row["low"]))
            for value, row in zip(state, state_rows)
        ]
        reference_rows = self.output_schema()
        normalized_reference = [
            (float(value) - float(row["low"]))
            / (float(row["high"]) - float(row["low"]))
            for value, row in zip(reference, reference_rows)
        ]
        normalized = np.clip([*normalized_state, *normalized_reference], 0.0, 1.0)
        error = normalized[:2] - normalized[4:6]
        return [*normalized.tolist(), *error.tolist()]

    def recompute_derived_observation(self, observation):
        values = np.asarray(observation, dtype=float).copy()
        values[6:] = values[:2] - values[4:6]
        return values

    def measurement(self, state, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        display = self.display_outputs(state)
        return {
            "x": list(state),
            "levels": list(display["levels"]),
            "temps": list(display["temps"]),
            "y": list(self.outputs(state)),
            **context,
        }

    def measurement_from_observation(self, observation, disturbances=None):
        values = np.asarray(observation, dtype=float).reshape(-1)
        expected = len(self.observation_schema())
        if values.shape != (expected,) or not np.isfinite(values).all():
            raise ValueError(
                "quadruple policy observation must match observation_schema"
            )
        state_rows = self.state_schema()
        state_dim = len(state_rows)
        low = np.asarray([row["low"] for row in state_rows], dtype=float)
        high = np.asarray([row["high"] for row in state_rows], dtype=float)
        state = low + values[:state_dim] * (high - low)
        return self.measurement(state, disturbances)

    def tracking_steady_state_action(self, reference, disturbances=None):
        del disturbances
        return super().tracking_steady_state_action(reference)

    def tracking_steady_state_state(self, reference, disturbances=None):
        action = self.tracking_steady_state_action(reference, disturbances)
        if action is None:
            return None
        maximum = float(self.p["max_voltage"])
        return self.equilibrium_state([value * maximum for value in action])

    def constraint_costs(self, state, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        display = self.display_outputs(state)
        reasons = self.hard_termination_reasons(
            state,
            display["levels"],
            display["temps"],
            context,
        )
        return {str(reason): 1.0 for reason in reasons}

    def hard_termination_reasons(self, x, levels, temps, env):
        del levels, temps, env
        values = [float(value) for value in x]
        reasons = []
        if any(value < 0.0 for value in values):
            reasons.append("negative_level")
        if any(value > float(self.p["max_level"]) for value in values):
            reasons.append("tank_overflow_limit")
        return tuple(reasons)

    def safety_margins(self, state, disturbances=None):
        del disturbances
        maximum = float(self.p["max_level"])
        values = [float(value) for value in state]
        return {
            **{
                f"tank_{index + 1}_lower": value / maximum
                for index, value in enumerate(values)
            },
            **{
                f"tank_{index + 1}_upper": (maximum - value) / maximum
                for index, value in enumerate(values)
            },
        }

    def step_info(self, state, action, disturbances=None):
        context = self._resolve_disturbances(
            {} if disturbances is None else disturbances
        )
        display = self.display_outputs(state)
        applied = self.default_action() if action is None else action
        info = self.process_info(
            state,
            display["levels"],
            display["temps"],
            context,
        )
        info["y"] = list(self.outputs(state))
        info["energy_kw"] = self.action_energy_kw(applied, state, context)
        return info


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
            f"unknown quadruple parameters: {unknown}; available: {sorted(defaults)}"
        )
    resolved = deepcopy(defaults)
    for name, value in supplied.items():
        default = defaults[name]
        if isinstance(default, list):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"quadruple parameter {name!r} must be a sequence")
            values = [_finite_number(name, item) for item in value]
            if len(values) != len(default):
                raise ValueError(
                    f"quadruple parameter {name!r} must contain {len(default)} values"
                )
            resolved[name] = values
        else:
            resolved[name] = _finite_number(name, value)
    for name in ("tank_area", "outlet_area", "pump_gain"):
        if any(value <= 0.0 for value in resolved[name]):
            raise ValueError(f"quadruple parameter {name!r} must be positive")
    for name in ("gravity", "max_voltage", "max_level"):
        if resolved[name] <= 0.0:
            raise ValueError(f"quadruple parameter {name!r} must be positive")
    if any(value < 0.0 or value > 1.0 for value in resolved["gamma"]):
        raise ValueError("quadruple parameter 'gamma' must stay within [0, 1]")
    if any(
        value < 0.0 or value > resolved["max_voltage"]
        for value in resolved["nominal_voltage"]
    ):
        raise ValueError("quadruple nominal_voltage must stay within max_voltage")
    return resolved


def _finite_number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"quadruple parameter {name!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"quadruple parameter {name!r} must be finite")
    return number


__all__ = ["QuadrupleModel"]
