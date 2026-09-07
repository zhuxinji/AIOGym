"""Shared numerical helpers for the built-in physics models."""

from __future__ import annotations

import math
from functools import cached_property

import numpy as np


# Volumetric heat capacity of liquid water, J/(m3*K).
RHO_CP = 4_186_000.0


def apply_action_slew(model, previous_action, requested_action):
    """Apply one model-owned action slew step."""

    previous = np.asarray(previous_action, dtype=float).reshape(-1)
    requested = np.asarray(requested_action, dtype=float).reshape(-1)
    if previous.shape != requested.shape:
        raise ValueError("previous and requested actions must have matching shapes")
    limits = model.action_slew_limits()
    if limits is None:
        return requested.copy()
    maximum_step = np.asarray(limits, dtype=float).reshape(-1)
    if (
        maximum_step.shape != requested.shape
        or not np.isfinite(maximum_step).all()
        or np.any(maximum_step < 0.0)
    ):
        raise ValueError("model action slew limits must match actions")
    return np.clip(requested, previous - maximum_step, previous + maximum_step)


def integrate_process_state(
    model,
    state,
    action,
    disturbances,
    *,
    duration,
):
    """Advance one process state with the same RK4 cadence used by the Env."""

    interval = float(duration)
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError("integration duration must be finite and positive")
    result = np.asarray(state, dtype=float).reshape(-1)
    applied = np.asarray(action, dtype=float).reshape(-1)
    if result.shape != (len(model.state_schema()),) or not np.isfinite(result).all():
        raise ValueError("integration state must match the model state dimension")
    if not np.isfinite(applied).all():
        raise ValueError("integration action must contain finite values")
    maximum_step = float(model.dt_micro)
    substeps = max(1, math.ceil(interval / maximum_step - 1e-12))
    step = interval / substeps

    dynamics = lambda values: model.dynamics(values, applied, disturbances=disturbances)
    if isinstance(model, PhysicsModelBase) and type(model).dynamics is PhysicsModelBase.dynamics:
        # Built-in numeric kernels consume the canonical action once per interval.
        applied = model.action_vector(applied)
        dynamics = lambda values: model._dynamics(values, applied, disturbances)

    def derivative(values):
        output = np.asarray(
            dynamics(values),
            dtype=float,
        ).reshape(-1)
        if output.shape != result.shape:
            raise ValueError("model dynamics shape does not match state shape")
        return output

    for _ in range(substeps):
        k1 = derivative(result)
        k2 = derivative(result + 0.5 * step * k1)
        k3 = derivative(result + 0.5 * step * k2)
        k4 = derivative(result + step * k3)
        result = result + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        result = np.asarray(model.clamp_state(result), dtype=float)
        if not np.isfinite(result).all():
            raise FloatingPointError("model produced a non-finite state")
    return result


def _copy_value(v):
    return list(v) if isinstance(v, list) else v


class PhysicsModelBase:
    """Shared vector, schema, disturbance, and numerical operations."""

    time_unit = "s"
    reference_observation_suffix = "setpoint"
    state_names = ()
    state_units = {}
    state_bounds = {}
    action_names = ()
    action_units = {}
    action_bounds = {}
    action_kinds = {}
    output_names = ()
    output_units = {}
    output_bounds = {}
    input_disturbances = ()
    event_disturbances = (
        {
            "name": "setpoint_move",
            "event": "setpoint_move",
            "kind": "setpoint",
            "description": "controlled-variable setpoint move",
        },
    )

    def _schema_row(self, name, units, bounds):
        low, high = bounds[name]
        return {
            "name": name,
            "unit": units[name],
            "low": -math.inf if low is None else float(low),
            "high": math.inf if high is None else float(high),
        }

    def state_schema(self):
        return [dict(row) for row in self._cached_state_schema]

    @cached_property
    def _cached_state_schema(self):
        names = self._vector_names(self.state_names, "x", len(self.initial_state()))
        return [
            self._schema_row(name, self.state_units, self.state_bounds)
            for name in names
        ]

    def action_schema(self):
        return [dict(row) for row in self._cached_action_schema]

    @cached_property
    def _cached_action_schema(self):
        names = self._action_names()
        counters = {}
        rows = []
        for i, name in enumerate(names):
            kind = self.action_kinds[name]
            counters.setdefault(kind, 0)
            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "index": i,
                    "kind_index": counters[kind],
                    "unit": self.action_units[name],
                    "low": float(self.action_bounds[name][0]),
                    "high": float(self.action_bounds[name][1]),
                }
            )
            counters[kind] += 1
        return rows

    def action_dim(self):
        return len(self.action_names or self.action_bounds)

    def _action_names(self):
        if self.action_names:
            return list(self.action_names)
        if self.action_bounds:
            return list(self.action_bounds.keys())
        return []

    def action_vector(self, act):
        """Validate and return the canonical flat actuator vector ``u``."""
        try:
            values = [float(v) for v in act]
        except (TypeError, ValueError) as ex:
            raise ValueError(f"{self.scenario} action must be a numeric vector") from ex
        expected = self.action_dim()
        if len(values) != expected:
            raise ValueError(
                f"{self.scenario} expected {expected} action values, got {len(values)}"
            )
        return values

    def state_vector(self, x):
        """Return the generic state vector x used by controllers and simulators."""
        return [float(v) for v in x]

    def _vector_names(self, names, prefix: str, length: int):
        values = list(names)
        if len(values) != int(length):
            raise ValueError(
                f"{prefix} schema declares {len(values)} names for {length} values"
            )
        return values

    def dynamics(self, x, u, disturbances=None):
        """Generic continuous numeric dynamics ``dx/dt = f(x, u, env)``."""
        context = {} if disturbances is None else disturbances
        return self._dynamics(self.state_vector(x), self.action_vector(u), context)

    def output_schema(self):
        return [dict(row) for row in self._cached_output_schema]

    @cached_property
    def _cached_output_schema(self):
        y0 = list(self.outputs(self.initial_state()))
        names = self._vector_names(self.output_names, "y", len(y0))
        return [
            self._schema_row(name, self.output_units, self.output_bounds)
            for name in names
        ]

    def output_scales(self):
        return list(self._cached_output_scales)

    @cached_property
    def _cached_output_scales(self):
        scales = []
        for row in self.output_schema():
            low = float(row["low"])
            high = float(row["high"])
            scale = high - low if math.isfinite(low) and math.isfinite(high) else 1.0
            scales.append(max(float(scale), 1e-12))
        return scales

    def observation_schema(self):
        return [dict(row) for row in self._cached_observation_schema]

    @cached_property
    def _cached_observation_schema(self):
        return [
            *(
                {**row, "kind": "measurement", "low": 0.0, "high": 1.0}
                for row in self.state_schema()
            ),
            *(
                {
                    **row,
                    "name": f"{row['name']}_{self.reference_observation_suffix}",
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
            *self._normalize(state, self._cached_state_schema),
            *self._normalize(reference, self._cached_output_schema),
        ]

    def recompute_derived_observation(self, observation):
        """Models declaring derived channels must rebuild them after sensor noise."""
        raise NotImplementedError(
            f"{self.scenario} must implement recompute_derived_observation "
            "for derived observation channels"
        )

    def measurement_from_observation(self, observation, disturbances=None):
        values = np.asarray(observation, dtype=float).reshape(-1)
        expected = len(self.observation_schema())
        if values.shape != (expected,) or not np.isfinite(values).all():
            raise ValueError(
                f"{self.scenario} policy observation must match observation_schema"
            )
        state_rows = self.state_schema()
        state_dimension = len(state_rows)
        low = np.asarray([row["low"] for row in state_rows], dtype=float)
        high = np.asarray([row["high"] for row in state_rows], dtype=float)
        state = low + values[:state_dimension] * (high - low)
        return self.measurement(state, disturbances)

    @staticmethod
    def _normalize(values, rows):
        return [
            (float(value) - float(row["low"]))
            / (float(row["high"]) - float(row["low"]))
            for value, row in zip(values, rows)
        ]

    def energy_kw(self, u):
        return 0.0

    def action_energy_kw(self, act, x=None, env=None):
        """Return total action energy rate in kW for numeric environment steps."""

        return float(self.energy_kw(self.action_vector(act)))

    def default_disturbances(self):
        defaults = {}
        for row in self.input_disturbances + self.event_disturbances:
            out = dict(row)
            if "name" not in out:
                raise ValueError("disturbance schema rows require a name")
            if out.get("kind") == "setpoint":
                continue
            if "default" not in out and out["name"] in self.p:
                out["default"] = _copy_value(self.p[out["name"]])
            if "default" not in out:
                raise ValueError(
                    f"disturbance {out['name']!r} requires an explicit default"
                )
            defaults[out["name"]] = _copy_value(out["default"])
        return defaults

    def _resolve_disturbances(self, disturbance_values):
        env = self.default_disturbances()
        env.update(
            {
                name: _copy_value(value)
                for name, value in dict(disturbance_values).items()
            }
        )
        return env
