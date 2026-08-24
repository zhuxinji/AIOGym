"""Shared numerical helpers for the built-in physics models."""

from __future__ import annotations

import math


# Volumetric heat capacity of liquid water, J/(m3*K).
RHO_CP = 4_186_000.0


def _copy_value(v):
    return list(v) if isinstance(v, list) else v


class PhysicsModelBase:
    """Shared vector, schema, disturbance, and numerical operations."""

    time_unit = "s"
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
        names = self._vector_names(self.state_names, "x", len(self.initial_state()))
        return [
            self._schema_row(name, self.state_units, self.state_bounds)
            for name in names
        ]

    def action_schema(self):
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
        return len(self._action_names())

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

    def default_action(self):
        return [0.5] * self.action_dim()

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

    def outputs(self, x):
        """Return the controlled-output vector derived from state ``x``."""

        return self.state_vector(x)

    def measurement(self, x, env=None):
        """Measured state dict exposed to controllers.

        Controllers should use x/y generically; scenario-specific consumers can
        also inspect levels, temps, conc, and disturbance names.
        """
        state = self.state_vector(x)
        display = self.display_outputs(state)
        measured = {
            "x": state,
            "levels": list(display["levels"]),
            "temps": list(display["temps"]),
            "y": list(self.outputs(state)),
        }
        for key, value in display.items():
            if key not in measured:
                measured[key] = value
        context = {} if env is None else dict(env)
        return {**measured, **context}

    def output_schema(self):
        y0 = list(self.outputs(self.initial_state()))
        names = self._vector_names(self.output_names, "y", len(y0))
        return [
            self._schema_row(name, self.output_units, self.output_bounds)
            for name in names
        ]

    def output_scales(self):
        scales = []
        for row in self.output_schema():
            low = float(row["low"])
            high = float(row["high"])
            scale = high - low if math.isfinite(low) and math.isfinite(high) else 1.0
            scales.append(max(float(scale), 1e-12))
        return scales

    def energy_kw(self, u):
        return 0.0

    def action_energy_kw(self, act, x=None, env=None):
        """Return total action energy rate in kW for numeric environment steps."""

        return float(self.energy_kw(self.action_vector(act)))

    def display_outputs(self, x):
        return {"levels": [], "temps": list(x)}

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

    def hard_termination_reasons(self, x, levels, temps, env):
        """Return unconditional physical termination reasons for a transition."""

        return ()

    def process_info(self, x, levels, temps, env):
        return {}
