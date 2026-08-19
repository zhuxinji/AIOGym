"""Shared numerical helpers for the built-in physics models."""
from __future__ import annotations


# Volumetric heat capacity of liquid water, J/(m3*K).
RHO_CP = 4_186_000.0


def _copy_value(v):
    return list(v) if isinstance(v, list) else v


class PhysicsModelBase:
    """Shared vector, schema, disturbance, and numerical operations."""

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
        {"name": "setpoint_move", "event": "setpoint_move", "kind": "setpoint", "description": "controlled-variable setpoint move"},
    )

    def _schema_row(self, name, units, bounds):
        return {"name": name, "unit": units[name], "bounds": bounds[name]}

    def state_schema(self):
        names = self._vector_names(self.state_names, "x", len(self.initial_state()))
        return [self._schema_row(name, self.state_units, self.state_bounds) for name in names]

    def action_schema(self):
        names = self._action_names()
        counters = {}
        rows = []
        for i, name in enumerate(names):
            kind = self.action_kinds[name]
            counters.setdefault(kind, 0)
            rows.append({
                "name": name,
                "kind": kind,
                "index": i,
                "kind_index": counters[kind],
                "unit": self.action_units[name],
                "bounds": self.action_bounds[name],
            })
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
            raise ValueError(f"{self.scenario} expected {expected} action values, got {len(values)}")
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
        """Semantic outputs derived from x.

        ``y`` is the generic controlled-output vector used for calculations.
        ``levels``/``temps`` are optional physical display channels for built-in
        process scenarios.
        """
        state = self.state_vector(x)
        display = self.display_outputs(state)
        out = {
            "x": state,
            "levels": list(display["levels"]),
            "temps": list(display["temps"]),
        }
        for key, value in display.items():
            if key not in out:
                out[key] = value
        out["y"] = self.controlled_output(state)
        return out

    def measurement(self, x, env=None):
        """Measured state dict exposed to controllers.

        Controllers should use x/y generically; scenario-specific consumers can
        also inspect levels, temps, conc, and disturbance names.
        """
        context = {} if env is None else dict(env)
        return {**self.outputs(x), **context}

    def controlled_output(self, x):
        return self.state_vector(x)

    def setpoint_vector(self, y_sp=None):
        return list(y_sp) if y_sp is not None else self.default_setpoint_vector()

    def controlled_output_schema(self):
        y0 = list(self.controlled_output(self.initial_state()))
        names = self._vector_names(self.output_names, "y", len(y0))
        rows = []
        for name in names:
            rows.append({
                "name": name,
                "unit": self.output_units[name],
                "bounds": self.output_bounds[name],
            })
        return rows

    def controlled_output_scales(self):
        scales = []
        for row in self.controlled_output_schema():
            bounds = row.get("bounds")
            scale = None
            if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
                lo, hi = bounds
                if lo is not None and hi is not None and float(hi) > float(lo):
                    scale = float(hi) - float(lo)
            scales.append(max(float(scale if scale is not None else 1.0), 1e-12))
        return scales

    def energy_kw(self, u):
        return 0.0

    def action_energy_kw(self, act, x=None, env=None):
        """Return total action energy rate in kW for numeric environment steps."""

        return float(self.energy_kw(self.action_vector(act)))

    def display_outputs(self, x):
        return {"levels": [], "temps": list(x)}

    def disturbance_schema(self):
        rows = []
        for row in self.input_disturbances + self.event_disturbances:
            out = dict(row)
            if "name" not in out:
                raise ValueError("disturbance schema rows require a name")
            out["dynamic"] = "event" in out
            if "default" not in out and out["name"] in self.p:
                out["default"] = _copy_value(self.p[out["name"]])
            rows.append(out)
        return rows

    def disturbance_defaults(self):
        defaults = {}
        for row in self.disturbance_schema():
            if "kind" in row and row["kind"] == "setpoint":
                continue
            defaults[row["name"]] = _copy_value(row["default"])
        return defaults

    def runtime_env(self, disturbance_values):
        env = self.disturbance_defaults()
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
