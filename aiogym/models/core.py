"""Native NumPy/pure-Python process models for AIO-Gym.

State layout:
  cascade / quadruple : x = [h0, T0, h1, T1, ...]   (level, temp interleaved)
  cstr                : x = [Ca, T]
  hvac                : x = [T0, T1]
  heater              : x = [T_firebox, T_out, O2]
Actions are the actuator vector in [0, 1]: [pumps..., valves..., heaters...].
"""
from __future__ import annotations
import copy
import math

from .._internal.identifiers import canonical_scenario_id
from .._internal.vocabulary import GOAL_NAMES
from .backends import _NUMERIC_OPS, _NumericOps, _casadi_ops, _maxv
from .integration import Integrator

RHO = 1000.0
CP = 4186.0
G = 9.81
# Volumetric heat capacity of liquid water, J/(m3*K).
RHO_CP = RHO * CP


def _copy_value(v):
    return list(v) if isinstance(v, list) else v


def _is_model_instance(obj):
    return hasattr(obj, "dynamics") and hasattr(obj, "initial_state") and not isinstance(obj, (str, type))


class ProcessModelContract:
    """Common read-only contract for process-model metadata.

    The numerical dynamics stay in each concrete model. This layer only exposes
    names, units, bounds, disturbances, and safety constraints for benchmark and
    custom-model tooling.
    """

    display_name = "Process model"
    summary = ""
    supported_goals = GOAL_NAMES
    benchmark_goals = GOAL_NAMES
    state_names = ()
    state_units = {}
    state_bounds = {}
    param_units = {}
    param_bounds = {}
    action_names = ()
    action_units = {}
    action_bounds = {}
    action_kinds = {}
    output_names = ()
    output_units = {}
    output_bounds = {}
    setpoint_names = ()
    setpoint_units = {}
    setpoint_bounds = {}
    default_y_sp = ()
    plant_regime = {}
    economic_config = {
        "temp_band": (),
        "level_band": (),
        "value": "none",
        "w_value": 0.0,
        "w_energy": 0.0,
        "w_viol": 0.0,
    }
    supervisory_layout = ()
    supports_integral_observation = True
    randomize_common_temperatures = True
    disturbance_attributes = {
        "t_cold": "t_cold",
        "t_amb": "t_amb",
        "extra_outflow": "extra_outflow",
        "Caf": "caf",
        "Tcool": "tcool",
        "pump_flow_factor": "pump_flow_factor",
        "heater_efficiency": "heater_efficiency",
        "heat_loss_factor": "heat_loss_factor",
        "heat_load": "heat_load",
        "hvac_efficiency": "hvac_efficiency",
        "growth_factor": "growth_factor",
        "nucleation_factor": "nucleation_factor",
        "solubility_bias": "solubility_bias",
    }
    input_disturbances = (
        {"name": "t_cold", "event": "cold_inlet_step", "unit": "degC", "bounds": (0.0, 40.0), "description": "inlet or cold-source temperature"},
        {"name": "t_amb", "event": "ambient_step", "unit": "degC", "bounds": (0.0, 45.0), "description": "ambient temperature"},
        {"name": "extra_outflow", "event": "demand_surge", "unit": "m3/s", "bounds": (0.0, 0.002), "default": 0.0, "description": "downstream demand outflow"},
    )
    event_disturbances = (
        {"name": "setpoint_move", "event": "setpoint_move", "kind": "setpoint", "description": "controlled-variable setpoint move"},
    )
    safety_constraints = ()

    def _extra_parameters(self):
        return {}

    def _schema_row(self, name, units, bounds):
        return {"name": name, "unit": units.get(name, ""), "bounds": bounds.get(name)}

    def state_schema(self):
        names = self._vector_names(self.state_names, "x", len(self.initial_state()))
        return [self._schema_row(name, self.state_units, self.state_bounds) for name in names]

    def action_schema(self):
        names = self._action_names()
        counters = {}
        rows = []
        for i, name in enumerate(names):
            kind = self.action_kinds.get(name, "input")
            counters.setdefault(kind, 0)
            rows.append({
                "name": name,
                "kind": kind,
                "index": i,
                "kind_index": counters[kind],
                "unit": self.action_units.get(name, "fraction"),
                "bounds": self.action_bounds.get(name, (0.0, 1.0)),
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

    def physical_action_vector(self, act):
        """Return actuator values in the model's physical tracking-cost units.

        Models whose canonical action vector is normalized should override this
        method.  The default preserves existing models whose public actuator
        vector is already expressed in its reporting unit.
        """

        return self.action_vector(act)

    def default_action(self):
        return [0.5] * self.action_dim()

    def state_vector(self, x):
        """Return the generic state vector x used by controllers and simulators."""
        return [float(v) for v in x]

    def _vector_names(self, names, prefix: str, length: int):
        values = list(names)
        if len(values) != int(length):
            values = [f"{prefix}{i}" for i in range(int(length))]
        return values

    def dynamics(self, x, u, env=None, backend="numeric", ca=None):
        """Generic continuous dynamics dx/dt = f(x, u, env).

        ``backend="numeric"`` returns numbers for simulation. ``backend="casadi"``
        returns a CasADi expression graph for NMPC, using the same model formula.
        """
        if callable(getattr(self, "_dynamics", None)):
            if backend == "casadi":
                if ca is None:
                    raise ValueError("backend='casadi' requires the casadi module as ca=...")
                return self._dynamics(x, u, self.dynamics_disturbance_map(env), _casadi_ops(ca))
            if backend != "numeric":
                raise ValueError(f"unknown dynamics backend: {backend!r}")
            return self._dynamics(self.state_vector(x), self.action_vector(u), env or {}, _NUMERIC_OPS)
        if backend != "numeric":
            raise NotImplementedError(f"{self.scenario} does not support backend={backend!r} dynamics")
        raise NotImplementedError(
            f"{self.scenario} must implement _dynamics(x, u, d, ops) or override dynamics()"
        )

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
            "levels": list(display.get("levels", [])),
            "temps": list(display.get("temps", [])),
        }
        for key, value in display.items():
            if key not in out:
                out[key] = value
        if callable(getattr(self, "conc", None)):
            out["conc"] = self.conc(state)
        out["y"] = self.controlled_output(state)
        return out

    def measurement(self, x, env=None):
        """Measured state dict exposed to controllers.

        Controllers should use x/y generically; scenario-specific consumers can
        also inspect levels, temps, conc, and disturbance names.
        """
        return {**self.outputs(x), **dict(env or {})}

    def controlled_output(self, x, backend="numeric", ca=None):
        if backend == "casadi":
            return [x[i] for i in range(len(self.initial_state()))]
        return self.state_vector(x)

    def setpoint_vector(self, y_sp=None):
        return list(y_sp) if y_sp is not None else self.default_setpoint_vector()

    def default_setpoint_vector(self):
        if self.default_y_sp:
            return list(self.default_y_sp)
        values = []
        for row in self.setpoint_schema():
            bounds = row.get("bounds")
            if isinstance(bounds, (tuple, list)) and len(bounds) == 2 and bounds[0] is not None and bounds[1] is not None:
                values.append(0.5 * (float(bounds[0]) + float(bounds[1])))
            else:
                values.append(0.0)
        return values

    def env_setpoint_vector(self, options=None):
        """Return initial setpoints for an environment instance."""

        return self.default_setpoint_vector()

    def sample_env_setpoints(self, y_sp, rng, options=None):
        """Sample one bounded, reproducible target for an environment episode.

        The generic sampler applies a conservative relative perturbation. Models
        with an analytic feasibility test can override ``is_setpoint_reachable``
        so rejected candidates are resampled from the same seeded RNG stream.
        """

        reference = [float(value) for value in y_sp]
        options = dict(options or {})
        if not options.get("randomize_setpoints", False):
            return reference
        schema = self.setpoint_schema()
        for _ in range(64):
            candidate = []
            for index, value in enumerate(reference):
                bounds = (
                    schema[index].get("bounds")
                    if index < len(schema) and isinstance(schema[index], dict)
                    else None
                )
                if (
                    isinstance(bounds, (tuple, list))
                    and len(bounds) == 2
                    and bounds[0] is not None
                    and bounds[1] is not None
                    and float(bounds[1]) > float(bounds[0])
                ):
                    lo, hi = float(bounds[0]), float(bounds[1])
                    span = hi - lo
                    radius = 0.10 * min(
                        max(abs(value), 0.01 * span),
                        span,
                    )
                    trial = float(rng.uniform(
                        max(lo, value - radius),
                        min(hi, value + radius),
                    ))
                else:
                    trial = float(
                        value * (1.0 + 0.10 * rng.uniform(-1.0, 1.0))
                    )
                candidate.append(trial)
            if self.is_setpoint_reachable(candidate):
                return candidate
        return reference

    def is_setpoint_reachable(self, y_sp):
        """Return whether a sampled target satisfies the model's SP contract.

        The base implementation validates dimensions, finiteness, and declared
        setpoint bounds. Models with a steady-state inverse should strengthen
        this check.
        """

        try:
            values = [float(value) for value in y_sp]
        except (TypeError, ValueError):
            return False
        schema = self.setpoint_schema()
        if len(values) != len(schema) or any(not math.isfinite(value) for value in values):
            return False
        for value, row in zip(values, schema):
            bounds = row.get("bounds") if isinstance(row, dict) else None
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                continue
            lo, hi = bounds
            if lo is not None and value < float(lo):
                return False
            if hi is not None and value > float(hi):
                return False
        return True

    def setpoint_schema(self):
        output_rows = self.controlled_output_schema()
        names = self._vector_names(self.setpoint_names or self.output_names, "y_sp", len(output_rows))
        rows = []
        for i, output in enumerate(output_rows):
            output_name = output.get("name", f"y{i}")
            name = names[i] if i < len(names) else f"y_sp{i}"
            rows.append({
                "name": name,
                "unit": self.setpoint_units.get(name, self.output_units.get(output_name, output.get("unit", ""))),
                "bounds": self.setpoint_bounds.get(name, output.get("bounds")),
                "output": output_name,
            })
        return rows

    def controlled_output_schema(self):
        y0 = list(self.controlled_output(self.initial_state()))
        names = self._vector_names(self.output_names, "y", len(y0))
        rows = []
        for name in names:
            rows.append({
                "name": name,
                "unit": self.output_units.get(name, ""),
                "bounds": self.output_bounds.get(name),
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

    def dynamics_disturbance_specs(self):
        specs = []
        defaults = self.disturbance_defaults()
        for row in self.disturbance_schema():
            if row.get("kind") == "setpoint":
                continue
            name = row.get("name")
            if not name:
                continue
            default = defaults.get(name, row.get("default", self.p.get(name, 0.0)))
            if isinstance(default, (list, tuple)):
                specs.extend((name, i) for i in range(len(default)))
            else:
                specs.append((name, None))
        return tuple(specs)

    def dynamics_disturbance_names(self):
        return tuple(name if idx is None else f"{name}[{idx}]" for name, idx in self.dynamics_disturbance_specs())

    def dynamics_disturbance_map(self, d):
        values = {}
        for j, (name, idx) in enumerate(self.dynamics_disturbance_specs()):
            if idx is None:
                values[name] = d[j]
            else:
                values.setdefault(name, []).append(d[j])
        return values

    def disturbance_vector(self, values=None):
        merged = self.disturbance_defaults()
        merged.update(dict(values or {}))
        if "Caf" not in merged and "caf" in merged:
            merged["Caf"] = merged["caf"]
        if "Tcool" not in merged and "tcool" in merged:
            merged["Tcool"] = merged["tcool"]
        out = []
        for name, idx in self.dynamics_disturbance_specs():
            value = merged.get(name, self.p.get(name, 0.0))
            if idx is None:
                out.append(value)
            elif isinstance(value, (int, float)):
                out.append(value)
            else:
                out.append(value[idx])
        return out

    def energy_kw(self, u, backend="numeric", ca=None):
        return 0.0

    def action_energy_kw(self, act, x=None, env=None):
        """Return total action energy rate in kW for numeric environment steps."""

        return float(self.energy_kw(self.action_vector(act)))

    def economic_value(self, x, u, env=None, backend="numeric", ca=None):
        return 0.0

    def display_outputs(self, x, backend="numeric", ca=None):
        if backend == "casadi":
            return {"levels": [], "temps": [x[i] for i in range(len(self.initial_state()))]}
        return {"levels": [], "temps": list(x)}

    def parameter_schema(self):
        params = {k: _copy_value(v) for k, v in self.p.items()}
        params.update(self._extra_parameters())
        schema = {
            name: {"value": value, "unit": self.param_units.get(name, ""), "bounds": self.param_bounds.get(name)}
            for name, value in params.items()
        }
        from .parameter_profiles import enrich_parameter_schema

        return enrich_parameter_schema(self.scenario, schema)

    def physical_metadata(self):
        """Return fidelity, provenance, validity-domain, and solver metadata."""

        from .parameter_profiles import model_physical_metadata

        return model_physical_metadata(
            self.scenario,
            dt_micro=float(getattr(self, "dt_micro", 0.02) or 0.02),
        )

    def solver_settings(self):
        """Return validated numerical integration settings for this model."""

        settings = copy.deepcopy(self.physical_metadata()["solver"])
        method = settings.get("method", "rk4")
        if method != "rk4":
            raise ValueError(f"unsupported integration method for {self.scenario!r}: {method!r}")
        max_step = float(settings.get("max_step", getattr(self, "dt_micro", 0.02) or 0.02))
        if not math.isfinite(max_step) or max_step <= 0:
            raise ValueError(f"solver max_step must be finite and positive, got {max_step!r}")
        settings["method"] = method
        settings["max_step"] = max_step
        return settings

    def disturbance_schema(self):
        rows = []
        for row in self.input_disturbances + self.event_disturbances:
            out = dict(row)
            out.setdefault("name", out.get("event", "disturbance"))
            out.setdefault("dynamic", "event" in out)
            if "default" not in out and out["name"] in getattr(self, "p", {}):
                out["default"] = _copy_value(self.p[out["name"]])
            rows.append(out)
        return rows

    def disturbance_defaults(self):
        defaults = {}
        for row in self.disturbance_schema():
            if row.get("kind") == "setpoint":
                continue
            name = row.get("name")
            if not name:
                continue
            if "default" in row:
                defaults[name] = _copy_value(row["default"])
            elif name in getattr(self, "p", {}):
                defaults[name] = _copy_value(self.p[name])
            else:
                defaults[name] = 0.0
        return defaults

    def disturbance_attribute_map(self):
        defaults = self.runtime_env(self.disturbance_defaults())
        return {
            name: attr
            for name, attr in self.disturbance_attributes.items()
            if name in defaults
        }

    def runtime_env(self, disturbance_values):
        env = {name: _copy_value(value) for name, value in dict(disturbance_values or {}).items()}
        env.setdefault("t_cold", self.p.get("t_cold", 15.0))
        env.setdefault("t_amb", self.p.get("t_amb", 20.0))
        env.setdefault("extra_outflow", 0.0)
        return env

    def sample_disturbance(self, event, current, rng):
        if event == "cold_inlet_step":
            return float(max(2.0, min(35.0, float(current) + rng.uniform(-8, 8))))
        if event == "ambient_step":
            return float(max(0.0, min(40.0, float(current) + rng.uniform(-8, 12))))
        if event == "demand_surge":
            return float(abs(rng.uniform(0, 8e-4)))
        row = next((row for row in self.disturbance_schema() if row.get("event") == event), None)
        return self._sample_schema_disturbance(current, row, rng)

    def _sample_schema_disturbance(self, default, row, rng):
        if not row:
            return _copy_value(default)
        if row.get("values"):
            values = list(row["values"])
            return _copy_value(values[int(rng.integers(0, len(values)))])
        bounds = row.get("bounds")
        if (
            isinstance(bounds, (tuple, list))
            and len(bounds) == 2
            and bounds[0] is not None
            and bounds[1] is not None
        ):
            lo, hi = float(bounds[0]), float(bounds[1])
            if isinstance(default, (list, tuple)):
                return [float(rng.uniform(lo, hi)) for _ in default]
            return float(rng.uniform(lo, hi))
        return _copy_value(default)

    def process_constraint_info(self, x, levels, temps, env):
        return {}

    def hard_termination_reasons(self, x, levels, temps, env):
        """Return unconditional physical termination reasons for a transition."""

        return ()

    def process_info(self, x, levels, temps, env):
        return {}

    def constraint_penalty_scales(self):
        return {}

    def common_constraint_info(self, levels, temps):
        hmax = self.height_max
        return {
            "temp_high": max((max(0.0, t - 80.0) for t in temps), default=0.0),
            "temp_trip": max((max(0.0, t - 92.0) for t in temps), default=0.0),
            "level_high": max((max(0.0, levels[i] - 0.90 * hmax[i]) for i in range(len(levels))), default=0.0),
            "level_low": max((max(0.0, 0.15 * hmax[i] - levels[i]) for i in range(len(levels))), default=0.0),
        }

    def runaway_state(self, levels, temps):
        hmax = self.height_max
        return any(t > 92.0 for t in temps) or any(levels[i] > 0.97 * hmax[i] for i in range(len(levels)))

    def constraint_schema(self):
        return [dict(row) for row in self.safety_constraints]

    def metadata(self):
        physical_metadata = self.physical_metadata()
        return {
            "scenario": canonical_scenario_id(self.scenario),
            "name": self.display_name,
            "summary": self.summary,
            "supported_goals": list(self.supported_goals),
            "benchmark_goals": list(self.benchmark_goals),
            "states": self.state_schema(),
            "actions": self.action_schema(),
            "controlled_outputs": self.controlled_output_schema(),
            "setpoints": self.setpoint_schema(),
            "state_vector": {"name": "x", "length": len(self.initial_state())},
            "action_vector": {"name": "u", "length": self.action_dim()},
            "controlled_output_vector": {"name": "y", "length": len(self.controlled_output(self.initial_state()))},
            "setpoint_vector": {"name": "y_sp", "length": len(self.default_setpoint_vector())},
            "dynamics_disturbances": list(self.dynamics_disturbance_names()),
            "parameters": self.parameter_schema(),
            "physical_metadata": physical_metadata,
            "solver": copy.deepcopy(physical_metadata["solver"]),
            "disturbances": self.disturbance_schema(),
            "disturbance_defaults": self.disturbance_defaults(),
            "constraints": self.constraint_schema(),
            "plant_regime": copy.deepcopy(self.plant_regime),
            "economic_config": copy.deepcopy(self.economic_config),
            "supervisory_layout": [list(row) for row in self.supervisory_layout],
            "dt_micro": self.dt_micro,
            "energy_scored": bool(getattr(self, "energy_scored", True)),
        }

    @property
    def height_max(self):
        return [1.0] * int(self.n)

    def ideal_energy_kw(self, x, y_sp, env, act):
        return 0.0
