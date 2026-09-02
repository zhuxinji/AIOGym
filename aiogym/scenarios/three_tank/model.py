"""Fixed BOM-backed hydraulic model for the laboratory three-tank system."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from aiogym.core.model import PhysicsModelBase


BOM_CONFIGURATION = {
    "process_tanks": {
        "count": 3,
        "material": "acrylic",
        "dimensions_m": [0.3, 0.3, 0.5],
        "effective_capacity_m3": 0.045,
        "wall_thickness_m": 0.01,
        "bottom_thickness_m": 0.015,
    },
    "reservoir": {
        "role": "constant_level_source_and_return",
        "dimensions_m": [0.8, 0.5, 0.55],
        "effective_capacity_m3": 0.18,
    },
    "pump": {
        "id": "P101",
        "motor_power_w": 370.0,
        "max_flow_m3s": 25.0 / 60000.0,
        "rated_max_flow_m3s": 8.0 / 3600.0,
        "rated_max_head_m": 12.0,
        "static_head_m": 1.7,
        "shutoff_head_m": 10.0,
    },
    "valves": ["V12", "V23", "V34"],
    "bypass_valves": ["BV12", "BV23", "BV34"],
    "flowmeters": [
        {"id": name, "range_l_min": [0.0, 10.0]}
        for name in ("FT101", "FT12", "FT23", "FT34")
    ],
    "source": (
        "2026-08-16 45 L redesign with the 2026-08-27 heated-system "
        "flow path and confirmed 0-10 L/min flowmeter span; final procurement "
        "list takes precedence"
    ),
}

NOMINAL_FLOW_M3S = 3.0 / 60000.0
TRACKING_ERROR_SCALES = (0.1, 0.1, 0.1)


class ThreeTankModel(PhysicsModelBase):
    """Three coupled liquid-level balances with direct hydraulic actions."""

    scenario = "three_tank"
    state_names = ("h1", "h2", "h3")
    state_units = {name: "m" for name in state_names}
    action_names = ("pump_P101", "valve_V12", "valve_V23", "valve_V34")
    action_kinds = {
        "pump_P101": "pump",
        "valve_V12": "valve",
        "valve_V23": "valve",
        "valve_V34": "valve",
    }
    action_units = {name: "fraction" for name in action_names}
    action_bounds = {name: (0.0, 1.0) for name in action_names}
    output_names = ("tank_1_level", "tank_2_level", "tank_3_level")
    output_units = {name: "m" for name in output_names}
    input_disturbances = (
        {
            "name": "reservoir_available",
            "event": "reservoir_low_level_trip",
            "unit": "binary",
            "bounds": (0.0, 1.0),
            "default": 1.0,
            "description": "hardwired reservoir dry-run permissive for P101",
        },
        {
            "name": "bv12_open",
            "event": "BV12_open",
            "unit": "binary",
            "bounds": (0.0, 1.0),
            "default": 0.0,
            "description": "BV12 on/off bypass position around the V12 branch",
        },
        {
            "name": "bv23_open",
            "event": "BV23_open",
            "unit": "binary",
            "bounds": (0.0, 1.0),
            "default": 0.0,
            "description": "BV23 on/off bypass position around the V23 branch",
        },
        {
            "name": "bv34_open",
            "event": "BV34_open",
            "unit": "binary",
            "bounds": (0.0, 1.0),
            "default": 0.0,
            "description": "BV34 on/off bypass position around the V34 branch",
        },
        {
            "name": "pump_flow_factor",
            "event": "pump_capacity_shift",
            "unit": "fraction",
            "bounds": (0.4, 1.4),
            "default": 1.0,
            "description": "P101 flow-capacity multiplier",
        },
        *(
            {
                "name": f"{valve.lower()}_flow_factor",
                "event": f"{valve.lower()}_capacity_shift",
                "unit": "fraction",
                "bounds": (0.5, 1.5),
                "default": 1.0,
                "description": f"{valve} installed flow-coefficient multiplier",
            }
            for valve in ("V12", "V23", "V34")
        ),
    )
    dt_micro = 0.1
    energy_scored = True
    parameter_units = MappingProxyType(
        {
            "area": "m^2",
            "height_max": "m",
            "level_sensor_range": "m",
            "cv_valves": "m^(5/2)/s",
            "cv_bypass": "m^(5/2)/s",
            "flow_observation_scale": "m^3/s",
            "gravity_drop": "m",
            "overflow_level": "m",
            "cv_overflow": "m^(5/2)/s",
            "overflow_head_floor": "m",
            "high_level_trip": "m",
            "nominal_level": "m",
            "pump_flow_max": "m^3/s",
            "pump_power_max": "W",
            "pump_static_head": "m",
            "pump_shutoff_head": "m",
        }
    )

    def __init__(self, parameters: Mapping[str, Any] | None = None):
        defaults = {
            "area": [0.09, 0.09, 0.09],
            "height_max": [0.50, 0.50, 0.50],
            "level_sensor_range": [0.50, 0.50, 0.50],
            "cv_valves": [0.0005, 0.0005, 0.0005],
            "cv_bypass": [0.00004, 0.00004, 0.00004],
            "flow_observation_scale": [10.0 / 60000.0] * 4,
            "gravity_drop": [0.30, 0.30, 0.30],
            "overflow_level": [0.45, 0.45, 0.45],
            "cv_overflow": [0.001, 0.001, 0.001],
            "overflow_head_floor": 1e-9,
            "high_level_trip": [0.425, 0.425, 0.425],
            "nominal_level": 0.225,
            "pump_flow_max": 25.0 / 60000.0,
            "pump_power_max": 370.0,
            "pump_static_head": 1.7,
            "pump_shutoff_head": 10.0,
        }
        self.p = _resolved_parameters(defaults, parameters)
        self._resolved_parameters = MappingProxyType(
            {
                name: tuple(value) if isinstance(value, list) else value
                for name, value in self.p.items()
            }
        )
        self._environment_bounds = {
            row["name"]: tuple(row["bounds"]) for row in self.input_disturbances
        }

    @property
    def resolved_parameters(self) -> Mapping[str, Any]:
        return self._resolved_parameters

    def parameter(self, name):
        try:
            return deepcopy(self.p[str(name)])
        except KeyError as error:
            raise KeyError(f"unknown three-tank parameter {name!r}") from error

    @property
    def height_max(self):
        return [float(value) for value in self.p["height_max"]]

    @property
    def state_bounds(self):
        return {
            name: (0.0, self.height_max[index])
            for index, name in enumerate(self.state_names)
        }

    @property
    def output_bounds(self):
        return {
            name: (0.0, self.height_max[index])
            for index, name in enumerate(self.output_names)
        }

    def initial_state(self):
        level = float(self.p["nominal_level"])
        return [level, level, level]

    def outputs(self, x):
        return self.state_vector(x)

    def display_outputs(self, x):
        return {"levels": [max(float(value), 0.0) for value in x], "temps": []}

    def nominal_steady_state(self, *, flow=NOMINAL_FLOW_M3S, levels=None, env=None):
        context = self._resolved_env(env)
        q = float(flow)
        if levels is None:
            levels = [self.p["nominal_level"]] * 3
        h = [float(value) for value in levels]
        if not math.isfinite(q) or q <= 0.0:
            raise ValueError("flow must be finite and positive")
        if len(h) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in h):
            raise ValueError("levels must contain three finite positive values")
        pump_capacity = self.p["pump_flow_max"] * context["pump_flow_factor"]
        static_head = self.p["pump_static_head"]
        shutoff_head = self.p["pump_shutoff_head"]
        pump = math.sqrt(
            (static_head + (shutoff_head - static_head) * (q / pump_capacity) ** 2)
            / shutoff_head
        )
        bypasses = [
            self.p["cv_bypass"][index]
            * context[f"bv{index + 1}{index + 2}_open"]
            * math.sqrt(h[index] + self.p["gravity_drop"][index])
            for index in range(3)
        ]
        valves = [
            (q - bypasses[index])
            / (
                self.p["cv_valves"][index]
                * context[f"v{index + 1}{index + 2}_flow_factor"]
                * math.sqrt(h[index] + self.p["gravity_drop"][index])
            )
            for index in range(3)
        ]
        action = [pump, *valves]
        reasons = [
            f"{name} command is outside [0, 1]"
            for name, value in zip(self.action_names, action)
            if not math.isfinite(value) or value < 0.0 or value > 1.0
        ]
        reasons.extend(
            f"BV{index + 1}{index + 2} bypass flow exceeds the requested circulation"
            for index, bypass in enumerate(bypasses)
            if bypass > q
        )
        if context["reservoir_available"] < 0.5:
            reasons.append("reservoir is unavailable")
        return {
            "feasible": not reasons,
            "infeasible_reasons": tuple(reasons),
            "state": h,
            "y_sp": h,
            "action": action,
            "flow_m3s": q,
        }

    def tracking_steady_state_action(self, y_sp, disturbances=None):
        equilibrium = self._tracking_steady_state(y_sp, disturbances)
        return list(equilibrium["action"]) if equilibrium is not None else None

    def tracking_steady_state_state(self, y_sp, disturbances=None):
        equilibrium = self._tracking_steady_state(y_sp, disturbances)
        return list(equilibrium["state"]) if equilibrium is not None else None

    def _tracking_steady_state(self, y_sp, disturbances):
        values = np.asarray(y_sp, dtype=float).reshape(-1)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            return None
        if np.any(values <= 0.0):
            return None
        equilibrium = self.nominal_steady_state(
            levels=values.tolist(), env=disturbances
        )
        return equilibrium if equilibrium["feasible"] else None

    def default_action(self):
        return list(self.nominal_steady_state()["action"])

    def default_setpoint_vector(self):
        return list(self.nominal_steady_state()["y_sp"])

    def default_disturbances(self):
        return dict(super().default_disturbances())

    def _resolved_env(self, env=None):
        supplied = {} if env is None else dict(env)
        clean = {}
        for row in self.input_disturbances:
            name = row["name"]
            try:
                value = float(supplied.get(name, row["default"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"three_tank disturbance {name!r} must be finite"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"three_tank disturbance {name!r} must be finite")
            lower, upper = self._environment_bounds[name]
            if value < float(lower) or value > float(upper):
                raise ValueError(
                    f"three_tank disturbance {name!r} must be within "
                    f"[{lower}, {upper}], got {value}"
                )
            if row["unit"] == "binary" and value not in (0.0, 1.0):
                raise ValueError(
                    f"three_tank disturbance {name!r} must be binary, got {value}"
                )
            clean[name] = value
        return clean

    def _resolve_disturbances(self, disturbance_values):
        return self._resolved_env(super()._resolve_disturbances(disturbance_values))

    def _effective_action(self, action):
        values = self.action_vector(action)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("three_tank action values must be finite")
        return [min(max(value, 0.0), 1.0) for value in values]

    @staticmethod
    def _gate(condition):
        return 1.0 if condition else 0.0

    def _flow_terms(self, levels, action, env):
        high_level_ok = self._gate(
            all(
                levels[index] < self.p["high_level_trip"][index]
                for index in range(3)
            )
        )
        reservoir_ok = self._gate(env["reservoir_available"] >= 0.5)
        pump_enabled = high_level_ok * reservoir_ok
        effective_max = self.p["pump_flow_max"] * env["pump_flow_factor"]
        head_margin = self.p["pump_shutoff_head"] - self.p["pump_static_head"]
        normalized_head = (
            self.p["pump_shutoff_head"] * action[0] ** 2
            - self.p["pump_static_head"]
        ) / head_margin
        pump_flow = effective_max * math.sqrt(max(normalized_head, 0.0)) * pump_enabled
        valve_flows = [
            self.p["cv_valves"][index]
            * env[f"v{index + 1}{index + 2}_flow_factor"]
            * action[1 + index]
            * math.sqrt(max(levels[index] + self.p["gravity_drop"][index], 0.0))
            for index in range(3)
        ]
        bypass_flows = [
            self.p["cv_bypass"][index]
            * env[f"bv{index + 1}{index + 2}_open"]
            * math.sqrt(max(levels[index] + self.p["gravity_drop"][index], 0.0))
            for index in range(3)
        ]
        overflow_flows = []
        for index in range(3):
            head = levels[index] - self.p["overflow_level"][index]
            enabled = self._gate(head > 0.0)
            overflow_flows.append(
                self.p["cv_overflow"][index]
                * enabled
                * math.sqrt(max(head, self.p["overflow_head_floor"]))
            )
        return (
            pump_flow,
            valve_flows,
            bypass_flows,
            overflow_flows,
            bool(pump_enabled),
        )

    def _dynamics(self, x, action, env):
        context = self._resolved_env(env)
        u = self._effective_action(action)
        pump, valves, bypasses, overflows, _ = self._flow_terms(x, u, context)
        q12, q23, q34 = (
            valves[index] + bypasses[index] for index in range(3)
        )
        flows_in = (pump, q12, q23)
        flows_out = (
            q12 + overflows[0],
            q23 + overflows[1],
            q34 + overflows[2],
        )
        return [
            (flows_in[index] - flows_out[index]) / self.p["area"][index]
            for index in range(3)
        ]

    def action_slew_limits(self):
        return None

    def observation_schema(self):
        measurement_names = [
            "normalized_tank_1_level",
            "normalized_tank_2_level",
            "normalized_tank_3_level",
            "normalized_FT101_flow",
            "normalized_FT12_flow",
            "normalized_FT23_flow",
            "normalized_FT34_flow",
        ]
        reference_names = [
            f"normalized_{row['name']}_reference" for row in self.output_schema()
        ]
        return [
            *(
                {
                    "name": name,
                    "kind": "measurement",
                    "unit": "normalized",
                    "low": 0.0,
                    "high": 1.0,
                }
                for name in measurement_names
            ),
            *(
                {
                    "name": name,
                    "kind": "reference",
                    "unit": "normalized",
                    "low": 0.0,
                    "high": 1.0,
                }
                for name in reference_names
            ),
        ]

    def observation(self, state, reference, previous_action, disturbances):
        context = self._resolved_env(disturbances)
        action = self.default_action() if previous_action is None else previous_action
        pump, valve_flows, _, _, _ = self._flow_terms(
            [float(value) for value in state],
            self._effective_action(action),
            context,
        )
        return self.observation_from_measurements(
            state,
            [pump, *valve_flows],
            reference,
        )

    def observation_from_measurements(self, levels, flows, reference):
        level_values = np.asarray(levels, dtype=float).reshape(-1)
        flow_values = np.asarray(flows, dtype=float).reshape(-1)
        reference_values = np.asarray(reference, dtype=float).reshape(-1)
        if (
            level_values.shape != (3,)
            or flow_values.shape != (4,)
            or reference_values.shape != (3,)
            or not np.isfinite(level_values).all()
            or not np.isfinite(flow_values).all()
            or not np.isfinite(reference_values).all()
        ):
            raise ValueError(
                "three-tank observation requires three finite levels, four finite "
                "flows, and three finite references"
            )
        normalized_state = np.clip(
            level_values / np.asarray(self.height_max), 0.0, 1.0
        )
        normalized_reference = np.clip(
            reference_values / np.asarray(self.output_scales(), dtype=float),
            0.0,
            1.0,
        )
        normalized_flow = np.clip(
            flow_values / np.asarray(self.p["flow_observation_scale"], dtype=float),
            0.0,
            1.0,
        )
        return [
            *normalized_state.tolist(),
            *normalized_flow.tolist(),
            *normalized_reference.tolist(),
        ]

    def measurement(self, state, disturbances=None):
        context = self._resolved_env(disturbances)
        values = self.state_vector(state)
        return {
            "x": values,
            "levels": values,
            "temps": [],
            "y": list(self.outputs(values)),
            **context,
        }

    def measurement_from_observation(self, observation, disturbances=None):
        values = np.asarray(observation, dtype=float).reshape(-1)
        if values.shape != (10,) or not np.isfinite(values).all():
            raise ValueError(
                "three-tank policy observation must match observation_schema"
            )
        state = values[:3] * np.asarray(self.height_max)
        flows = values[3:7] * np.asarray(self.p["flow_observation_scale"])
        measurement = self.measurement(state, disturbances)
        measurement.update(
            {
                "flow_measurement_m3s": flows.tolist(),
                "FT101_flow_m3s": float(flows[0]),
                "FT12_flow_m3s": float(flows[1]),
                "FT23_flow_m3s": float(flows[2]),
                "FT34_flow_m3s": float(flows[3]),
            }
        )
        return measurement

    def clamp_state(self, state):
        return list(state)

    def hard_termination_reasons(self, state):
        levels = [float(value) for value in state]
        reasons = []
        if any(value < 0.0 for value in levels):
            reasons.append("negative_level")
        if any(
            value > self.p["height_max"][index]
            for index, value in enumerate(levels)
        ):
            reasons.append("tank_overflow_limit")
        return tuple(reasons)

    def constraint_costs(self, state, disturbances=None):
        del disturbances
        return {reason: 1.0 for reason in self.hard_termination_reasons(state)}

    def safety_margins(self, state, disturbances=None):
        del disturbances
        margins = {}
        for index, (level, maximum) in enumerate(
            zip(state, self.p["height_max"]), start=1
        ):
            level = float(level)
            maximum = float(maximum)
            margins[f"tank_{index}_level_lower"] = level / maximum
            margins[f"tank_{index}_level_upper"] = (maximum - level) / maximum
        return margins

    def process_info(self, state, action, disturbances=None):
        context = self._resolved_env(disturbances)
        u = self._effective_action(self.default_action() if action is None else action)
        levels = [float(value) for value in state]
        pump, valves, bypass_flows, overflows, pump_enabled = self._flow_terms(
            levels, u, context
        )
        return {
            "P101_flow_m3s": float(pump),
            "V12_flow_m3s": float(valves[0]),
            "V23_flow_m3s": float(valves[1]),
            "V34_flow_m3s": float(valves[2]),
            "BV12_flow_m3s": float(bypass_flows[0]),
            "BV23_flow_m3s": float(bypass_flows[1]),
            "BV34_flow_m3s": float(bypass_flows[2]),
            "FT101_flow_m3s": float(pump),
            "FT12_flow_m3s": float(valves[0]),
            "FT23_flow_m3s": float(valves[1]),
            "FT34_flow_m3s": float(valves[2]),
            "overflow_flow_m3s": [float(value) for value in overflows],
            "P101_enabled": pump_enabled,
        }

    def action_energy_kw(self, act, x=None, env=None):
        action = self._effective_action(act)
        pump_enabled = True
        if x is not None:
            context = self._resolved_env(env)
            _, _, _, _, pump_enabled = self._flow_terms(x, action, context)
        pump = action[0] ** 3 * self.p["pump_power_max"] * float(pump_enabled)
        return float(pump / 1000.0)

    def energy_kw(self, action):
        effective = self._effective_action(action)
        return effective[0] ** 3 * self.p["pump_power_max"] / 1000.0

    def step_info(self, state, action, disturbances=None):
        applied = self.default_action() if action is None else list(action)
        info = self.process_info(state, applied, disturbances)
        info["y"] = list(self.outputs(state))
        info["energy_kw"] = self.action_energy_kw(applied, state, disturbances)
        return info

    def physical_io_schema(self):
        return {
            "measurements": [
                *[
                    {
                        "name": f"LT{tank}01",
                        "quantity": f"tank_{tank}_level",
                        "range_m": [0.0, 0.5],
                        "signal": "RS485",
                    }
                    for tank in (1, 2, 3)
                ],
                *[
                    {
                        "name": name,
                        "quantity": quantity,
                        "range_l_min": [0.0, 10.0],
                        "signal": "4-20mA / Modbus",
                    }
                    for name, quantity in (
                        ("FT101", "P101_flow"),
                        ("FT12", "tank_1_to_2_flow"),
                        ("FT23", "tank_2_to_3_flow"),
                        ("FT34", "tank_3_to_reservoir_flow"),
                    )
                ],
            ],
            "actuators": [
                {"name": "P101", "command": "VFD speed"},
                {"name": "V12", "command": "valve position"},
                {"name": "V23", "command": "valve position"},
                {"name": "V34", "command": "valve position"},
            ],
            "disturbance_actuators": [
                {"name": "BV12", "command": "on/off bypass"},
                {"name": "BV23", "command": "on/off bypass"},
                {"name": "BV34", "command": "on/off bypass"},
            ],
            "boundary": dict(BOM_CONFIGURATION["reservoir"]),
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
            f"unknown three-tank parameters: {unknown}; available: {sorted(defaults)}"
        )
    resolved = deepcopy(defaults)
    for name, value in supplied.items():
        default = defaults[name]
        if isinstance(default, list):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"three-tank parameter {name!r} must be a sequence")
            values = [_finite_number(name, item) for item in value]
            if len(values) != len(default):
                raise ValueError(
                    f"three-tank parameter {name!r} must contain {len(default)} values"
                )
            resolved[name] = values
        else:
            resolved[name] = _finite_number(name, value)
    for name, value in resolved.items():
        values = value if isinstance(value, list) else [value]
        if any(item <= 0.0 for item in values):
            raise ValueError(f"three-tank parameter {name!r} must be positive")
    if any(
        not nominal < high < overflow < maximum
        for nominal, high, overflow, maximum in zip(
            [resolved["nominal_level"]] * 3,
            resolved["high_level_trip"],
            resolved["overflow_level"],
            resolved["height_max"],
        )
    ):
        raise ValueError(
            "three-tank level parameters must satisfy nominal < high < overflow < maximum"
        )
    if any(
        sensor < maximum
        for sensor, maximum in zip(
            resolved["level_sensor_range"], resolved["height_max"]
        )
    ):
        raise ValueError("three-tank sensor ranges must cover tank heights")
    if resolved["pump_static_head"] >= resolved["pump_shutoff_head"]:
        raise ValueError("pump_static_head must be below pump_shutoff_head")
    return resolved


def _finite_number(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"three-tank parameter {name!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"three-tank parameter {name!r} must be finite")
    return number


__all__ = [
    "BOM_CONFIGURATION",
    "NOMINAL_FLOW_M3S",
    "TRACKING_ERROR_SCALES",
    "ThreeTankModel",
]
