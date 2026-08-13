"""Fixed BOM-backed model for the laboratory three-tank heating system."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from aiogym.core.backends import _NUMERIC_OPS
from aiogym.core.model import RHO_CP

from .physics import _ThreeTankPhysicsKernel


BOM_CONFIGURATION = {
    "process_tanks": {
        "count": 3,
        "material": "acrylic",
        "dimensions_m": [0.5, 0.3, 0.4],
        "effective_capacity_m3": 0.06,
        "wall_thickness_m": 0.01,
        "bottom_thickness_m": 0.015,
    },
    "reservoir": {
        "role": "boundary_source_and_return",
        "dimensions_m": [0.8, 0.5, 0.55],
        "effective_capacity_m3": 0.18,
    },
    "pump": {
        "id": "P101",
        "motor_power_w": 370.0,
        "max_flow_m3s": 25.0 / 60000.0,
        "static_head_m": 1.7,
        "shutoff_head_m": 10.0,
    },
    "valves": ["V12", "V23", "V34"],
    "installed_heaters": [{"id": "H1", "tank": 1, "power_w": 2000.0}],
    "reserved_heater_ports": [2, 3],
    "source": "三级水箱加热系统.pdf; detailed BOM and final procurement list take precedence",
}

TANK3_INTERNAL_CONTROL = {
    "feedback_source": "controller_observation",
    "level_tolerance_m": 0.01,
    "pump_h1_kp": 0.08,
    "v12_h2_kp": 0.08,
    "maximum_correction": 0.25,
    "v23_nominal_flow_feedforward": True,
}

TANK3_MAXIMUM_ACTION_STEP = (0.05, 0.08, 0.08, 0.08, 0.05)


def _schema(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for raw in rows:
        row = dict(raw)
        low, high = row.pop("bounds")
        if "name" not in row:
            raise ValueError("schema row has no name")
        row["low"] = -np.inf if low is None else float(low)
        row["high"] = np.inf if high is None else float(high)
        result.append(row)
    return result


class _BOMThreeTankKernel(_ThreeTankPhysicsKernel):
    """Three process tanks with a separate, non-dynamic 180 L reservoir."""

    scenario = "three_tank"
    state_names = ("h1", "T1", "h2", "T2", "h3", "T3")
    state_units = {
        "h1": "m",
        "h2": "m",
        "h3": "m",
        "T1": "degC",
        "T2": "degC",
        "T3": "degC",
    }
    action_names = (
        "pump_P101",
        "valve_V12",
        "valve_V23",
        "valve_V34",
        "heater_H1",
    )
    action_kinds = {
        "pump_P101": "pump",
        "valve_V12": "valve",
        "valve_V23": "valve",
        "valve_V34": "valve",
        "heater_H1": "heater",
    }
    action_units = {name: "fraction" for name in action_names}
    action_bounds = {name: (0.0, 1.0) for name in action_names}
    output_names = (
        "tank_1_level",
        "tank_2_level",
        "tank_3_level",
        "tank_1_temperature",
        "tank_2_temperature",
        "tank_3_temperature",
    )
    output_units = {
        "tank_1_level": "m",
        "tank_2_level": "m",
        "tank_3_level": "m",
        "tank_1_temperature": "degC",
        "tank_2_temperature": "degC",
        "tank_3_temperature": "degC",
    }
    input_disturbances = (
        {
            "name": "t_amb",
            "event": "ambient_step",
            "unit": "degC",
            "bounds": (0.0, 45.0),
            "default": 20.0,
            "description": "ambient air temperature",
        },
        {
            "name": "t_reservoir",
            "event": "reservoir_temperature_step",
            "unit": "degC",
            "bounds": (0.0, 45.0),
            "default": 20.0,
            "description": "well-mixed boundary temperature of the 180 L reservoir",
        },
        {
            "name": "reservoir_available",
            "event": "reservoir_low_level_trip",
            "unit": "binary",
            "bounds": (0.0, 1.0),
            "default": 1.0,
            "description": "hardwired reservoir dry-run permissive for P101",
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
        {
            "name": "heater_efficiency",
            "event": "heater_efficiency_shift",
            "unit": "fraction",
            "bounds": (0.4, 1.0),
            "default": 1.0,
            "description": "fraction of H1 electric power transferred to Tank 1",
        },
        {
            "name": "heat_loss_factor",
            "event": "heat_loss_shift",
            "unit": "fraction",
            "bounds": (0.3, 3.0),
            "default": 1.0,
            "description": "common multiplier on provisional heat-loss coefficients",
        },
    )
    dt_micro = 0.1
    energy_scored = True

    def __init__(self):
        self.p = {
            "area": [0.15, 0.15, 0.15],
            "height_max": [0.40, 0.40, 0.40],
            "level_sensor_range": [0.50, 0.50, 0.50],
            "cv_valves": [0.0005, 0.0005, 0.0005],
            "gravity_drop": [0.30, 0.30, 0.30],
            "overflow_level": [0.36, 0.36, 0.36],
            "cv_overflow": [0.001, 0.001, 0.001],
            "overflow_head_floor": 1e-9,
            "high_level_trip": [0.34, 0.34, 0.34],
            "low_level_trip": [0.08, 0.08, 0.08],
            "nominal_level": 0.18,
            "ua_loss": [40.0, 40.0, 40.0],
            "heater_power": 2000.0,
            "pump_flow_max": 25.0 / 60000.0,
            "pump_power_max": 370.0,
            "pump_static_head": 1.7,
            "pump_shutoff_head": 10.0,
            "h_floor": 1e-3,
            "temperature_trip": 80.0,
            "temperature_hard_limit": 100.0,
        }
        self._environment_bounds = {
            row["name"]: tuple(row["bounds"]) for row in self.input_disturbances
        }

    @property
    def height_max(self):
        return [float(value) for value in self.p["height_max"]]

    def initial_state(self):
        level = float(self.p["nominal_level"])
        return [level, 20.0, level, 20.0, level, 20.0]

    def nominal_steady_state(
        self,
        *,
        flow=5.0 / 60000.0,
        tank_1_temperature=24.0,
        levels=None,
        env=None,
    ):
        context = self._resolved_env(env)
        q = float(flow)
        if levels is None:
            levels = [self.p["nominal_level"]] * 3
        h = [float(value) for value in levels]
        t1 = float(tank_1_temperature)
        if not math.isfinite(q) or q <= 0.0:
            raise ValueError("flow must be finite and positive")
        if len(h) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in h):
            raise ValueError("levels must contain three finite positive values")
        heat_capacity_flow = RHO_CP * q
        ambient = context["t_amb"]
        loss_factor = context["heat_loss_factor"]
        t2 = (heat_capacity_flow * t1 + self.p["ua_loss"][1] * loss_factor * ambient) / (
            heat_capacity_flow + self.p["ua_loss"][1] * loss_factor
        )
        t3 = (heat_capacity_flow * t2 + self.p["ua_loss"][2] * loss_factor * ambient) / (
            heat_capacity_flow + self.p["ua_loss"][2] * loss_factor
        )
        liquid_heat = (
            heat_capacity_flow * (t1 - context["t_reservoir"])
            + self.p["ua_loss"][0] * loss_factor * (t1 - ambient)
        )
        electric_heat = liquid_heat / context["heater_efficiency"]
        pump_capacity = self.p["pump_flow_max"] * context["pump_flow_factor"]
        static_head = self.p["pump_static_head"]
        shutoff_head = self.p["pump_shutoff_head"]
        pump = math.sqrt(
            (static_head + (shutoff_head - static_head) * (q / pump_capacity) ** 2)
            / shutoff_head
        )
        valves = [
            q
            / (
                self.p["cv_valves"][index]
                * context[f"v{index + 1}{index + 2}_flow_factor"]
                * math.sqrt(h[index] + self.p["gravity_drop"][index])
            )
            for index in range(3)
        ]
        action = [pump, *valves, electric_heat / self.p["heater_power"]]
        reasons = [
            f"{name} command is outside [0, 1]"
            for name, value in zip(self.action_names, action)
            if not math.isfinite(value) or value < 0.0 or value > 1.0
        ]
        state = [h[0], t1, h[1], t2, h[2], t3]
        return {
            "feasible": not reasons,
            "infeasible_reasons": tuple(reasons),
            "state": state,
            "y_sp": [*h, t1, t2, t3],
            "action": action,
            "flow_m3s": q,
            "H1_to_liquid_power_w": liquid_heat,
            "H1_electric_power_w": electric_heat,
        }

    def tracking_steady_state_action(self, y_sp, disturbances=None):
        """Invert a physically consistent tracking target into actuator commands."""
        values = np.asarray(y_sp, dtype=float).reshape(-1)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            return None
        levels = values[:3].tolist()
        temperatures = values[3:].tolist()
        if any(level <= 0.0 for level in levels):
            return None
        context = self._resolved_env(disturbances)
        ambient = context["t_amb"]
        loss_factor = context["heat_loss_factor"]
        flow_candidates = []
        for index in (1, 2):
            upstream = temperatures[index - 1]
            downstream = temperatures[index]
            numerator = (
                self.p["ua_loss"][index]
                * loss_factor
                * (downstream - ambient)
            )
            denominator = RHO_CP * (upstream - downstream)
            if abs(denominator) > 1e-12:
                candidate = numerator / denominator
                if math.isfinite(candidate) and candidate > 0.0:
                    flow_candidates.append(candidate)
            elif abs(numerator) > 1e-9:
                return None
        if len(flow_candidates) == 1 or (
            len(flow_candidates) == 2
            and not np.allclose(
                flow_candidates,
                flow_candidates[0],
                rtol=1e-3,
                atol=1e-12,
            )
        ):
            return None
        flow = (
            float(np.mean(flow_candidates))
            if flow_candidates
            else 5.0 / 60000.0
        )
        pump_capacity = self.p["pump_flow_max"] * context["pump_flow_factor"]
        normalized_flow = flow / pump_capacity
        pump = math.sqrt(
            (
                self.p["pump_static_head"]
                + (
                    self.p["pump_shutoff_head"]
                    - self.p["pump_static_head"]
                )
                * normalized_flow**2
            )
            / self.p["pump_shutoff_head"]
        )
        valves = [
            flow
            / (
                self.p["cv_valves"][index]
                * context[f"v{index + 1}{index + 2}_flow_factor"]
                * math.sqrt(levels[index] + self.p["gravity_drop"][index])
            )
            for index in range(3)
        ]
        liquid_heat = (
            RHO_CP * flow * (temperatures[0] - context["t_reservoir"])
            + self.p["ua_loss"][0]
            * loss_factor
            * (temperatures[0] - ambient)
        )
        heater = liquid_heat / (
            self.p["heater_power"] * context["heater_efficiency"]
        )
        action = [pump, *valves, heater]
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in action
        ):
            return None
        return action

    def nominal_steady_state_for_tank3(
        self,
        *,
        tank_3_level,
        tank_3_temperature,
        flow=5.0 / 60000.0,
        upstream_levels=None,
        env=None,
    ):
        """Construct a physically consistent equilibrium from a Tank 3 target."""
        context = self._resolved_env(env)
        q = float(flow)
        h3 = float(tank_3_level)
        t3 = float(tank_3_temperature)
        if q <= 0.0 or not math.isfinite(q):
            raise ValueError("flow must be finite and positive")
        if not math.isfinite(h3) or h3 <= 0.0:
            raise ValueError("tank_3_level must be finite and positive")
        if not math.isfinite(t3):
            raise ValueError("tank_3_temperature must be finite")
        heat_capacity_flow = RHO_CP * q
        ambient = context["t_amb"]
        loss_factor = context["heat_loss_factor"]
        temperatures = [0.0, 0.0, t3]
        for index in (2, 1):
            downstream = temperatures[index]
            temperatures[index - 1] = (
                downstream
                * (heat_capacity_flow + self.p["ua_loss"][index] * loss_factor)
                - self.p["ua_loss"][index] * loss_factor * ambient
            ) / heat_capacity_flow
        if upstream_levels is None:
            upstream_levels = [self.p["nominal_level"]] * 2
        levels = [*map(float, upstream_levels), h3]
        return self.nominal_steady_state(
            flow=q,
            tank_1_temperature=temperatures[0],
            levels=levels,
            env=context,
        )

    def default_action(self):
        return list(self.nominal_steady_state()["action"])

    def default_setpoint_vector(self):
        return list(self.nominal_steady_state()["y_sp"])

    @staticmethod
    def _gate(condition, ops):
        return ops.if_else(condition, 1.0, 0.0)

    def _flow_terms(self, levels, u, env, ops):
        high_level_ok = self._gate(
            (levels[0] < self.p["high_level_trip"][0])
            * (levels[1] < self.p["high_level_trip"][1])
            * (levels[2] < self.p["high_level_trip"][2]),
            ops,
        )
        reservoir_ok = self._gate(env["reservoir_available"] >= 0.5, ops)
        pump_enabled = high_level_ok * reservoir_ok
        effective_max = self.p["pump_flow_max"] * env["pump_flow_factor"]
        head_margin = self.p["pump_shutoff_head"] - self.p["pump_static_head"]
        normalized_head = (
            self.p["pump_shutoff_head"] * u[0] * u[0]
            - self.p["pump_static_head"]
        ) / head_margin
        pump_flow = effective_max * ops.sqrt(ops.max(normalized_head, 0.0)) * pump_enabled
        valve_flows = [
            self.p["cv_valves"][index]
            * env[f"v{index + 1}{index + 2}_flow_factor"]
            * u[1 + index]
            * ops.sqrt(ops.max(levels[index] + self.p["gravity_drop"][index], 0.0))
            for index in range(3)
        ]
        overflow_flows = []
        for index in range(3):
            head = levels[index] - self.p["overflow_level"][index]
            enabled = self._gate(head > 0.0, ops)
            overflow_flows.append(
                self.p["cv_overflow"][index]
                * enabled
                * ops.sqrt(ops.max(head, self.p["overflow_head_floor"]))
            )
        return pump_flow, valve_flows, overflow_flows, pump_enabled

    def _heater_terms(self, levels, temperatures, u, env, ops):
        level_ok = self._gate(levels[0] >= self.p["low_level_trip"][0], ops)
        temperature_ok = self._gate(temperatures[0] < self.p["temperature_trip"], ops)
        enabled = level_ok * temperature_ok
        electric = u[4] * self.p["heater_power"] * enabled
        return electric * env["heater_efficiency"], electric, enabled

    def _dynamics(self, x, u, env, ops):
        context = self._resolved_env(env, ops)
        u = self._effective_action(u, ops)
        levels, temperatures = self._levels_temperatures(x)
        pump, valves, overflows, _ = self._flow_terms(levels, u, context, ops)
        q12, q23, q34 = valves
        heat_h1, _, _ = self._heater_terms(levels, temperatures, u, context, ops)
        flows_in = [pump, q12, q23]
        flows_out = [q12 + overflows[0], q23 + overflows[1], q34 + overflows[2]]
        mixing_terms = [
            pump * (context["t_reservoir"] - temperatures[0]),
            q12 * (temperatures[0] - temperatures[1]),
            q23 * (temperatures[1] - temperatures[2]),
        ]
        return self._assemble_dynamics(
            levels,
            temperatures,
            flows_in,
            flows_out,
            mixing_terms,
            [heat_h1, 0.0, 0.0],
            context,
            ops,
        )

    def hard_termination_reasons(self, x, levels, temps, env):
        del levels, temps, env
        h = [float(x[0]), float(x[2]), float(x[4])]
        temperatures = [float(x[1]), float(x[3]), float(x[5])]
        reasons = []
        if any(value < 0.0 for value in h):
            reasons.append("negative_level")
        if any(value > self.p["height_max"][index] for index, value in enumerate(h)):
            reasons.append("tank_overflow_limit")
        if any(value >= self.p["temperature_hard_limit"] for value in temperatures):
            reasons.append("temperature_hard_limit")
        return tuple(reasons)

    def process_info(self, x, levels, temps, env, action=None):
        del levels, temps
        context = self._resolved_env(env)
        u = self._effective_action(
            self.action_vector(self.default_action() if action is None else action),
            _NUMERIC_OPS,
        )
        physical_levels = [float(x[0]), float(x[2]), float(x[4])]
        temperatures = [float(x[1]), float(x[3]), float(x[5])]
        pump, valves, overflows, pump_enabled = self._flow_terms(
            physical_levels, u, context, _NUMERIC_OPS
        )
        heat, electric, heater_enabled = self._heater_terms(
            physical_levels, temperatures, u, context, _NUMERIC_OPS
        )
        return {
            "P101_flow_m3s": float(pump),
            "V12_flow_m3s": float(valves[0]),
            "V23_flow_m3s": float(valves[1]),
            "V34_flow_m3s": float(valves[2]),
            "product_flow_m3s": float(valves[2]),
            "overflow_flow_m3s": [float(value) for value in overflows],
            "P101_enabled": bool(pump_enabled),
            "H1_enabled": bool(heater_enabled),
            "H1_electric_power_w": float(electric),
            "H1_to_liquid_power_w": float(heat),
            "reservoir_temperature_degC": float(context["t_reservoir"]),
        }

    def action_energy_kw(self, act, x=None, env=None):
        u = self._effective_action(self.action_vector(act), _NUMERIC_OPS)
        context = self._resolved_env(env)
        pump_enabled = 1.0
        heater = u[4] * self.p["heater_power"]
        if x is not None:
            levels = [float(x[0]), float(x[2]), float(x[4])]
            temperatures = [float(x[1]), float(x[3]), float(x[5])]
            _, _, _, pump_enabled = self._flow_terms(levels, u, context, _NUMERIC_OPS)
            _, heater, _ = self._heater_terms(levels, temperatures, u, context, _NUMERIC_OPS)
        pump = u[0] ** 3 * self.p["pump_power_max"] * pump_enabled
        return float((pump + heater) / 1000.0)

    def energy_kw(self, u, backend="numeric", ca=None):
        if backend == "numeric":
            values, ops = self.action_vector(u), _NUMERIC_OPS
        elif backend == "casadi":
            if ca is None:
                raise ValueError("backend='casadi' requires the casadi module as ca=...")
            from aiogym.core.backends import _casadi_ops

            values, ops = u, _casadi_ops(ca)
        else:
            raise ValueError(f"unknown dynamics backend: {backend!r}")
        effective = self._effective_action(values, ops)
        return (
            effective[0] ** 3 * self.p["pump_power_max"]
            + effective[4] * self.p["heater_power"]
        ) / 1000.0

    def production(self, x, act, env=None):
        levels = [float(x[0]), float(x[2]), float(x[4])]
        context = self._resolved_env(env)
        u = self._effective_action(self.action_vector(act), _NUMERIC_OPS)
        _, valves, _, _ = self._flow_terms(levels, u, context, _NUMERIC_OPS)
        return float(valves[2])

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
                        "name": f"TT{tank}01",
                        "quantity": f"tank_{tank}_temperature",
                        "signal": "RS485",
                    }
                    for tank in (1, 2, 3)
                ],
                *[
                    {
                        "name": f"FT{start}{end}",
                        "quantity": f"V{start}{end}_flow",
                        "signal": "RS485",
                    }
                    for start, end in ((1, 2), (2, 3), (3, 4))
                ],
            ],
            "actuators": [
                {"name": "P101", "command": "VFD speed"},
                {"name": "V12", "command": "valve position"},
                {"name": "V23", "command": "valve position"},
                {"name": "V34", "command": "valve position"},
                {"name": "H1", "command": "SSR duty"},
            ],
            "boundary": dict(BOM_CONFIGURATION["reservoir"]),
        }


class ThreeTankModel(_BOMThreeTankKernel):
    """Fixed BOM Three-Tank model with validated parameter overrides."""

    parameter_units = MappingProxyType(
        {
            "area": "m^2",
            "height_max": "m",
            "level_sensor_range": "m",
            "cv_valves": "m^(5/2)/s",
            "gravity_drop": "m",
            "overflow_level": "m",
            "cv_overflow": "m^(5/2)/s",
            "overflow_head_floor": "m",
            "high_level_trip": "m",
            "low_level_trip": "m",
            "nominal_level": "m",
            "ua_loss": "W/K",
            "heater_power": "W",
            "pump_flow_max": "m^3/s",
            "pump_power_max": "W",
            "pump_static_head": "m",
            "pump_shutoff_head": "m",
            "h_floor": "m",
            "temperature_trip": "degC",
            "temperature_hard_limit": "degC",
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
            raise KeyError(f"unknown three-tank parameter {name!r}") from error

    def outputs(self, state):
        return self.controlled_output(state)

    def action_schema(self):
        return _schema(super().action_schema())

    def state_schema(self):
        return _schema(super().state_schema())

    def output_schema(self):
        return _schema(super().controlled_output_schema())

    def default_disturbances(self):
        return dict(self.disturbance_defaults())

    def action_slew_limits(self):
        return list(TANK3_MAXIMUM_ACTION_STEP)

    def observation_schema(self):
        output_names = [f"normalized_{row['name']}" for row in self.output_schema()]
        reference_names = [
            f"normalized_{row['name']}_reference" for row in self.output_schema()
        ]
        action_names = [f"previous_{row['name']}" for row in self.action_schema()]
        return [
            {"name": name, "unit": "normalized", "low": 0.0, "high": 1.0}
            for name in (*output_names, *reference_names, *action_names)
        ]

    def observation(self, state, reference, previous_action, disturbances):
        del disturbances
        scales = np.asarray(self.controlled_output_scales(), dtype=float)
        output = np.asarray(self.outputs(state), dtype=float)
        normalized_output = np.clip(output / scales, 0.0, 1.0)
        normalized_reference = np.clip(
            np.asarray(reference, dtype=float) / scales,
            0.0,
            1.0,
        )
        action = np.clip(np.asarray(previous_action, dtype=float), 0.0, 1.0)
        return [
            *normalized_output.tolist(),
            *normalized_reference.tolist(),
            *action.tolist(),
        ]

    def measurement(self, state, disturbances=None):
        context = self.runtime_env({} if disturbances is None else disturbances)
        display = self.display_outputs(state)
        return {
            "x": list(state),
            "levels": list(display["levels"]),
            "temps": list(display["temps"]),
            "y": list(self.controlled_output(state)),
            **context,
        }

    def measurement_from_observation(self, observation, disturbances=None):
        values = np.asarray(observation, dtype=float).reshape(-1)
        expected = len(self.observation_schema())
        if values.shape != (expected,) or not np.isfinite(values).all():
            raise ValueError(
                "three-tank policy observation must match observation_schema"
            )
        output_dim = len(self.output_schema())
        if len(self.state_schema()) != output_dim:
            raise ValueError(
                "three-tank policy observation must expose the complete state"
            )
        scales = np.asarray(self.controlled_output_scales(), dtype=float)
        output = values[:output_dim] * scales
        state = np.empty(output_dim, dtype=float)
        state[0::2] = output[:3]
        state[1::2] = output[3:]
        return self.measurement(state, disturbances)

    def clamp_state(self, state):
        return list(state)

    def constraint_costs(self, state, disturbances=None):
        values = list(state)
        reasons = self.hard_termination_reasons(
            values,
            values[0::2],
            values[1::2],
            self._resolved_env(disturbances),
        )
        return {str(reason): 1.0 for reason in reasons}

    def safety_margins(self, state, disturbances=None):
        del disturbances
        values = [float(value) for value in state]
        levels = values[0::2]
        temperatures = values[1::2]
        temperature_limit = float(self.p["temperature_hard_limit"])
        margins = {}
        for index, (level, maximum) in enumerate(
            zip(levels, self.p["height_max"]), start=1
        ):
            maximum = float(maximum)
            margins[f"tank_{index}_level_lower"] = level / maximum
            margins[f"tank_{index}_level_upper"] = (maximum - level) / maximum
        for index, temperature in enumerate(temperatures, start=1):
            margins[f"tank_{index}_temperature_upper"] = (
                temperature_limit - temperature
            ) / temperature_limit
        return margins

    def step_info(self, state, action, disturbances=None):
        values = list(state)
        context = self._resolved_env(disturbances)
        applied = self.default_action() if action is None else list(action)
        info = self.process_info(
            values,
            values[0::2],
            values[1::2],
            context,
            applied,
        )
        info["y"] = list(self.outputs(values))
        info["energy_kw"] = self.action_energy_kw(applied, values, context)
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
        not low < nominal < high < overflow < maximum
        for low, nominal, high, overflow, maximum in zip(
            resolved["low_level_trip"],
            [resolved["nominal_level"]] * 3,
            resolved["high_level_trip"],
            resolved["overflow_level"],
            resolved["height_max"],
        )
    ):
        raise ValueError(
            "three-tank level parameters must satisfy low < nominal < high < overflow < maximum"
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
    if resolved["temperature_trip"] >= resolved["temperature_hard_limit"]:
        raise ValueError("temperature_trip must be below temperature_hard_limit")
    if resolved["h_floor"] >= min(resolved["low_level_trip"]):
        raise ValueError("h_floor must be below low_level_trip")
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
    "TANK3_INTERNAL_CONTROL",
    "TANK3_MAXIMUM_ACTION_STEP",
    "ThreeTankModel",
]
