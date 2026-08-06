"""Closed-loop three-tank rig described by the retrofit design PDF.

The public ``cascade`` scenario is an open, three-heater benchmark.  This model
is intentionally separate because the retrofit rig recirculates Tank 3 through
P101, has only two gravity valves, and heats only Tank 1.
"""
from __future__ import annotations

import math

from aiogym.core.backends import _NUMERIC_OPS, _casadi_ops, _maxv
from aiogym.core.model import RHO_CP, ProcessModelContract


class RecirculatingCascadeModel(ProcessModelContract):
    scenario = "cascade_recirculating"
    display_name = "Recirculating heated-tank cascade"
    summary = (
        "Three non-identical tanks in a closed P101-Tank 1-V12-Tank 2-V23-Tank 3 "
        "loop with one 2 kW heater."
    )
    supported_goals = ("regulation",)
    benchmark_goals = supported_goals
    n = 3
    dt_micro = 0.02

    state_names = ("h1", "T1", "h2", "T2", "h3", "T3")
    state_units = {
        "h1": "m", "h2": "m", "h3": "m",
        "T1": "degC", "T2": "degC", "T3": "degC",
    }
    action_names = ("pump_P101", "valve_V12", "valve_V23", "heater_H1")
    action_kinds = {
        "pump_P101": "pump",
        "valve_V12": "valve",
        "valve_V23": "valve",
        "heater_H1": "heater",
    }
    output_names = (
        "tank_1_level", "tank_2_level", "tank_3_level",
        "tank_1_temperature", "tank_2_temperature", "tank_3_temperature",
    )
    output_units = {
        "tank_1_level": "m", "tank_2_level": "m", "tank_3_level": "m",
        "tank_1_temperature": "degC",
        "tank_2_temperature": "degC",
        "tank_3_temperature": "degC",
    }
    default_y_sp = (
        0.24, 0.24, 0.24,
        30.0, 28.97128161165881, 27.654664660905787,
    )
    supervisory_layout = (("y_sp", 3, 20.0, 75.0),)

    # These ranges describe benchmark mutability, not equipment tolerances.
    plant_regime = {
        "cv_interstage": (0.7, 1.3),
        "ua_loss": (0.5, 2.0),
        "pump_flow_max": (0.7, 1.3),
        "heater_power": (0.9, 1.0),
    }
    economic_config = {
        "temp_band": (),
        "level_band": (),
        "value": "none",
        "w_value": 0.0,
        "w_energy": 0.0,
        "w_viol": 0.0,
    }

    param_units = {
        "area": "m2",
        "height_max": "m",
        "level_sensor_range": "m",
        "cv_interstage": "m2.5/s",
        "gravity_drop": "m",
        "overflow_level": "m",
        "cv_overflow": "m2.5/s",
        "overflow_head_floor": "m",
        "high_level_trip": "m",
        "ua_loss": "W/K",
        "heater_power": "W",
        "pump_flow_max": "m3/s",
        "pump_power_max": "W",
        "pump_static_head": "m",
        "pump_shutoff_head": "m",
        "t_amb": "degC",
        "h_floor": "m",
        "low_level_trip": "m",
        "temperature_trip": "degC",
        "temperature_hard_limit": "degC",
    }
    param_bounds = {
        "area": (0.01, 5.0),
        "height_max": (0.1, 5.0),
        "level_sensor_range": (0.1, 5.0),
        "cv_interstage": (0.0, 0.02),
        "gravity_drop": (0.0, 5.0),
        "overflow_level": (0.05, 5.0),
        "cv_overflow": (0.0, 0.05),
        "overflow_head_floor": (1e-12, 1e-3),
        "high_level_trip": (0.05, 5.0),
        "ua_loss": (0.0, 2000.0),
        "heater_power": (0.0, 5000.0),
        "pump_flow_max": (0.0, 0.02),
        "pump_power_max": (0.0, 5000.0),
        "pump_static_head": (0.0, 20.0),
        "pump_shutoff_head": (0.0, 50.0),
        "t_amb": (0.0, 45.0),
        "h_floor": (1e-6, 0.1),
        "low_level_trip": (0.0, 1.0),
        "temperature_trip": (40.0, 100.0),
        "temperature_hard_limit": (60.0, 120.0),
    }
    input_disturbances = (
        {
            "name": "t_amb", "event": "ambient_step", "unit": "degC",
            "bounds": (0.0, 45.0), "default": 20.0,
            "description": "ambient air temperature",
        },
        {
            "name": "pump_flow_factor", "event": "pump_capacity_shift",
            "unit": "fraction", "bounds": (0.4, 1.4), "default": 1.0,
            "description": "P101 circulation-flow capacity multiplier",
        },
        {
            "name": "heater_efficiency", "event": "heater_efficiency_shift",
            "unit": "fraction", "bounds": (0.4, 1.0), "default": 1.0,
            "description": "fraction of H1 electrical power transferred to Tank 1 liquid",
        },
        {
            "name": "heat_loss_factor", "event": "heat_loss_shift",
            "unit": "fraction", "bounds": (0.3, 3.0), "default": 1.0,
            "description": "common multiplier on the three provisional UA values",
        },
    )

    def __init__(self):
        # V2.0 is the executable design baseline. Conflicting procurement
        # candidates remain documented in the parameter profile and must not be
        # mixed into this numerical parameter set.
        self.p = {
            "area": [0.075, 0.075, 0.075],
            "height_max": [0.40, 0.40, 0.40],
            "level_sensor_range": [0.50, 0.50, 0.50],
            "cv_interstage": [0.0005, 0.0005],
            "gravity_drop": [0.30, 0.30],
            "overflow_level": [0.36, 0.36],
            "cv_overflow": [0.001, 0.001],
            "overflow_head_floor": 1e-9,
            "high_level_trip": [0.34, 0.34, 0.34],
            "low_level_trip": [0.08, 0.08, 0.08],
            "ua_loss": [40.0, 40.0, 60.0],
            "heater_power": 2000.0,
            "pump_flow_max": 25.0 / 60000.0,
            "pump_power_max": 370.0,
            "pump_static_head": 1.7,
            "pump_shutoff_head": 10.0,
            "t_amb": 20.0,
            "h_floor": 1e-3,
            "temperature_trip": 80.0,
            "temperature_hard_limit": 100.0,
        }
        self._environment_bounds = {
            row["name"]: tuple(row["bounds"])
            for row in self.input_disturbances
        }

    @property
    def height_max(self):
        return [float(value) for value in self.p["height_max"]]

    @property
    def state_bounds(self):
        hmax = self.height_max
        tmax = float(self.p["temperature_hard_limit"])
        return {
            "h1": (0.0, hmax[0]), "h2": (0.0, hmax[1]), "h3": (0.0, hmax[2]),
            "T1": (0.0, tmax), "T2": (0.0, tmax), "T3": (0.0, tmax),
        }

    @property
    def output_bounds(self):
        hmax = self.height_max
        tmax = float(self.p["temperature_hard_limit"])
        return {
            "tank_1_level": (0.0, hmax[0]),
            "tank_2_level": (0.0, hmax[1]),
            "tank_3_level": (0.0, hmax[2]),
            "tank_1_temperature": (0.0, tmax),
            "tank_2_temperature": (0.0, tmax),
            "tank_3_temperature": (0.0, tmax),
        }

    @property
    def setpoint_bounds(self):
        bounds = dict(self.output_bounds)
        bounds.update({
            "tank_1_temperature": (20.0, 75.0),
            "tank_2_temperature": (20.0, 75.0),
            "tank_3_temperature": (20.0, 75.0),
        })
        return bounds

    @property
    def safety_constraints(self):
        return (
            {
                "name": "level_bounds",
                "states": ("h1", "h2", "h3"),
                "bounds": (0.0, max(self.height_max)),
            },
            {
                "name": "P101_low_level_interlock",
                "states": ("h3",),
                "bounds": (float(self.p["low_level_trip"][2]), None),
            },
            {
                "name": "passive_overflow_onset",
                "states": ("h1", "h2"),
                "bounds": (None, max(float(value) for value in self.p["overflow_level"])),
            },
            {
                "name": "P101_high_level_interlock",
                "states": ("h1", "h2"),
                "bounds": (
                    None,
                    max(float(value) for value in self.p["high_level_trip"][:2]),
                ),
            },
            {
                "name": "H1_low_level_interlock",
                "states": ("h1",),
                "bounds": (float(self.p["low_level_trip"][0]), None),
            },
            {
                "name": "H1_temperature_trip",
                "states": ("T1",),
                "bounds": (None, float(self.p["temperature_trip"])),
            },
            {
                "name": "temperature_hard_limit",
                "states": ("T1", "T2", "T3"),
                "bounds": (None, float(self.p["temperature_hard_limit"])),
            },
        )

    def initial_state(self):
        return [0.24, 20.0, 0.24, 20.0, 0.24, 20.0]

    def nominal_steady_state(
        self,
        *,
        circulation_flow=5.0 / 60000.0,
        tank_1_temperature=30.0,
        levels=(0.24, 0.24, 0.24),
        env=None,
    ):
        """Return a model-consistent benchmark equilibrium.

        The PDF does not provide a commissioned operating point.  This helper
        therefore starts from an explicitly assumed circulation flow and Tank 1
        temperature, then derives the passive Tank 2/Tank 3 temperatures and
        required actuator commands from the same balances used by the model.
        """

        context = self._resolved_env(env)
        flow = float(circulation_flow)
        t1 = float(tank_1_temperature)
        h = [float(value) for value in levels]
        if not math.isfinite(flow) or flow <= 0.0:
            raise ValueError("circulation_flow must be finite and positive")
        if not math.isfinite(t1):
            raise ValueError("tank_1_temperature must be finite")
        if len(h) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in h):
            raise ValueError("levels must contain three finite positive values")

        heat_capacity_flow = RHO_CP * flow
        loss_factor = context["heat_loss_factor"]
        ambient = context["t_amb"]
        ua2 = self.p["ua_loss"][1] * loss_factor
        ua3 = self.p["ua_loss"][2] * loss_factor
        t2 = (heat_capacity_flow * t1 + ua2 * ambient) / (heat_capacity_flow + ua2)
        t3 = (heat_capacity_flow * t2 + ua3 * ambient) / (heat_capacity_flow + ua3)

        liquid_heat = (
            heat_capacity_flow * (t1 - t3)
            + self.p["ua_loss"][0] * loss_factor * (t1 - ambient)
        )
        efficiency = context["heater_efficiency"]
        electric_heat = liquid_heat / efficiency if efficiency > 0.0 else math.inf
        pump_capacity = self.p["pump_flow_max"] * context["pump_flow_factor"]
        static_head = float(self.p["pump_static_head"])
        shutoff_head = float(self.p["pump_shutoff_head"])
        pump_command = math.inf
        if pump_capacity > 0.0 and shutoff_head > static_head:
            pump_command = math.sqrt(
                (
                    static_head
                    + (shutoff_head - static_head)
                    * (flow / pump_capacity) ** 2
                )
                / shutoff_head
            )
        heater_capacity = float(self.p["heater_power"])
        heater_command = (
            electric_heat / heater_capacity
            if heater_capacity > 0.0
            else (0.0 if abs(electric_heat) <= 1e-12 else math.inf)
        )
        action = [
            pump_command,
            flow / (
                self.p["cv_interstage"][0]
                * math.sqrt(h[0] + self.p["gravity_drop"][0])
            ),
            flow / (
                self.p["cv_interstage"][1]
                * math.sqrt(h[1] + self.p["gravity_drop"][1])
            ),
            heater_command,
        ]
        reasons = []
        labels = ("pump_P101", "valve_V12", "valve_V23", "heater_H1")
        for label, command in zip(labels, action):
            if not math.isfinite(command) or command < 0.0 or command > 1.0:
                reasons.append(f"{label} command is outside [0, 1]")
        if h[2] < self.p["low_level_trip"][2]:
            reasons.append("P101 is blocked by the Tank 3 low-level interlock")
        if h[0] < self.p["low_level_trip"][0]:
            reasons.append("H1 is blocked by the Tank 1 low-level interlock")
        if any(h[i] >= self.p["high_level_trip"][i] for i in range(2)):
            reasons.append("P101 is blocked by the L3 high-level interlock")
        if any(h[i] > self.p["overflow_level"][i] for i in range(2)):
            reasons.append("requested levels activate passive overflow and are not steady")
        if t1 >= self.p["temperature_trip"]:
            reasons.append("H1 is blocked by the Tank 1 temperature trip")

        state = [h[0], t1, h[1], t2, h[2], t3]
        return {
            "feasible": not reasons,
            "infeasible_reasons": tuple(reasons),
            "circulation_flow_m3s": flow,
            "state": state,
            "y_sp": [h[0], h[1], h[2], t1, t2, t3],
            "action": action,
            "H1_to_liquid_power_w": liquid_heat,
            "H1_electric_power_w": electric_heat,
            "P101_electric_power_w": action[0] ** 3 * self.p["pump_power_max"],
            "ideal_energy_kw": (
                action[0] ** 3 * self.p["pump_power_max"] + electric_heat
            ) / 1000.0,
        }

    def default_action(self):
        if self.p["heater_power"] <= 0.0:
            return list(
                self.nominal_steady_state(
                    tank_1_temperature=self.p["t_amb"]
                )["action"]
            )
        return list(self.nominal_steady_state()["action"])

    def mpc_init(self):
        return self.default_action()

    def default_setpoint_vector(self):
        return list(self.nominal_steady_state()["y_sp"])

    def tracking_steady_state_action(self, y_sp):
        nominal = self.nominal_steady_state()
        requested = [float(value) for value in y_sp]
        if len(requested) != len(nominal["y_sp"]):
            return None
        if all(
            math.isclose(requested[i], nominal["y_sp"][i], rel_tol=0.0, abs_tol=1e-9)
            for i in range(len(requested))
        ):
            return list(nominal["action"])
        return None

    def controlled_output(self, x, backend="numeric", ca=None):
        return [x[0], x[2], x[4], x[1], x[3], x[5]]

    def integral_observation_limits(self):
        return [8.0, 8.0, 8.0, 300.0, 300.0, 300.0]

    def display_outputs(self, x, backend="numeric", ca=None):
        levels = [x[0], x[2], x[4]]
        if backend != "casadi":
            levels = [_maxv(value, 0.0) for value in levels]
        return {"levels": levels, "temps": [x[1], x[3], x[5]]}

    def physical_io_schema(self):
        """Return the V2.0 field-instrument and actuator contract."""

        return {
            "analog_measurements": [
                *[
                    {
                        "name": f"LT{tank}01",
                        "quantity": f"tank_{tank}_level",
                        "range": [0.0, self.p["level_sensor_range"][tank - 1]],
                        "unit": "m",
                        "signal": "4-20mA",
                    }
                    for tank in (1, 2, 3)
                ],
                *[
                    {
                        "name": f"TT{tank}01",
                        "quantity": f"tank_{tank}_temperature",
                        "range": [-20.0, 150.0],
                        "unit": "degC",
                        "signal": "PT100/4-20mA",
                    }
                    for tank in (1, 2, 3)
                ],
                {
                    "name": "FT12",
                    "quantity": "V12_flow",
                    "range": [0.0, self.p["pump_flow_max"]],
                    "unit": "m3/s",
                    "signal": "4-20mA/RS485",
                },
                {
                    "name": "FT23",
                    "quantity": "V23_flow",
                    "range": [0.0, self.p["pump_flow_max"]],
                    "unit": "m3/s",
                    "signal": "4-20mA/RS485",
                },
            ],
            "digital_inputs": [
                *[
                    {
                        "name": f"LSL{tank}01",
                        "quantity": f"tank_{tank}_low_level",
                        "threshold": self.p["low_level_trip"][tank - 1],
                        "unit": "m",
                    }
                    for tank in (1, 2, 3)
                ],
                *[
                    {
                        "name": f"LSH{tank}01",
                        "quantity": f"tank_{tank}_high_level",
                        "threshold": self.p["high_level_trip"][tank - 1],
                        "unit": "m",
                    }
                    for tank in (1, 2, 3)
                ],
            ],
            "actuators": [
                {"name": "P101", "signal": "VFD/Modbus", "command": "speed"},
                {"name": "V12", "signal": "4-20mA/Modbus", "command": "position"},
                {"name": "V23", "signal": "4-20mA/Modbus", "command": "position"},
                {"name": "H1", "signal": "SSR", "command": "duty"},
            ],
            "network": "Modbus RTU (RS485) through RTU-to-TCP gateway",
        }

    def metadata(self):
        metadata = super().metadata()
        metadata["physical_io"] = self.physical_io_schema()
        return metadata

    def _resolved_env(self, env=None, ops=None):
        values = dict(env or {})
        resolved = {
            "t_amb": values.get("t_amb", self.p["t_amb"]),
            "pump_flow_factor": values.get("pump_flow_factor", 1.0),
            "heater_efficiency": values.get("heater_efficiency", 1.0),
            "heat_loss_factor": values.get("heat_loss_factor", 1.0),
        }
        if bool(getattr(ops, "symbolic", False)):
            return resolved
        clean = {}
        for name, value in resolved.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"cascade_recirculating disturbance {name!r} must be finite"
                ) from exc
            if not math.isfinite(number):
                raise ValueError(
                    f"cascade_recirculating disturbance {name!r} must be finite"
                )
            lower, upper = self._environment_bounds[name]
            if number < lower or number > upper:
                raise ValueError(
                    f"cascade_recirculating disturbance {name!r} must be within "
                    f"[{lower}, {upper}], got {number}"
                )
            clean[name] = number
        return clean

    def runtime_env(self, disturbance_values):
        return self._resolved_env(disturbance_values)

    def disturbance_vector(self, values=None):
        return super().disturbance_vector(self.runtime_env(values or {}))

    def _effective_action(self, u, ops):
        if not bool(getattr(ops, "symbolic", False)):
            for value in u:
                if not math.isfinite(float(value)):
                    raise ValueError("cascade_recirculating action values must be finite")
        return [
            ops.min(ops.max(u[i], 0.0), 1.0)
            for i in range(self.action_dim())
        ]

    @staticmethod
    def _gate(condition, ops):
        return ops.if_else(condition, 1.0, 0.0)

    def _pump_interlock_terms(self, levels, ops):
        low_level_ok = self._gate(
            levels[2] >= self.p["low_level_trip"][2], ops
        )
        high_level_ok = self._gate(
            (levels[0] < self.p["high_level_trip"][0])
            * (levels[1] < self.p["high_level_trip"][1]),
            ops,
        )
        return low_level_ok, high_level_ok, low_level_ok * high_level_ok

    def _pump_curve_flow(self, speed, env, ops):
        """Provisional affinity-law curve fitted to the V2.0 endpoints."""

        effective_max_flow = (
            self.p["pump_flow_max"] * env["pump_flow_factor"]
        )
        head_margin = self.p["pump_shutoff_head"] - self.p["pump_static_head"]
        if head_margin <= 0.0:
            raise ValueError(
                "pump_shutoff_head must be greater than pump_static_head"
            )
        normalized_head = (
            self.p["pump_shutoff_head"] * speed * speed
            - self.p["pump_static_head"]
        ) / head_margin
        return effective_max_flow * ops.sqrt(ops.max(normalized_head, 0.0))

    def _flow_terms(self, levels, u, env, ops):
        _, _, pump_enabled = self._pump_interlock_terms(levels, ops)
        pump_flow = self._pump_curve_flow(u[0], env, ops) * pump_enabled
        q12 = (
            self.p["cv_interstage"][0]
            * u[1]
            * ops.sqrt(
                ops.max(levels[0] + self.p["gravity_drop"][0], 0.0)
            )
        )
        q23 = (
            self.p["cv_interstage"][1]
            * u[2]
            * ops.sqrt(
                ops.max(levels[1] + self.p["gravity_drop"][1], 0.0)
            )
        )
        overflow_flows = []
        for i in range(2):
            overflow_head = levels[i] - self.p["overflow_level"][i]
            overflow_enabled = self._gate(overflow_head > 0.0, ops)
            overflow_flows.append(
                self.p["cv_overflow"][i]
                * overflow_enabled
                * ops.sqrt(
                    ops.max(overflow_head, self.p["overflow_head_floor"])
                )
            )
        return pump_flow, q12, q23, overflow_flows, pump_enabled

    def _heater_terms(self, levels, temperatures, u, env, ops):
        level_ok = self._gate(
            levels[0] >= self.p["low_level_trip"][0], ops
        )
        temperature_ok = self._gate(
            temperatures[0] < self.p["temperature_trip"], ops
        )
        enabled = level_ok * temperature_ok
        electric_power = u[3] * self.p["heater_power"] * enabled
        return (
            electric_power * env["heater_efficiency"],
            electric_power,
            enabled,
            level_ok,
            temperature_ok,
        )

    def _dynamics(self, x, u, env, ops):
        env = self._resolved_env(env, ops)
        u = self._effective_action(u, ops)
        levels = [x[0], x[2], x[4]]
        temperatures = [x[1], x[3], x[5]]
        pump_flow, q12, q23, overflow_flows, _ = self._flow_terms(
            levels, u, env, ops
        )
        overflow_1, overflow_2 = overflow_flows
        heat_h1, _, _, _, _ = self._heater_terms(
            levels, temperatures, u, env, ops
        )
        flows_in = [pump_flow, q12, q23 + overflow_1 + overflow_2]
        flows_out = [q12 + overflow_1, q23 + overflow_2, pump_flow]
        mixing_terms = [
            pump_flow * (temperatures[2] - temperatures[0]),
            q12 * (temperatures[0] - temperatures[1]),
            q23 * (temperatures[1] - temperatures[2])
            + overflow_1 * (temperatures[0] - temperatures[2])
            + overflow_2 * (temperatures[1] - temperatures[2]),
        ]
        heat_inputs = [heat_h1, 0.0, 0.0]

        dx = []
        for i in range(3):
            area = self.p["area"][i]
            volume = area * ops.max(levels[i], self.p["h_floor"])
            heat_loss = (
                self.p["ua_loss"][i]
                * env["heat_loss_factor"]
                * (temperatures[i] - env["t_amb"])
            )
            dx.extend((
                (flows_in[i] - flows_out[i]) / area,
                mixing_terms[i] / volume
                + (heat_inputs[i] - heat_loss) / (RHO_CP * volume),
            ))
        return ops.vector(dx)

    def balance_residuals(self, x, u, env=None):
        """Independently reconstruct numeric mass and energy residuals.

        This audit surface is defined for positive physical levels above the
        numerical ``h_floor``.  It does not reuse the temperature-derivative
        formula when constructing the right-hand side of the energy balance.
        """

        state = self.state_vector(x)
        action = self._effective_action(self.action_vector(u), _NUMERIC_OPS)
        context = self._resolved_env(env)
        levels = [state[0], state[2], state[4]]
        temperatures = [state[1], state[3], state[5]]
        if any(level <= self.p["h_floor"] for level in levels):
            raise ValueError("balance_residuals requires levels above h_floor")

        dx = list(self.dynamics(state, action, context))
        pump_flow, q12, q23, overflow_flows, _ = self._flow_terms(
            levels, action, context, _NUMERIC_OPS
        )
        overflow_1, overflow_2 = overflow_flows
        heat_h1, _, _, _, _ = self._heater_terms(
            levels, temperatures, action, context, _NUMERIC_OPS
        )
        flows_in = [pump_flow, q12, q23 + overflow_1 + overflow_2]
        inlet_enthalpy_flows = [
            pump_flow * temperatures[2],
            q12 * temperatures[0],
            q23 * temperatures[1]
            + overflow_1 * temperatures[0]
            + overflow_2 * temperatures[1],
        ]
        flows_out = [q12 + overflow_1, q23 + overflow_2, pump_flow]
        heat_inputs = [heat_h1, 0.0, 0.0]

        mass_residuals = []
        energy_residuals = []
        stored_energy_rates = []
        external_energy_rates = []
        for i in range(3):
            area = self.p["area"][i]
            dh = dx[2 * i]
            dtemperature = dx[2 * i + 1]
            mass_residuals.append(
                area * dh - (flows_in[i] - flows_out[i])
            )
            stored_energy_rate = RHO_CP * area * (
                levels[i] * dtemperature + temperatures[i] * dh
            )
            heat_loss = (
                self.p["ua_loss"][i]
                * context["heat_loss_factor"]
                * (temperatures[i] - context["t_amb"])
            )
            external_energy_rate = (
                RHO_CP
                * (
                    inlet_enthalpy_flows[i]
                    - flows_out[i] * temperatures[i]
                )
                + heat_inputs[i]
                - heat_loss
            )
            stored_energy_rates.append(stored_energy_rate)
            external_energy_rates.append(external_energy_rate)
            energy_residuals.append(stored_energy_rate - external_energy_rate)

        return {
            "tank_mass_balance_m3s": mass_residuals,
            "total_mass_balance_m3s": sum(
                self.p["area"][i] * dx[2 * i] for i in range(3)
            ),
            "tank_energy_balance_w": energy_residuals,
            "total_energy_balance_w": (
                sum(stored_energy_rates) - sum(external_energy_rates)
            ),
        }

    def physical_validation_checks(self):
        samples = (
            (
                [0.20, 25.0, 0.30, 24.0, 0.32, 23.0],
                [0.04, 0.05, 0.04, 0.30],
                {"t_amb": 20.0, "pump_flow_factor": 1.0,
                 "heater_efficiency": 0.9, "heat_loss_factor": 1.0},
            ),
            (
                [0.37, 45.0, 0.30, 36.0, 0.25, 28.0],
                [0.08, 0.03, 0.07, 0.70],
                {"t_amb": 18.0, "pump_flow_factor": 0.8,
                 "heater_efficiency": 0.7, "heat_loss_factor": 1.6},
            ),
            (
                [0.12, 70.0, 0.35, 55.0, 0.25, 40.0],
                [0.02, 0.09, 0.05, 1.0],
                {"t_amb": 25.0, "pump_flow_factor": 1.2,
                 "heater_efficiency": 1.0, "heat_loss_factor": 0.6},
            ),
        )
        residuals = [self.balance_residuals(*sample) for sample in samples]
        max_mass = max(
            abs(value)
            for row in residuals
            for value in (*row["tank_mass_balance_m3s"], row["total_mass_balance_m3s"])
        )
        max_energy = max(
            abs(value)
            for row in residuals
            for value in (*row["tank_energy_balance_w"], row["total_energy_balance_w"])
        )
        return (
            {
                "name": "mass_balance",
                "passed": max_mass <= 1e-12,
                "detail": f"max residual={max_mass:.3e} m3/s across {len(samples)} points",
            },
            {
                "name": "energy_balance",
                "passed": max_energy <= 1e-7,
                "detail": f"max residual={max_energy:.3e} W across {len(samples)} points",
            },
        )

    energy_scored = True

    def energy_kw(self, u, backend="numeric", ca=None):
        if backend == "numeric":
            values, ops = self.action_vector(u), _NUMERIC_OPS
        elif backend == "casadi":
            if ca is None:
                raise ValueError("backend='casadi' requires the casadi module as ca=...")
            values, ops = u, _casadi_ops(ca)
        else:
            raise ValueError(f"unknown dynamics backend: {backend!r}")
        effective = self._effective_action(values, ops)
        return (
            effective[0] ** 3 * self.p["pump_power_max"]
            + effective[3] * self.p["heater_power"]
        ) / 1000.0

    def action_energy_kw(self, act, x=None, env=None):
        u = self._effective_action(self.action_vector(act), _NUMERIC_OPS)
        if x is None:
            return float(
                (
                    u[0] ** 3 * self.p["pump_power_max"]
                    + u[3] * self.p["heater_power"]
                )
                / 1000.0
            )
        context = self._resolved_env(env)
        levels = [float(x[0]), float(x[2]), float(x[4])]
        temperatures = [float(x[1]), float(x[3]), float(x[5])]
        _, _, _, _, pump_enabled = self._flow_terms(
            levels, u, context, _NUMERIC_OPS
        )
        _, heater_power, _, _, _ = self._heater_terms(
            levels, temperatures, u, context, _NUMERIC_OPS
        )
        pump_power = u[0] ** 3 * self.p["pump_power_max"] * pump_enabled
        return float((pump_power + heater_power) / 1000.0)

    def ideal_energy_kw(self, x, y_sp, env, act):
        target = [float(value) for value in y_sp]
        if len(target) != 6:
            raise ValueError("cascade_recirculating setpoint must contain 6 values")
        requirements = self.nominal_steady_state(
            tank_1_temperature=target[3],
            levels=target[:3],
            env=env,
        )
        return float(requirements["ideal_energy_kw"])

    def process_info(self, x, levels, temps, env, action=None):
        context = self._resolved_env(env)
        u = self._effective_action(
            self.action_vector(self.default_action() if action is None else action),
            _NUMERIC_OPS,
        )
        physical_levels = [float(x[0]), float(x[2]), float(x[4])]
        temperatures = [float(x[1]), float(x[3]), float(x[5])]
        pump_flow, q12, q23, overflow_flows, pump_enabled = self._flow_terms(
            physical_levels, u, context, _NUMERIC_OPS
        )
        (
            heat_to_liquid,
            heater_power,
            heater_enabled,
            heater_level_ok,
            heater_temperature_ok,
        ) = self._heater_terms(
            physical_levels, temperatures, u, context, _NUMERIC_OPS
        )
        pump_low_level_ok, pump_high_level_ok, _ = self._pump_interlock_terms(
            physical_levels, _NUMERIC_OPS
        )
        hardware_interlocks = []
        if not bool(pump_low_level_ok):
            hardware_interlocks.append("P101_tank_3_low_level")
        if not bool(pump_high_level_ok):
            hardware_interlocks.append("L3_P101_high_level")
        if not bool(heater_level_ok):
            hardware_interlocks.append("L2_H1_dry_fire")
        if not bool(heater_temperature_ok):
            hardware_interlocks.append("L4_H1_over_temperature")
        low_switches = {
            f"LSL{tank}01": physical_levels[index]
            < self.p["low_level_trip"][index]
            for index, tank in enumerate((1, 2, 3))
        }
        high_switches = {
            f"LSH{tank}01": physical_levels[index]
            >= self.p["high_level_trip"][index]
            for index, tank in enumerate((1, 2, 3))
        }
        passive_safety_events = [
            f"tank_{i + 1}_passive_overflow"
            for i, flow in enumerate(overflow_flows)
            if float(flow) > 0.0
        ]
        return {
            "ambient_temperature_degC": float(context["t_amb"]),
            "pump_flow_factor": float(context["pump_flow_factor"]),
            "heater_efficiency": float(context["heater_efficiency"]),
            "heat_loss_factor": float(context["heat_loss_factor"]),
            "circulation_flow_m3s": float(pump_flow),
            "V12_flow_m3s": float(q12),
            "V23_flow_m3s": float(q23),
            "FT12_flow_m3s": float(q12),
            "FT23_flow_m3s": float(q23),
            "tank_1_overflow_return_m3s": float(overflow_flows[0]),
            "tank_2_overflow_return_m3s": float(overflow_flows[1]),
            "total_overflow_return_m3s": float(sum(overflow_flows)),
            "P101_enabled": bool(pump_enabled),
            "P101_speed_fraction": float(u[0]),
            "P101_electric_power_w": float(
                u[0] ** 3 * self.p["pump_power_max"] * pump_enabled
            ),
            "H1_enabled": bool(heater_enabled),
            "H1_electric_power_w": float(heater_power),
            "H1_to_liquid_power_w": float(heat_to_liquid),
            "hardware_interlocks_active": hardware_interlocks,
            "digital_inputs": {**low_switches, **high_switches},
            "passive_safety_events": passive_safety_events,
            "protection_events": [*hardware_interlocks, *passive_safety_events],
            "closed_loop_nominal": not hardware_interlocks and not passive_safety_events,
        }

    def process_constraint_info(self, x, levels, temps, env):
        physical_levels = [float(x[0]), float(x[2]), float(x[4])]
        temperatures = [float(x[1]), float(x[3]), float(x[5])]
        return {
            "level_negative": max(
                (max(0.0, -value) for value in physical_levels), default=0.0
            ),
            "level_overflow": max(
                (
                    max(0.0, physical_levels[i] - self.height_max[i])
                    for i in range(3)
                ),
                default=0.0,
            ),
            "passive_overflow_head": max(
                (
                    max(0.0, physical_levels[i] - self.p["overflow_level"][i])
                    for i in range(2)
                ),
                default=0.0,
            ),
            "high_level_alarm": max(
                (
                    max(0.0, physical_levels[i] - self.p["high_level_trip"][i])
                    for i in range(3)
                ),
                default=0.0,
            ),
            "P101_high_level_trip": max(
                (
                    max(0.0, physical_levels[i] - self.p["high_level_trip"][i])
                    for i in range(2)
                ),
                default=0.0,
            ),
            "temperature_hard_limit": max(
                (
                    max(0.0, value - self.p["temperature_hard_limit"])
                    for value in temperatures
                ),
                default=0.0,
            ),
        }

    def hard_termination_reasons(self, x, levels, temps, env):
        physical_levels = [float(x[0]), float(x[2]), float(x[4])]
        temperatures = [float(x[1]), float(x[3]), float(x[5])]
        reasons = []
        if any(value < 0.0 for value in physical_levels):
            reasons.append("negative_level")
        for i in range(3):
            if physical_levels[i] > self.height_max[i]:
                reasons.append(f"tank_{i + 1}_hard_overflow")
        if any(
            value >= self.p["temperature_hard_limit"]
            for value in temperatures
        ):
            reasons.append("temperature_hard_limit")
        return tuple(reasons)

    def sample_disturbance(self, event, current, rng):
        if event == "pump_capacity_shift":
            return float(max(0.6, min(1.3, float(current) + rng.uniform(-0.3, 0.3))))
        if event == "heater_efficiency_shift":
            return float(max(0.55, min(1.0, float(current) + rng.uniform(-0.3, 0.1))))
        if event == "heat_loss_shift":
            return float(max(0.5, min(2.4, float(current) + rng.uniform(-0.4, 1.0))))
        return super().sample_disturbance(event, current, rng)


# Public topology name used by the unified compiler. The implementation owns
# the shared recirculating mass/energy balance used by the laboratory subclass.
RecirculatingTopology = RecirculatingCascadeModel

__all__ = ["RecirculatingTopology"]
