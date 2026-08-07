import math

from aiogym.core.backends import _NUMERIC_OPS, _casadi_ops
from aiogym.core.model import RHO_CP

from .physics import ThreeTankPhysicsKernel


class OpenCascadeTopology(ThreeTankPhysicsKernel):
    scenario = "three_tank"
    display_name = "Heated-tank cascade"
    summary = "Three interlinked heated tanks with level and temperature dynamics."
    state_names = ("h0", "T0", "h1", "T1", "h2", "T2")
    state_units = {"h0": "m", "h1": "m", "h2": "m", "T0": "degC", "T1": "degC", "T2": "degC"}
    action_names = ("feed_pump", "outlet_valve_0", "outlet_valve_1", "outlet_valve_2", "heater_0", "heater_1", "heater_2")
    action_kinds = {
        "feed_pump": "pump",
        "outlet_valve_0": "valve", "outlet_valve_1": "valve", "outlet_valve_2": "valve",
        "heater_0": "heater", "heater_1": "heater", "heater_2": "heater",
    }
    output_names = ("tank_0_level", "tank_1_level", "tank_2_level", "tank_0_temperature", "tank_1_temperature", "tank_2_temperature")
    output_units = {"tank_0_level": "m", "tank_1_level": "m", "tank_2_level": "m", "tank_0_temperature": "degC", "tank_1_temperature": "degC", "tank_2_temperature": "degC"}
    default_y_sp = (0.45, 0.45, 0.45, 35.0, 50.0, 65.0)
    plant_regime = {"ua_loss": (0.4, 2.6), "heater_max": (0.6, 1.15), "pump_flow_max": (0.7, 1.3), "cv_out": (0.7, 1.4)}
    economic_config = {
        "value_unit": "normalized_value",
        "product_value_per_m3": 100000.0,
        "electricity_price_per_kwh": 0.7,
    }
    supervisory_layout = (("y_sp", 3, 25, 80), ("y_sp", 4, 30, 82), ("y_sp", 5, 35, 85))
    param_units = {"area": "m2", "height_max": "m", "cv_out": "m2.5/s", "ua_loss": "W/K", "heater_max": "W", "pump_flow_max": "m3/s", "pump_power_max": "W", "t_cold": "degC", "t_amb": "degC", "h_floor": "m", "heater_min_level": "m", "temperature_trip": "degC", "temperature_hard_limit": "degC"}
    param_bounds = {"area": (0.01, 2.0), "height_max": (0.1, 5.0), "cv_out": (0.0, 0.02), "ua_loss": (0.0, 1000.0), "heater_max": (0.0, 500000.0), "pump_flow_max": (0.0, 0.02), "pump_power_max": (0.0, 10000.0), "t_cold": (0.0, 40.0), "t_amb": (0.0, 45.0), "h_floor": (1e-6, 0.1), "heater_min_level": (0.0, 0.8), "temperature_trip": (40.0, 120.0), "temperature_hard_limit": (92.0, 150.0)}
    input_disturbances = ThreeTankPhysicsKernel.input_disturbances + (
        {"name": "pump_flow_factor", "event": "pump_capacity_shift", "unit": "fraction", "bounds": (0.4, 1.4), "default": 1.0, "description": "feed-pump flow capacity multiplier"},
        {"name": "heater_efficiency", "event": "heater_efficiency_shift", "unit": "fraction", "bounds": (0.4, 1.0), "default": 1.0, "description": "fraction of heater electrical power transferred to the liquid"},
        {"name": "heat_loss_factor", "event": "heat_loss_shift", "unit": "fraction", "bounds": (0.3, 3.0), "default": 1.0, "description": "ambient heat-loss multiplier"},
    )
    def __init__(self):
        self.p = dict(area=0.15, height_max=0.80, cv_out=0.0026, ua_loss=40.0,
                      heater_max=90000.0, pump_flow_max=0.0016, pump_power_max=1500.0,
                      t_cold=15.0, t_amb=20.0, h_floor=1e-3,
                      heater_min_level=0.05, temperature_trip=92.0,
                      temperature_hard_limit=120.0)
        self._environment_bounds = {
            row["name"]: tuple(row["bounds"])
            for row in self.input_disturbances
            if isinstance(row.get("bounds"), (tuple, list)) and len(row["bounds"]) == 2
        }
        self.operation = {
            "product_flow_sp": 0.0,
            "min_product_flow": 0.0,
        }

    def configure_operation(self, operation):
        """Configure throughput economics without mutating physical parameters."""

        values = dict(operation or {})
        unknown = set(values) - {"product_flow_sp", "min_product_flow"}
        if unknown:
            raise ValueError(
                f"unknown cascade operation fields: {', '.join(sorted(unknown))}"
            )
        product_flow_sp = self._finite_nonnegative(
            "product_flow_sp", values.get("product_flow_sp", 0.0)
        )
        min_product_flow = self._finite_nonnegative(
            "min_product_flow",
            values.get("min_product_flow", product_flow_sp),
        )
        if min_product_flow > product_flow_sp:
            raise ValueError("min_product_flow must not exceed product_flow_sp")
        self.operation = {
            "product_flow_sp": product_flow_sp,
            "min_product_flow": min_product_flow,
        }
        return self

    @staticmethod
    def _finite_nonnegative(name, value):
        if isinstance(value, bool):
            raise TypeError(f"cascade {name} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"cascade {name} must be numeric") from exc
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"cascade {name} must be finite and non-negative")
        return number

    @property
    def height_max(self):
        return [float(self.p["height_max"])] * 3

    @property
    def setpoint_bounds(self):
        height_max = float(self.p["height_max"])
        return {
            "tank_0_level": (0.0, height_max),
            "tank_1_level": (0.0, height_max),
            "tank_2_level": (0.0, height_max),
            "tank_0_temperature": (25.0, 80.0),
            "tank_1_temperature": (30.0, 82.0),
            "tank_2_temperature": (35.0, 85.0),
        }

    @property
    def safety_constraints(self):
        return (
            {
                "name": "level_bounds",
                "states": ("h0", "h1", "h2"),
                "bounds": (0.0, float(self.p["height_max"])),
            },
            {
                "name": "heater_min_level",
                "states": ("h0", "h1", "h2"),
                "bounds": (float(self.p["heater_min_level"]), None),
            },
            {
                "name": "temperature_trip",
                "states": ("T0", "T1", "T2"),
                "bounds": (None, float(self.p["temperature_trip"])),
            },
            {
                "name": "temperature_hard_limit",
                "states": ("T0", "T1", "T2"),
                "bounds": (None, float(self.p["temperature_hard_limit"])),
            },
        )

    @staticmethod
    def _hard_gate(condition, ops):
        return ops.if_else(condition, 1.0, 0.0)

    def _flow_terms(self, h, u, env, ops):
        p = self.p
        pump_flow = u[0] * p["pump_flow_max"] * env["pump_flow_factor"]
        valve_flows = [
            p["cv_out"] * u[1 + i] * ops.sqrt(ops.max(h[i], 0.0))
            for i in range(3)
        ]
        extra_enabled = self._hard_gate(h[2] > 0.0, ops)
        extra_outflow = env["extra_outflow"] * extra_enabled
        total_outflows = [valve_flows[0], valve_flows[1], valve_flows[2] + extra_outflow]
        return pump_flow, valve_flows, extra_outflow, total_outflows

    def _heater_terms(self, h, temperatures, u, env, ops):
        heat_to_liquid = []
        electric_power = []
        interlocked = []
        low_level_active = []
        temperature_trip_active = []
        for i in range(3):
            level_ok = self._hard_gate(h[i] >= self.p["heater_min_level"], ops)
            temperature_ok = self._hard_gate(
                temperatures[i] < self.p["temperature_trip"], ops
            )
            enabled = level_ok * temperature_ok
            requested_electric_power = u[4 + i] * self.p["heater_max"]
            effective_electric_power = requested_electric_power * enabled
            electric_power.append(effective_electric_power)
            heat_to_liquid.append(effective_electric_power * env["heater_efficiency"])
            interlocked.append(1.0 - enabled)
            low_level_active.append(1.0 - level_ok)
            temperature_trip_active.append(1.0 - temperature_ok)
        return (
            heat_to_liquid,
            electric_power,
            interlocked,
            low_level_active,
            temperature_trip_active,
        )

    def sample_disturbance(self, event, current, rng):
        if event == "pump_capacity_shift":
            return float(max(0.6, min(1.3, float(current) + rng.uniform(-0.30, 0.30))))
        if event == "heater_efficiency_shift":
            return float(max(0.55, min(1.0, float(current) + rng.uniform(-0.35, 0.15))))
        if event == "heat_loss_shift":
            return float(max(0.5, min(2.4, float(current) + rng.uniform(-0.4, 1.2))))
        return super().sample_disturbance(event, current, rng)

    def process_info(self, x, levels, temps, env, action=None):
        env = self._resolved_env(env)
        u = self._effective_action(
            self.action_vector(self.default_action() if action is None else action),
            _NUMERIC_OPS,
        )
        h = [float(x[0]), float(x[2]), float(x[4])]
        temperatures = [float(x[1]), float(x[3]), float(x[5])]
        pump_flow, valve_flows, extra_outflow, total_outflows = self._flow_terms(
            h, u, env, _NUMERIC_OPS
        )
        (
            heat_to_liquid,
            electric_power,
            interlocked,
            low_level_active,
            temperature_trip_active,
        ) = self._heater_terms(h, temperatures, u, env, _NUMERIC_OPS)
        product_flow = float(total_outflows[2])
        min_product_flow = float(self.operation["min_product_flow"])
        product_flow_shortfall = max(0.0, min_product_flow - product_flow)
        return {
            "pump_flow_factor": env.get("pump_flow_factor", 1.0),
            "heater_efficiency": env.get("heater_efficiency", 1.0),
            "heat_loss_factor": env.get("heat_loss_factor", 1.0),
            "feed_flow_m3s": float(pump_flow),
            "interstage_flow_01_m3s": float(valve_flows[0]),
            "interstage_flow_12_m3s": float(valve_flows[1]),
            "product_flow_m3s": product_flow,
            "product_flow_sp_m3s": float(self.operation["product_flow_sp"]),
            "min_product_flow_m3s": min_product_flow,
            "product_flow_shortfall_m3s": product_flow_shortfall,
            "extra_outflow_m3s": float(extra_outflow),
            "heater_electric_power_w": [float(value) for value in electric_power],
            "heater_to_liquid_power_w": [float(value) for value in heat_to_liquid],
            "heater_interlocked": [bool(value) for value in interlocked],
            "temperature_trip_active": [bool(value) for value in temperature_trip_active],
            "low_level_interlock_active": [bool(value) for value in low_level_active],
        }

    def process_constraint_info(self, x, levels, temps, env):
        h = [float(x[0]), float(x[2]), float(x[4])]
        temperatures = [float(x[1]), float(x[3]), float(x[5])]
        return {
            "level_negative": max((max(0.0, -value) for value in h), default=0.0),
            "level_overflow": max(
                (max(0.0, value - self.p["height_max"]) for value in h),
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
        h = [float(x[0]), float(x[2]), float(x[4])]
        temperatures = [float(x[1]), float(x[3]), float(x[5])]
        reasons = []
        if any(value < 0.0 for value in h):
            reasons.append("negative_level")
        if any(value > self.p["height_max"] for value in h):
            reasons.append("overflow")
        if any(value >= self.p["temperature_hard_limit"] for value in temperatures):
            reasons.append("temperature_hard_limit")
        return tuple(reasons)

    def _dynamics(self, x, u, env, ops):
        env = self._resolved_env(env, ops)
        u = self._effective_action(u, ops)
        levels, temperatures = self._levels_temperatures(x)
        pump_flow, valve_flows, _, flows_out = self._flow_terms(
            levels, u, env, ops
        )
        heat_inputs, _, _, _, _ = self._heater_terms(
            levels, temperatures, u, env, ops
        )
        flows_in = [pump_flow, valve_flows[0], valve_flows[1]]
        inlet_temperatures = [
            env["t_cold"],
            temperatures[0],
            temperatures[1],
        ]
        mixing_terms = [
            flows_in[index]
            * (inlet_temperatures[index] - temperatures[index])
            for index in range(self.n)
        ]
        return self._assemble_dynamics(
            levels,
            temperatures,
            flows_in,
            flows_out,
            mixing_terms,
            heat_inputs,
            env,
            ops,
        )

    def initial_state(self):
        return [0.30, 20.0, 0.30, 20.0, 0.30, 20.0]

    # ---- KPI support ----
    energy_scored = True

    def energy_kw(self, u, backend="numeric", ca=None):
        if backend == "numeric":
            values = self.action_vector(u)
            ops = _NUMERIC_OPS
        elif backend == "casadi":
            if ca is None:
                raise ValueError("backend='casadi' requires the casadi module as ca=...")
            values = u
            ops = _casadi_ops(ca)
        else:
            raise ValueError(f"unknown dynamics backend: {backend!r}")
        u = self._effective_action(values, ops)
        input_energy = u[0] * self.p["pump_power_max"]
        thermal_energy = sum(u[4 + i] * self.p["heater_max"] for i in range(3))
        return (input_energy + thermal_energy) / 1000.0

    def action_energy_kw(self, act, x=None, env=None):
        """Return actual plant electrical power after heater interlocks."""

        u = self._effective_action(self.action_vector(act), _NUMERIC_OPS)
        pump_power = u[0] * self.p["pump_power_max"]
        if x is None:
            heater_power = [u[4 + i] * self.p["heater_max"] for i in range(3)]
        else:
            context = self._resolved_env(env)
            h = [float(x[0]), float(x[2]), float(x[4])]
            temperatures = [float(x[1]), float(x[3]), float(x[5])]
            _, heater_power, _, _, _ = self._heater_terms(
                h, temperatures, u, context, _NUMERIC_OPS
            )
        return float((pump_power + sum(heater_power)) / 1000.0)

    def steady_state_requirements(self, y_sp, env=None, product_flow_sp=None):
        """Return actuator and power requirements for a requested steady state.

        The requested product flow is a Case-level throughput target, not a
        physical model parameter.  Returned commands are the unconstrained
        requirements; ``feasible`` reports whether every requirement lies
        within the available normalized actuator range and safety interlocks.
        """

        target = [float(value) for value in y_sp]
        if len(target) != 6 or not all(math.isfinite(value) for value in target):
            raise ValueError("cascade steady-state setpoint must contain six finite values")
        levels = target[:3]
        temperatures = target[3:]
        context = self._resolved_env(env)
        flow = self._finite_nonnegative(
            "product_flow_sp",
            self.operation["product_flow_sp"]
            if product_flow_sp is None
            else product_flow_sp,
        )
        reasons = {name: [] for name in self.action_names}
        reasons["setpoint"] = []

        for i, level in enumerate(levels):
            if level < 0.0 or level > self.p["height_max"]:
                reasons["setpoint"].append(
                    f"tank_{i}_level is outside [0, {self.p['height_max']}]"
                )
        for i, temperature in enumerate(temperatures):
            if temperature < 0.0 or temperature >= self.p["temperature_hard_limit"]:
                reasons["setpoint"].append(
                    f"tank_{i}_temperature is outside [0, {self.p['temperature_hard_limit']})"
                )

        pump_capacity = self.p["pump_flow_max"] * context["pump_flow_factor"]
        pump_command = flow / pump_capacity if pump_capacity > 0.0 else 0.0
        if flow > 0.0 and pump_capacity <= 0.0:
            reasons["feed_pump"].append("feed-pump capacity is zero")
        elif pump_command > 1.0:
            reasons["feed_pump"].append("required flow exceeds feed-pump capacity")

        required_valve_flows = [flow, flow, flow - context["extra_outflow"]]
        valve_commands = []
        for i, (level, required_flow) in enumerate(zip(levels, required_valve_flows)):
            name = f"outlet_valve_{i}"
            if required_flow < 0.0:
                reasons[name].append("extra outflow exceeds the product-flow target")
                valve_commands.append(0.0)
                continue
            capacity = self.p["cv_out"] * math.sqrt(max(level, 0.0))
            command = required_flow / capacity if capacity > 0.0 else 0.0
            valve_commands.append(command)
            if required_flow > 0.0 and capacity <= 0.0:
                reasons[name].append("positive flow requires a positive liquid level")
            elif command > 1.0:
                reasons[name].append("required flow exceeds valve capacity")

        thermal_load_w = []
        heater_electric_power_w = []
        heater_commands = []
        for i, temperature in enumerate(temperatures):
            inlet_temperature = context["t_cold"] if i == 0 else temperatures[i - 1]
            load = (
                RHO_CP * flow * (temperature - inlet_temperature)
                + self.p["ua_loss"]
                * context["heat_loss_factor"]
                * (temperature - context["t_amb"])
            )
            thermal_load_w.append(load)
            electric_power = max(0.0, load) / context["heater_efficiency"]
            heater_electric_power_w.append(electric_power)
            command = (
                electric_power / self.p["heater_max"]
                if self.p["heater_max"] > 0.0
                else 0.0
            )
            heater_commands.append(command)
            name = f"heater_{i}"
            if load < 0.0:
                reasons[name].append("steady state requires cooling but no cooling actuator exists")
            if electric_power > 0.0 and self.p["heater_max"] <= 0.0:
                reasons[name].append("heater capacity is zero")
            elif command > 1.0:
                reasons[name].append("required electrical power exceeds heater capacity")
            if electric_power > 0.0 and levels[i] < self.p["heater_min_level"]:
                reasons[name].append("required heating is blocked by the low-level interlock")
            if electric_power > 0.0 and temperatures[i] >= self.p["temperature_trip"]:
                reasons[name].append("required heating is blocked by the temperature trip")

        pump_power_w = pump_command * self.p["pump_power_max"]
        ideal_energy_kw = (pump_power_w + sum(heater_electric_power_w)) / 1000.0
        infeasible_reasons = {
            name: tuple(messages) for name, messages in reasons.items() if messages
        }
        action = [pump_command, *valve_commands, *heater_commands]
        return {
            "feasible": not infeasible_reasons,
            "infeasible_reasons": infeasible_reasons,
            "product_flow_sp_m3s": flow,
            "feed_flow_m3s": flow,
            "pump_command": pump_command,
            "valve_commands": valve_commands,
            "heater_commands": heater_commands,
            "action": action,
            "thermal_load_w": thermal_load_w,
            "heater_electric_power_w": heater_electric_power_w,
            "pump_power_w": pump_power_w,
            "ideal_energy_kw": ideal_energy_kw,
        }

    def default_action(self):
        """Return the analytic steady input for the configured throughput."""

        requirements = self.steady_state_requirements(self.default_setpoint_vector())
        if requirements["feasible"]:
            return list(requirements["action"])
        return super().default_action()

    def tracking_steady_state_action(self, y_sp):
        """Return the nominal steady input for a feasible tracking target."""

        requirements = self.steady_state_requirements(y_sp)
        if not requirements["feasible"]:
            return None
        return list(requirements["action"])

    def is_setpoint_reachable(self, y_sp):
        if not super().is_setpoint_reachable(y_sp):
            return False
        try:
            return bool(self.steady_state_requirements(y_sp)["feasible"])
        except (TypeError, ValueError, KeyError):
            return False

    def _economic_product_flow(self, x, u, env, ops):
        context = self._resolved_env(env, ops)
        effective_action = self._effective_action(u, ops)
        levels = [x[0], x[2], x[4]]
        _, _, _, total_outflows = self._flow_terms(
            levels, effective_action, context, ops
        )
        return total_outflows[2]

    def production(self, x, act, env=None):
        return float(
            self._economic_product_flow(
                self.state_vector(x),
                self.action_vector(act),
                env or {},
                _NUMERIC_OPS,
            )
        )

    def economic_value(self, x, u, env=None, backend="numeric", ca=None):
        if backend == "numeric":
            return self._economic_product_flow(
                self.state_vector(x), self.action_vector(u), env or {}, _NUMERIC_OPS
            )
        if backend == "casadi":
            if ca is None:
                raise ValueError("backend='casadi' requires the casadi module as ca=...")
            return self._economic_product_flow(x, u, env or {}, _casadi_ops(ca))
        raise ValueError(f"unknown dynamics backend: {backend!r}")

    def product_flow_shortfall(self, production, backend="numeric", ca=None):
        target = float(self.operation["product_flow_sp"])
        if target <= 0.0:
            return 0.0
        minimum = float(self.operation["min_product_flow"])
        if backend == "numeric":
            return max(0.0, minimum - float(production)) / target
        if backend == "casadi":
            if ca is None:
                raise ValueError("backend='casadi' requires the casadi module as ca=...")
            return ca.fmax(0.0, minimum - production) / target
        raise ValueError(f"unknown backend: {backend!r}")

    def ideal_energy_kw(self, x, y_sp, env, act):
        return float(
            self.steady_state_requirements(
                y_sp,
                env,
                product_flow_sp=self.operation["product_flow_sp"],
            )["ideal_energy_kw"]
        )


class RecirculatingTopology(ThreeTankPhysicsKernel):
    scenario = "three_tank"
    display_name = "Recirculating heated-tank cascade"
    summary = (
        "Three non-identical tanks in a closed P101-Tank 1-V12-Tank 2-V23-Tank 3 "
        "loop with one 2 kW heater."
    )
    supported_goals = ("regulation",)
    benchmark_goals = supported_goals
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

        return self._assemble_dynamics(
            levels,
            temperatures,
            flows_in,
            flows_out,
            mixing_terms,
            heat_inputs,
            env,
            ops,
        )

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
            raise ValueError("three_tank setpoint must contain 6 values")
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


__all__ = ["OpenCascadeTopology", "RecirculatingTopology"]
