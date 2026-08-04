import math

from ..backends import _NUMERIC_OPS, _casadi_ops, _maxv
from ..core import RHO_CP, ProcessModelContract


class CascadeModel(ProcessModelContract):
    scenario = "cascade"
    display_name = "Heated-tank cascade"
    summary = "Three interlinked heated tanks with level and temperature dynamics."
    n = 3
    dt_micro = 0.02
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
        "temp_band": [(34, 44), (48, 58), (60, 72)],
        "level_band": [(0.32, 0.58)] * 3,
        "value": "production",
        "w_value": 100000.0,
        "w_energy": 0.7,
        "w_viol": 29.0,
        "w_product_shortfall": 100.0,
    }
    supervisory_layout = (("y_sp", 3, 25, 80), ("y_sp", 4, 30, 82), ("y_sp", 5, 35, 85))
    param_units = {"area": "m2", "height_max": "m", "cv_out": "m2.5/s", "ua_loss": "W/K", "heater_max": "W", "pump_flow_max": "m3/s", "pump_power_max": "W", "t_cold": "degC", "t_amb": "degC", "h_floor": "m", "heater_min_level": "m", "temperature_trip": "degC", "temperature_hard_limit": "degC"}
    param_bounds = {"area": (0.01, 2.0), "height_max": (0.1, 5.0), "cv_out": (0.0, 0.02), "ua_loss": (0.0, 1000.0), "heater_max": (0.0, 500000.0), "pump_flow_max": (0.0, 0.02), "pump_power_max": (0.0, 10000.0), "t_cold": (0.0, 40.0), "t_amb": (0.0, 45.0), "h_floor": (1e-6, 0.1), "heater_min_level": (0.0, 0.8), "temperature_trip": (40.0, 120.0), "temperature_hard_limit": (92.0, 150.0)}
    input_disturbances = ProcessModelContract.input_disturbances + (
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
    def state_bounds(self):
        height_max = float(self.p["height_max"])
        temperature_max = float(self.p["temperature_hard_limit"])
        return {
            "h0": (0.0, height_max),
            "h1": (0.0, height_max),
            "h2": (0.0, height_max),
            "T0": (0.0, temperature_max),
            "T1": (0.0, temperature_max),
            "T2": (0.0, temperature_max),
        }

    @property
    def output_bounds(self):
        height_max = float(self.p["height_max"])
        temperature_max = float(self.p["temperature_hard_limit"])
        return {
            "tank_0_level": (0.0, height_max),
            "tank_1_level": (0.0, height_max),
            "tank_2_level": (0.0, height_max),
            "tank_0_temperature": (0.0, temperature_max),
            "tank_1_temperature": (0.0, temperature_max),
            "tank_2_temperature": (0.0, temperature_max),
        }

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

    def pump_flow_factor(self, env=None):
        env = env or {}
        return env.get("pump_flow_factor", 1.0)

    def heater_efficiency(self, env=None):
        env = env or {}
        return env.get("heater_efficiency", 1.0)

    def heat_loss_factor(self, env=None):
        env = env or {}
        return env.get("heat_loss_factor", 1.0)

    def _resolved_env(self, env=None, ops=None):
        """Resolve and validate the six runtime inputs used by the dynamics.

        CasADi builds the prediction graph with symbolic disturbance values, so
        symbolic values cannot be checked for finiteness while the graph is
        being constructed. Numeric values are validated here and also pass
        through ``runtime_env``/``disturbance_vector`` before controller or
        environment use.
        """

        values = dict(env or {})
        resolved = {
            "t_cold": values.get("t_cold", self.p["t_cold"]),
            "t_amb": values.get("t_amb", self.p["t_amb"]),
            "extra_outflow": values.get("extra_outflow", 0.0),
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
                raise ValueError(f"cascade disturbance {name!r} must be a finite number") from exc
            if not math.isfinite(number):
                raise ValueError(f"cascade disturbance {name!r} must be a finite number")
            lower, upper = self._environment_bounds[name]
            if lower is not None and number < float(lower):
                raise ValueError(
                    f"cascade disturbance {name!r} must be within [{lower}, {upper}], got {number}"
                )
            if upper is not None and number > float(upper):
                raise ValueError(
                    f"cascade disturbance {name!r} must be within [{lower}, {upper}], got {number}"
                )
            clean[name] = number
        return clean

    def runtime_env(self, disturbance_values):
        return self._resolved_env(super().runtime_env(disturbance_values))

    def disturbance_vector(self, values=None):
        return super().disturbance_vector(self.runtime_env(values or {}))

    def _effective_action(self, u, ops):
        """Return plant-side bounded commands for numeric and CasADi dynamics."""

        if not bool(getattr(ops, "symbolic", False)):
            for value in u:
                if not math.isfinite(float(value)):
                    raise ValueError("cascade action values must be finite")
        return [ops.min(ops.max(u[i], 0.0), 1.0) for i in range(self.action_dim())]

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
        p = self.p
        env = self._resolved_env(env, ops)
        u = self._effective_action(u, ops)
        t_cold, t_amb = env["t_cold"], env["t_amb"]
        heat_loss_factor = env.get("heat_loss_factor", 1.0)
        h = [x[0], x[2], x[4]]
        T = [x[1], x[3], x[5]]
        qp, valve_flows, _, total_outflows = self._flow_terms(h, u, env, ops)
        pheat, _, _, _, _ = self._heater_terms(h, T, u, env, ops)
        dx = []
        for i in range(3):
            qin = qp if i == 0 else valve_flows[i - 1]
            tin = t_cold if i == 0 else T[i - 1]
            vol = p["area"] * ops.max(h[i], p["h_floor"])
            qloss = p["ua_loss"] * heat_loss_factor * (T[i] - t_amb)
            dx += [
                (qin - total_outflows[i]) / p["area"],
                qin * (tin - T[i]) / vol + (pheat[i] - qloss) / (RHO_CP * vol),
            ]
        return ops.vector(dx)

    def display_outputs(self, x, backend="numeric", ca=None):
        if backend == "casadi":
            return {"levels": [x[0], x[2], x[4]], "temps": [x[1], x[3], x[5]]}
        return {"levels": [_maxv(x[0], 0.0), _maxv(x[2], 0.0), _maxv(x[4], 0.0)], "temps": [x[1], x[3], x[5]]}

    def initial_state(self):
        return [0.30, 20.0, 0.30, 20.0, 0.30, 20.0]

    def controlled_output(self, x, backend="numeric", ca=None):
        # Control, KPI, and constraint consumers receive the same raw physical
        # state for every backend.  Any cosmetic clipping belongs exclusively
        # to ``display_outputs``.
        return [x[0], x[2], x[4], x[1], x[3], x[5]]

    def integral_observation_limits(self):
        return [8.0, 8.0, 8.0, 300.0, 300.0, 300.0]

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

    def mpc_init(self):
        return self.default_action()

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
        if self.operation["product_flow_sp"] <= 0.0:
            return 0.0
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
