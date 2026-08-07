"""PlantConfig-v2 equipment compiler for the laboratory three-tank rig."""
from __future__ import annotations

import math

from aiogym.core.backends import _NUMERIC_OPS, _casadi_ops
from aiogym.core.model import RHO_CP
from aiogym.core.specs import OperatingCondition, ResolvedPlant

from .topologies import RecirculatingTopology


class ThreeTankDesignModel(RecirculatingTopology):
    """Three-tank recirculating design with stable H1/H2/H3 actuator slots."""

    scenario = "three_tank"
    display_name = "Parameterised recirculating three-tank design"
    summary = (
        "Three-tank P101-V12-V23 recirculating loop with independently optional "
        "H1, H2, and H3 heater slots."
    )
    action_names = (
        "pump_P101",
        "valve_V12",
        "valve_V23",
        "heater_H1",
        "heater_H2",
        "heater_H3",
    )
    action_kinds = {
        "pump_P101": "pump",
        "valve_V12": "valve",
        "valve_V23": "valve",
        "heater_H1": "heater",
        "heater_H2": "heater",
        "heater_H3": "heater",
    }
    dt_micro = 0.1

    def __init__(self, plant: ResolvedPlant):
        super().__init__()
        if not isinstance(plant, ResolvedPlant):
            raise TypeError("ThreeTankDesignModel requires a resolved PlantConfig v2")
        declaration = plant.config.plant
        if declaration.get("topology") != "recirculating_loop":
            raise ValueError("laboratory equipment requires recirculating_loop topology")
        self.plant = plant
        self.plant_id = plant.id
        self.plant_hash = plant.plant_hash
        tanks = [dict(row) for row in declaration["tanks"]]
        pump = dict(declaration["pump"])
        hydraulics = dict(declaration["hydraulics"])
        requirements = dict(plant.config.study.get("requirements", {}))
        requirements.update(dict(declaration.get("safety", {})))
        self._circulation_flow_m3s = float(
            hydraulics["nominal_circulation_flow_m3s"]
        )
        self.requirements = requirements
        self.uncertainty = dict(plant.config.study.get("uncertainty", {}))
        self.references = list(plant.config.references)
        powers = [0.0, 0.0, 0.0]
        efficiencies = [1.0, 1.0, 1.0]
        for heater in declaration["heaters"]:
            index = heater["tank"] - 1
            powers[index] = heater["power_w"]
            efficiencies[index] = heater["efficiency"]
        self.heater_mask = tuple(power > 0.0 for power in powers)
        self.heater_efficiencies = tuple(efficiencies)
        self.p.update(
            {
                "area": [tank["area_m2"] for tank in tanks],
                "height_max": [tank["height_m"] for tank in tanks],
                "level_sensor_range": [tank["level_sensor_range_m"] for tank in tanks],
                "ua_loss": [tank["ua_w_per_k"] for tank in tanks],
                "low_level_trip": [tank["low_level_trip_m"] for tank in tanks],
                "high_level_trip": [tank["high_level_trip_m"] for tank in tanks],
                "overflow_level": [
                    tanks[0]["overflow_level_m"],
                    tanks[1]["overflow_level_m"],
                ],
                "heater_power": powers,
                "pump_flow_max": pump["max_flow_m3s"],
                "pump_power_max": pump["motor_power_w"],
                "pump_static_head": pump["static_head_m"],
                "pump_shutoff_head": pump["shutoff_head_m"],
                "cv_interstage": list(hydraulics["cv_interstage"]),
                "gravity_drop": list(hydraulics["gravity_drop_m"]),
                "cv_overflow": list(hydraulics["cv_overflow"]),
                "overflow_head_floor": hydraulics["overflow_head_floor_m"],
                "t_amb": 20.0,
                "temperature_trip": requirements["temperature_trip_degC"],
                "temperature_hard_limit": requirements[
                    "temperature_hard_limit_degC"
                ],
            }
        )
        # These neutral values are replaced by bind_condition() before an
        # environment is exposed. Construction itself compiles only equipment.
        self.operation = {
            "circulation_flow_m3s": self._circulation_flow_m3s,
            "target_levels_m": [0.1, 0.1, 0.1],
            "target_temperatures_degC": [20.0, 20.0, 20.0],
            "initial_levels_m": [0.1, 0.1, 0.1],
            "initial_temperatures_degC": [20.0, 20.0, 20.0],
            "ambient_temperature_degC": 20.0,
            "control_dt_s": 1.0,
            "duration_s": 1.0,
        }
        self._bound_condition = None
        self._apply_operation(self.operation)

    def __deepcopy__(self, memo):
        del memo
        copied = type(self)(self.plant)
        if self._bound_condition is not None:
            copied.bind_condition(self._bound_condition)
        return copied

    def study_context(self, condition: OperatingCondition):
        """Build engineering inputs from one explicitly resolved condition."""

        available_duration = float(condition.control_dt * condition.horizon)
        requirements = dict(self.requirements)
        maximum_heatup = float(
            requirements.get("maximum_heatup_time_s", available_duration)
        )
        requirements["maximum_heatup_time_s"] = maximum_heatup
        requirements["available_duration_s"] = available_duration
        requirements["assessment_horizon_sufficient"] = (
            maximum_heatup <= available_duration
        )
        operation = {
            "circulation_flow_m3s": self._circulation_flow_m3s,
            "target_levels_m": list(condition.reference[:3]),
            "target_temperatures_degC": list(condition.reference[3:]),
            "initial_levels_m": list(condition.initial_state[0::2]),
            "initial_temperatures_degC": list(condition.initial_state[1::2]),
            "ambient_temperature_degC": float(
                condition.disturbances.get("t_amb", self.p["t_amb"])
            ),
            "control_dt_s": float(condition.control_dt),
            "duration_s": available_duration,
        }

        return {
            "operation": operation,
            "requirements": requirements,
            "uncertainty": dict(self.uncertainty),
            "references": list(self.references),
        }

    def bind_condition(self, condition: OperatingCondition):
        context = self.study_context(condition)
        self._bound_condition = condition
        self.operation = dict(context["operation"])
        self.p["t_amb"] = self.operation["ambient_temperature_degC"]
        self._apply_operation(self.operation)

    def _apply_operation(self, operation):
        self._initial_state = _interleave(
            operation["initial_levels_m"],
            operation["initial_temperatures_degC"],
        )
        self._target_levels = list(operation["target_levels_m"])
        self._target_temperatures = list(operation["target_temperatures_degC"])

    @property
    def safety_constraints(self):
        constraints = list(super().safety_constraints)
        constraints = [
            row
            for row in constraints
            if not row["name"].startswith("H1_")
        ]
        for index, installed in enumerate(self.heater_mask):
            if not installed:
                continue
            constraints.extend(
                (
                    {
                        "name": f"H{index + 1}_low_level_interlock",
                        "states": (f"h{index + 1}",),
                        "bounds": (float(self.p["low_level_trip"][index]), None),
                    },
                    {
                        "name": f"H{index + 1}_temperature_trip",
                        "states": (f"T{index + 1}",),
                        "bounds": (None, float(self.p["temperature_trip"])),
                    },
                )
            )
        return tuple(constraints)

    def initial_state(self):
        return list(self._initial_state)

    def default_setpoint_vector(self):
        return [*self._target_levels, *self._target_temperatures]

    def nominal_steady_state(
        self,
        *,
        circulation_flow=None,
        target_temperatures=None,
        levels=None,
        env=None,
        tank_1_temperature=None,
    ):
        """Solve the algebraic hydraulic and per-tank thermal requirements."""

        operation = self.operation
        flow = float(
            operation["circulation_flow_m3s"]
            if circulation_flow is None
            else circulation_flow
        )
        h = [
            float(value)
            for value in (
                operation["target_levels_m"] if levels is None else levels
            )
        ]
        temperatures = [
            float(value)
            for value in (
                operation["target_temperatures_degC"]
                if target_temperatures is None
                else target_temperatures
            )
        ]
        if tank_1_temperature is not None:
            temperatures[0] = float(tank_1_temperature)
        if not math.isfinite(flow) or flow <= 0.0:
            raise ValueError("circulation_flow must be finite and positive")
        if len(h) != 3 or len(temperatures) != 3:
            raise ValueError("levels and target_temperatures must contain three values")
        if any(not math.isfinite(value) or value <= 0.0 for value in h):
            raise ValueError("levels must contain three finite positive values")
        if any(not math.isfinite(value) for value in temperatures):
            raise ValueError("target temperatures must be finite")

        context = self._resolved_env(env)
        ambient = context["t_amb"]
        loss_factor = context["heat_loss_factor"]
        inlet_temperatures = [temperatures[2], temperatures[0], temperatures[1]]
        liquid_heat = [
            RHO_CP * flow * (temperatures[i] - inlet_temperatures[i])
            + self.p["ua_loss"][i] * loss_factor * (temperatures[i] - ambient)
            for i in range(3)
        ]
        electric_heat = []
        heater_commands = []
        reasons = []
        for index, required_heat in enumerate(liquid_heat):
            if abs(required_heat) <= 1e-7:
                required_heat = 0.0
                liquid_heat[index] = 0.0
            installed_efficiency = (
                self.heater_efficiencies[index] * context["heater_efficiency"]
            )
            if required_heat < -1e-7:
                reasons.append(
                    f"Tank {index + 1} requires {abs(required_heat):.3f} W of cooling"
                )
            required_electric = (
                max(required_heat, 0.0) / installed_efficiency
                if installed_efficiency > 0.0
                else math.inf
            )
            capacity = self.p["heater_power"][index]
            command = (
                required_electric / capacity
                if capacity > 0.0
                else (0.0 if required_electric <= 1e-12 else math.inf)
            )
            electric_heat.append(required_electric)
            heater_commands.append(command)

        pump_capacity = self.p["pump_flow_max"] * context["pump_flow_factor"]
        static_head = float(self.p["pump_static_head"])
        shutoff_head = float(self.p["pump_shutoff_head"])
        pump_command = math.inf
        if pump_capacity > 0.0 and shutoff_head > static_head:
            pump_command = math.sqrt(
                (
                    static_head
                    + (shutoff_head - static_head) * (flow / pump_capacity) ** 2
                )
                / shutoff_head
            )
        valve_commands = [
            flow
            / (
                self.p["cv_interstage"][i]
                * math.sqrt(h[i] + self.p["gravity_drop"][i])
            )
            for i in range(2)
        ]
        action = [pump_command, *valve_commands, *heater_commands]
        for label, command in zip(self.action_names, action):
            if not math.isfinite(command) or command < 0.0 or command > 1.0:
                reasons.append(f"{label} command is outside [0, 1]")
        if h[2] < self.p["low_level_trip"][2]:
            reasons.append("P101 is blocked by the Tank 3 low-level interlock")
        if any(h[i] >= self.p["high_level_trip"][i] for i in range(2)):
            reasons.append("P101 is blocked by a high-level interlock")
        for index, installed in enumerate(self.heater_mask):
            if not installed:
                continue
            if h[index] < self.p["low_level_trip"][index]:
                reasons.append(f"H{index + 1} is blocked by its low-level interlock")
            if temperatures[index] >= self.p["temperature_trip"]:
                reasons.append(f"H{index + 1} is blocked by the temperature trip")
        if any(h[i] > self.p["overflow_level"][i] for i in range(2)):
            reasons.append("requested levels activate passive overflow")

        state = _interleave(h, temperatures)
        pump_power = pump_command**3 * self.p["pump_power_max"]
        return {
            "feasible": not reasons,
            "infeasible_reasons": tuple(dict.fromkeys(reasons)),
            "circulation_flow_m3s": flow,
            "state": state,
            "y_sp": [*h, *temperatures],
            "action": action,
            "heater_to_liquid_power_w": liquid_heat,
            "heater_electric_power_w": electric_heat,
            "P101_electric_power_w": pump_power,
            "ideal_energy_kw": (pump_power + sum(electric_heat)) / 1000.0,
        }

    def default_action(self):
        return [_finite_clip01(value) for value in self.nominal_steady_state()["action"]]

    def tracking_steady_state_action(self, y_sp):
        requested = [float(value) for value in y_sp]
        if len(requested) != 6:
            return None
        equilibrium = self.nominal_steady_state(
            levels=requested[:3],
            target_temperatures=requested[3:],
        )
        return list(equilibrium["action"]) if equilibrium["feasible"] else None

    def _heater_vectors(self, levels, temperatures, u, env, ops):
        heat_to_liquid = []
        electric_power = []
        enabled = []
        level_ok = []
        temperature_ok = []
        for index in range(3):
            level_gate = self._gate(
                levels[index] >= self.p["low_level_trip"][index], ops
            )
            temperature_gate = self._gate(
                temperatures[index] < self.p["temperature_trip"], ops
            )
            installed = 1.0 if self.heater_mask[index] else 0.0
            gate = level_gate * temperature_gate * installed
            power = u[3 + index] * self.p["heater_power"][index] * gate
            electric_power.append(power)
            heat_to_liquid.append(
                power * self.heater_efficiencies[index] * env["heater_efficiency"]
            )
            enabled.append(gate)
            level_ok.append(level_gate)
            temperature_ok.append(temperature_gate)
        return heat_to_liquid, electric_power, enabled, level_ok, temperature_ok

    def _dynamics(self, x, u, env, ops):
        env = self._resolved_env(env, ops)
        u = self._effective_action(u, ops)
        levels = [x[0], x[2], x[4]]
        temperatures = [x[1], x[3], x[5]]
        pump_flow, q12, q23, overflow_flows, _ = self._flow_terms(
            levels, u, env, ops
        )
        overflow_1, overflow_2 = overflow_flows
        heat_inputs, _, _, _, _ = self._heater_vectors(
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
        heat_inputs, _, _, _, _ = self._heater_vectors(
            levels, temperatures, action, context, _NUMERIC_OPS
        )
        flows_in = [pump_flow, q12, q23 + overflow_1 + overflow_2]
        flows_out = [q12 + overflow_1, q23 + overflow_2, pump_flow]
        inlet_enthalpy_flows = [
            pump_flow * temperatures[2],
            q12 * temperatures[0],
            q23 * temperatures[1]
            + overflow_1 * temperatures[0]
            + overflow_2 * temperatures[1],
        ]
        mass_residuals = []
        energy_residuals = []
        stored_energy_rates = []
        external_energy_rates = []
        for index in range(3):
            area = self.p["area"][index]
            dh = dx[2 * index]
            dtemperature = dx[2 * index + 1]
            mass_residuals.append(area * dh - (flows_in[index] - flows_out[index]))
            stored = RHO_CP * area * (
                levels[index] * dtemperature + temperatures[index] * dh
            )
            heat_loss = (
                self.p["ua_loss"][index]
                * context["heat_loss_factor"]
                * (temperatures[index] - context["t_amb"])
            )
            external = (
                RHO_CP
                * (
                    inlet_enthalpy_flows[index]
                    - flows_out[index] * temperatures[index]
                )
                + heat_inputs[index]
                - heat_loss
            )
            stored_energy_rates.append(stored)
            external_energy_rates.append(external)
            energy_residuals.append(stored - external)
        return {
            "tank_mass_balance_m3s": mass_residuals,
            "total_mass_balance_m3s": sum(
                self.p["area"][index] * dx[2 * index] for index in range(3)
            ),
            "tank_energy_balance_w": energy_residuals,
            "total_energy_balance_w": sum(stored_energy_rates)
            - sum(external_energy_rates),
        }

    def physical_validation_checks(self):
        levels = self._target_levels
        temperatures = self._target_temperatures
        residuals = self.balance_residuals(
            _interleave(levels, temperatures),
            self.default_action(),
            self.disturbance_defaults(),
        )
        max_mass = max(
            abs(value)
            for value in (
                *residuals["tank_mass_balance_m3s"],
                residuals["total_mass_balance_m3s"],
            )
        )
        max_energy = max(
            abs(value)
            for value in (
                *residuals["tank_energy_balance_w"],
                residuals["total_energy_balance_w"],
            )
        )
        return (
            {
                "name": "mass_balance",
                "passed": max_mass <= 1e-12,
                "detail": f"max residual={max_mass:.3e} m3/s",
            },
            {
                "name": "energy_balance",
                "passed": max_energy <= 1e-7,
                "detail": f"max residual={max_energy:.3e} W",
            },
        )

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
        energy = (
            effective[0] ** 3 * self.p["pump_power_max"]
            + sum(
                effective[3 + index] * self.p["heater_power"][index]
                for index in range(3)
            )
        ) / 1000.0
        return float(energy) if backend == "numeric" else energy

    def action_energy_kw(self, act, x=None, env=None):
        action = self._effective_action(self.action_vector(act), _NUMERIC_OPS)
        if x is None:
            return self.energy_kw(action)
        context = self._resolved_env(env)
        levels = [float(x[0]), float(x[2]), float(x[4])]
        temperatures = [float(x[1]), float(x[3]), float(x[5])]
        _, _, _, _, pump_enabled = self._flow_terms(
            levels, action, context, _NUMERIC_OPS
        )
        _, heater_power, _, _, _ = self._heater_vectors(
            levels, temperatures, action, context, _NUMERIC_OPS
        )
        pump_power = action[0] ** 3 * self.p["pump_power_max"] * pump_enabled
        return float((pump_power + sum(heater_power)) / 1000.0)

    def ideal_energy_kw(self, x, y_sp, env, act):
        target = [float(value) for value in y_sp]
        if len(target) != 6:
            raise ValueError("design setpoint must contain 6 values")
        return float(
            self.nominal_steady_state(
                levels=target[:3],
                target_temperatures=target[3:],
                env=env,
            )["ideal_energy_kw"]
        )

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
        ) = self._heater_vectors(
            physical_levels, temperatures, u, context, _NUMERIC_OPS
        )
        pump_low_ok, pump_high_ok, _ = self._pump_interlock_terms(
            physical_levels, _NUMERIC_OPS
        )
        hardware_interlocks = []
        if not bool(pump_low_ok):
            hardware_interlocks.append("P101_tank_3_low_level")
        if not bool(pump_high_ok):
            hardware_interlocks.append("L3_P101_high_level")
        for index, installed in enumerate(self.heater_mask):
            if not installed:
                continue
            if not bool(heater_level_ok[index]):
                hardware_interlocks.append(f"L2_H{index + 1}_dry_fire")
            if not bool(heater_temperature_ok[index]):
                hardware_interlocks.append(f"L4_H{index + 1}_over_temperature")
        passive_events = [
            f"tank_{index + 1}_passive_overflow"
            for index, flow in enumerate(overflow_flows)
            if float(flow) > 0.0
        ]
        payload = {
            "ambient_temperature_degC": float(context["t_amb"]),
            "circulation_flow_m3s": float(pump_flow),
            "V12_flow_m3s": float(q12),
            "V23_flow_m3s": float(q23),
            "P101_enabled": bool(pump_enabled),
            "P101_electric_power_w": float(
                u[0] ** 3 * self.p["pump_power_max"] * pump_enabled
            ),
            "heater_mask": list(self.heater_mask),
            "hardware_interlocks_active": hardware_interlocks,
            "passive_safety_events": passive_events,
            "protection_events": [*hardware_interlocks, *passive_events],
            "closed_loop_nominal": not hardware_interlocks and not passive_events,
        }
        for index in range(3):
            payload.update(
                {
                    f"H{index + 1}_enabled": bool(heater_enabled[index]),
                    f"H{index + 1}_electric_power_w": float(heater_power[index]),
                    f"H{index + 1}_to_liquid_power_w": float(
                        heat_to_liquid[index]
                    ),
                }
            )
        return payload

    def physical_io_schema(self):
        schema = super().physical_io_schema()
        schema["actuators"] = [
            {"name": "P101", "signal": "VFD/Modbus", "command": "speed"},
            {"name": "V12", "signal": "4-20mA/Modbus", "command": "position"},
            {"name": "V23", "signal": "4-20mA/Modbus", "command": "position"},
            *[
                {
                    "name": f"H{index + 1}",
                    "signal": "SSR",
                    "command": "duty",
                    "installed": bool(self.heater_mask[index]),
                    "power_w": float(self.p["heater_power"][index]),
                }
                for index in range(3)
            ],
        ]
        return schema

    def metadata(self):
        metadata = super().metadata()
        metadata["plant_id"] = self.plant_id
        metadata["plant_hash"] = self.plant_hash
        metadata["heater_mask"] = list(self.heater_mask)
        metadata["parameter_status"] = "user-supplied-design"
        return metadata


def _interleave(levels, temperatures):
    return [
        value
        for pair in zip(levels, temperatures)
        for value in (float(pair[0]), float(pair[1]))
    ]


def _finite_clip01(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


__all__ = ["ThreeTankDesignModel"]
