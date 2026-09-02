"""Three-tank cascade with one independently commanded heater per tank."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from types import MappingProxyType
from typing import Any

import numpy as np

from aiogym.core.model import PhysicsModelBase, RHO_CP
from aiogym.scenarios.three_tank.model import (
    BOM_CONFIGURATION as THREE_TANK_BOM_CONFIGURATION,
    NOMINAL_FLOW_M3S,
    ThreeTankModel,
)


HEATER_POWER_W = 2000.0
DEFAULT_HEATER = (1, 0, 0)
_DEFAULT_HEATER_DUTIES = (0.08, 0.10, 0.12)

BOM_CONFIGURATION = {
    **deepcopy(THREE_TANK_BOM_CONFIGURATION),
    "installed_heaters": [
        {"id": f"H{tank}", "tank": tank, "power_w": HEATER_POWER_W}
        for tank in (1, 2, 3)
    ],
    "source": (
        "three_tank hydraulic BOM with one independently commanded 2 kW "
        "immersion heater per process tank"
    ),
}

TRACKING_ERROR_SCALES = (0.1, 0.1, 0.1, 5.0, 5.0, 5.0)
TRACKING_SETTLING_TOLERANCES = (0.005, 0.005, 0.005, 0.2, 0.2, 0.2)

_HYDRAULIC_PARAMETER_NAMES = {
    "area",
    "height_max",
    "level_sensor_range",
    "cv_valves",
    "cv_bypass",
    "flow_observation_scale",
    "gravity_drop",
    "overflow_level",
    "cv_overflow",
    "overflow_head_floor",
    "high_level_trip",
    "nominal_level",
    "pump_flow_max",
    "pump_power_max",
    "pump_static_head",
    "pump_shutoff_head",
}


class CascadeModel(ThreeTankModel):
    """Hydraulic cascade plus three variable-volume thermal balances."""

    scenario = "cascade"
    state_names = (
        "h1",
        "T1",
        "h2",
        "T2",
        "h3",
        "T3",
        "reservoir_volume",
        "reservoir_temperature",
    )
    state_units = {
        "h1": "m",
        "h2": "m",
        "h3": "m",
        "T1": "degC",
        "T2": "degC",
        "T3": "degC",
        "reservoir_volume": "m3",
        "reservoir_temperature": "degC",
    }
    action_names = (
        "pump_P101",
        "valve_V12",
        "valve_V23",
        "valve_V34",
        "heater_H1",
        "heater_H2",
        "heater_H3",
    )
    action_kinds = {
        "pump_P101": "pump",
        "valve_V12": "valve",
        "valve_V23": "valve",
        "valve_V34": "valve",
        "heater_H1": "heater",
        "heater_H2": "heater",
        "heater_H3": "heater",
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
        *ThreeTankModel.input_disturbances,
        {
            "name": "ambient_temperature",
            "event": "ambient_temperature_step",
            "unit": "degC",
            "bounds": (0.0, 45.0),
            "default": 20.0,
            "description": "ambient air temperature around the process tanks",
        },
        {
            "name": "heater_H1_efficiency_factor",
            "event": "heater_H1_efficiency_shift",
            "unit": "fraction",
            "bounds": (0.4, 1.2),
            "default": 1.0,
            "description": "Tank 1 heater heat-transfer multiplier",
        },
        {
            "name": "heater_H2_efficiency_factor",
            "event": "heater_H2_efficiency_shift",
            "unit": "fraction",
            "bounds": (0.4, 1.2),
            "default": 1.0,
            "description": "Tank 2 heater heat-transfer multiplier",
        },
        {
            "name": "heater_H3_efficiency_factor",
            "event": "heater_H3_efficiency_shift",
            "unit": "fraction",
            "bounds": (0.4, 1.2),
            "default": 1.0,
            "description": "Tank 3 heater heat-transfer multiplier",
        },
        {
            "name": "heat_loss_factor",
            "event": "heat_loss_shift",
            "unit": "fraction",
            "bounds": (0.3, 3.0),
            "default": 1.0,
            "description": "common tank-to-ambient heat-loss multiplier",
        },
    )
    energy_scored = True
    parameter_units = MappingProxyType(
        {
            **dict(ThreeTankModel.parameter_units),
            "heater_power_max": "W",
            "heat_loss_coefficient": "W/K",
            "ambient_temperature": "degC",
            "reservoir_capacity": "m3",
            "reservoir_initial_volume": "m3",
            "reservoir_heat_loss_coefficient": "W/K",
            "heater_min_level": "m",
            "temperature_trip": "degC",
            "temperature_hard_limit": "degC",
            "thermal_level_floor": "m",
        }
    )

    def __init__(
        self,
        parameters: Mapping[str, Any] | None = None,
        *,
        heater: Sequence[int] | None = None,
    ):
        supplied = _parameter_mapping(parameters)
        self.heater = _resolved_heater(DEFAULT_HEATER if heater is None else heater)
        thermal_defaults = {
            "heater_power_max": [HEATER_POWER_W] * 3,
            "heat_loss_coefficient": [40.0, 40.0, 40.0],
            "ambient_temperature": 20.0,
            "reservoir_capacity": 0.180,
            "reservoir_initial_volume": 0.090,
            "reservoir_heat_loss_coefficient": 40.0,
            "heater_min_level": 0.10,
            "temperature_trip": 80.0,
            "temperature_hard_limit": 90.0,
            "thermal_level_floor": 0.005,
        }
        known = _HYDRAULIC_PARAMETER_NAMES | set(thermal_defaults)
        unknown = sorted(set(supplied) - known)
        if unknown:
            raise ValueError(
                f"unknown cascade parameters: {unknown}; available: {sorted(known)}"
            )
        hydraulic = {
            name: value
            for name, value in supplied.items()
            if name in _HYDRAULIC_PARAMETER_NAMES
        }
        super().__init__(hydraulic)
        thermal = _resolved_thermal_parameters(
            thermal_defaults,
            {
                name: value
                for name, value in supplied.items()
                if name in thermal_defaults
            },
        )
        self.p.update(thermal)
        if self.p["heater_min_level"] >= self.p["nominal_level"]:
            raise ValueError(
                "cascade heater_min_level must be below the nominal level"
            )
        if self.p["thermal_level_floor"] > self.p["heater_min_level"]:
            raise ValueError(
                "cascade thermal_level_floor must not exceed heater_min_level"
            )
        if self.p["reservoir_initial_volume"] >= self.p["reservoir_capacity"]:
            raise ValueError(
                "cascade reservoir_initial_volume must be below reservoir_capacity"
            )
        if not 30.0 < self.p["temperature_trip"] < self.p["temperature_hard_limit"]:
            raise ValueError(
                "cascade temperatures must satisfy 30 < trip < hard limit"
            )
        self._resolved_parameters = MappingProxyType(
            {
                name: tuple(value) if isinstance(value, list) else value
                for name, value in self.p.items()
            }
        )
        self._environment_bounds = {
            row["name"]: tuple(row["bounds"]) for row in self.input_disturbances
        }

    def parameter(self, name):
        try:
            return deepcopy(self.p[str(name)])
        except KeyError as error:
            raise KeyError(f"unknown cascade parameter {name!r}") from error

    @property
    def state_bounds(self):
        maximum_temperature = float(self.p["temperature_hard_limit"])
        result = {}
        for index in range(3):
            result[f"h{index + 1}"] = (0.0, self.height_max[index])
            result[f"T{index + 1}"] = (0.0, maximum_temperature)
        result["reservoir_volume"] = (0.0, self.p["reservoir_capacity"])
        result["reservoir_temperature"] = (0.0, maximum_temperature)
        return result

    @property
    def output_bounds(self):
        maximum_temperature = float(self.p["temperature_hard_limit"])
        return {
            **{
                f"tank_{index + 1}_level": (0.0, self.height_max[index])
                for index in range(3)
            },
            **{
                f"tank_{index + 1}_temperature": (0.0, maximum_temperature)
                for index in range(3)
            },
        }

    def initial_state(self):
        equilibrium = self.nominal_steady_state()
        if not equilibrium["feasible"]:
            raise ValueError("default cascade operating point is infeasible")
        state = list(equilibrium["state"])
        state[7] = float(self.p["ambient_temperature"])
        return state

    def outputs(self, state):
        values = self.state_vector(state)
        return [values[0], values[2], values[4], values[1], values[3], values[5]]

    def display_outputs(self, state):
        values = self.state_vector(state)
        return {
            "levels": [values[0], values[2], values[4]],
            "temps": [values[1], values[3], values[5]],
        }

    def steady_temperature_profile(
        self,
        *,
        flow=NOMINAL_FLOW_M3S,
        heater_duties=_DEFAULT_HEATER_DUTIES,
        env=None,
    ):
        context = self._resolved_env(env)
        q = float(flow)
        if not math.isfinite(q) or q <= 0.0:
            raise ValueError("flow must be finite and positive")
        try:
            duties = [float(value) for value in heater_duties]
        except (TypeError, ValueError) as error:
            raise ValueError("heater_duties must contain three numeric values") from error
        if (
            len(duties) != 3
            or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in duties)
        ):
            raise ValueError("heater_duties must contain three values in [0, 1]")
        return self._steady_thermal_state(q, duties, context)[:3]

    def _steady_thermal_state(self, flow, heater_duties, context):
        q = float(flow)
        transport = RHO_CP * q
        ambient = float(context["ambient_temperature"])
        tank_losses = np.asarray(self.p["heat_loss_coefficient"], dtype=float)
        tank_losses *= float(context["heat_loss_factor"])
        reservoir_loss = (
            self.p["reservoir_heat_loss_coefficient"]
            * float(context["heat_loss_factor"])
        )
        delivered_power = np.asarray(
            [
                heater_duties[index]
                * self.heater[index]
                * self.p["heater_power_max"][index]
                * context[f"heater_H{index + 1}_efficiency_factor"]
                for index in range(3)
            ],
            dtype=float,
        )
        matrix = np.asarray(
            [
                [transport + tank_losses[0], 0.0, 0.0, -transport],
                [-transport, transport + tank_losses[1], 0.0, 0.0],
                [0.0, -transport, transport + tank_losses[2], 0.0],
                [0.0, 0.0, -transport, transport + reservoir_loss],
            ],
            dtype=float,
        )
        right_hand_side = np.asarray(
            [
                delivered_power[0] + tank_losses[0] * ambient,
                delivered_power[1] + tank_losses[1] * ambient,
                delivered_power[2] + tank_losses[2] * ambient,
                reservoir_loss * ambient,
            ],
            dtype=float,
        )
        temperatures = np.linalg.solve(matrix, right_hand_side)
        return [float(value) for value in temperatures]

    def nominal_steady_state(
        self,
        *,
        flow=NOMINAL_FLOW_M3S,
        levels=None,
        temperatures=None,
        reservoir_volume=None,
        env=None,
    ):
        context = self._resolved_env(env)
        hydraulic = ThreeTankModel.nominal_steady_state(
            self,
            flow=flow,
            levels=levels,
            env=context,
        )
        h = [float(value) for value in hydraulic["state"]]
        q = float(hydraulic["flow_m3s"])
        if temperatures is None:
            thermal_state = self._steady_thermal_state(
                q,
                _DEFAULT_HEATER_DUTIES,
                context,
            )
            temperatures = thermal_state[:3]
            reservoir_temperature = thermal_state[3]
        else:
            supplied_temperatures = [float(value) for value in temperatures]
            if len(supplied_temperatures) != 3 or any(
                not math.isfinite(value) for value in supplied_temperatures
            ):
                raise ValueError("temperatures must contain three finite values")
            transport = RHO_CP * q
            reservoir_loss = (
                self.p["reservoir_heat_loss_coefficient"]
                * context["heat_loss_factor"]
            )
            ambient = context["ambient_temperature"]
            reservoir_temperature = (
                transport * supplied_temperatures[2] + reservoir_loss * ambient
            ) / (transport + reservoir_loss)
        t = [float(value) for value in temperatures]
        if len(t) != 3 or any(not math.isfinite(value) for value in t):
            raise ValueError("temperatures must contain three finite values")
        volume = float(
            self.p["reservoir_initial_volume"]
            if reservoir_volume is None
            else reservoir_volume
        )
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError("reservoir_volume must be finite and positive")

        inlet_temperatures = [reservoir_temperature, t[0], t[1]]
        heater_commands = []
        thermal_loads = []
        reasons = list(hydraulic["infeasible_reasons"])
        for index, (level, temperature, inlet_temperature) in enumerate(
            zip(h, t, inlet_temperatures)
        ):
            load = (
                RHO_CP * q * (temperature - inlet_temperature)
                + self.p["heat_loss_coefficient"][index]
                * context["heat_loss_factor"]
                * (temperature - context["ambient_temperature"])
            )
            thermal_loads.append(float(load))
            efficiency = context[f"heater_H{index + 1}_efficiency_factor"]
            capacity = self.p["heater_power_max"][index] * efficiency
            command = (
                max(0.0, load) / capacity
                if self.heater[index]
                else 0.0
            )
            heater_commands.append(float(command))
            if temperature < 0.0 or temperature >= self.p["temperature_hard_limit"]:
                reasons.append(f"tank_{index + 1} temperature is outside its hard bounds")
            if load < -1e-9:
                reasons.append(f"tank_{index + 1} steady state requires cooling")
            if not self.heater[index] and load > 1e-7:
                reasons.append(f"heater_H{index + 1} is unavailable")
            if command > 1.0 or not math.isfinite(command):
                reasons.append(f"heater_H{index + 1} command is outside [0, 1]")
            if command > 0.0 and level < self.p["heater_min_level"]:
                reasons.append(f"heater_H{index + 1} is blocked by low liquid level")
            if command > 0.0 and temperature >= self.p["temperature_trip"]:
                reasons.append(f"heater_H{index + 1} is blocked by temperature trip")
        if volume > self.p["reservoir_capacity"]:
            reasons.append("reservoir volume exceeds reservoir_capacity")
        if reservoir_temperature < 0.0 or reservoir_temperature >= self.p[
            "temperature_hard_limit"
        ]:
            reasons.append("reservoir temperature is outside its hard bounds")

        state = [
            h[0],
            t[0],
            h[1],
            t[1],
            h[2],
            t[2],
            volume,
            float(reservoir_temperature),
        ]
        action = [*hydraulic["action"], *heater_commands]
        return {
            "feasible": not reasons,
            "infeasible_reasons": tuple(reasons),
            "state": state,
            "y_sp": [*h, *t],
            "action": action,
            "flow_m3s": q,
            "thermal_load_w": thermal_loads,
            "reservoir_volume_m3": volume,
            "reservoir_temperature": float(reservoir_temperature),
        }

    def _tracking_steady_state(self, reference, disturbances):
        values = np.asarray(reference, dtype=float).reshape(-1)
        if values.shape != (6,) or not np.all(np.isfinite(values)):
            return None
        flow = self._tracking_flow(values[3:], disturbances)
        try:
            equilibrium = self.nominal_steady_state(
                flow=flow,
                levels=values[:3].tolist(),
                temperatures=values[3:].tolist(),
                env=disturbances,
            )
        except ValueError:
            return None
        return equilibrium if equilibrium["feasible"] else None

    def _tracking_flow(self, temperatures, disturbances):
        supplied = {} if disturbances is None else dict(disturbances)
        measured_flow = supplied.get("FT101_flow_m3s")
        if measured_flow is not None:
            flow = float(measured_flow)
            if math.isfinite(flow) and flow > 0.0:
                return flow

        context = self._resolved_env(disturbances)
        values = [float(value) for value in temperatures]
        inlet_temperatures = [
            None,
            values[0],
            values[1],
        ]
        candidates = []
        if not self.heater[0]:
            tank_1 = values[0]
            tank_3 = values[2]
            ambient = context["ambient_temperature"]
            tank_loss = (
                self.p["heat_loss_coefficient"][0]
                * context["heat_loss_factor"]
            )
            reservoir_loss = (
                self.p["reservoir_heat_loss_coefficient"]
                * context["heat_loss_factor"]
            )
            roots = np.roots(
                [
                    tank_1 - tank_3,
                    (reservoir_loss + tank_loss) * (tank_1 - ambient),
                    tank_loss * reservoir_loss * (tank_1 - ambient),
                ]
            )
            candidates.extend(
                float(root.real / RHO_CP)
                for root in roots
                if abs(float(root.imag)) <= 1e-9
                and root.real > 0.0
                and root.real / RHO_CP <= self.p["pump_flow_max"]
            )
        for index, (temperature, inlet_temperature) in enumerate(
            zip(values, inlet_temperatures)
        ):
            if self.heater[index] or inlet_temperature is None:
                continue
            denominator = RHO_CP * (temperature - inlet_temperature)
            if abs(denominator) <= 1e-12:
                continue
            loss = (
                self.p["heat_loss_coefficient"][index]
                * context["heat_loss_factor"]
                * (temperature - context["ambient_temperature"])
            )
            candidate = -loss / denominator
            if math.isfinite(candidate) and candidate > 0.0:
                candidates.append(float(candidate))
        return (
            float(np.median(candidates))
            if candidates
            else float(NOMINAL_FLOW_M3S)
        )

    def default_action(self):
        equilibrium = self.nominal_steady_state()
        if not equilibrium["feasible"]:
            raise ValueError("default cascade operating point is infeasible")
        return list(equilibrium["action"])

    def default_setpoint_vector(self):
        return list(self.outputs(self.initial_state()))

    def default_disturbances(self):
        defaults = dict(PhysicsModelBase.default_disturbances(self))
        defaults.update({"ambient_temperature": float(self.p["ambient_temperature"])})
        return defaults

    def _resolved_env(self, env=None):
        supplied = self.default_disturbances()
        supplied.update({} if env is None else dict(env))
        clean = {}
        for row in self.input_disturbances:
            name = row["name"]
            try:
                value = float(supplied[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"cascade disturbance {name!r} must be finite") from exc
            if not math.isfinite(value):
                raise ValueError(f"cascade disturbance {name!r} must be finite")
            lower, upper = self._environment_bounds[name]
            if value < float(lower) or value > float(upper):
                raise ValueError(
                    f"cascade disturbance {name!r} must be within "
                    f"[{lower}, {upper}], got {value}"
                )
            if row["unit"] == "binary" and value not in (0.0, 1.0):
                raise ValueError(
                    f"cascade disturbance {name!r} must be binary, got {value}"
                )
            clean[name] = value
        return clean

    def _effective_action(self, action):
        values = self.action_vector(action)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("cascade action values must be finite")
        return [min(max(value, 0.0), 1.0) for value in values]

    def action_vector(self, action):
        values = list(super().action_vector(action))
        for index, enabled in enumerate(self.heater):
            if not enabled:
                values[4 + index] = 0.0
        return values

    def action_slew_limits(self):
        return [1.0] * 4 + [float(enabled) for enabled in self.heater]

    def _heater_terms(self, levels, temperatures, action, env):
        heat_to_liquid = []
        electric_power = []
        interlocked = []
        for index in range(3):
            enabled = (
                bool(self.heater[index])
                and levels[index] >= self.p["heater_min_level"]
                and temperatures[index] < self.p["temperature_trip"]
            )
            electric = (
                action[4 + index] * self.p["heater_power_max"][index]
                if enabled
                else 0.0
            )
            electric_power.append(float(electric))
            heat_to_liquid.append(
                float(electric * env[f"heater_H{index + 1}_efficiency_factor"])
            )
            interlocked.append(not enabled)
        return heat_to_liquid, electric_power, interlocked

    def _dynamics(self, state, action, env):
        context = self._resolved_env(env)
        u = self._effective_action(action)
        levels = [float(state[0]), float(state[2]), float(state[4])]
        temperatures = [float(state[1]), float(state[3]), float(state[5])]
        reservoir_volume = float(state[6])
        reservoir_temperature = float(state[7])
        context["reservoir_available"] *= float(reservoir_volume > 0.0)
        pump, valves, bypasses, overflows, _ = self._flow_terms(levels, u, context)
        transfer_flows = [
            valves[index] + bypasses[index] for index in range(3)
        ]
        heat_inputs, _, _ = self._heater_terms(
            levels, temperatures, u, context
        )
        flows_in = [pump, transfer_flows[0], transfer_flows[1]]
        flows_out = [
            transfer_flows[0] + overflows[0],
            transfer_flows[1] + overflows[1],
            transfer_flows[2] + overflows[2],
        ]
        inlet_temperatures = [
            reservoir_temperature,
            temperatures[0],
            temperatures[1],
        ]
        derivatives = []
        for index in range(3):
            level_derivative = (
                flows_in[index] - flows_out[index]
            ) / self.p["area"][index]
            volume = self.p["area"][index] * max(
                levels[index], self.p["thermal_level_floor"]
            )
            heat_loss = (
                self.p["heat_loss_coefficient"][index]
                * context["heat_loss_factor"]
                * (temperatures[index] - context["ambient_temperature"])
            )
            temperature_derivative = (
                flows_in[index]
                * (inlet_temperatures[index] - temperatures[index])
                + (heat_inputs[index] - heat_loss) / RHO_CP
            ) / volume
            derivatives.extend([level_derivative, temperature_derivative])
        reservoir_inflows = [
            (transfer_flows[2], temperatures[2]),
            *zip(overflows, temperatures),
        ]
        reservoir_volume_derivative = (
            sum(flow for flow, _temperature in reservoir_inflows) - pump
        )
        reservoir_heat_loss = (
            self.p["reservoir_heat_loss_coefficient"]
            * context["heat_loss_factor"]
            * (reservoir_temperature - context["ambient_temperature"])
        )
        reservoir_temperature_derivative = (
            sum(
                flow * (temperature - reservoir_temperature)
                for flow, temperature in reservoir_inflows
            )
            - reservoir_heat_loss / RHO_CP
        ) / max(reservoir_volume, 1e-6)
        derivatives.extend(
            [reservoir_volume_derivative, reservoir_temperature_derivative]
        )
        return derivatives

    def observation_schema(self):
        measurement_names = [
            "normalized_tank_1_level",
            "normalized_tank_1_temperature",
            "normalized_tank_2_level",
            "normalized_tank_2_temperature",
            "normalized_tank_3_level",
            "normalized_tank_3_temperature",
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
        state_values = np.asarray(state, dtype=float)
        process_state = state_values[:6]
        state_scales = np.asarray(
            [
                self.height_max[0],
                self.p["temperature_hard_limit"],
                self.height_max[1],
                self.p["temperature_hard_limit"],
                self.height_max[2],
                self.p["temperature_hard_limit"],
            ],
            dtype=float,
        )
        normalized_state = np.clip(process_state / state_scales, 0.0, 1.0)
        normalized_reference = np.clip(
            np.asarray(reference, dtype=float)
            / np.asarray(self.output_scales(), dtype=float),
            0.0,
            1.0,
        )
        action = self.default_action() if previous_action is None else previous_action
        levels = [
            float(process_state[0]),
            float(process_state[2]),
            float(process_state[4]),
        ]
        context["reservoir_available"] *= float(state_values[6] > 0.0)
        pump, valve_flows, _, _, _ = self._flow_terms(
            levels,
            self._effective_action(action),
            context,
        )
        normalized_flow = np.clip(
            np.asarray([pump, *valve_flows], dtype=float)
            / np.asarray(self.p["flow_observation_scale"], dtype=float),
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
        display = self.display_outputs(values)
        return {
            "x": values,
            "levels": display["levels"],
            "temps": display["temps"],
            "y": list(self.outputs(values)),
            "reservoir_volume_m3": float(values[6]),
            "reservoir_temperature": float(values[7]),
            **context,
        }

    def measurement_from_observation(self, observation, disturbances=None):
        values = np.asarray(observation, dtype=float).reshape(-1)
        if values.shape != (16,) or not np.isfinite(values).all():
            raise ValueError("cascade policy observation must match observation_schema")
        state_scales = np.asarray(
            [
                self.height_max[0],
                self.p["temperature_hard_limit"],
                self.height_max[1],
                self.p["temperature_hard_limit"],
                self.height_max[2],
                self.p["temperature_hard_limit"],
            ],
            dtype=float,
        )
        process_state = values[:6] * state_scales
        flows = values[6:10] * np.asarray(self.p["flow_observation_scale"])
        context = self._resolved_env(disturbances)
        transport = RHO_CP * max(float(flows[0]), 1e-12)
        reservoir_loss = (
            self.p["reservoir_heat_loss_coefficient"]
            * context["heat_loss_factor"]
        )
        reservoir_temperature = (
            transport * float(process_state[5])
            + reservoir_loss * context["ambient_temperature"]
        ) / (transport + reservoir_loss)
        estimated_state = [
            *process_state.tolist(),
            self.p["reservoir_initial_volume"],
            reservoir_temperature,
        ]
        measurement = self.measurement(estimated_state, disturbances)
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

    def hard_termination_reasons(self, state):
        values = self.state_vector(state)
        levels = [values[0], values[2], values[4]]
        temperatures = [values[1], values[3], values[5]]
        reservoir_volume = values[6]
        reservoir_temperature = values[7]
        reasons = []
        if any(value < 0.0 for value in levels):
            reasons.append("negative_level")
        if any(
            value > self.p["height_max"][index]
            for index, value in enumerate(levels)
        ):
            reasons.append("tank_overflow_limit")
        if any(value < 0.0 for value in temperatures):
            reasons.append("negative_temperature")
        if any(value >= self.p["temperature_hard_limit"] for value in temperatures):
            reasons.append("temperature_hard_limit")
        if reservoir_volume <= 0.0:
            reasons.append("reservoir_empty")
        if reservoir_volume > self.p["reservoir_capacity"]:
            reasons.append("reservoir_overflow_limit")
        if reservoir_temperature < 0.0:
            reasons.append("negative_reservoir_temperature")
        if reservoir_temperature >= self.p["temperature_hard_limit"]:
            reasons.append("reservoir_temperature_hard_limit")
        return tuple(reasons)

    def constraint_costs(self, state, disturbances=None):
        del disturbances
        return {reason: 1.0 for reason in self.hard_termination_reasons(state)}

    def safety_margins(self, state, disturbances=None):
        del disturbances
        values = self.state_vector(state)
        levels = [values[0], values[2], values[4]]
        temperatures = [values[1], values[3], values[5]]
        margins = {}
        for index, (level, maximum) in enumerate(
            zip(levels, self.p["height_max"]), start=1
        ):
            margins[f"tank_{index}_level_lower"] = level / maximum
            margins[f"tank_{index}_level_upper"] = (maximum - level) / maximum
        hard_limit = float(self.p["temperature_hard_limit"])
        for index, temperature in enumerate(temperatures, start=1):
            margins[f"tank_{index}_temperature_lower"] = temperature / hard_limit
            margins[f"tank_{index}_temperature_upper"] = (
                hard_limit - temperature
            ) / hard_limit
        reservoir_capacity = float(self.p["reservoir_capacity"])
        reservoir_volume = float(values[6])
        margins["reservoir_volume_lower"] = reservoir_volume / reservoir_capacity
        margins["reservoir_volume_upper"] = (
            reservoir_capacity - reservoir_volume
        ) / reservoir_capacity
        reservoir_temperature = float(values[7])
        margins["reservoir_temperature_lower"] = (
            reservoir_temperature / hard_limit
        )
        margins["reservoir_temperature_upper"] = (
            hard_limit - reservoir_temperature
        ) / hard_limit
        return margins

    def process_info(self, state, action, disturbances=None):
        context = self._resolved_env(disturbances)
        applied = self._effective_action(
            self.default_action() if action is None else action
        )
        values = self.state_vector(state)
        levels = [values[0], values[2], values[4]]
        temperatures = [values[1], values[3], values[5]]
        context["reservoir_available"] *= float(values[6] > 0.0)
        pump, valves, bypass_flows, overflows, pump_enabled = self._flow_terms(
            levels, applied, context
        )
        heat, electric, interlocked = self._heater_terms(
            levels, temperatures, applied, context
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
            "heater_available": [bool(value) for value in self.heater],
            "heater_electric_power_w": electric,
            "heater_to_liquid_power_w": heat,
            "heater_interlocked": interlocked,
            "reservoir_volume_m3": float(values[6]),
            "reservoir_temperature": float(values[7]),
        }

    def action_energy_kw(self, act, x=None, env=None):
        applied = self._effective_action(act)
        pump_enabled = True
        heater_power = [
            applied[4 + index] * self.p["heater_power_max"][index]
            for index in range(3)
        ]
        if x is not None:
            context = self._resolved_env(env)
            values = self.state_vector(x)
            levels = [values[0], values[2], values[4]]
            temperatures = [values[1], values[3], values[5]]
            context["reservoir_available"] *= float(values[6] > 0.0)
            _, _, _, _, pump_enabled = self._flow_terms(levels, applied, context)
            _, heater_power, _ = self._heater_terms(
                levels, temperatures, applied, context
            )
        pump_power = (
            applied[0] ** 3 * self.p["pump_power_max"] * float(pump_enabled)
        )
        return float((pump_power + sum(heater_power)) / 1000.0)

    def energy_kw(self, action):
        return self.action_energy_kw(action)

    def step_info(self, state, action, disturbances=None):
        applied = self.default_action() if action is None else list(action)
        info = self.process_info(state, applied, disturbances)
        info["y"] = list(self.outputs(state))
        info["energy_kw"] = self.action_energy_kw(applied, state, disturbances)
        return info

    def physical_io_schema(self):
        return {
            "measurements": [
                *(
                    {
                        "name": f"LT{tank}01",
                        "quantity": f"tank_{tank}_level",
                        "range_m": [0.0, 0.5],
                        "signal": "RS485",
                    }
                    for tank in (1, 2, 3)
                ),
                *(
                    {
                        "name": f"TT{tank}01",
                        "quantity": f"tank_{tank}_temperature",
                        "range_degC": [0.0, self.p["temperature_hard_limit"]],
                        "signal": "RS485",
                    }
                    for tank in (1, 2, 3)
                ),
                *(
                    {
                        "name": name,
                        "quantity": quantity,
                        "range_l_min": [0.0, 100.0],
                        "signal": "4-20mA / Modbus",
                    }
                    for name, quantity in (
                        ("FT101", "P101_flow"),
                        ("FT12", "tank_1_to_2_flow"),
                        ("FT23", "tank_2_to_3_flow"),
                        ("FT34", "tank_3_to_reservoir_flow"),
                    )
                ),
            ],
            "actuators": [
                {"name": "P101", "command": "VFD speed"},
                {"name": "V12", "command": "valve position"},
                {"name": "V23", "command": "valve position"},
                {"name": "V34", "command": "valve position"},
                *(
                    {
                        "name": f"H{tank}",
                        "command": "heater duty",
                        "available": bool(self.heater[tank - 1]),
                    }
                    for tank in (1, 2, 3)
                ),
            ],
            "disturbance_actuators": [
                {"name": "BV12", "command": "on/off bypass"},
                {"name": "BV23", "command": "on/off bypass"},
                {"name": "BV34", "command": "on/off bypass"},
            ],
            "boundary": dict(BOM_CONFIGURATION["reservoir"]),
        }


def _parameter_mapping(parameters):
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping or None")
    supplied = dict(parameters)
    if any(not isinstance(name, str) for name in supplied):
        raise TypeError("parameter keys must be strings")
    return supplied


def _resolved_heater(value):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("heater must be a three-value binary sequence")
    values = list(value)
    if len(values) != 3:
        raise ValueError("heater must contain exactly three values")
    if any(
        not isinstance(item, (bool, int)) or int(item) not in (0, 1)
        for item in values
    ):
        raise ValueError("heater values must be binary 0 or 1")
    return tuple(int(item) for item in values)


def _resolved_thermal_parameters(defaults, overrides):
    resolved = deepcopy(defaults)
    for name, value in overrides.items():
        default = defaults[name]
        if isinstance(default, list):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"cascade parameter {name!r} must be a sequence")
            values = [_finite_positive(name, item) for item in value]
            if len(values) != len(default):
                raise ValueError(
                    f"cascade parameter {name!r} must contain {len(default)} values"
                )
            resolved[name] = values
        else:
            resolved[name] = _finite_positive(name, value)
    if resolved["temperature_trip"] >= resolved["temperature_hard_limit"]:
        raise ValueError(
            "cascade temperature_trip must be below temperature_hard_limit"
        )
    return resolved


def _finite_positive(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"cascade parameter {name!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"cascade parameter {name!r} must be finite")
    if number <= 0.0:
        raise ValueError(f"cascade parameter {name!r} must be positive")
    return number


__all__ = [
    "BOM_CONFIGURATION",
    "CascadeModel",
    "DEFAULT_HEATER",
    "HEATER_POWER_W",
    "TRACKING_ERROR_SCALES",
    "TRACKING_SETTLING_TOLERANCES",
]
