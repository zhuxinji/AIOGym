"""Engineering assessments for resolved three-tank PlantConfig v2 models."""
from __future__ import annotations

import math
from collections.abc import Mapping

from aiogym.core.model import RHO_CP

from .equipment import ThreeTankDesignModel


def _static_assessment(model: ThreeTankDesignModel, context: Mapping) -> dict:
    operation = context["operation"]
    requirements = context["requirements"]
    checks = []

    def record(name, passed, value, limit, detail):
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "value": value,
                "limit": limit,
                "detail": detail,
            }
        )

    tank_volumes = [
        model.p["area"][index] * model.p["height_max"][index]
        for index in range(3)
    ]
    operating_volumes = [
        model.p["area"][index] * operation["target_levels_m"][index]
        for index in range(3)
    ]
    flow = operation["circulation_flow_m3s"]
    record(
        "pump_flow_capacity",
        flow <= model.p["pump_flow_max"],
        flow,
        model.p["pump_flow_max"],
        "requested circulation must not exceed the supplied maximum flow",
    )
    valve_capacities = [
        model.p["cv_interstage"][index]
        * math.sqrt(
            operation["target_levels_m"][index]
            + model.p["gravity_drop"][index]
        )
        for index in range(2)
    ]
    record(
        "interstage_valve_capacity",
        all(flow <= capacity for capacity in valve_capacities),
        valve_capacities,
        flow,
        "each fully open gravity valve must pass the requested circulation flow",
    )
    installed_heat = sum(
        model.p["heater_power"][index] * model.heater_efficiencies[index]
        for index in range(3)
    )
    stored_energy = sum(
        RHO_CP
        * operating_volumes[index]
        * max(
            operation["target_temperatures_degC"][index]
            - operation["initial_temperatures_degC"][index],
            0.0,
        )
        for index in range(3)
    )
    average_temperatures = [
        0.5
        * (
            operation["target_temperatures_degC"][index]
            + operation["initial_temperatures_degC"][index]
        )
        for index in range(3)
    ]
    approximate_loss = sum(
        model.p["ua_loss"][index]
        * max(
            average_temperatures[index]
            - operation["ambient_temperature_degC"],
            0.0,
        )
        for index in range(3)
    )
    net_heat = installed_heat - approximate_loss
    theoretical_heatup = stored_energy / net_heat if net_heat > 0.0 else None
    record(
        "theoretical_heatup_lower_bound",
        theoretical_heatup is not None
        and theoretical_heatup <= requirements["maximum_heatup_time_s"],
        theoretical_heatup,
        requirements["maximum_heatup_time_s"],
        "lumped stored energy divided by approximate net installed heat",
    )
    references = context["references"]
    record(
        "parameter_provenance_present",
        bool(references),
        len(references),
        1,
        "at least one design, datasheet, or measurement reference is required",
    )
    return {
        "passed": all(check["passed"] for check in checks),
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
        "checks": checks,
        "tank_total_volumes_m3": tank_volumes,
        "tank_operating_volumes_m3": operating_volumes,
        "installed_heater_count": sum(model.heater_mask),
        "installed_electric_heater_power_w": sum(model.p["heater_power"]),
        "installed_liquid_heat_capacity_w": installed_heat,
        "theoretical_heatup_lower_bound_s": theoretical_heatup,
    }


def _steady_state_assessment(model, context, *, env=None):
    equilibrium = model.nominal_steady_state(env=env)
    requirements = context["requirements"]
    active_indices = [0, 1, 2] + [
        3 + index for index, installed in enumerate(model.heater_mask) if installed
    ]
    finite_actions = [
        float(equilibrium["action"][index])
        for index in active_indices
        if math.isfinite(float(equilibrium["action"][index]))
    ]
    max_action = max(finite_actions, default=0.0)
    required_maximum = 1.0 - requirements["minimum_actuator_margin"]
    reasons = list(equilibrium["infeasible_reasons"])
    if max_action > required_maximum:
        reasons.append(
            f"steady actuator command {max_action:.4f} exceeds margin limit {required_maximum:.4f}"
        )
    state = equilibrium["state"]
    action = [
        min(1.0, max(0.0, float(value))) if math.isfinite(float(value)) else 0.0
        for value in equilibrium["action"]
    ]
    residual = model.dynamics(state, action, model._resolved_env(env))
    max_derivative = max(abs(float(value)) for value in residual)
    if equilibrium["feasible"] and max_derivative > 1e-8:
        reasons.append(f"steady derivative residual is {max_derivative:.3e}")
    return {
        "passed": not reasons,
        "feasible": bool(equilibrium["feasible"]),
        "reasons": list(dict.fromkeys(reasons)),
        "state": list(equilibrium["state"]),
        "action": [_json_number(value) for value in equilibrium["action"]],
        "action_names": list(model.action_names),
        "max_active_action": max_action,
        "required_maximum_action": required_maximum,
        "actuator_margin": 1.0 - max_action,
        "ideal_energy_kw": _json_number(equilibrium["ideal_energy_kw"]),
        "heater_to_liquid_power_w": [
            _json_number(value)
            for value in equilibrium["heater_to_liquid_power_w"]
        ],
        "heater_electric_power_w": [
            _json_number(value)
            for value in equilibrium["heater_electric_power_w"]
        ],
        "maximum_derivative_residual": max_derivative,
        "environment": model._resolved_env(env),
    }


def _safety_interlock_assessment(model, context):
    operation = context["operation"]
    target_levels = list(operation["target_levels_m"])
    target_temperatures = list(operation["target_temperatures_degC"])
    full_action = [1.0] * model.action_dim()
    checks = []

    low_levels = list(target_levels)
    low_levels[2] = max(0.0, model.p["low_level_trip"][2] - 0.001)
    state = _interleave(low_levels, target_temperatures)
    info = model.process_info(state, low_levels, target_temperatures, {}, full_action)
    checks.append(
        {
            "name": "P101_dry_run_interlock",
            "passed": not info["P101_enabled"] and info["P101_electric_power_w"] == 0.0,
        }
    )
    for index, installed in enumerate(model.heater_mask):
        if not installed:
            checks.append(
                {
                    "name": f"H{index + 1}_absent_actuator_mask",
                    "passed": info[f"H{index + 1}_electric_power_w"] == 0.0,
                }
            )
            continue
        heater_low_levels = list(target_levels)
        heater_low_levels[index] = max(
            0.0, model.p["low_level_trip"][index] - 0.001
        )
        heater_low_state = _interleave(heater_low_levels, target_temperatures)
        low_info = model.process_info(
            heater_low_state,
            heater_low_levels,
            target_temperatures,
            {},
            full_action,
        )
        checks.append(
            {
                "name": f"H{index + 1}_dry_fire_interlock",
                "passed": low_info[f"H{index + 1}_electric_power_w"] == 0.0,
            }
        )
        hot_temperatures = list(target_temperatures)
        hot_temperatures[index] = model.p["temperature_trip"]
        hot_state = _interleave(target_levels, hot_temperatures)
        hot_info = model.process_info(
            hot_state,
            target_levels,
            hot_temperatures,
            {},
            full_action,
        )
        checks.append(
            {
                "name": f"H{index + 1}_over_temperature_interlock",
                "passed": hot_info[f"H{index + 1}_electric_power_w"] == 0.0,
            }
        )
    return {
        "passed": all(check["passed"] for check in checks),
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
        "checks": checks,
    }



def _interleave(levels, temperatures):
    return [
        item
        for pair in zip(levels, temperatures)
        for item in (float(pair[0]), float(pair[1]))
    ]



def _json_number(value):
    number = float(value)
    return number if math.isfinite(number) else None


__all__ = []
