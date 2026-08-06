"""Deterministic and uncertainty-aware equipment design studies."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence

import numpy as np

from aiogym.core.integration import Integrator
from aiogym.core.model import RHO_CP
from aiogym.core.validation import validate_model_readiness

from .equipment import ThreeTankDesignModel, compile_design_model
from .schema import load_design_spec


DESIGN_RESULT_SCHEMA_VERSION = "aiogym.design_result.v1"
DESIGN_SWEEP_SCHEMA_VERSION = "aiogym.design_sweep.v1"


def run_design_study(
    source,
    *,
    robustness_samples: int | None = None,
    seed: int | None = None,
) -> dict:
    """Run the Phase-1 design gates and return one self-contained result."""

    spec = load_design_spec(source)
    model = compile_design_model(spec)
    sample_count = (
        spec["uncertainties"]["samples"]
        if robustness_samples is None
        else _nonnegative_int("robustness_samples", robustness_samples)
    )
    resolved_seed = (
        spec["uncertainties"]["seed"]
        if seed is None
        else _nonnegative_int("seed", seed)
    )
    static = _static_assessment(model, spec)
    readiness = validate_model_readiness(model)
    steady = _steady_state_assessment(model, spec)
    safety = _safety_interlock_assessment(model, spec)
    dynamic = _dynamic_assessment(model, spec, env=None)
    robustness = _robustness_assessment(
        model,
        spec,
        samples=sample_count,
        seed=resolved_seed,
    )
    gates = [
        _gate("static_engineering", static["passed"], static["failed_checks"]),
        _gate(
            "model_readiness",
            readiness["passed"],
            [row["name"] for row in readiness["checks"] if not row["passed"]],
        ),
        _gate("steady_state_feasibility", steady["passed"], steady["reasons"]),
        _gate("hardware_interlocks", safety["passed"], safety["failed_checks"]),
        _gate("dynamic_commissioning", dynamic["passed"], dynamic["reasons"]),
        _gate("robustness", robustness["passed"], robustness["reasons"]),
    ]
    passed = all(gate["passed"] for gate in gates)
    return {
        "schema_version": DESIGN_RESULT_SCHEMA_VERSION,
        "design_id": spec["id"],
        "design_hash": spec["design_hash"],
        "verdict": "PASS" if passed else "FAIL",
        "evidence_status": "simulation-screening-only",
        "model_status": "user-supplied-design",
        "limitations": [
            "This result is a simulation screening result, not a safety certification or final equipment-selection calculation.",
            "Pump and valve conclusions depend on the supplied hydraulic curve parameters.",
            "Field commissioning must identify installed flow, heat-loss, efficiency, sensor, actuator, and protection settings.",
        ],
        "gates": gates,
        "design_spec": spec,
        "model": {
            "scenario": model.scenario,
            "state_names": list(model.state_names),
            "action_names": list(model.action_names),
            "heater_mask": list(model.heater_mask),
        },
        "static": static,
        "model_readiness": readiness,
        "steady_state": steady,
        "safety": safety,
        "dynamic": dynamic,
        "robustness": robustness,
    }


def run_design_sweep(
    source,
    *,
    parameter: str,
    values: Sequence[float],
    robustness_samples: int | None = None,
    seed: int | None = None,
) -> dict:
    """Evaluate a one-dimensional equipment sweep with exact design hashes."""

    if not isinstance(parameter, str) or not parameter:
        raise ValueError("sweep parameter must be a non-empty dotted path")
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError("sweep values must be a non-empty numeric sequence")
    base = load_design_spec(source)
    base.pop("design_hash", None)
    results = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"sweep value {index} must be numeric")
        candidate = copy.deepcopy(base)
        _set_dotted(candidate, parameter, float(value))
        candidate["id"] = f"{base['id']}-{_slug(parameter)}-{index + 1}"
        results.append(
            run_design_study(
                candidate,
                robustness_samples=robustness_samples,
                seed=seed,
            )
        )
    return {
        "schema_version": DESIGN_SWEEP_SCHEMA_VERSION,
        "base_design_id": base["id"],
        "parameter": parameter,
        "values": [float(value) for value in values],
        "candidates": [
            {
                "design_id": result["design_id"],
                "design_hash": result["design_hash"],
                "value": float(values[index]),
                "verdict": result["verdict"],
                "steady_max_action": result["steady_state"]["max_active_action"],
                "heatup_time_s": result["dynamic"]["heatup_time_s"],
                "energy_kwh": result["dynamic"]["energy_kwh"],
                "robustness_pass_rate": result["robustness"]["pass_rate"],
                "result": result,
            }
            for index, result in enumerate(results)
        ],
    }


def _static_assessment(model: ThreeTankDesignModel, spec: Mapping) -> dict:
    operation = spec["operation"]
    requirements = spec["requirements"]
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
    references = spec["references"]
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


def _steady_state_assessment(model, spec, *, env=None):
    equilibrium = model.nominal_steady_state(env=env)
    requirements = spec["requirements"]
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


def _safety_interlock_assessment(model, spec):
    operation = spec["operation"]
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


def _dynamic_assessment(model, spec, *, env=None):
    operation = spec["operation"]
    requirements = spec["requirements"]
    context = model._resolved_env(env)
    dt = operation["control_dt_s"]
    steps = int(math.ceil(operation["duration_s"] / dt))
    integrator = Integrator(model)
    integrator.reset(model.initial_state())
    controller = _CommissioningController(model, context)
    target_levels = operation["target_levels_m"]
    target_temperatures = operation["target_temperatures_degC"]
    tolerance_t = requirements["temperature_tolerance_degC"]
    tolerance_h = requirements["level_tolerance_m"]
    heatup_time = None
    energy_kwh = 0.0
    max_overshoot = 0.0
    max_level_error = 0.0
    hard_reasons = []
    protection_events = set()
    max_action = 0.0
    final_action = model.default_action()
    final_state = model.initial_state()
    for step in range(steps):
        state = list(integrator.x)
        action = controller.compute(state, dt)
        max_action = max(max_action, *(float(value) for value in action))
        energy_kwh += model.action_energy_kw(action, state, context) * dt / 3600.0
        next_state = list(integrator.step(dt, action, context))
        levels = [next_state[0], next_state[2], next_state[4]]
        temperatures = [next_state[1], next_state[3], next_state[5]]
        info = model.process_info(next_state, levels, temperatures, context, action)
        protection_events.update(info["protection_events"])
        hard = model.hard_termination_reasons(
            next_state, levels, temperatures, context
        )
        hard_reasons.extend(hard)
        max_overshoot = max(
            max_overshoot,
            *(max(0.0, temperatures[index] - target_temperatures[index]) for index in range(3)),
        )
        max_level_error = max(
            max_level_error,
            *(abs(levels[index] - target_levels[index]) for index in range(3)),
        )
        if heatup_time is None and all(
            abs(temperatures[index] - target_temperatures[index]) <= tolerance_t
            for index in range(3)
        ) and all(
            abs(levels[index] - target_levels[index]) <= tolerance_h
            for index in range(3)
        ):
            heatup_time = (step + 1) * dt
        final_action = action
        final_state = next_state
        if hard:
            break
    final_levels = [final_state[0], final_state[2], final_state[4]]
    final_temperatures = [final_state[1], final_state[3], final_state[5]]
    final_temperature_error = max(
        abs(final_temperatures[index] - target_temperatures[index])
        for index in range(3)
    )
    final_level_error = max(
        abs(final_levels[index] - target_levels[index]) for index in range(3)
    )
    reasons = []
    if hard_reasons:
        reasons.append("hard process limit reached: " + ", ".join(sorted(set(hard_reasons))))
    if heatup_time is None or heatup_time > requirements["maximum_heatup_time_s"]:
        reasons.append("heat-up requirement was not met")
    if max_overshoot > requirements["maximum_overshoot_degC"]:
        reasons.append("temperature overshoot exceeds requirement")
    if energy_kwh > requirements["maximum_energy_kwh"]:
        reasons.append("energy consumption exceeds requirement")
    if final_temperature_error > tolerance_t or final_level_error > tolerance_h:
        reasons.append("final state is outside the target tolerance")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "controller": "commissioning-pi-v1",
        "duration_s": min(steps * dt, integrator.t),
        "heatup_time_s": heatup_time,
        "energy_kwh": energy_kwh,
        "maximum_temperature_overshoot_degC": max_overshoot,
        "maximum_level_error_m": max_level_error,
        "maximum_action_command": max_action,
        "final_state": final_state,
        "final_action": final_action,
        "final_temperature_error_degC": final_temperature_error,
        "final_level_error_m": final_level_error,
        "hard_termination_reasons": sorted(set(hard_reasons)),
        "protection_events": sorted(protection_events),
        "environment": context,
    }


def _robustness_assessment(model, spec, *, samples, seed):
    if samples == 0:
        return {
            "passed": True,
            "reasons": [],
            "samples": 0,
            "seed": seed,
            "pass_count": 0,
            "pass_rate": 1.0,
            "required_pass_rate": spec["requirements"]["robustness_pass_rate"],
            "cases": [],
        }
    rng = np.random.default_rng(seed)
    uncertainty = spec["uncertainties"]
    cases = []
    for index in range(samples):
        env = {
            "t_amb": spec["operation"]["ambient_temperature_degC"],
            "pump_flow_factor": float(rng.uniform(*uncertainty["pump_flow_factor"])),
            "heater_efficiency": float(
                rng.uniform(*uncertainty["heater_efficiency_factor"])
            ),
            "heat_loss_factor": float(rng.uniform(*uncertainty["heat_loss_factor"])),
        }
        steady = _steady_state_assessment(model, spec, env=env)
        dynamic = _dynamic_assessment(model, spec, env=env)
        passed = steady["passed"] and dynamic["passed"]
        cases.append(
            {
                "sample": index,
                "passed": passed,
                "environment": env,
                "steady_max_action": steady["max_active_action"],
                "steady_reasons": steady["reasons"],
                "heatup_time_s": dynamic["heatup_time_s"],
                "energy_kwh": dynamic["energy_kwh"],
                "dynamic_reasons": dynamic["reasons"],
            }
        )
    pass_count = sum(case["passed"] for case in cases)
    pass_rate = pass_count / samples
    required = spec["requirements"]["robustness_pass_rate"]
    reasons = [] if pass_rate >= required else [
        f"robustness pass rate {pass_rate:.3f} is below required {required:.3f}"
    ]
    return {
        "passed": not reasons,
        "reasons": reasons,
        "samples": samples,
        "seed": seed,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "required_pass_rate": required,
        "cases": cases,
    }


class _CommissioningController:
    """Conservative fixed-interface baseline used only for design screening."""

    def __init__(self, model, env):
        self.model = model
        self.env = env
        equilibrium = model.nominal_steady_state(env=env)
        self.bias = [
            min(1.0, max(0.0, float(value))) if math.isfinite(float(value)) else 0.0
            for value in equilibrium["action"]
        ]
        self.integral = [0.0, 0.0, 0.0]

    def compute(self, state, dt):
        levels = [state[0], state[2], state[4]]
        temperatures = [state[1], state[3], state[5]]
        target_levels = self.model._target_levels
        target_temperatures = self.model._target_temperatures
        action = list(self.bias)
        action[0] += 4.0 * (target_levels[0] - levels[0])
        action[1] += 3.0 * (levels[0] - target_levels[0]) + 3.0 * (
            target_levels[1] - levels[1]
        )
        action[2] += 3.0 * (levels[1] - target_levels[1]) + 3.0 * (
            target_levels[2] - levels[2]
        )
        for index, installed in enumerate(self.model.heater_mask):
            if not installed:
                action[3 + index] = 0.0
                continue
            error = target_temperatures[index] - temperatures[index]
            candidate = self.integral[index] + error * dt
            raw = self.bias[3 + index] + 0.12 * error + 0.0005 * candidate
            clipped = min(1.0, max(0.0, raw))
            if clipped == raw or (clipped >= 1.0 and error < 0.0) or (
                clipped <= 0.0 and error > 0.0
            ):
                self.integral[index] = candidate
            action[3 + index] = clipped
        return [min(1.0, max(0.0, float(value))) for value in action]


def _gate(name, passed, reasons):
    return {"name": name, "passed": bool(passed), "reasons": list(reasons)}


def _interleave(levels, temperatures):
    return [
        item
        for pair in zip(levels, temperatures)
        for item in (float(pair[0]), float(pair[1]))
    ]


def _nonnegative_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _set_dotted(target, path, value):
    current = target
    parts = path.split(".")
    if not all(parts):
        raise ValueError("sweep parameter contains an empty path segment")
    for part in parts[:-1]:
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"invalid sweep list index in {path!r}") from exc
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ValueError(f"unknown sweep parameter path: {path}")
    final = parts[-1]
    if isinstance(current, list):
        try:
            current[int(final)] = value
        except (ValueError, IndexError) as exc:
            raise ValueError(f"invalid sweep list index in {path!r}") from exc
    elif isinstance(current, dict) and final in current:
        current[final] = value
    else:
        raise ValueError(f"unknown sweep parameter path: {path}")


def _slug(value):
    return "-".join(part for part in value.replace("_", "-").split(".") if part)


def _json_number(value):
    number = float(value)
    return number if math.isfinite(number) else None


__all__ = [
    "DESIGN_RESULT_SCHEMA_VERSION",
    "DESIGN_SWEEP_SCHEMA_VERSION",
    "run_design_study",
    "run_design_sweep",
]
