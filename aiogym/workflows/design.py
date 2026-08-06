"""Generic PlantConfig validation, design studies, sweeps, and reports."""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import aiogym.scenarios  # noqa: F401
from aiogym.controllers.base import make_controller
from aiogym.core import (
    CheckResult,
    PlantConfig,
    get_scenario,
    make_env,
    rollout,
)

from .artifacts import write_run_bundle


STUDY_SCHEMA_VERSION = "aiogym.study.v1"
SWEEP_SCHEMA_VERSION = "aiogym.study_sweep.v1"


def load_plant(source: PlantConfig | str | Path | Mapping[str, Any]) -> PlantConfig:
    if isinstance(source, PlantConfig):
        return source
    if isinstance(source, Mapping):
        raw = copy.deepcopy(dict(source))
    else:
        path = Path(source)
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            matches = []
            from aiogym.core import list_scenarios

            for scenario in list_scenarios():
                plugin = get_scenario(scenario)
                if str(source) in plugin.built_in_plants:
                    matches.append(plugin.built_in_plants[str(source)]())
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"unknown or ambiguous built-in plant {str(source)!r}"
                )
            raw = matches[0]
    if raw.get("schema_version") == "aiogym.design_spec.v1":
        raw = convert_design_spec_v1(raw)
    return PlantConfig.from_mapping(raw)


def convert_design_spec_v1(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        raw = copy.deepcopy(dict(source))
    else:
        raw = json.loads(Path(source).read_text(encoding="utf-8"))
    raw.pop("design_hash", None)
    if raw.get("schema_version") != "aiogym.design_spec.v1":
        raise ValueError("source is not an aiogym.design_spec.v1 declaration")
    from aiogym.scenarios.three_tank.migration import design_v1_to_plant_v2

    return design_v1_to_plant_v2(raw)


def validate_plant(source):
    config = load_plant(source)
    plugin = get_scenario(config.scenario)
    return plugin.resolve_plant(config)


def study(
    source,
    *,
    condition=None,
    controller="pid",
    robustness_samples: int | None = None,
    seed: int | None = None,
    output: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    plant = load_plant(source)
    plugin = get_scenario(plant.scenario)
    resolved = plugin.resolve_plant(plant)
    provider = plugin.study_provider
    if provider is None:
        raise ValueError(f"scenario {plant.scenario!r} does not provide design studies")
    checks = list(provider.checks(resolved))
    default_samples, default_seed = provider.default_robustness(resolved)
    sample_count = _nonnegative_int(
        "robustness_samples",
        default_samples if robustness_samples is None else robustness_samples,
    )
    resolved_seed = _nonnegative_int("seed", default_seed if seed is None else seed)
    identity_env = make_env(
        f"{plant.scenario}/regulation", plant=plant, condition=condition
    )
    identity = identity_env.identity
    identity_env.close()
    dynamic = _dynamic_check(
        plant,
        provider,
        condition=condition,
        controller=controller,
        disturbances={},
    )
    checks.append(dynamic)
    robustness = _robustness_check(
        plant,
        resolved,
        provider,
        condition=condition,
        controller=controller,
        samples=sample_count,
        seed=resolved_seed,
    )
    checks.append(robustness)
    passed = all(check.passed for check in checks)
    result = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "workflow": "design",
        "plant_id": plant.id,
        "plant_hash": plant.plant_hash,
        "condition_id": identity.condition_id,
        "condition_hash": identity.condition_hash,
        "study_hash": plant.study_hash,
        "interface_hash": identity.interface_hash,
        "env_hash": identity.env_hash,
        "scenario": plant.scenario,
        "task_id": f"{plant.scenario}/regulation",
        "task_hash": get_scenario(plant.scenario).tasks["regulation"].task_hash,
        "seed": resolved_seed,
        "robustness_samples": sample_count,
        "verdict": "PASS" if passed else "FAIL",
        "evidence_status": "simulation-screening-only",
        "limitations": [
            "This is a simulation screening result, not a safety certification.",
            "Installed equipment and protection settings require field commissioning.",
        ],
        "plant_config": plant.as_dict(),
        "checks": [_check_dict(check) for check in checks],
        "dynamic": dict(dynamic.metrics),
        "robustness": dict(robustness.metrics),
    }
    if output is not None:
        result["artifacts"] = write_run_bundle(
            result,
            output,
            workflow="design",
            report_markdown=render_study_report(result),
            overwrite=overwrite,
        )
    return result


def sweep(
    source,
    *,
    parameter: str,
    values: Sequence[float],
    robustness_samples: int | None = None,
    seed: int | None = None,
    output: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not parameter or not isinstance(parameter, str):
        raise ValueError("sweep parameter must be a non-empty dotted path")
    if isinstance(values, (str, bytes)) or not values:
        raise ValueError("sweep values must be a non-empty numeric sequence")
    base = load_plant(source).as_dict(include_hash=False)
    candidates = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"sweep value {index} must be numeric")
        candidate = copy.deepcopy(base)
        _set_dotted(candidate, parameter, float(value))
        candidate["id"] = f"{base['id']}-{_slug(parameter)}-{index + 1}"
        result = study(
            candidate,
            robustness_samples=robustness_samples,
            seed=seed,
        )
        candidates.append(
            {
                "value": float(value),
                "plant_id": result["plant_id"],
                "plant_hash": result["plant_hash"],
                "verdict": result["verdict"],
                "result": result,
            }
        )
    result = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "workflow": "design-sweep",
        "base_plant_id": base["id"],
        "parameter": parameter,
        "values": [float(value) for value in values],
        "candidates": candidates,
    }
    if output is not None:
        result["artifacts"] = write_run_bundle(
            result,
            output,
            workflow="design-sweep",
            report_markdown=render_sweep_report(result),
            overwrite=overwrite,
        )
    return result


def _dynamic_check(
    plant, provider, *, condition, controller, disturbances
):
    env = make_env(
        f"{plant.scenario}/regulation", plant=plant, condition=condition
    )
    env.set_disturbances(disturbances)
    policy = make_controller(controller, env=env)
    requirements = provider.dynamic_requirements(env.plant)
    limits = requirements["requirements"]
    try:
        episode = rollout(env, policy, seed=0)
    finally:
        env.close()
    target_levels = np.asarray(env.condition.reference[:3], dtype=float)
    target_temperatures = np.asarray(env.condition.reference[3:], dtype=float)
    tolerance_h = float(limits["level_tolerance_m"])
    tolerance_t = float(limits["temperature_tolerance_degC"])
    heatup_time = None
    energy_kwh = 0.0
    maximum_overshoot = 0.0
    maximum_level_error = 0.0
    maximum_action = 0.0
    protection_events: set[str] = set()
    hard_reasons: set[str] = set()
    for transition in episode.transitions:
        output = np.asarray(transition.info["y"], dtype=float)
        levels = output[:3]
        temperatures = output[3:6]
        maximum_overshoot = max(
            maximum_overshoot,
            float(np.max(np.maximum(temperatures - target_temperatures, 0.0))),
        )
        maximum_level_error = max(
            maximum_level_error, float(np.max(np.abs(levels - target_levels)))
        )
        maximum_action = max(maximum_action, float(np.max(transition.action)))
        energy_kwh += (
            env.model.action_energy_kw(
                transition.action,
                transition.info["true_state"],
                disturbances,
            )
            * env.control_dt
            / 3600.0
        )
        protection_events.update(transition.info.get("protection_events", ()))
        hard_reasons.update(transition.info.get("constraint_costs", {}))
        if heatup_time is None and np.all(
            np.abs(levels - target_levels) <= tolerance_h
        ) and np.all(np.abs(temperatures - target_temperatures) <= tolerance_t):
            heatup_time = transition.physical_time
    final = np.asarray(episode.transitions[-1].info["y"], dtype=float)
    final_level_error = float(np.max(np.abs(final[:3] - target_levels)))
    final_temperature_error = float(
        np.max(np.abs(final[3:6] - target_temperatures))
    )
    reasons = []
    if hard_reasons:
        reasons.append("hard process limit reached: " + ", ".join(sorted(hard_reasons)))
    if heatup_time is None or heatup_time > limits["maximum_heatup_time_s"]:
        reasons.append("heat-up requirement was not met")
    if maximum_overshoot > limits["maximum_overshoot_degC"]:
        reasons.append("temperature overshoot exceeds requirement")
    if energy_kwh > limits["maximum_energy_kwh"]:
        reasons.append("energy consumption exceeds requirement")
    if final_temperature_error > tolerance_t or final_level_error > tolerance_h:
        reasons.append("final state is outside the target tolerance")
    metrics = {
        "passed": not reasons,
        "reasons": reasons,
        "controller": policy.metadata(),
        "rollout_executor": "aiogym.core.rollout",
        "steps": len(episode.transitions),
        "duration_s": len(episode.transitions) * env.control_dt,
        "heatup_time_s": heatup_time,
        "energy_kwh": energy_kwh,
        "maximum_temperature_overshoot_degC": maximum_overshoot,
        "maximum_level_error_m": maximum_level_error,
        "maximum_action_command": maximum_action,
        "final_observation": final.tolist(),
        "final_action": episode.transitions[-1].action.tolist(),
        "final_temperature_error_degC": final_temperature_error,
        "final_level_error_m": final_level_error,
        "hard_termination_reasons": sorted(hard_reasons),
        "protection_events": sorted(protection_events),
        "disturbance": dict(disturbances),
    }
    return CheckResult(
        name="dynamic_commissioning",
        category="dynamic",
        passed=not reasons,
        summary="passed" if not reasons else "; ".join(reasons),
        metrics=metrics,
    )


def _robustness_check(
    plant, resolved, provider, *, condition, controller, samples, seed
):
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(samples):
        disturbances = provider.sample_disturbances(resolved, rng)
        steady = provider.steady_check(resolved, disturbances)
        dynamic = _dynamic_check(
            plant,
            provider,
            condition=condition,
            controller=controller,
            disturbances=disturbances,
        )
        passed = steady.passed and dynamic.passed
        cases.append(
            {
                "sample": index,
                "passed": passed,
                "disturbance": disturbances,
                "steady": _check_dict(steady),
                "dynamic": dict(dynamic.metrics),
            }
        )
    pass_count = sum(case["passed"] for case in cases)
    pass_rate = 1.0 if samples == 0 else pass_count / samples
    required = float(
        plant.study.get("requirements", {}).get("robustness_pass_rate", 0.0)
    )
    passed = pass_rate >= required
    metrics = {
        "passed": passed,
        "samples": samples,
        "seed": seed,
        "pass_count": pass_count,
        "pass_rate": pass_rate,
        "required_pass_rate": required,
        "cases": cases,
    }
    return CheckResult(
        name="robustness",
        category="robustness",
        passed=passed,
        summary=(
            "passed"
            if passed
            else f"pass rate {pass_rate:.3f} is below required {required:.3f}"
        ),
        metrics=metrics,
    )


def _check_dict(check: CheckResult) -> dict[str, Any]:
    return {
        "name": check.name,
        "category": check.category,
        "passed": check.passed,
        "summary": check.summary,
        "metrics": dict(check.metrics),
        "warnings": list(check.warnings),
    }


def render_study_report(result: Mapping[str, Any]) -> str:
    lines = [
        f"# Design study: {result['plant_id']}",
        "",
        f"Verdict: **{result['verdict']}**",
        "",
        f"Plant hash: `{result['plant_hash']}`",
        "",
        "## Checks",
        "",
    ]
    for check in result["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- {check['name']}: {status} — {check['summary']}")
    lines.extend(("", "## Limitations", ""))
    lines.extend(f"- {item}" for item in result["limitations"])
    return "\n".join(lines)


def render_sweep_report(result: Mapping[str, Any]) -> str:
    lines = [f"# Design sweep: {result['base_plant_id']}", ""]
    for candidate in result["candidates"]:
        lines.append(
            f"- {candidate['value']}: {candidate['verdict']} "
            f"(`{candidate['plant_hash']}`)"
        )
    return "\n".join(lines)


def _set_dotted(target, path, value):
    current = target
    parts = path.split(".")
    if not all(parts):
        raise ValueError("sweep parameter contains an empty path segment")
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ValueError(f"unknown sweep parameter path: {path}")
    final = parts[-1]
    if isinstance(current, list):
        current[int(final)] = value
    elif isinstance(current, dict) and final in current:
        current[final] = value
    else:
        raise ValueError(f"unknown sweep parameter path: {path}")


def _slug(value):
    return "-".join(part for part in value.replace("_", "-").split(".") if part)


def _nonnegative_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "STUDY_SCHEMA_VERSION",
    "SWEEP_SCHEMA_VERSION",
    "convert_design_spec_v1",
    "load_plant",
    "render_study_report",
    "study",
    "sweep",
    "validate_plant",
]
