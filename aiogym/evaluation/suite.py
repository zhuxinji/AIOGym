"""Benchmark-suite loading, case expansion, and result preparation."""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path

from .._internal.config import parse_seed_list
from .._internal.identifiers import (
    require_canonical_scenario_id,
    suite_catalog_text,
)
from .._internal.paths import (
    run_path,
    timestamped_artifact_path,
)
from ..catalog import list_scenarios, list_suites, list_tasks
from ..models.registry import make_model
from .cases import BenchmarkCase
from .protocols import resolve_protocol


CONFIG_DIR = Path(__file__).resolve().parent / "suites"
PRESET_DIR = CONFIG_DIR / "presets"


def scenario_groups() -> dict[str, tuple[str, ...]]:
    """Build scenario groups from the current runtime model registry."""

    scenarios = list_scenarios()
    economic = tuple(
        scenario
        for scenario in scenarios
        if "economic" in tuple(make_model(scenario).benchmark_objectives)
    )
    return {
        "ALL_SCENARIOS": scenarios,
        "ECONOMIC_SCENARIOS": economic,
    }


def _reject_removed_options(data: dict, context: str) -> None:
    removed = sorted(set(data) & {"dynamic", "reward_mode"})
    if removed:
        raise ValueError(f"unsupported {context} option(s): {', '.join(removed)}")


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge suite declarations, replacing lists and recursively merging mappings."""

    result = dict(base)
    for key, value in dict(override).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _reference_list(value, field: str) -> list[str]:
    refs = [value] if isinstance(value, str) else value
    if not isinstance(refs, list) or not refs:
        raise TypeError(f"suite {field} must be a non-empty string or list of strings")
    if any(not isinstance(ref, str) or not ref for ref in refs):
        raise TypeError(f"suite {field} must contain non-empty strings")
    return refs


def _load_reuse_registry() -> tuple[dict, dict]:
    presets = {}
    cases = {}
    for path in sorted(PRESET_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise TypeError(f"suite preset file must contain a mapping: {path}")
        unknown = set(data) - {"presets", "cases"}
        if unknown:
            raise ValueError(
                f"unknown suite preset sections in {path.name}: "
                f"{', '.join(sorted(unknown))}"
            )
        for section, target in (("presets", presets), ("cases", cases)):
            entries = data.get(section, {})
            if not isinstance(entries, dict):
                raise TypeError(f"suite preset {section} must be a mapping: {path}")
            for name, declaration in entries.items():
                if not isinstance(name, str) or not name or not isinstance(declaration, dict):
                    raise TypeError(
                        f"suite preset {section} entries must map non-empty IDs to mappings"
                    )
                if name in target:
                    raise ValueError(f"duplicate suite {section[:-1]} ID: {name!r}")
                target[name] = declaration
    return presets, cases


def _unknown_reference(kind: str, name: str, registry: dict) -> ValueError:
    available = ", ".join(sorted(registry)) if registry else "none"
    return ValueError(
        f"unknown suite {kind} ID {name!r}; available {kind} IDs: {available}"
    )


def _apply_preset(
    declaration: dict,
    presets: dict,
    *,
    stack: tuple[str, ...] = (),
) -> dict:
    current = dict(declaration)
    raw_refs = current.pop("preset", None)
    if raw_refs is None:
        return current
    resolved = {}
    for name in _reference_list(raw_refs, "preset"):
        if name in stack:
            cycle = " -> ".join((*stack, name))
            raise ValueError(f"cyclic suite preset reference: {cycle}")
        if name not in presets:
            raise _unknown_reference("preset", name, presets)
        fragment = _apply_preset(
            presets[name], presets, stack=(*stack, name)
        )
        resolved = _deep_merge(resolved, fragment)
    return _deep_merge(resolved, current)


def _resolve_case_reference(
    declaration: dict,
    presets: dict,
    case_refs: dict,
    *,
    stack: tuple[str, ...] = (),
) -> dict:
    current = dict(declaration)
    reference = current.pop("case_ref", None)
    current = _apply_preset(current, presets)
    if isinstance(current.get("environment"), dict):
        current["environment"] = _apply_preset(
            current["environment"], presets
        )
    if reference is not None:
        if not isinstance(reference, str) or not reference:
            raise TypeError("suite case_ref must be a non-empty string")
        if reference in stack:
            cycle = " -> ".join((*stack, reference))
            raise ValueError(f"cyclic suite case reference: {cycle}")
        if reference not in case_refs:
            raise _unknown_reference("case", reference, case_refs)
        base = _resolve_case_reference(
            case_refs[reference],
            presets,
            case_refs,
            stack=(*stack, reference),
        )
        current = _deep_merge(base, current)
    return current


def _resolve_suite_path(reference: str, *, relative_to: Path | None = None) -> Path:
    path = Path(reference)
    if os.path.sep not in reference and not path.suffix:
        available_suites = list_suites()
        path = CONFIG_DIR / f"{reference}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"unknown suite ID {reference!r}; available suite IDs: "
                f"{suite_catalog_text(available_suites)}"
            )
    elif not path.is_absolute() and relative_to is not None:
        path = relative_to.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"suite declaration not found: {reference}")
    return path.resolve()


def _load_suite_declaration(path: Path, *, stack: tuple[Path, ...] = ()) -> dict:
    resolved_path = path.resolve()
    if resolved_path in stack:
        cycle = " -> ".join(item.name for item in (*stack, resolved_path))
        raise ValueError(f"cyclic suite extends reference: {cycle}")
    with resolved_path.open(encoding="utf-8") as stream:
        declaration = json.load(stream)
    if not isinstance(declaration, dict):
        raise TypeError(f"suite declaration must be a mapping: {resolved_path}")
    parents = declaration.pop("extends", None)
    merged = {}
    if parents is not None:
        for reference in _reference_list(parents, "extends"):
            parent_path = _resolve_suite_path(reference, relative_to=resolved_path)
            parent = _load_suite_declaration(
                parent_path, stack=(*stack, resolved_path)
            )
            merged = _deep_merge(merged, parent)
    return _deep_merge(merged, declaration)


def load_suite(name_or_path: str):
    path = _resolve_suite_path(name_or_path)
    suite = _load_suite_declaration(path)
    presets, case_refs = _load_reuse_registry()
    suite = _apply_preset(suite, presets)
    if isinstance(suite.get("environment"), dict):
        suite["environment"] = _apply_preset(suite["environment"], presets)
    if isinstance(suite.get("cases"), list):
        suite["cases"] = [
            _resolve_case_reference(case, presets, case_refs)
            if isinstance(case, dict) else case
            for case in suite["cases"]
        ]
    _reject_removed_options(suite, "suite")
    if isinstance(suite.get("environment"), dict):
        _reject_removed_options(suite["environment"], "suite environment")
    if isinstance(suite.get("cases"), list):
        for case in suite["cases"]:
            if not isinstance(case, dict):
                continue
            _reject_removed_options(case, "suite case")
            if isinstance(case.get("environment"), dict):
                _reject_removed_options(case["environment"], "suite case environment")
    suite.setdefault("name", path.stem)
    suite["scenarios"] = expand_scenarios(suite.get("scenarios", ()))
    suite["objectives"] = list(suite.get("objectives", ()))
    suite["controllers"] = list(suite.get("controllers", ()))
    task_selector = suite.pop("tasks", None)
    if task_selector is not None:
        if "cases" in suite:
            raise ValueError("suite tasks and cases cannot be used together")
        suite["cases"] = _task_cases(suite["scenarios"], task_selector)
    if "cases" in suite:
        if not isinstance(suite["cases"], list) or not suite["cases"]:
            raise ValueError("suite cases must be a non-empty list")
        if any(not isinstance(case, dict) for case in suite["cases"]):
            raise TypeError("each suite case must be a mapping")
    suite.setdefault("action_mode", "actuator")
    suite.setdefault("description", "")
    if "cases" not in suite and suite.get("task") is None:
        suite.setdefault("episode_steps", 80)
        suite.setdefault("control_dt", 0.5)
    return suite


def expand_scenarios(value):
    groups = scenario_groups()
    if isinstance(value, str):
        if value in groups:
            return list(groups[value])
        require_canonical_scenario_id(value)
        return [value]
    out = []
    for item in value:
        if item in groups:
            out.extend(groups[item])
        else:
            require_canonical_scenario_id(item)
            out.append(item)
    return out


def _task_cases(scenarios: list[str], selector) -> list[dict]:
    """Expand a task selector from the model-owned task registry."""

    if selector == "ALL_TASKS":
        include = None
        exclude = set()
    elif isinstance(selector, Mapping):
        unknown = set(selector) - {"include", "exclude"}
        if unknown:
            raise ValueError(
                "unknown suite task selector fields: "
                + ", ".join(sorted(unknown))
            )
        include = _task_name_set(selector.get("include"), "include")
        exclude = _task_name_set(selector.get("exclude"), "exclude") or set()
    else:
        raise TypeError(
            "suite tasks must be 'ALL_TASKS' or a mapping with include/exclude"
        )

    cases = []
    available_names = set()
    for scenario in scenarios:
        for task_id in list_tasks(scenario):
            task_name = task_id.split("/", 1)[1]
            available_names.add(task_name)
            if include is not None and task_name not in include:
                continue
            if task_name in exclude:
                continue
            cases.append({"scenario": scenario, "task": task_name})
    if include is not None:
        missing = include - available_names
        if missing:
            raise ValueError(
                "suite task selector includes unknown tasks: "
                + ", ".join(sorted(missing))
            )
    if not cases:
        raise ValueError("suite task selector selected no tasks")
    return cases


def _task_name_set(value, field: str) -> set[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise TypeError(f"suite task selector {field} must be a list of task names")
    return set(value)


def parse_csv(raw: str | None, default):
    if raw is None:
        return list(default)
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("comma-separated options must contain at least one value")
    return values


def build_cases(args, *, load_suite_fn=None):
    """Resolve one suite declaration into canonical benchmark cases."""

    loader = load_suite_fn or load_suite
    suite = loader(args.suite)
    explicit_cases = bool(suite.get("cases"))
    scenario_filter = (
        set(expand_scenarios(parse_csv(args.scenarios, suite["scenarios"])))
        if explicit_cases and args.scenarios is not None else None
    )
    explicit_objectives = (
        parse_csv(args.objectives, ()) if args.objectives is not None else None
    )
    controller_filter = (
        set(parse_csv(args.controllers, suite["controllers"]))
        if explicit_cases and args.controllers is not None else None
    )
    seeds = parse_seed_list(args.seed_list, args.seed, args.episodes)
    cases = []
    declarations = suite.get("cases") or [{
        "scenarios": parse_csv(args.scenarios, suite["scenarios"]),
        "controllers": parse_csv(args.controllers, suite["controllers"]),
    }]
    for declaration in declarations:
        scenarios = expand_scenarios(
            declaration.get("scenarios", declaration.get("scenario", suite["scenarios"]))
        )
        if explicit_objectives is not None:
            objective_candidates = [("explicit", value) for value in explicit_objectives]
        elif "objectives" in declaration:
            objective_candidates = [
                ("case-config", value) for value in declaration["objectives"]
            ]
        elif "objective" in declaration:
            objective_candidates = [("case-config", declaration["objective"])]
        elif suite["objectives"]:
            objective_candidates = [
                ("suite-config", value) for value in suite["objectives"]
            ]
        else:
            objective_candidates = [("task-default", None)]
        controllers = list(
            declaration.get(
                "controllers",
                [declaration["controller"]]
                if "controller" in declaration else suite["controllers"],
            )
        )
        for scenario in scenarios:
            if scenario_filter is not None and scenario not in scenario_filter:
                continue
            for objective_source, objective_value in objective_candidates:
                action_mode = declaration.get("action_mode", suite["action_mode"])
                protocol_config = {"action_mode": action_mode}
                declaration_environment = dict(declaration.get("environment", {}))
                suite_environment = dict(suite.get("environment", {}))
                task = declaration.get("task", suite.get("task"))
                if task is not None:
                    protocol_config["task"] = task
                for key in (
                    "auto_events", "randomize", "randomize_setpoints",
                    "randomize_plant", "plant_drift", "integral_obs",
                    "disturbance_obs", "previous_action_obs",
                    "normalize_observations", "tracking_error_obs",
                    "terminate_on_runaway", "noise", "noise_pct",
                    "tracking_q_y", "tracking_r_move",
                    "initial_setpoint", "setpoint_schedule",
                    "model_params",
                ):
                    if key in declaration:
                        protocol_config[key] = declaration[key]
                    elif key in declaration_environment:
                        protocol_config[key] = declaration_environment[key]
                    elif key in suite:
                        protocol_config[key] = suite[key]
                    elif key in suite_environment:
                        protocol_config[key] = suite_environment[key]
                cli_overrides = {
                    "tracking_q_y": getattr(args, "tracking_q_y", None),
                    "tracking_r_move": getattr(args, "tracking_r_move", None),
                    "disturbance_obs": getattr(args, "disturbance_obs", None),
                    "previous_action_obs": getattr(
                        args, "previous_action_obs", None
                    ),
                    "normalize_observations": getattr(
                        args, "normalize_observations", None
                    ),
                    "tracking_error_obs": getattr(
                        args, "tracking_error_obs", None
                    ),
                    "initial_setpoint": getattr(
                        args, "parsed_initial_setpoint", None
                    ),
                    "setpoint_schedule": getattr(
                        args, "parsed_setpoint_schedule", None
                    ),
                }
                protocol_config.update({
                    key: value
                    for key, value in cli_overrides.items()
                    if value is not None
                })
                randomize_setpoints = getattr(args, "randomize_setpoints", None)
                if randomize_setpoints is not None:
                    protocol_config["randomize_setpoints"] = bool(
                        randomize_setpoints
                    )
                if args.episode_steps is not None:
                    protocol_config["episode_steps"] = int(args.episode_steps)
                elif "episode_steps" in declaration:
                    protocol_config["episode_steps"] = int(declaration["episode_steps"])
                elif "episode_steps" in suite:
                    protocol_config["episode_steps"] = int(suite["episode_steps"])
                if args.control_dt is not None:
                    protocol_config["control_dt"] = float(args.control_dt)
                elif "control_dt" in declaration:
                    protocol_config["control_dt"] = float(declaration["control_dt"])
                elif "control_dt" in suite:
                    protocol_config["control_dt"] = float(suite["control_dt"])
                protocol = resolve_protocol(
                    scenario,
                    objective_value if objective_source == "explicit" else None,
                    protocol_config,
                    case_objective=(
                        objective_value if objective_source == "case-config" else None
                    ),
                    suite_objective=(
                        objective_value if objective_source == "suite-config" else None
                    ),
                )
                task_meta = protocol.metadata()["task_identity"]
                objective = protocol.objective
                for controller in controllers:
                    if controller_filter is not None and controller not in controller_filter:
                        continue
                    controller_config = controller_config_for(
                        args, controller, action_mode, objective
                    )
                    controller_config = _merge_config(
                        controller_config,
                        suite.get("controller_configs", {}).get(controller, {}),
                    )
                    controller_config = _merge_config(
                        controller_config,
                        declaration.get("controller_configs", {}).get(controller, {}),
                    )
                    case_name = (
                        f"{objective}:{scenario}:{task_meta['name']}:{controller}"
                    )
                    case_spec = BenchmarkCase.from_protocol(
                        protocol,
                        controller=controller,
                        seeds=seeds,
                        controller_config=controller_config,
                        case_id=case_name,
                    )
                    cases.append({
                        "name": case_name,
                        "scenario": scenario,
                        "task": task_meta["name"],
                        "task_status": task_meta["status"],
                        "task_profile_hash": task_meta["profile_hash"],
                        "objective": objective,
                        "objective_source": protocol.objective_source,
                        "controller": controller,
                        "action_mode": action_mode,
                        "controller_config": dict(case_spec.controller_config),
                        "protocol": protocol,
                        "case_spec": case_spec,
                        "seeds": seeds,
                    })
    if not cases:
        raise ValueError("suite filters selected no benchmark cases")
    return suite, cases


def _merge_config(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in dict(override or {}).items():
        if key == "parameters" and isinstance(value, dict):
            out[key] = {**dict(out.get(key) or {}), **value}
        else:
            out[key] = value
    return out


def controller_config_for(
    args,
    controller: str,
    action_mode: str,
    objective: str | None = None,
):
    if controller == "sb3":
        if not args.sb3_path:
            raise SystemExit("controller 'sb3' requires --sb3-path")
        return {
            "path": args.sb3_path,
            "algo": args.sb3_algo,
            "action_mode": getattr(args, "sb3_action_mode", None) or action_mode,
        }
    if controller == "onnx":
        if not args.onnx_path:
            raise SystemExit("controller 'onnx' requires --onnx-path")
        return {
            "path": args.onnx_path,
            "action_mode": (
                getattr(args, "onnx_action_mode", None) or action_mode
            ),
        }
    config = {}
    profile = getattr(args, "controller_profile", None)
    if controller == "oracle":
        profile = getattr(args, "oracle_profile", None) or profile
    if profile:
        config["profile"] = profile
    if controller == "mpc" and getattr(args, "mpc_horizon", None) is not None:
        config["P"] = args.mpc_horizon
    return config


SUMMARY_COLUMNS = (
    "suite_case", "scenario", "task", "task_status", "task_profile_hash",
    "objective", "objective_source", "objective_status", "action_mode",
    "controller", "control_structure", "execution_status", "metric",
    "metric_mean", "metric_std", "normalized_score", "profit", "production",
    "return", "track", "tracking_cost", "tracking_return",
    "tracking_error_cost", "tracking_move_cost",
    "tracking_mse", "tracking_iae", "energy_kwh", "constraint",
    "constraint_violation_count", "constraint_violation_severity",
    "safety_margin_min", "runtime_seconds_per_step",
    "controller_fallback_count", "controller_solver_failure_count", "episodes",
    "seed_list",
)


def build_summary_table(rows: list[dict]):
    table = []
    for row in rows:
        metric = row.get("metric")
        out = {key: row.get(key) for key in SUMMARY_COLUMNS}
        out["metric_mean"] = row.get(metric) if metric else None
        out["metric_std"] = row.get(f"{metric}_std") if metric else None
        table.append(out)
    return table


def artifact_dir_for(
    suite_name: str,
    artifact_dir: str | None = None,
    overwrite: bool = True,
    run_id: str | None = None,
):
    if artifact_dir is not None and not str(artifact_dir).strip():
        raise ValueError("--artifact-dir must not be empty")
    safe_suite = re.sub(r"[^A-Za-z0-9_.-]+", "-", suite_name).strip("-") or "suite"
    base = Path(artifact_dir) if artifact_dir is not None else run_path(
        f"{safe_suite}_suite"
    )
    if overwrite:
        return str(base)
    return str(timestamped_artifact_path(base, run_id))


def effective_suite_config(
    suite: dict,
    cases: list[dict],
    episode_steps: int | None,
    control_dt: float | None,
):
    config = dict(suite)
    config["scenarios"] = list(dict.fromkeys(case["scenario"] for case in cases))
    config["objectives"] = list(dict.fromkeys(case["objective"] for case in cases))
    config["controllers"] = list(dict.fromkeys(case["controller"] for case in cases))
    config["tasks"] = list(
        dict.fromkeys(case.get("task", "default") for case in cases)
    )
    config["cases"] = [
        {
            "name": case.get(
                "name",
                f"{case['objective']}:{case['scenario']}:"
                f"{case.get('task', 'default')}:{case['controller']}",
            ),
            "scenario": case["scenario"],
            "task": case.get("task", "default"),
            "objective": case["objective"],
            "objective_source": case.get("objective_source"),
            "controller": case["controller"],
            "episode_steps": (
                case["protocol"].episode_steps
                if case.get("protocol") else episode_steps
            ),
            "control_dt": (
                case["protocol"].control_dt
                if case.get("protocol") else control_dt
            ),
        }
        for case in cases
    ]
    resolved_steps = list(dict.fromkeys(
        case["protocol"].episode_steps for case in cases if case.get("protocol")
    ))
    resolved_dt = list(dict.fromkeys(
        case["protocol"].control_dt for case in cases if case.get("protocol")
    ))
    config["episode_steps"] = (
        episode_steps
        if episode_steps is not None
        else (resolved_steps[0] if len(resolved_steps) == 1 else None)
    )
    config["control_dt"] = (
        control_dt
        if control_dt is not None
        else (resolved_dt[0] if len(resolved_dt) == 1 else None)
    )
    return config
