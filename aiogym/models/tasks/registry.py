"""Loading and runtime resolution for model-bound task specifications."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

from aiogym._internal.identifiers import (
    canonical_scenario_id,
    canonical_task_id,
    internal_scenario_id,
    require_canonical_scenario_id,
)

from .schema import (
    ENVIRONMENT_BOOLEAN_FIELDS,
    TASK_PROFILE_SCHEMA_VERSION,
    validate_task_profile,
)


_TASK_DIR = Path(__file__).with_name("builtin")


def list_tasks(scenario: str | None = None) -> tuple[str, ...]:
    """List bundled task identifiers as ``scenario/name`` strings."""

    if scenario:
        require_canonical_scenario_id(scenario)
    storage_scenario = internal_scenario_id(scenario) if scenario else None
    paths = (
        (_TASK_DIR / storage_scenario).glob("*.json")
        if storage_scenario else _TASK_DIR.glob("*/*.json")
    )
    return tuple(sorted(
        canonical_task_id(f"{path.parent.name}/{path.stem}")
        for path in paths
    ))


def load_task_profile(
    source: str | Path | Mapping[str, Any],
    *,
    scenario: str | None = None,
) -> dict[str, Any]:
    """Load and validate a bundled, file-based, or in-memory task profile."""

    if isinstance(source, Mapping):
        data = copy.deepcopy(dict(source))
    else:
        path = Path(source)
        named_scenario = None
        named_task = False
        if isinstance(source, str) and not path.exists():
            parts = source.split("/", 1)
            if len(parts) == 2 and all(parts) and "/" not in parts[1] and not path.suffix:
                task_scenario, task_name = parts
                require_canonical_scenario_id(task_scenario)
                storage_scenario = internal_scenario_id(task_scenario)
                path = _TASK_DIR / storage_scenario / f"{task_name}.json"
                named_scenario = canonical_scenario_id(storage_scenario)
                named_task = True
            elif scenario and "/" not in source and not path.suffix:
                require_canonical_scenario_id(scenario)
                storage_scenario = internal_scenario_id(scenario)
                path = _TASK_DIR / storage_scenario / f"{source}.json"
                named_scenario = canonical_scenario_id(storage_scenario)
                named_task = True
        if not path.is_file():
            if named_task:
                available = list_tasks(named_scenario)
                available_text = ", ".join(available) if available else "none"
                raise FileNotFoundError(
                    f"unknown task ID {source!r}; available task IDs for "
                    f"scenario {named_scenario!r}: {available_text}"
                )
            raise FileNotFoundError(f"task profile not found: {source}")
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    validate_task_profile(data, expected_scenario=scenario)
    return data


def task_environment(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the executable environment conditions owned by a task."""

    validate_task_profile(profile)
    return copy.deepcopy(profile["environment"])


def task_objective_options(
    profile: Mapping[str, Any] | None,
    objective: str,
) -> dict[str, Any]:
    """Return objective options owned by a task, if declared."""

    if profile is None:
        return {}
    validate_task_profile(profile)
    return copy.deepcopy(profile.get("objectives", {}).get(objective, {}))


def task_controller_config(
    profile: Mapping[str, Any] | None,
    controller: str,
) -> dict[str, Any]:
    """Return task-owned defaults for one controller."""

    if profile is None:
        return {}
    validate_task_profile(profile)
    return copy.deepcopy(profile.get("controllers", {}).get(controller, {}))


def resolve_environment_options(
    *,
    scenario: str,
    task: str | Path | Mapping[str, Any] | None,
    explicit: Mapping[str, Any],
    defaults: Mapping[str, Any],
    default_control_dt: float,
    default_episode_steps: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Resolve and validate task-owned environment settings in one place."""

    profile = (
        load_task_profile(task, scenario=scenario)
        if task is not None
        else None
    )
    task_defaults = task_environment(profile) if profile is not None else {}
    provided = dict(explicit)

    resolved = {}
    for name in ("action_mode", *ENVIRONMENT_BOOLEAN_FIELDS):
        value = provided.get(name)
        resolved[name] = (
            value
            if value is not None
            else task_defaults.get(name, defaults[name])
        )

    control_dt = provided.get("control_dt")
    if control_dt is None:
        control_dt = task_defaults.get("control_dt", default_control_dt)
    if isinstance(control_dt, bool):
        raise TypeError("control_dt must be numeric")
    control_dt = float(control_dt)
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")

    episode_steps = provided.get("episode_steps")
    if episode_steps is None:
        episode_steps = task_defaults.get("episode_steps", default_episode_steps)
    if (
        isinstance(episode_steps, bool)
        or not isinstance(episode_steps, Integral)
        or int(episode_steps) <= 0
    ):
        raise ValueError("episode_steps must be a positive integer")

    noise_pct = provided.get("noise_pct")
    if noise_pct is None:
        noise_pct = task_defaults.get("noise_pct", defaults["noise_pct"])
    if isinstance(noise_pct, bool):
        raise TypeError("noise_pct must be numeric")
    noise_pct = float(noise_pct)
    if not math.isfinite(noise_pct) or noise_pct < 0:
        raise ValueError("noise_pct must be finite and non-negative")

    if resolved["action_mode"] not in {"actuator", "setpoint"}:
        raise ValueError("action_mode must be one of: actuator, setpoint")
    for name in ENVIRONMENT_BOOLEAN_FIELDS:
        if not isinstance(resolved[name], bool):
            raise TypeError(f"{name} must be a boolean")

    explicit_model_params = provided.get("model_params")
    if explicit_model_params is not None and not isinstance(explicit_model_params, Mapping):
        raise TypeError("model_params must be a mapping")
    model_params = dict((profile or {}).get("model_params", {}))
    model_params.update(dict(explicit_model_params or {}))

    resolved.update({
        "control_dt": control_dt,
        "episode_steps": int(episode_steps),
        "noise_pct": noise_pct,
        "model_params": model_params,
    })
    return profile, resolved


def task_operation(profile: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a normalized throughput declaration, if present."""

    validate_task_profile(profile)
    operation = profile.get("operation")
    if operation is None:
        return None
    product_flow_sp = float(operation["product_flow_sp"])
    min_product_flow = float(
        operation.get("min_product_flow", product_flow_sp)
    )
    return {
        "product_flow_sp": product_flow_sp,
        "min_product_flow": min_product_flow,
    }


def configure_model_for_task(model, profile: Mapping[str, Any] | None):
    """Apply task context that affects model economics without changing ``p``."""

    if profile is None:
        return model
    operation = task_operation(profile)
    if operation is None:
        return model
    configure = getattr(model, "configure_operation", None)
    if not callable(configure):
        raise ValueError(
            f"task {profile['name']!r} declares operation settings, but model "
            f"{model.scenario!r} does not support them"
        )
    configure(operation)
    return model


def task_identity(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the stable task identity recorded in benchmark rows and artifacts."""

    if profile is None:
        return {
            "name": "default",
            "status": "implicit-default",
            "schema_version": TASK_PROFILE_SCHEMA_VERSION,
            "profile_hash": None,
        }
    validate_task_profile(profile)
    canonical_profile = copy.deepcopy(dict(profile))
    canonical = json.dumps(
        canonical_profile,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return {
        "name": str(profile["name"]),
        "status": str(profile["status"]),
        "schema_version": str(profile["schema_version"]),
        "profile_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


__all__ = [
    "configure_model_for_task",
    "list_tasks",
    "load_task_profile",
    "resolve_environment_options",
    "task_controller_config",
    "task_environment",
    "task_identity",
    "task_objective_options",
    "task_operation",
]
