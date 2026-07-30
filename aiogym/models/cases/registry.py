"""Loading and runtime resolution for Case v2 profiles."""
from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

from aiogym._internal.identifiers import (
    canonical_scenario_id,
    canonical_case_id,
    internal_scenario_id,
    require_canonical_scenario_id,
)

from .schema import (
    CASE_PROFILE_SCHEMA_VERSION,
    ENVIRONMENT_BOOLEAN_FIELDS,
    apply_case_overrides,
    case_profile_hash,
    validate_case_profile,
)


_CASE_DIR = Path(__file__).with_name("builtin")
def _list_builtin_profiles(scenario: str | None = None) -> tuple[str, ...]:
    if scenario:
        require_canonical_scenario_id(scenario)
    storage_scenario = internal_scenario_id(scenario) if scenario else None
    paths = (
        (_CASE_DIR / storage_scenario).glob("*.json")
        if storage_scenario
        else _CASE_DIR.glob("*/*.json")
    )
    return tuple(
        sorted(
            canonical_case_id(f"{path.parent.name}/{path.stem}")
            for path in paths
        )
    )


def list_cases(scenario: str | None = None) -> tuple[str, ...]:
    """List bundled Case identifiers as ``scenario/name`` strings."""

    return _list_builtin_profiles(scenario)


def _load_profile_source(
    source: str | Path | Mapping[str, Any],
    *,
    scenario: str | None,
    noun: str,
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))

    path = Path(source)
    named_scenario = None
    named_profile = False
    if isinstance(source, str) and not path.exists():
        parts = source.split("/", 1)
        if len(parts) == 2 and all(parts) and "/" not in parts[1] and not path.suffix:
            profile_scenario, profile_name = parts
            require_canonical_scenario_id(profile_scenario)
            storage_scenario = internal_scenario_id(profile_scenario)
            path = _CASE_DIR / storage_scenario / f"{profile_name}.json"
            named_scenario = canonical_scenario_id(storage_scenario)
            named_profile = True
        elif scenario and "/" not in source and not path.suffix:
            require_canonical_scenario_id(scenario)
            storage_scenario = internal_scenario_id(scenario)
            path = _CASE_DIR / storage_scenario / f"{source}.json"
            named_scenario = canonical_scenario_id(storage_scenario)
            named_profile = True
    if not path.is_file():
        if named_profile:
            available = _list_builtin_profiles(named_scenario)
            available_text = ", ".join(available) if available else "none"
            raise FileNotFoundError(
                f"unknown {noun} ID {source!r}; available {noun} IDs for "
                f"scenario {named_scenario!r}: {available_text}"
            )
        raise FileNotFoundError(f"{noun} profile not found: {source}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_case(
    source: str | Path | Mapping[str, Any],
    *,
    scenario: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate a Case v2 profile."""

    data = _load_profile_source(source, scenario=scenario, noun="case")
    schema_version = data.get("schema_version")
    if schema_version != CASE_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"unsupported case profile schema: {schema_version!r}")
    validate_case_profile(data, expected_scenario=scenario)
    return apply_case_overrides(data, overrides)


def case_environment(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the executable environment conditions owned by a Case."""

    validate_case_profile(profile)
    return copy.deepcopy(profile["environment"])


def case_controller_config(
    profile: Mapping[str, Any] | None,
    controller: str,
) -> dict[str, Any]:
    """Return specialist controller metadata retained during migration."""

    if profile is None:
        return {}
    validate_case_profile(profile)
    return copy.deepcopy(
        profile.get("specialist_metadata", {})
        .get("controllers", {})
        .get(controller, {})
    )


def resolve_environment_options(
    *,
    scenario: str,
    explicit: Mapping[str, Any],
    defaults: Mapping[str, Any],
    default_control_dt: float,
    default_episode_steps: int,
    case: str | Path | Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Resolve and validate Case-owned environment settings in one place."""

    if case is not None:
        profile = load_case(case, scenario=scenario)
    else:
        profile = None
    profile_defaults = case_environment(profile) if profile is not None else {}
    provided = dict(explicit)

    resolved = {}
    for name in ("action_mode", *ENVIRONMENT_BOOLEAN_FIELDS):
        value = provided.get(name)
        resolved[name] = (
            value
            if value is not None
            else profile_defaults.get(name, defaults[name])
        )
    observation_mode = provided.get("observation_mode")
    resolved["observation_mode"] = (
        observation_mode
        if observation_mode is not None
        else profile_defaults.get(
            "observation_mode",
            defaults["observation_mode"],
        )
    )

    control_dt = provided.get("control_dt")
    if control_dt is None:
        control_dt = profile_defaults.get("control_dt", default_control_dt)
    if isinstance(control_dt, bool):
        raise TypeError("control_dt must be numeric")
    control_dt = float(control_dt)
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")

    episode_steps = provided.get("episode_steps")
    if episode_steps is None:
        episode_steps = profile_defaults.get("episode_steps", default_episode_steps)
    if (
        isinstance(episode_steps, bool)
        or not isinstance(episode_steps, Integral)
        or int(episode_steps) <= 0
    ):
        raise ValueError("episode_steps must be a positive integer")

    noise_pct = provided.get("noise_pct")
    if noise_pct is None:
        noise_pct = profile_defaults.get("noise_pct", defaults["noise_pct"])
    if isinstance(noise_pct, bool):
        raise TypeError("noise_pct must be numeric")
    noise_pct = float(noise_pct)
    if not math.isfinite(noise_pct) or noise_pct < 0:
        raise ValueError("noise_pct must be finite and non-negative")

    if resolved["action_mode"] not in {"actuator", "setpoint"}:
        raise ValueError("action_mode must be one of: actuator, setpoint")
    if resolved["observation_mode"] not in {
        "full_state",
        "measured_output",
    }:
        raise ValueError(
            "observation_mode must be one of: full_state, measured_output"
        )
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


def case_operation(profile: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a normalized throughput declaration, if present."""

    validate_case_profile(profile)
    operation = profile.get("operation")
    if operation is None:
        return None
    product_flow_sp = float(operation["product_flow_sp"])
    min_product_flow = float(operation.get("min_product_flow", product_flow_sp))
    return {
        "product_flow_sp": product_flow_sp,
        "min_product_flow": min_product_flow,
    }


def configure_model_for_case(model, profile: Mapping[str, Any] | None):
    """Apply Case context that affects model economics without changing ``p``."""

    if profile is None:
        return model
    operation = case_operation(profile)
    if operation is None:
        return model
    configure = getattr(model, "configure_operation", None)
    if not callable(configure):
        raise ValueError(
            f"case {profile['name']!r} declares operation settings, but model "
            f"{model.scenario!r} does not support them"
        )
    configure(operation)
    return model


def case_identity(profile: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the stable identity recorded for a resolved Case."""

    if profile is None:
        return {
            "name": "default",
            "status": "implicit-default",
            "schema_version": CASE_PROFILE_SCHEMA_VERSION,
            "profile_hash": None,
        }
    validate_case_profile(profile)
    return {
        "name": str(profile["name"]),
        "status": str(profile.get("status", "unspecified")),
        "schema_version": str(profile["schema_version"]),
        "profile_hash": case_profile_hash(profile),
    }


__all__ = [
    "case_controller_config",
    "case_environment",
    "case_identity",
    "case_operation",
    "configure_model_for_case",
    "list_cases",
    "load_case",
    "resolve_environment_options",
]
