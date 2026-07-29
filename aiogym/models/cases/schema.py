"""Schema validation for model-bound AIO-Gym Case specifications."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

from aiogym._internal.identifiers import internal_scenario_id


CASE_PROFILE_SCHEMA_VERSION = "aiogym.case_profile.v2"
CASE_OVERRIDE_FIELDS = frozenset({
    "environment",
    "initialization",
    "setpoints",
    "disturbances",
    "constraints",
    "operation",
    "model_params",
    "acceptance",
    "evaluation",
})
CASE_FORBIDDEN_GOAL_FIELDS = frozenset({
    "goal",
    "default_goal",
    "reward_spec",
    "default_objective",
    "supported_objectives",
    "objectives",
})
CASE_FORBIDDEN_CONTROLLER_FIELDS = frozenset({"controllers"})
CASE_ENVIRONMENT_FIELDS = frozenset({
    "control_dt",
    "episode_steps",
    "action_mode",
    "auto_events",
    "randomize",
    "randomize_setpoints",
    "randomize_plant",
    "plant_drift",
    "integral_obs",
    "disturbance_obs",
    "previous_action_obs",
    "normalize_observations",
    "tracking_error_obs",
    "terminate_on_runaway",
    "noise",
    "noise_pct",
})
CASE_OPERATION_FIELDS = frozenset({"product_flow_sp", "min_product_flow"})
ENVIRONMENT_BOOLEAN_FIELDS = (
    "auto_events",
    "randomize",
    "randomize_setpoints",
    "randomize_plant",
    "plant_drift",
    "integral_obs",
    "disturbance_obs",
    "previous_action_obs",
    "normalize_observations",
    "tracking_error_obs",
    "terminate_on_runaway",
    "noise",
)


def _canonical_profile_json(profile: Mapping) -> str:
    return json.dumps(
        profile,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def case_profile_hash(profile: Mapping) -> str:
    """Return the stable SHA-256 identity of a resolved Case v2 profile."""

    validate_case_profile(profile)
    return hashlib.sha256(_canonical_profile_json(profile).encode("utf-8")).hexdigest()


@dataclass(frozen=True, init=False)
class CaseSpec:
    """Immutable validated snapshot of a resolved Case v2 profile."""

    _canonical_json: str

    def __init__(self, profile: Mapping) -> None:
        resolved = deepcopy(dict(profile))
        validate_case_profile(resolved)
        object.__setattr__(self, "_canonical_json", _canonical_profile_json(resolved))

    @classmethod
    def from_profile(cls, profile: Mapping) -> "CaseSpec":
        return cls(profile)

    @property
    def profile(self) -> dict:
        return json.loads(self._canonical_json)

    @property
    def name(self) -> str:
        return str(self.profile["name"])

    @property
    def scenario(self) -> str:
        return str(self.profile["scenario"])

    @property
    def profile_hash(self) -> str:
        return hashlib.sha256(self._canonical_json.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict:
        return self.profile


def _validate_case_contents(
    profile: Mapping,
    *,
    expected_scenario: str | None = None,
) -> None:
    """Validate Case identity, timing, and extension sections."""

    if not isinstance(profile, Mapping):
        raise TypeError("case profile must be a mapping")
    retired = sorted(set(profile) & CASE_FORBIDDEN_GOAL_FIELDS)
    if retired:
        raise ValueError(
            "Case profiles must not define Goal or RewardSpec fields; "
            "declare them in a Track: "
            + ", ".join(retired)
        )
    required = ("schema_version", "name", "scenario", "environment")
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"case profile is missing required fields: {', '.join(missing)}")
    if profile["schema_version"] != CASE_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"unsupported case profile schema: {profile['schema_version']!r}")
    for key in ("name", "scenario"):
        if not isinstance(profile[key], str) or not profile[key]:
            raise ValueError(f"case profile {key} must be a non-empty string")
    if "status" in profile and (
        not isinstance(profile["status"], str) or not profile["status"]
    ):
        raise ValueError("case profile status must be a non-empty string")
    if (
        expected_scenario is not None
        and internal_scenario_id(profile["scenario"])
        != internal_scenario_id(expected_scenario)
    ):
        raise ValueError(f"expected case for {expected_scenario!r}, got {profile['scenario']!r}")
    environment = profile["environment"]
    if not isinstance(environment, Mapping):
        raise TypeError("case profile environment must be a mapping")
    unknown = set(environment) - CASE_ENVIRONMENT_FIELDS
    if unknown:
        raise ValueError(f"unknown case environment fields: {', '.join(sorted(unknown))}")
    if "control_dt" in environment:
        value = float(environment["control_dt"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("case control_dt must be finite and positive")
    if "episode_steps" in environment:
        value = environment["episode_steps"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("case episode_steps must be a positive integer")
    if "action_mode" in environment and environment["action_mode"] not in {"actuator", "setpoint"}:
        raise ValueError("case action_mode must be one of: actuator, setpoint")
    for key in ENVIRONMENT_BOOLEAN_FIELDS:
        if key in environment and not isinstance(environment[key], bool):
            raise TypeError(f"case {key} must be a boolean")
    if "noise_pct" in environment:
        value = float(environment["noise_pct"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("case noise_pct must be finite and non-negative")
    if "model_params" in profile and not isinstance(profile["model_params"], Mapping):
        raise TypeError("case profile model_params must be a mapping")
    controllers = profile.get("specialist_metadata", {}).get("controllers", {})
    if not isinstance(controllers, Mapping):
        raise TypeError("case profile controllers must be a mapping")
    for controller, config in controllers.items():
        if not isinstance(controller, str) or not controller:
            raise TypeError("case controller IDs must be non-empty strings")
        if not isinstance(config, Mapping):
            raise TypeError(
                f"case controller config for {controller!r} must be a mapping"
            )
        profile_name = config.get("profile")
        if profile_name is not None and (
            not isinstance(profile_name, str) or not profile_name
        ):
            raise TypeError(
                f"case controller profile for {controller!r} must be a non-empty string"
            )
        parameters = config.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError(
                f"case controller parameters for {controller!r} must be a mapping"
            )
    _validate_operation(profile.get("operation"))
    for section in ("initialization", "setpoints", "disturbances", "constraints", "acceptance"):
        if section in profile and not isinstance(profile[section], (Mapping, list)):
            raise TypeError(f"case profile {section} must be a mapping or list")
    acceptance = profile.get("acceptance", {})
    if isinstance(acceptance, Mapping) and "metrics" in acceptance:
        _validate_acceptance_metrics(acceptance["metrics"])
    initialization = profile.get("initialization", {})
    if isinstance(initialization, Mapping) and "state" in initialization:
        _finite_numeric_vector("case initialization state", initialization["state"])
    setpoints = profile.get("setpoints", {})
    if isinstance(setpoints, Mapping):
        if "initial" in setpoints:
            _finite_numeric_vector("case initial setpoint", setpoints["initial"])
        schedule = setpoints.get("schedule", [])
        if not isinstance(schedule, list):
            raise TypeError("case setpoint schedule must be a list")
        for event in schedule:
            if not isinstance(event, Mapping):
                raise TypeError("each case setpoint event must be a mapping")
            at_step = event.get("at_step")
            if isinstance(at_step, bool) or not isinstance(at_step, int) or at_step < 0:
                raise ValueError("case setpoint event at_step must be a non-negative integer")
            _finite_numeric_vector("case scheduled setpoint", event.get("values"))
    disturbances = profile.get("disturbances", [])
    if not isinstance(disturbances, list):
        raise TypeError("case disturbances must be a list")
    for event in disturbances:
        if not isinstance(event, Mapping):
            raise TypeError("each case disturbance event must be a mapping")
        at_step = event.get("at_step")
        if isinstance(at_step, bool) or not isinstance(at_step, int) or at_step < 0:
            raise ValueError("case disturbance event at_step must be a non-negative integer")
        name = event.get("name")
        if not isinstance(name, str) or not name:
            raise TypeError("case disturbance event name must be a non-empty string")
        _finite_numeric_value("case disturbance value", event.get("value"))
    if "references" in profile and not isinstance(profile["references"], list):
        raise TypeError("case profile references must be a list")
def validate_case_profile(
    profile: Mapping,
    *,
    expected_scenario: str | None = None,
) -> None:
    """Validate a Goal-independent Case v2 profile."""

    if not isinstance(profile, Mapping):
        raise TypeError("case profile must be a mapping")
    required = ("schema_version", "name", "scenario", "environment")
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"case profile is missing required fields: {', '.join(missing)}")
    if profile["schema_version"] != CASE_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"unsupported case profile schema: {profile['schema_version']!r}")
    forbidden = CASE_FORBIDDEN_GOAL_FIELDS.intersection(profile)
    if forbidden:
        raise ValueError(
            "case profiles must not define Goal or RewardSpec fields: "
            + ", ".join(sorted(forbidden))
        )
    controller_fields = CASE_FORBIDDEN_CONTROLLER_FIELDS.intersection(profile)
    if controller_fields:
        raise ValueError(
            "case profiles must not bind controller defaults; "
            "use specialist_metadata for historical specialist configurations"
        )
    if "summary" in profile and (
        not isinstance(profile["summary"], str) or not profile["summary"]
    ):
        raise TypeError("case profile summary must be a non-empty string")
    if "description" in profile and (
        not isinstance(profile["description"], str) or not profile["description"]
    ):
        raise TypeError("case profile description must be a non-empty string")
    tags = profile.get("tags")
    if tags is not None and (
        not isinstance(tags, list)
        or any(not isinstance(tag, str) or not tag for tag in tags)
    ):
        raise TypeError("case profile tags must be a list of non-empty strings")
    if "evaluation" in profile and not isinstance(profile["evaluation"], Mapping):
        raise TypeError("case profile evaluation must be a mapping")
    specialist = profile.get("specialist_metadata")
    if specialist is not None:
        if not isinstance(specialist, Mapping):
            raise TypeError("case profile specialist_metadata must be a mapping")
        unknown_specialist = set(specialist) - {"controllers"}
        if unknown_specialist:
            raise ValueError(
                "unknown case specialist metadata: "
                + ", ".join(sorted(unknown_specialist))
            )
        controllers = specialist.get("controllers", {})
        if not isinstance(controllers, Mapping):
            raise TypeError("specialist controller metadata must be a mapping")

    _validate_case_contents(profile, expected_scenario=expected_scenario)


def apply_case_overrides(profile: Mapping, overrides: Mapping | None) -> dict:
    """Resolve a whitelisted Case variant without mutating its base profile."""

    validate_case_profile(profile)
    resolved = deepcopy(dict(profile))
    if overrides is None:
        return resolved
    if not isinstance(overrides, Mapping):
        raise TypeError("case overrides must be a mapping")
    unknown = set(overrides) - CASE_OVERRIDE_FIELDS
    if unknown:
        raise ValueError(
            "case overrides may not change fields: " + ", ".join(sorted(unknown))
        )
    for key, value in overrides.items():
        current = resolved.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            resolved[key] = _deep_merge(current, value)
        else:
            resolved[key] = deepcopy(value)
    validate_case_profile(resolved)
    return resolved


def _deep_merge(base: Mapping, override: Mapping) -> dict:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _finite_numeric_vector(name: str, values) -> None:
    if not isinstance(values, list) or not values:
        raise TypeError(f"{name} must be a non-empty list")
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError(f"{name} values must be finite")


def _finite_numeric_value(name: str, value) -> None:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise TypeError(f"{name} must not be empty")
    try:
        numbers = [float(item) for item in values]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric or a numeric list") from exc
    if not all(math.isfinite(item) for item in numbers):
        raise ValueError(f"{name} must be finite")


def _validate_operation(operation) -> None:
    if operation is None:
        return
    if not isinstance(operation, Mapping):
        raise TypeError("case profile operation must be a mapping")
    unknown = set(operation) - CASE_OPERATION_FIELDS
    if unknown:
        raise ValueError(f"unknown case operation fields: {', '.join(sorted(unknown))}")
    if "product_flow_sp" not in operation:
        raise ValueError("case operation requires product_flow_sp")
    product_flow_sp = _nonnegative_operation_value(
        "case operation product_flow_sp", operation["product_flow_sp"]
    )
    min_product_flow = _nonnegative_operation_value(
        "case operation min_product_flow",
        operation.get("min_product_flow", product_flow_sp),
    )
    if min_product_flow > product_flow_sp:
        raise ValueError("case operation min_product_flow must not exceed product_flow_sp")


def _nonnegative_operation_value(name: str, value) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _validate_acceptance_metrics(metrics) -> None:
    if not isinstance(metrics, Mapping) or not metrics:
        raise TypeError("case acceptance metrics must be a non-empty mapping")
    for metric, bounds in metrics.items():
        if not isinstance(metric, str) or not metric:
            raise TypeError("case acceptance metric names must be non-empty strings")
        if not isinstance(bounds, Mapping) or not bounds:
            raise TypeError("case acceptance metric bounds must be a non-empty mapping")
        unknown = set(bounds) - {"min", "max"}
        if unknown:
            raise ValueError(
                f"unknown case acceptance bounds for {metric}: {', '.join(sorted(unknown))}"
            )
        for key, value in bounds.items():
            _finite_numeric_value(f"case acceptance {metric} {key}", value)
        if "min" in bounds and "max" in bounds:
            if float(bounds["min"]) > float(bounds["max"]):
                raise ValueError(
                    f"case acceptance minimum exceeds maximum for {metric}"
                )


__all__ = [
    "CASE_ENVIRONMENT_FIELDS",
    "CASE_OPERATION_FIELDS",
    "CASE_PROFILE_SCHEMA_VERSION",
    "ENVIRONMENT_BOOLEAN_FIELDS",
    "CaseSpec",
    "apply_case_overrides",
    "case_profile_hash",
    "validate_case_profile",
]
