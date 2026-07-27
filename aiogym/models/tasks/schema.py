"""Schema validation for model-bound AIO-Gym task specifications."""
from __future__ import annotations

import math
from collections.abc import Mapping

from aiogym._internal.identifiers import internal_scenario_id
from aiogym._internal.vocabulary import OBJECTIVE_NAMES


TASK_PROFILE_SCHEMA_VERSION = "aiogym.task_profile.v1"
TASK_ENVIRONMENT_FIELDS = frozenset({
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
TASK_OPERATION_FIELDS = frozenset({"product_flow_sp", "min_product_flow"})
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


def validate_task_profile(
    profile: Mapping,
    *,
    expected_scenario: str | None = None,
) -> None:
    """Validate task identity, timing, and extension sections."""

    if not isinstance(profile, Mapping):
        raise TypeError("task profile must be a mapping")
    required = ("schema_version", "name", "scenario", "status", "environment")
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"task profile is missing required fields: {', '.join(missing)}")
    if profile["schema_version"] != TASK_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"unsupported task profile schema: {profile['schema_version']!r}")
    for key in ("name", "scenario", "status"):
        if not isinstance(profile[key], str) or not profile[key]:
            raise ValueError(f"task profile {key} must be a non-empty string")
    if (
        expected_scenario is not None
        and internal_scenario_id(profile["scenario"])
        != internal_scenario_id(expected_scenario)
    ):
        raise ValueError(f"expected task for {expected_scenario!r}, got {profile['scenario']!r}")
    environment = profile["environment"]
    if not isinstance(environment, Mapping):
        raise TypeError("task profile environment must be a mapping")
    unknown = set(environment) - TASK_ENVIRONMENT_FIELDS
    if unknown:
        raise ValueError(f"unknown task environment fields: {', '.join(sorted(unknown))}")
    if "control_dt" in environment:
        value = float(environment["control_dt"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("task control_dt must be finite and positive")
    if "episode_steps" in environment:
        value = environment["episode_steps"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("task episode_steps must be a positive integer")
    if "action_mode" in environment and environment["action_mode"] not in {"actuator", "setpoint"}:
        raise ValueError("task action_mode must be one of: actuator, setpoint")
    for key in ENVIRONMENT_BOOLEAN_FIELDS:
        if key in environment and not isinstance(environment[key], bool):
            raise TypeError(f"task {key} must be a boolean")
    if "noise_pct" in environment:
        value = float(environment["noise_pct"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("task noise_pct must be finite and non-negative")
    if "model_params" in profile and not isinstance(profile["model_params"], Mapping):
        raise TypeError("task profile model_params must be a mapping")
    controllers = profile.get("controllers", {})
    if not isinstance(controllers, Mapping):
        raise TypeError("task profile controllers must be a mapping")
    for controller, config in controllers.items():
        if not isinstance(controller, str) or not controller:
            raise TypeError("task controller IDs must be non-empty strings")
        if not isinstance(config, Mapping):
            raise TypeError(
                f"task controller config for {controller!r} must be a mapping"
            )
        profile_name = config.get("profile")
        if profile_name is not None and (
            not isinstance(profile_name, str) or not profile_name
        ):
            raise TypeError(
                f"task controller profile for {controller!r} must be a non-empty string"
            )
        parameters = config.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError(
                f"task controller parameters for {controller!r} must be a mapping"
            )
    _validate_operation(profile.get("operation"))
    for section in ("initialization", "setpoints", "disturbances", "constraints", "acceptance"):
        if section in profile and not isinstance(profile[section], (Mapping, list)):
            raise TypeError(f"task profile {section} must be a mapping or list")
    acceptance = profile.get("acceptance", {})
    if isinstance(acceptance, Mapping) and "metrics" in acceptance:
        _validate_acceptance_metrics(acceptance["metrics"])
    initialization = profile.get("initialization", {})
    if isinstance(initialization, Mapping) and "state" in initialization:
        _finite_numeric_vector("task initialization state", initialization["state"])
    setpoints = profile.get("setpoints", {})
    if isinstance(setpoints, Mapping):
        if "initial" in setpoints:
            _finite_numeric_vector("task initial setpoint", setpoints["initial"])
        schedule = setpoints.get("schedule", [])
        if not isinstance(schedule, list):
            raise TypeError("task setpoint schedule must be a list")
        for event in schedule:
            if not isinstance(event, Mapping):
                raise TypeError("each task setpoint event must be a mapping")
            at_step = event.get("at_step")
            if isinstance(at_step, bool) or not isinstance(at_step, int) or at_step < 0:
                raise ValueError("task setpoint event at_step must be a non-negative integer")
            _finite_numeric_vector("task scheduled setpoint", event.get("values"))
    disturbances = profile.get("disturbances", [])
    if not isinstance(disturbances, list):
        raise TypeError("task disturbances must be a list")
    for event in disturbances:
        if not isinstance(event, Mapping):
            raise TypeError("each task disturbance event must be a mapping")
        at_step = event.get("at_step")
        if isinstance(at_step, bool) or not isinstance(at_step, int) or at_step < 0:
            raise ValueError("task disturbance event at_step must be a non-negative integer")
        name = event.get("name")
        if not isinstance(name, str) or not name:
            raise TypeError("task disturbance event name must be a non-empty string")
        _finite_numeric_value("task disturbance value", event.get("value"))
    if "references" in profile and not isinstance(profile["references"], list):
        raise TypeError("task profile references must be a list")
    supported = profile.get("supported_objectives")
    if supported is not None:
        if not isinstance(supported, list) or not supported:
            raise TypeError("task supported_objectives must be a non-empty list")
        if any(not isinstance(value, str) or not value for value in supported):
            raise TypeError("task supported_objectives must contain non-empty strings")
        unknown_supported = set(supported) - set(OBJECTIVE_NAMES)
        if unknown_supported:
            raise ValueError(
                "unknown task supported objectives: "
                + ", ".join(sorted(unknown_supported))
            )
    default_objective = profile.get("default_objective")
    if default_objective is not None:
        if not isinstance(default_objective, str) or not default_objective:
            raise TypeError("task default_objective must be a non-empty string")
        if supported is not None and default_objective not in supported:
            raise ValueError("task default_objective must be listed in supported_objectives")
    objectives = profile.get("objectives", {})
    if not isinstance(objectives, Mapping):
        raise TypeError("task objectives must be a mapping")
    unknown_objectives = set(objectives) - {"tracking"}
    if unknown_objectives:
        raise ValueError(
            f"unsupported task objective options: {', '.join(sorted(unknown_objectives))}"
        )
    tracking = objectives.get("tracking", {})
    if not isinstance(tracking, Mapping):
        raise TypeError("task tracking objective options must be a mapping")
    unknown_tracking = set(tracking) - {"tracking_q_y", "tracking_r_move"}
    if unknown_tracking:
        raise ValueError(
            f"unknown task tracking options: {', '.join(sorted(unknown_tracking))}"
        )
    if "tracking_q_y" in tracking:
        _finite_numeric_value("task tracking_q_y", tracking["tracking_q_y"])
        values = (
            tracking["tracking_q_y"]
            if isinstance(tracking["tracking_q_y"], list)
            else [tracking["tracking_q_y"]]
        )
        if any(float(value) < 0.0 for value in values):
            raise ValueError("task tracking_q_y must be non-negative")
    if "tracking_r_move" in tracking:
        _finite_numeric_value("task tracking_r_move", tracking["tracking_r_move"])
        if float(tracking["tracking_r_move"]) < 0.0:
            raise ValueError("task tracking_r_move must be non-negative")


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
        raise TypeError("task profile operation must be a mapping")
    unknown = set(operation) - TASK_OPERATION_FIELDS
    if unknown:
        raise ValueError(f"unknown task operation fields: {', '.join(sorted(unknown))}")
    if "product_flow_sp" not in operation:
        raise ValueError("task operation requires product_flow_sp")
    product_flow_sp = _nonnegative_operation_value(
        "task operation product_flow_sp", operation["product_flow_sp"]
    )
    min_product_flow = _nonnegative_operation_value(
        "task operation min_product_flow",
        operation.get("min_product_flow", product_flow_sp),
    )
    if min_product_flow > product_flow_sp:
        raise ValueError("task operation min_product_flow must not exceed product_flow_sp")


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
        raise TypeError("task acceptance metrics must be a non-empty mapping")
    for metric, bounds in metrics.items():
        if not isinstance(metric, str) or not metric:
            raise TypeError("task acceptance metric names must be non-empty strings")
        if not isinstance(bounds, Mapping) or not bounds:
            raise TypeError("task acceptance metric bounds must be a non-empty mapping")
        unknown = set(bounds) - {"min", "max"}
        if unknown:
            raise ValueError(
                f"unknown task acceptance bounds for {metric}: {', '.join(sorted(unknown))}"
            )
        for key, value in bounds.items():
            _finite_numeric_value(f"task acceptance {metric} {key}", value)
        if "min" in bounds and "max" in bounds:
            if float(bounds["min"]) > float(bounds["max"]):
                raise ValueError(
                    f"task acceptance minimum exceeds maximum for {metric}"
                )


__all__ = [
    "ENVIRONMENT_BOOLEAN_FIELDS",
    "TASK_ENVIRONMENT_FIELDS",
    "TASK_OPERATION_FIELDS",
    "TASK_PROFILE_SCHEMA_VERSION",
    "validate_task_profile",
]
