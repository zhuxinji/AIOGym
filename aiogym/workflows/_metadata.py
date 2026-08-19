"""Shared environment metadata for workflow artifacts."""
from __future__ import annotations

from collections.abc import Mapping

from aiogym.core.io import jsonable


ENVIRONMENT_COMPATIBILITY_FIELDS = (
    "scenario",
    "reward",
    "parameters",
    "control_dt",
    "observation_shape",
    "action_shape",
)


def environment_metadata(env):
    base = env.unwrapped
    config = base.runtime_config
    return {
        "scenario": config["scenario"],
        "reward": config["reward"],
        "parameters": jsonable(config["parameters"]),
        "benchmark": config["benchmark"],
        "randomize": config["randomize"],
        "disturbance": config["disturbance"],
        "noise": jsonable(config["noise"]),
        "delay": jsonable(config["delay"]),
        "fault": jsonable(config["fault"]),
        "control_dt": config["control_dt"],
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
    }


def validate_environment_compatibility(expected, env) -> None:
    if not isinstance(expected, Mapping):
        raise TypeError("checkpoint environment must be a JSON object")
    missing = [
        field for field in ENVIRONMENT_COMPATIBILITY_FIELDS if field not in expected
    ]
    if missing:
        raise ValueError(
            f"checkpoint environment is missing compatibility fields: {missing}"
        )
    actual = environment_metadata(env)
    for field in ENVIRONMENT_COMPATIBILITY_FIELDS:
        if expected[field] != actual[field]:
            raise ValueError(
                f"checkpoint {field} is incompatible with the supplied environment"
            )


__all__ = ["environment_metadata", "validate_environment_compatibility"]
