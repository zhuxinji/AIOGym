"""Shared environment metadata for workflow artifacts."""
from __future__ import annotations

from collections.abc import Mapping
import warnings

from aiogym.core.contracts import environment_interface, policy_metadata
from aiogym.core.io import jsonable


ENVIRONMENT_COMPATIBILITY_FIELDS = (
    "scenario",
    "reward",
    "parameters",
    "control_dt",
    "observation_shape",
    "action_shape",
    "policy_interface",
)


def environment_metadata(env):
    """Return a JSON-compatible environment snapshot for policies and artifacts.

    Save this alongside external weights when training, not from a later test
    environment. Compatibility checks compare the fields declared above;
    operating cases and runtime variation may differ at evaluation time.
    """
    base = env.unwrapped
    config = base.runtime_config
    metadata = {
        "scenario": config["scenario"],
        "reward": config["reward"],
        "parameters": jsonable(config["parameters"]),
        "initial_state": jsonable(config["initial_state"]),
        "benchmark": config["benchmark"],
        "randomize": config["randomize"],
        "boundary_probability": config["boundary_probability"],
        "disturbance": config["disturbance"],
        "noise": jsonable(config["noise"]),
        "delay": jsonable(config["delay"]),
        "fault": jsonable(config["fault"]),
        "control_dt": config["control_dt"],
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "policy_interface": environment_interface(env),
    }
    if config["disturbance_schedule"] is not None:
        metadata["disturbance_schedule"] = jsonable(
            config["disturbance_schedule"]
        )
    if "disturbance_overrides" in config:
        metadata["disturbance_overrides"] = jsonable(config["disturbance_overrides"])
    return metadata


def validate_environment_compatibility(
    expected, env, *, artifact="checkpoint", allow_legacy=False,
) -> dict:
    if not isinstance(expected, Mapping):
        raise TypeError(f"{artifact} environment must be a JSON object")
    fields = ENVIRONMENT_COMPATIBILITY_FIELDS
    legacy = allow_legacy and "policy_interface" not in expected
    if legacy:
        fields = tuple(field for field in fields if field != "policy_interface")
    missing = [
        field for field in fields if field not in expected
    ]
    if missing:
        raise ValueError(
            f"{artifact} environment is missing compatibility fields: {missing}"
        )
    actual = environment_metadata(env)
    for field in fields:
        if expected[field] != actual[field]:
            raise ValueError(
                f"{artifact} {field} is incompatible with the supplied environment"
            )
    if legacy:
        warnings.warn(
            f"{artifact} has no policy_interface metadata; only legacy compatibility "
            "checks (scenario, reward, parameters, control interval, and shapes) "
            "were performed. Channel semantics and interface version are unverified.",
            UserWarning,
            stacklevel=3,
        )
    return actual


def policy_metadata_for_environment(policy, env):
    """Validate a policy's optional declaration once at the workflow boundary."""
    metadata = policy_metadata(policy)
    if "environment" in metadata:
        validate_environment_compatibility(
            metadata["environment"], env, artifact="policy",
        )
        status = "matched"
    else:
        status = "not_provided"
    metadata["declared_environment_check"] = status
    return metadata


__all__ = ["environment_metadata", "validate_environment_compatibility"]
