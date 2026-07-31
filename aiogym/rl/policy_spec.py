"""Versioned native-policy descriptions embedded in Torch checkpoints."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence


POLICY_SPEC_SCHEMA_VERSION = "aiogym.policy_spec.v1"
NATIVE_ACTION_CONTRACT = "normalized[-1,1]"
ENVIRONMENT_ACTION_CONTRACT = "command[0,1]"
ENVIRONMENT_ACTION_TRANSFORM = (
    "clip(0.5*(action+1.0),0.0,1.0)"
)


def behavior_cloning_policy_spec(
    observation_dim: int,
    action_dim: int,
    hidden: int,
    *,
    scenario: str | None,
    action_mode: str,
) -> dict:
    """Describe the deterministic MLP stored by behavior cloning."""

    return _policy_spec(
        algorithm_id="bc",
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden=hidden,
        scenario=scenario,
        action_mode=action_mode,
        network={
            "kind": "deterministic_mlp",
            "hidden_sizes": [int(hidden), int(hidden)],
            "activation": "relu",
            "deterministic_output": "tanh(logits)",
        },
    )


def rlpd_policy_spec(
    observation_dim: int,
    action_dim: int,
    hidden: int,
    *,
    scenario: str | None,
    action_mode: str,
    log_std_bounds: Sequence[float],
) -> dict:
    """Describe the actor stored by the native RLPD trainer."""

    bounds = [float(value) for value in log_std_bounds]
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ValueError("log_std_bounds must contain increasing bounds")
    return _policy_spec(
        algorithm_id="rlpd",
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden=hidden,
        scenario=scenario,
        action_mode=action_mode,
        network={
            "kind": "squashed_gaussian_actor",
            "hidden_sizes": [int(hidden), int(hidden)],
            "activation": "relu",
            "deterministic_output": "tanh(mu)",
            "log_std_bounds": bounds,
        },
    )


def validate_policy_spec(value: Mapping) -> dict:
    """Validate and defensively copy a native policy specification."""

    if not isinstance(value, Mapping):
        raise TypeError("policy_spec must be a mapping")
    spec = copy.deepcopy(dict(value))
    required = {
        "schema_version",
        "algorithm_id",
        "scenario",
        "action_mode",
        "observation_dim",
        "action_dim",
        "action_contract",
        "network",
        "state_dict_key",
    }
    if set(spec) != required:
        missing = sorted(required - set(spec))
        unknown = sorted(set(spec) - required)
        raise ValueError(
            "policy_spec fields do not match v1"
            f"; missing={missing}; unknown={unknown}"
        )
    if spec["schema_version"] != POLICY_SPEC_SCHEMA_VERSION:
        raise ValueError("unsupported policy_spec schema")
    if spec["algorithm_id"] not in {"bc", "rlpd"}:
        raise ValueError("policy_spec algorithm_id must be bc or rlpd")
    scenario = spec["scenario"]
    if scenario is not None and (
        not isinstance(scenario, str) or not scenario
    ):
        raise ValueError("policy_spec scenario must be null or non-empty")
    if spec["action_mode"] not in {"actuator", "setpoint"}:
        raise ValueError(
            "policy_spec action_mode must be actuator or setpoint"
        )
    for name in ("observation_dim", "action_dim"):
        value = spec[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"policy_spec {name} must be positive")
    if spec["state_dict_key"] != "policy_state_dict":
        raise ValueError(
            "policy_spec state_dict_key must be policy_state_dict"
        )
    action_contract = spec["action_contract"]
    if not isinstance(action_contract, Mapping) or dict(
        action_contract
    ) != _action_contract():
        raise ValueError("policy_spec action_contract is not canonical")
    network = spec["network"]
    if not isinstance(network, Mapping):
        raise TypeError("policy_spec network must be a mapping")
    hidden_sizes = network.get("hidden_sizes")
    if (
        isinstance(hidden_sizes, (str, bytes))
        or not isinstance(hidden_sizes, Sequence)
        or len(hidden_sizes) != 2
        or any(
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            for width in hidden_sizes
        )
    ):
        raise ValueError(
            "policy_spec network.hidden_sizes must contain two "
            "positive integers"
        )
    expected_kind = {
        "bc": "deterministic_mlp",
        "rlpd": "squashed_gaussian_actor",
    }[spec["algorithm_id"]]
    if network.get("kind") != expected_kind:
        raise ValueError(
            "policy_spec network kind does not match algorithm"
        )
    if network.get("activation") != "relu":
        raise ValueError("policy_spec activation must be relu")
    if spec["algorithm_id"] == "bc":
        expected_network_fields = {
            "kind",
            "hidden_sizes",
            "activation",
            "deterministic_output",
        }
        if network.get("deterministic_output") != "tanh(logits)":
            raise ValueError("BC deterministic output must be tanh(logits)")
    else:
        expected_network_fields = {
            "kind",
            "hidden_sizes",
            "activation",
            "deterministic_output",
            "log_std_bounds",
        }
        if network.get("deterministic_output") != "tanh(mu)":
            raise ValueError("RLPD deterministic output must be tanh(mu)")
        bounds = network.get("log_std_bounds")
        if (
            not isinstance(bounds, Sequence)
            or len(bounds) != 2
            or any(
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or not math.isfinite(float(bound))
                for bound in bounds
            )
            or float(bounds[0]) >= float(bounds[1])
        ):
            raise ValueError(
                "RLPD log_std_bounds must contain increasing numbers"
            )
    if set(network) != expected_network_fields:
        raise ValueError(
            "policy_spec network fields do not match algorithm"
        )
    return spec


def _policy_spec(
    *,
    algorithm_id: str,
    observation_dim: int,
    action_dim: int,
    hidden: int,
    scenario: str | None,
    action_mode: str,
    network: dict,
) -> dict:
    for name, value in (
        ("observation_dim", observation_dim),
        ("action_dim", action_dim),
        ("hidden", hidden),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    spec = {
        "schema_version": POLICY_SPEC_SCHEMA_VERSION,
        "algorithm_id": algorithm_id,
        "scenario": scenario,
        "action_mode": action_mode,
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "action_contract": _action_contract(),
        "network": network,
        "state_dict_key": "policy_state_dict",
    }
    return validate_policy_spec(spec)


def _action_contract() -> dict[str, object]:
    return {
        "native": NATIVE_ACTION_CONTRACT,
        "environment": ENVIRONMENT_ACTION_CONTRACT,
        "environment_transform": ENVIRONMENT_ACTION_TRANSFORM,
        "apply_transform_exactly_once": True,
    }


__all__ = [
    "ENVIRONMENT_ACTION_CONTRACT",
    "ENVIRONMENT_ACTION_TRANSFORM",
    "NATIVE_ACTION_CONTRACT",
    "POLICY_SPEC_SCHEMA_VERSION",
    "behavior_cloning_policy_spec",
    "rlpd_policy_spec",
    "validate_policy_spec",
]
