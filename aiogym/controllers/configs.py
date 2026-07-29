"""Controller configuration loading and merge rules."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_CONFIG_META_KEYS = {
    "action_mode",
    "control_structure",
    "name",
    "class",
    "adapter",
    "scenario",
    "scenarios",
    "profile",
    "track",
    "case",
    "policy_scope",
    "goal",
    "reward_spec",
    "access_level",
}


def load_controller_config(
    name: str,
    scenario: str | None = None,
    profile: str | None = None,
    *,
    goal: str | None = None,
    reward_spec: str | None = None,
    track: str | None = None,
    case: str | None = None,
    policy_scope: str | None = None,
) -> dict[str, Any]:
    """Load one controller profile under goal/reward and policy-scope semantics."""

    from .._internal.identifiers import canonical_scenario_id, internal_scenario_id

    if goal is not None and goal not in {"regulation", "economic"}:
        raise ValueError("controller goal must be one of: economic, regulation")
    if policy_scope is not None and policy_scope not in {
        "generalist",
        "specialist",
    }:
        raise ValueError(
            "controller policy_scope must be one of: generalist, specialist"
        )
    key = name.lower()
    path = Path(__file__).resolve().parent / "configs" / f"{key}.json"
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    params = dict(data.get("parameters", {}))
    if scenario:
        params.update(data.get("scenarios", {}).get(internal_scenario_id(scenario), {}))
    profile_data = data.get("profiles", {}).get(profile, {}) if profile else {}
    if profile and not profile_data:
        raise ValueError(f"unknown controller profile {profile!r} for {key!r}")
    params.update(profile_data.get("parameters", {}))
    if scenario:
        params.update(
            profile_data.get("scenarios", {}).get(internal_scenario_id(scenario), {})
        )
    if goal:
        params.update(profile_data.get("goals", {}).get(goal, {}))
    if reward_spec:
        params.update(
            profile_data.get("reward_specs", {}).get(reward_spec, {})
        )
    out = {
        k: v
        for k, v in data.items()
        if k not in {"parameters", "profiles"}
    }
    if isinstance(out.get("scenarios"), dict):
        out["scenarios"] = {
            canonical_scenario_id(key): value
            for key, value in out["scenarios"].items()
        }
    for meta_key in _CONFIG_META_KEYS:
        if meta_key in profile_data:
            out[meta_key] = profile_data[meta_key]
    requested_identity = {
        "track": track,
        "case": case,
        "policy_scope": policy_scope,
        "goal": goal,
        "reward_spec": reward_spec,
    }
    for meta_key, requested in requested_identity.items():
        configured = out.get(meta_key)
        if requested is not None and configured is not None and requested != configured:
            raise ValueError(
                f"controller profile {profile!r} {meta_key}={configured!r} "
                f"conflicts with requested {requested!r}"
            )
        if requested is not None:
            out[meta_key] = requested
    _validate_profile_identity(out, profile)
    out["parameters"] = params
    return out


def _merged_controller_config(
    name: str, scenario: str | None, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    override = dict(config or {})
    explicit_profile = override.pop("profile", None)
    base = load_controller_config(
        name,
        scenario,
        profile=explicit_profile,
        goal=override.get("goal"),
        reward_spec=override.get("reward_spec"),
        track=override.get("track"),
        case=override.get("case"),
        policy_scope=override.get("policy_scope"),
    )
    params = dict(base.get("parameters", {}))
    params.update(override.pop("parameters", {}))
    flat = {k: v for k, v in override.items() if k not in _CONFIG_META_KEYS}
    params.update(flat)
    merged = {k: v for k, v in base.items() if k != "parameters"}
    for k, v in override.items():
        if k in _CONFIG_META_KEYS:
            merged[k] = v
    if explicit_profile is not None:
        merged["profile"] = explicit_profile
    merged["parameters"] = params
    return merged


def _controller_params(config: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    params = dict(cfg.get("parameters", {}))
    for k, v in cfg.items():
        if k not in _CONFIG_META_KEYS and k != "parameters":
            params[k] = v
    return params


def _validate_profile_identity(config: Mapping[str, Any], profile: str | None) -> None:
    scope = config.get("policy_scope")
    track = config.get("track")
    case = config.get("case")
    if scope == "generalist":
        if not track:
            raise ValueError(
                f"generalist controller profile {profile!r} must bind a track"
            )
        if case is not None:
            raise ValueError(
                f"generalist controller profile {profile!r} cannot bind a case"
            )
    if scope == "specialist":
        if not case:
            raise ValueError(
                f"specialist controller profile {profile!r} must bind a case"
            )
        if track is not None:
            raise ValueError(
                f"specialist controller profile {profile!r} cannot bind a track"
            )
