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
}


def load_controller_config(
    name: str,
    scenario: str | None = None,
    profile: str | None = None,
    objective: str | None = None,
) -> dict[str, Any]:
    from .._internal.identifiers import canonical_scenario_id, internal_scenario_id

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
    params.update(profile_data.get("parameters", {}))
    if scenario:
        params.update(
            profile_data.get("scenarios", {}).get(internal_scenario_id(scenario), {})
        )
    if objective:
        params.update(profile_data.get("objectives", {}).get(objective, {}))
    out = {
        k: v
        for k, v in data.items()
        if k not in {"parameters", "profiles", "objectives"}
    }
    if isinstance(out.get("scenarios"), dict):
        out["scenarios"] = {
            canonical_scenario_id(key): value
            for key, value in out["scenarios"].items()
        }
    out["parameters"] = params
    return out


def _merged_controller_config(
    name: str, scenario: str | None, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    override = dict(config or {})
    explicit_profile = override.pop("profile", None)
    requested_objective = override.get("objective")
    base = load_controller_config(
        name,
        scenario,
        profile=explicit_profile,
        objective=requested_objective,
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
