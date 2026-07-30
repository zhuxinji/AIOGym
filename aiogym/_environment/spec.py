"""Resolution boundary for the single public environment factory."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from aiogym._internal.config import load_config, resolve_auto_events
from aiogym._internal.identifiers import canonical_scenario_id
from aiogym.models import make_model
from aiogym.models.cases import resolve_environment_options
from aiogym.rewards import RewardSpec, resolve_reward_spec

from .config import DIRECT_ENV_DEFAULTS


_CONFIG_FIELDS = frozenset(
    {"scenario", "case", "reward_spec", "environment", "info_level"}
)
_ENVIRONMENT_FIELDS = frozenset(
    {
        *DIRECT_ENV_DEFAULTS,
        "control_dt",
        "episode_steps",
        "model_params",
        "initial_setpoint",
        "setpoint_schedule",
        "profile_timing",
        "crystal_ln_sp",
        "crystal_cv_sp",
        "crystal_random_targets",
        "crystal_ln_range",
        "crystal_cv_range",
    }
)


@dataclass(frozen=True)
class ResolvedEnvSpec:
    """Immutable, fully resolved input consumed by ``_AIOGymEnv``."""

    scenario: str
    case_profile: Mapping[str, Any] | None
    reward_spec: RewardSpec
    control_dt: float
    episode_steps: int
    action_mode: str
    observation: Mapping[str, Any]
    realism: Mapping[str, Any]
    events: Mapping[str, Any]
    model_params: Mapping[str, Any]
    info_level: str
    profile_timing: bool = False
    spec_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "case_profile",
            "observation",
            "realism",
            "events",
            "model_params",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _freeze(deepcopy(dict(value))),
                )
        payload = {
            "scenario": self.scenario,
            "case_profile": _thaw(self.case_profile),
            "reward_spec": self.reward_spec.id,
            "control_dt": self.control_dt,
            "episode_steps": self.episode_steps,
            "action_mode": self.action_mode,
            "observation": _thaw(self.observation),
            "realism": _thaw(self.realism),
            "events": _thaw(self.events),
            "model_params": _thaw(self.model_params),
            "info_level": self.info_level,
            "profile_timing": self.profile_timing,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        object.__setattr__(
            self,
            "spec_hash",
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def runtime(self) -> dict[str, Any]:
        """Return a detached mutable view for the internal environment."""

        return {
            "case_profile": _thaw(self.case_profile),
            "observation": _thaw(self.observation),
            "realism": _thaw(self.realism),
            "events": _thaw(self.events),
            "model_params": _thaw(self.model_params),
        }


def resolve_env_spec(
    scenario: str | None = None,
    *,
    case=None,
    reward_spec=None,
    config: str | Path | Mapping[str, Any] | None = None,
    info_level: str = "full",
) -> ResolvedEnvSpec:
    """Resolve direct or config mode into one immutable environment spec."""

    if config is not None:
        if (
            scenario is not None
            or case is not None
            or reward_spec is not None
            or info_level != "full"
        ):
            raise ValueError(
                "config mode is mutually exclusive with direct arguments"
            )
        declaration = load_config(config)
    else:
        declaration = {
            "scenario": "cascade" if scenario is None else scenario,
            "case": case,
            "reward_spec": reward_spec,
            "info_level": info_level,
            "environment": {},
        }
    unknown = set(declaration) - _CONFIG_FIELDS
    if unknown:
        raise ValueError(
            "unknown environment config fields: "
            + ", ".join(sorted(unknown))
        )
    selected_scenario = declaration.get("scenario", "cascade")
    if not isinstance(selected_scenario, str) or not selected_scenario:
        raise ValueError("scenario must be a non-empty string")
    base_model = make_model(selected_scenario)
    canonical_scenario = canonical_scenario_id(base_model.scenario)
    environment = declaration.get("environment", {})
    if not isinstance(environment, Mapping):
        raise TypeError("config['environment'] must be a mapping")
    environment = dict(environment)
    unknown_environment = set(environment) - _ENVIRONMENT_FIELDS
    if unknown_environment:
        raise ValueError(
            "unknown environment options: "
            + ", ".join(sorted(unknown_environment))
        )
    explicit = {
        name: environment.get(name)
        for name in (
            *DIRECT_ENV_DEFAULTS,
            "control_dt",
            "episode_steps",
            "model_params",
        )
    }
    explicit["auto_events"] = resolve_auto_events(
        explicit["auto_events"]
    )
    profile, options = resolve_environment_options(
        scenario=canonical_scenario,
        case=declaration.get("case"),
        explicit=explicit,
        defaults=DIRECT_ENV_DEFAULTS,
        default_control_dt=0.5,
        default_episode_steps=600,
    )
    resolved_info_level = declaration.get("info_level", "full")
    if resolved_info_level not in {"minimal", "full"}:
        raise ValueError("info_level must be one of: minimal, full")
    observation = {
        name: options[name]
        for name in (
            "integral_obs",
            "disturbance_obs",
            "previous_action_obs",
            "normalize_observations",
            "tracking_error_obs",
            "observation_mode",
        )
    }
    realism = {
        name: options[name]
        for name in (
            "auto_events",
            "randomize",
            "randomize_setpoints",
            "randomize_plant",
            "plant_drift",
            "noise",
            "noise_pct",
            "terminate_on_runaway",
        )
    }
    realism.update(
        {
            name: deepcopy(environment.get(name, default))
            for name, default in (
                ("crystal_ln_sp", None),
                ("crystal_cv_sp", None),
                ("crystal_random_targets", False),
                ("crystal_ln_range", (10.0, 11.5)),
                ("crystal_cv_range", (0.75, 0.95)),
            )
        }
    )
    events = {
        "initial_setpoint": deepcopy(
            environment.get("initial_setpoint")
        ),
        "setpoint_schedule": deepcopy(
            environment.get("setpoint_schedule")
        ),
    }
    return ResolvedEnvSpec(
        scenario=canonical_scenario,
        case_profile=profile,
        reward_spec=resolve_reward_spec(
            declaration.get("reward_spec")
        ),
        control_dt=options["control_dt"],
        episode_steps=options["episode_steps"],
        action_mode=options["action_mode"],
        observation=observation,
        realism=realism,
        events=events,
        model_params=options["model_params"],
        info_level=str(resolved_info_level),
        profile_timing=bool(environment.get("profile_timing", False)),
    )


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return deepcopy(value)


__all__ = ["ResolvedEnvSpec", "resolve_env_spec"]
