"""Public environment construction from direct arguments or config mappings."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ._internal.config import load_config
from .env import AIOGymEnv


def make_env(
    scenario: Any = "cascade",
    seed: int | None = None,
    config: str | Path | Mapping[str, Any] | None = None,
    **overrides,
) -> AIOGymEnv:
    """Create an AIO-Gym environment from direct arguments or a config mapping."""

    data = load_config(config)
    removed = sorted(set(data) & {"model", "env"})
    if removed:
        raise ValueError(
            f"unsupported environment config field(s): {', '.join(removed)}; "
            "use 'scenario' and 'environment'"
        )
    nested_environment = data.pop("environment", {})
    if not isinstance(nested_environment, Mapping):
        raise TypeError("config['environment'] must be a mapping of environment options")
    data.update(nested_environment)
    data.update(overrides)
    scenario = data.pop("scenario", scenario)
    env = AIOGymEnv(scenario, **data)
    if seed is not None:
        env.reset(seed=seed)
    return env
