"""The single public environment-construction entry point."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym

from aiogym.models.cases import CaseSpec
from aiogym.rewards.specs import RewardSpec

from .env import _AIOGymEnv
from .spec import resolve_env_spec


def make_env(
    scenario: str | None = None,
    *,
    case: str | Mapping[str, Any] | CaseSpec | None = None,
    reward_spec: str | RewardSpec | None = None,
    config: str | Path | Mapping[str, Any] | None = None,
    info_level: str = "full",
) -> gym.Env:
    """Create one resolved physical environment.

    Seeding belongs to ``env.reset(seed=...)``. Track sampling, split
    orchestration, and algorithm preprocessing are intentionally outside this
    public factory.
    """

    if isinstance(case, CaseSpec):
        case = case.profile
    spec = resolve_env_spec(
        scenario,
        case=case,
        reward_spec=reward_spec,
        config=config,
        info_level=info_level,
    )
    return _AIOGymEnv(spec)


__all__ = ["make_env"]
