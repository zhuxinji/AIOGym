"""Canonical RewardSpec evaluation."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .canonical import canonical_stage_reward
from .registry import resolve_reward_spec
from .specs import RewardResult, RewardSpec, StageRewardOverride


def stage_reward(
    model,
    state: Sequence[float],
    action,
    next_state: Sequence[float],
    *,
    setpoint: Sequence[float],
    disturbance: Mapping[str, Any],
    previous_action,
    terminate_on_runaway: bool,
    dt: float = 1.0,
    economic_config: Mapping[str, Any] | None = None,
    reward_override: StageRewardOverride | None = None,
    reward_spec: str | RewardSpec | None = None,
    remaining_physical_time: float = 0.0,
) -> RewardResult:
    spec = resolve_reward_spec(reward_spec)
    return canonical_stage_reward(
        model,
        state,
        action,
        next_state,
        setpoint=setpoint,
        disturbance=disturbance,
        previous_action=previous_action,
        terminate_on_runaway=terminate_on_runaway,
        dt=dt,
        economic_config=economic_config,
        reward_override=reward_override,
        reward_spec=spec,
        remaining_physical_time=remaining_physical_time,
    )
