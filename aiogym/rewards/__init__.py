"""Versioned reward specifications and reward-engine entry points."""
from .engine import stage_reward
from .registry import (
    get_reward_spec,
    list_reward_specs,
    resolve_reward_spec,
)
from .scalarizers import RewardScaleWrapper
from .specs import RewardResult, RewardSpec, StageRewardContext

__all__ = [
    "RewardResult",
    "RewardScaleWrapper",
    "RewardSpec",
    "StageRewardContext",
    "get_reward_spec",
    "list_reward_specs",
    "resolve_reward_spec",
    "stage_reward",
]
