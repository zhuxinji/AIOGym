"""Adapters from goal rewards and applied costs to Gym scalar rewards."""
from __future__ import annotations

import math
from typing import Mapping

import gymnasium as gym


def fixed_penalty_scalarizer(
    goal_reward: float,
    applied_cost_penalties: Mapping[str, float],
    *,
    terminal_failure_cost: float = 0.0,
) -> float:
    """Subtract every applied cost channel from the goal reward."""

    return float(
        goal_reward
        - sum(float(value) for value in applied_cost_penalties.values())
        - float(terminal_failure_cost)
    )


class RewardScaleWrapper(gym.Wrapper):
    """Scale only the scalar seen by an algorithm, preserving reward semantics."""

    def __init__(self, env: gym.Env, scale: float):
        super().__init__(env)
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("reward wrapper scale must be finite and positive")
        self.algorithm_reward_scale = scale

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        wrapped_info = dict(info)
        wrapped_info["unscaled_reward"] = float(reward)
        wrapped_info["reward_wrapper_scale"] = self.algorithm_reward_scale
        return (
            observation,
            float(reward) * self.algorithm_reward_scale,
            terminated,
            truncated,
            wrapped_info,
        )
