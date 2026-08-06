"""The one episode loop shared by all workflows."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .contracts import Policy


@dataclass(frozen=True)
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    terminated: bool
    truncated: bool
    step_index: int
    physical_time: float
    info: Mapping[str, Any]


@dataclass(frozen=True)
class RolloutResult:
    seed: int
    transitions: tuple[Transition, ...]
    policy_metadata: Mapping[str, Any]

    @property
    def episode_return(self) -> float:
        return float(sum(row.reward for row in self.transitions))


def rollout(
    env: gym.Env,
    policy: Policy,
    *,
    seed: int = 0,
    max_steps: int | None = None,
) -> RolloutResult:
    observation, reset_info = env.reset(seed=seed)
    policy.reset(seed=seed)
    info: Mapping[str, Any] = reset_info or {}
    rows: list[Transition] = []
    limit = max_steps
    if limit is None:
        limit = int(getattr(env, "episode_steps", 0)) or None
    while limit is None or len(rows) < limit:
        context = {
            "env": env,
            "info": info,
            "step_index": len(rows),
            "physical_time": float(info.get("physical_time", 0.0)),
            "reference": info.get("reference"),
        }
        action = np.asarray(policy.act(np.asarray(observation), context), dtype=np.float32)
        if action.shape != env.action_space.shape or not env.action_space.contains(action):
            raise ValueError("policy output must belong directly to env.action_space")
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        rows.append(
            Transition(
                observation=np.asarray(observation).copy(),
                action=action.copy(),
                reward=float(reward),
                next_observation=np.asarray(next_observation).copy(),
                terminated=bool(terminated),
                truncated=bool(truncated),
                step_index=len(rows),
                physical_time=float(next_info.get("physical_time", len(rows) + 1)),
                info=dict(next_info),
            )
        )
        observation = next_observation
        info = next_info
        if terminated or truncated:
            break
    return RolloutResult(
        seed=int(seed),
        transitions=tuple(rows),
        policy_metadata=dict(policy.metadata()),
    )


__all__ = ["RolloutResult", "Transition", "rollout"]
