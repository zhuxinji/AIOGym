"""Small non-learning baseline policies."""
from __future__ import annotations

import numpy as np


class HoldPolicy:
    def __init__(self, env, action=None):
        self.env = env
        source = env.model.default_action() if action is None else action
        self.action = np.asarray(source, dtype=np.float32)
        if self.action.shape != env.action_space.shape or not env.action_space.contains(
            self.action
        ):
            raise ValueError("hold action must belong to env.action_space")

    def reset(self, seed=None):
        del seed

    def act(self, observation, context):
        del observation, context
        return self.action.copy()

    def metadata(self):
        return {
            "id": "hold",
            "kind": "fixed_action",
            "action": self.action.tolist(),
        }


class RandomPolicy:
    def __init__(self, env):
        self.env = env
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, observation, context):
        del observation, context
        return self.rng.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
        ).astype(np.float32)

    def metadata(self):
        return {
            "id": "random",
            "kind": "uniform_random",
        }


__all__ = ["HoldPolicy", "RandomPolicy"]
