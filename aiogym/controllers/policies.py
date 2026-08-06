"""Small baseline and learned-policy adapters."""
from __future__ import annotations

from pathlib import Path

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
            "action_contract": "env.action_space",
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
            "action_contract": "env.action_space",
        }


class SB3CheckpointPolicy:
    def __init__(self, model, *, algorithm: str, checkpoint: str | Path):
        self.model = model
        self.algorithm = algorithm.lower()
        self.checkpoint = str(checkpoint)

    @classmethod
    def load(
        cls,
        checkpoint: str | Path,
        *,
        algorithm: str,
        device: str = "auto",
    ):
        key = algorithm.lower()
        try:
            from stable_baselines3 import DDPG, PPO, SAC, TD3
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Stable-Baselines3 is required for checkpoint policies; "
                "install `aiogym[rl]`"
            ) from error
        algorithms = {"sac": SAC, "ppo": PPO, "td3": TD3, "ddpg": DDPG}
        if key not in algorithms:
            raise ValueError(f"unsupported SB3 algorithm {algorithm!r}")
        model = algorithms[key].load(str(checkpoint), device=device)
        return cls(model, algorithm=key, checkpoint=checkpoint)

    def reset(self, seed=None):
        del seed

    def act(self, observation, context):
        del context
        predicted = self.model.predict(observation, deterministic=True)
        action = predicted[0] if isinstance(predicted, tuple) else predicted
        return np.asarray(action, dtype=np.float32)

    def metadata(self):
        return {
            "id": "sb3_checkpoint",
            "kind": "learned_policy",
            "algorithm": self.algorithm,
            "checkpoint": self.checkpoint,
            "action_contract": "env.action_space",
        }


__all__ = ["HoldPolicy", "RandomPolicy", "SB3CheckpointPolicy"]
