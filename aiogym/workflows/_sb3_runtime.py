"""Stable-Baselines3 construction helpers used by training."""
from __future__ import annotations

import copy
import platform
from collections.abc import Mapping

from aiogym import __version__
from aiogym.core.io import jsonable


ALGORITHMS = ("ddpg", "ppo", "sac", "td3")


def algorithm_class(algorithm: str):
    try:
        from stable_baselines3 import DDPG, PPO, SAC, TD3
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Stable-Baselines3 is required for train(); install `aiogym[rl]`"
        ) from error
    return {"ddpg": DDPG, "ppo": PPO, "sac": SAC, "td3": TD3}[algorithm]


def effective_algorithm_kwargs(
    algorithm: str,
    steps: int,
    values: Mapping,
) -> dict:
    if algorithm == "ppo":
        n_steps = min(2048, max(2, steps))
        resolved = {
            "n_steps": n_steps,
            "batch_size": min(64, n_steps),
            "n_epochs": 1,
            "verbose": 0,
            "device": "cpu",
        }
    else:
        resolved = {
            "learning_starts": 0,
            "buffer_size": max(100, steps + 1),
            "batch_size": min(64, max(2, steps)),
            "train_freq": 1,
            "gradient_steps": 1,
            "verbose": 0,
            "device": "cpu",
        }
    resolved.update(copy.deepcopy(dict(values)))
    resolved.setdefault("policy", "MlpPolicy")
    serialized = jsonable(resolved)
    if not isinstance(serialized, dict):
        raise TypeError("effective algorithm kwargs must be a mapping")
    return serialized


def runtime_versions() -> dict[str, str]:
    try:
        import stable_baselines3
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Stable-Baselines3 is required for train(); install `aiogym[rl]`"
        ) from error
    return {
        "python": platform.python_version(),
        "aiogym": __version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
    }


__all__ = [
    "ALGORITHMS",
    "algorithm_class",
    "effective_algorithm_kwargs",
    "runtime_versions",
]
