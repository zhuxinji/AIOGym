"""Small Stable-Baselines3 train, save, and load workflow."""
from __future__ import annotations

import copy
import math
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from aiogym.controllers.policies import SB3CheckpointPolicy
from aiogym.core.io import jsonable, write_json

from ._metadata import environment_metadata
from ._sb3_runtime import (
    ALGORITHMS,
    algorithm_class,
    effective_algorithm_kwargs,
    runtime_versions,
)
from .training_curve import (
    TRAINING_CURVE_SCHEMA_VERSION,
    plot_training_curve,
)


TRAINING_SCHEMA_VERSION = "aiogym.training.v4"


def train(
    *,
    env,
    algorithm: str,
    steps: int,
    output: str | Path,
    seed: int = 0,
    algorithm_kwargs: Mapping | None = None,
    record_every: int = 500,
):
    """Train one SB3 policy without taking ownership of ``env``."""

    if env.unwrapped.benchmark is not None:
        raise ValueError("train does not accept a benchmark environment")
    key = str(algorithm).lower()
    if key not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {', '.join(ALGORITHMS)}")
    resolved_steps = _positive_integer("steps", steps)
    resolved_seed = _nonnegative_integer("seed", seed)
    resolved_record_every = _positive_integer("record_every", record_every)
    requested_kwargs = copy.deepcopy(
        {} if algorithm_kwargs is None else dict(algorithm_kwargs)
    )
    if any(not isinstance(name, str) for name in requested_kwargs):
        raise TypeError("algorithm_kwargs keys must be strings")
    serialized_kwargs = jsonable(requested_kwargs)
    if not isinstance(serialized_kwargs, dict):
        raise TypeError("algorithm_kwargs must be a mapping")

    directory = Path(output)
    if directory.exists() and not directory.is_dir():
        raise FileExistsError(f"training output is not a directory: {directory}")
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(
            f"refusing to create training output in non-empty directory: {directory}"
        )
    directory.mkdir(parents=True, exist_ok=True)

    resolved_kwargs = effective_algorithm_kwargs(
        key,
        resolved_steps,
        serialized_kwargs,
    )
    model_kwargs = copy.deepcopy(resolved_kwargs)
    policy = model_kwargs.pop("policy")
    action_noise = model_kwargs.get("action_noise")
    if action_noise is not None:
        if key not in {"ddpg", "td3"}:
            raise ValueError("action_noise is supported only for ddpg and td3")
        if not isinstance(action_noise, dict):
            raise TypeError("action_noise must be a JSON object")
        if set(action_noise) != {"type", "std"}:
            raise ValueError("action_noise requires exactly 'type' and 'std'")
        noise_type = action_noise["type"]
        if noise_type not in {"normal", "ornstein-uhlenbeck"}:
            raise ValueError(
                "action_noise type must be 'normal' or 'ornstein-uhlenbeck'"
            )
        noise_std = action_noise["std"]
        if isinstance(noise_std, bool) or not isinstance(noise_std, (int, float)):
            raise TypeError("action_noise std must be a positive number")
        if noise_std <= 0:
            raise ValueError("action_noise std must be a positive number")
        from stable_baselines3.common.noise import (
            NormalActionNoise,
            OrnsteinUhlenbeckActionNoise,
        )

        action_dimension = int(np.prod(env.action_space.shape))
        noise_class = (
            NormalActionNoise
            if noise_type == "normal"
            else OrnsteinUhlenbeckActionNoise
        )
        model_kwargs["action_noise"] = noise_class(
            mean=np.zeros(action_dimension, dtype=float),
            sigma=np.full(action_dimension, float(noise_std), dtype=float),
        )
    model = algorithm_class(key)(
        policy,
        env,
        seed=resolved_seed,
        **model_kwargs,
    )
    curve_callback = _training_curve_callback(resolved_record_every)
    model.learn(total_timesteps=resolved_steps, callback=curve_callback)
    curve = curve_callback.payload()
    checkpoint = directory / "model.zip"
    model.save(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SB3 did not create checkpoint: {checkpoint}")

    git_executable = shutil.which("git")
    git_commit = None
    if git_executable is not None:
        completed = subprocess.run(
            [git_executable, "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode == 0:
            git_commit = completed.stdout.strip()

    metadata = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "algorithm": key,
        "steps": resolved_steps,
        "actual_steps": curve["actual_steps"],
        "seed": resolved_seed,
        "record_every": resolved_record_every,
        "algorithm_kwargs": resolved_kwargs,
        "environment": environment_metadata(env),
        "runtime": runtime_versions(),
        "git_commit": git_commit,
    }
    write_json(directory / "metadata.json", metadata)
    curve_path = write_json(directory / "training_curve.json", curve)
    curve_figure = plot_training_curve(
        curve,
        output=directory / "training_curve.svg",
    )
    return {
        **metadata,
        "path": str(directory.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "training_curve": str(curve_path.resolve()),
        "training_curve_figure": str(curve_figure.resolve()),
    }


def load_policy(
    checkpoint: str | Path,
    *,
    algorithm: str,
    env=None,
):
    """Load one explicit SB3 ``model.zip`` as an AIO-Gym policy."""

    key = str(algorithm).lower()
    if key not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {', '.join(ALGORITHMS)}")
    path = Path(checkpoint)
    if path.name != "model.zip" or not path.is_file():
        raise ValueError("checkpoint must be an existing model.zip file")
    model = algorithm_class(key).load(str(path), env=env)
    return SB3CheckpointPolicy(model, algorithm=key, checkpoint=path)


def _positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _training_curve_callback(record_every):
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Stable-Baselines3 is required for train(); install `aiogym[rl]`"
        ) from error

    class TrainingCurveCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self._window_start = 0
            self._window_rewards = []
            self._window_completed = 0
            self._window_terminated = 0
            self._window_truncated = 0
            self._episode_return = 0.0
            self._episode_length = 0
            self._records = []
            self._episodes = []

        def _on_step(self):
            rewards = np.asarray(self.locals["rewards"], dtype=float).reshape(-1)
            dones = np.asarray(self.locals["dones"], dtype=bool).reshape(-1)
            infos = self.locals["infos"]
            if rewards.shape != (1,) or dones.shape != (1,) or len(infos) != 1:
                raise ValueError("train supports exactly one environment")
            reward = float(rewards[0])
            if not math.isfinite(reward):
                raise FloatingPointError("training reward must be finite")
            self._window_rewards.append(reward)
            self._episode_return += reward
            self._episode_length += 1
            if bool(dones[0]):
                info = infos[0]
                if "TimeLimit.truncated" not in info:
                    raise KeyError("SB3 terminal info is missing TimeLimit.truncated")
                truncated = bool(info["TimeLimit.truncated"])
                outcome = "truncated" if truncated else "terminated"
                self._episodes.append(
                    {
                        "end_step": int(self.num_timesteps),
                        "return": float(self._episode_return),
                        "length": int(self._episode_length),
                        "outcome": outcome,
                    }
                )
                self._window_completed += 1
                self._window_truncated += int(truncated)
                self._window_terminated += int(not truncated)
                self._episode_return = 0.0
                self._episode_length = 0
            if int(self.num_timesteps) - self._window_start == record_every:
                self._flush_window(int(self.num_timesteps))
            return True

        def _on_training_end(self):
            if self._window_rewards:
                self._flush_window(int(self.num_timesteps))

        def _flush_window(self, end_step):
            rewards = np.asarray(self._window_rewards, dtype=float)
            self._records.append(
                {
                    "start_step": int(self._window_start),
                    "end_step": int(end_step),
                    "transition_count": int(rewards.size),
                    "mean_reward": float(np.mean(rewards)),
                    "reward_std": float(np.std(rewards)),
                    "minimum_reward": float(np.min(rewards)),
                    "maximum_reward": float(np.max(rewards)),
                    "completed_episodes": int(self._window_completed),
                    "terminated_episodes": int(self._window_terminated),
                    "truncated_episodes": int(self._window_truncated),
                }
            )
            self._window_start = int(end_step)
            self._window_rewards = []
            self._window_completed = 0
            self._window_terminated = 0
            self._window_truncated = 0

        def payload(self):
            if self._window_rewards:
                raise RuntimeError("training curve was requested before training ended")
            actual_steps = int(self.num_timesteps)
            if actual_steps <= 0 or not self._records:
                raise RuntimeError("training produced no curve records")
            return {
                "schema_version": TRAINING_CURVE_SCHEMA_VERSION,
                "record_every": int(record_every),
                "actual_steps": actual_steps,
                "records": list(self._records),
                "episodes": list(self._episodes),
            }

    return TrainingCurveCallback()


__all__ = ["ALGORITHMS", "TRAINING_SCHEMA_VERSION", "load_policy", "train"]
