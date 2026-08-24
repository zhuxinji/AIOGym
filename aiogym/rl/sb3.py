"""Stable-Baselines3 implementation of the algorithm-backend contract."""
from __future__ import annotations

import copy
import importlib
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from aiogym import __version__
from aiogym.core.io import jsonable

from .algorithms import (
    BehaviorCloningHook,
    TrainingStep,
    TrainingStepCallback,
)
from .behavior_cloning import BEHAVIOR_CLONING_ALGORITHMS, behavior_clone


_BUILTIN_MODEL_CLASSES = {
    "ddpg": "stable_baselines3:DDPG",
    "ppo": "stable_baselines3:PPO",
    "sac": "stable_baselines3:SAC",
    "td3": "stable_baselines3:TD3",
}


class SB3CheckpointPolicy:
    """Expose deterministic SB3 prediction through the AIO-Gym Policy API."""

    def __init__(
        self,
        model,
        *,
        algorithm: str,
        checkpoint: str | Path,
    ):
        self.env = None
        self.model = model
        self.algorithm = algorithm.lower()
        self.checkpoint = str(checkpoint)

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
        }


@dataclass(frozen=True)
class SB3AlgorithmBackend:
    """Adapt one SB3 ``BaseAlgorithm`` class to the AIO-Gym workflow."""

    id: str
    model_class: type | str
    behavior_cloning: BehaviorCloningHook | None = None
    requires_dataset: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.model_class, str):
            module_name, separator, attribute_name = self.model_class.partition(
                ":"
            )
            if not separator or not module_name or not attribute_name:
                raise ValueError(
                    "SB3 model_class import path must be 'module:ClassName'"
                )
            return
        if not isinstance(self.model_class, type):
            raise TypeError(
                "model_class must be an SB3 BaseAlgorithm subclass or import path"
            )
        self.resolve_model_class()

    def effective_kwargs(
        self,
        *,
        steps: int,
        values: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del steps
        resolved = copy.deepcopy(dict(values))
        resolved.setdefault("policy", "MlpPolicy")
        serialized = jsonable(resolved)
        if not isinstance(serialized, dict):
            raise TypeError("effective algorithm kwargs must be a mapping")
        return serialized

    def create(
        self,
        *,
        env,
        seed: int,
        algorithm_kwargs: Mapping[str, Any],
    ):
        model_kwargs = copy.deepcopy(dict(algorithm_kwargs))
        try:
            policy = model_kwargs.pop("policy")
        except KeyError as error:
            raise KeyError("SB3 algorithm kwargs are missing policy") from error
        action_noise = model_kwargs.get("action_noise")
        if action_noise is not None:
            model_kwargs["action_noise"] = _materialize_action_noise(
                action_noise,
                action_shape=env.action_space.shape,
            )
        return self.resolve_model_class()(
            policy,
            env,
            seed=seed,
            **model_kwargs,
        )

    def learn(
        self,
        model,
        *,
        steps: int,
        dataset,
        on_step: TrainingStepCallback,
    ) -> int:
        del dataset
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Stable-Baselines3 is required for this algorithm; "
                "install `aiogym[rl]`"
            ) from error
        initial_num_timesteps = int(model.num_timesteps)

        class TransitionCallback(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self.last_step = 0

            def _on_step(self):
                rewards = np.asarray(self.locals["rewards"], dtype=float).reshape(-1)
                dones = np.asarray(self.locals["dones"], dtype=bool).reshape(-1)
                infos = self.locals["infos"]
                if rewards.shape != (1,) or dones.shape != (1,) or len(infos) != 1:
                    raise ValueError("train supports exactly one environment")
                done = bool(dones[0])
                truncated = False
                if done:
                    info = infos[0]
                    if "TimeLimit.truncated" not in info:
                        raise KeyError(
                            "SB3 terminal info is missing TimeLimit.truncated"
                        )
                    truncated = bool(info["TimeLimit.truncated"])
                self.last_step = int(self.num_timesteps) - initial_num_timesteps
                on_step(
                    TrainingStep(
                        step=self.last_step,
                        reward=float(rewards[0]),
                        terminated=done and not truncated,
                        truncated=truncated,
                    )
                )
                return True

        callback = TransitionCallback()
        model.learn(
            total_timesteps=steps,
            callback=callback,
            reset_num_timesteps=False,
        )
        return callback.last_step

    def save(self, model, payload: Path) -> None:
        include = (
            ["replay_buffer"]
            if getattr(model, "replay_buffer", None) is not None
            else None
        )
        model.save(payload, include=include)

    def load(self, payload: Path, *, env=None):
        return self.resolve_model_class().load(str(payload), env=env)

    def policy(self, model, *, checkpoint: Path):
        return SB3CheckpointPolicy(
            model,
            algorithm=self.id,
            checkpoint=checkpoint,
        )

    def runtime_metadata(self) -> Mapping[str, Any]:
        try:
            import stable_baselines3
            import torch
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Stable-Baselines3 is required for this algorithm; "
                "install `aiogym[rl]`"
            ) from error
        model_class = self.resolve_model_class()
        return {
            "python": platform.python_version(),
            "aiogym": __version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
            "model_class": (
                f"{model_class.__module__}:{model_class.__qualname__}"
            ),
        }

    def resolve_model_class(self):
        model_class = self.model_class
        if isinstance(model_class, str):
            module_name, _separator, attribute_name = model_class.partition(":")
            try:
                module = importlib.import_module(module_name)
                model_class = getattr(module, attribute_name)
            except (AttributeError, ModuleNotFoundError) as error:
                raise RuntimeError(
                    f"could not import SB3 model class {self.model_class!r}"
                ) from error
        try:
            from stable_baselines3.common.base_class import BaseAlgorithm
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Stable-Baselines3 is required for this algorithm; "
                "install `aiogym[rl]`"
            ) from error
        if not isinstance(model_class, type) or not issubclass(
            model_class, BaseAlgorithm
        ):
            raise TypeError("model_class must be an SB3 BaseAlgorithm subclass")
        return model_class


def built_in_backends() -> tuple[SB3AlgorithmBackend, ...]:
    return tuple(
        SB3AlgorithmBackend(
            id=algorithm,
            model_class=model_class,
            behavior_cloning=(
                partial(behavior_clone, algorithm=algorithm)
                if algorithm in BEHAVIOR_CLONING_ALGORITHMS
                else None
            ),
        )
        for algorithm, model_class in _BUILTIN_MODEL_CLASSES.items()
    )


def _materialize_action_noise(action_noise, *, action_shape):
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
    try:
        from stable_baselines3.common.noise import (
            NormalActionNoise,
            OrnsteinUhlenbeckActionNoise,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Stable-Baselines3 is required for this algorithm; install `aiogym[rl]`"
        ) from error

    action_dimension = int(np.prod(action_shape))
    noise_class = (
        NormalActionNoise
        if noise_type == "normal"
        else OrnsteinUhlenbeckActionNoise
    )
    return noise_class(
        mean=np.zeros(action_dimension, dtype=float),
        sigma=np.full(action_dimension, float(noise_std), dtype=float),
    )


__all__ = [
    "SB3AlgorithmBackend",
    "SB3CheckpointPolicy",
    "built_in_backends",
]
