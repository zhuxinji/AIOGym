"""Private SAC, TD3, and PPO construction adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..config import RLTrainingConfig
from ..utd import sb3_update_schedule


@dataclass(frozen=True)
class AlgorithmAdapter:
    algorithm_id: str
    off_policy: bool
    builder: Callable

    def build(self, config: RLTrainingConfig, env, *, verbose: int = 0):
        return self.builder(config, env, verbose=verbose)


def _common_kwargs(config: RLTrainingConfig, env, verbose: int) -> dict:
    algorithm = config.algorithm
    return {
        "policy": algorithm["policy"],
        "policy_kwargs": _materialize_policy_kwargs(
            algorithm["policy_kwargs"]
        ),
        "env": env,
        "verbose": int(verbose),
        "seed": config.training_seed,
        "device": config.device,
        "learning_rate": float(algorithm["learning_rate"]),
        "gamma": float(algorithm["gamma"]),
        "tensorboard_log": algorithm["tensorboard_log"],
    }


def _build_off_policy(name: str, config: RLTrainingConfig, env, *, verbose: int):
    try:
        from stable_baselines3 import SAC, TD3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "stable-baselines3 is required; install AIO-Gym with "
            "`pip install 'aiogym[rl]'`"
        ) from exc
    algorithm = config.algorithm
    train_freq, gradient_steps = sb3_update_schedule(
        utd_ratio=config.utd_ratio,
        n_envs=config.n_envs,
        vector_steps=int(algorithm["rollout_vector_steps"]),
    )
    cls = SAC if name == "sac" else TD3
    kwargs = dict(
        batch_size=int(algorithm["batch_size"]),
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        buffer_size=int(config.replay["capacity"]),
        learning_starts=int(config.replay["learning_starts"]),
        tau=float(algorithm["tau"]),
        **_common_kwargs(config, env, verbose),
    )
    if name == "sac":
        kwargs.update(
            ent_coef=algorithm["ent_coef"],
            target_entropy=algorithm["target_entropy"],
        )
    else:
        kwargs.update(
            action_noise=_td3_action_noise(algorithm, env),
            policy_delay=int(algorithm["policy_delay"]),
            target_policy_noise=float(
                algorithm["target_policy_noise"]
            ),
            target_noise_clip=float(algorithm["target_noise_clip"]),
        )
    return cls(**kwargs)


def _build_sac(config, env, *, verbose=0):
    return _build_off_policy("sac", config, env, verbose=verbose)


def _build_td3(config, env, *, verbose=0):
    return _build_off_policy("td3", config, env, verbose=verbose)


def _build_ppo(config: RLTrainingConfig, env, *, verbose: int = 0):
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "stable-baselines3 is required; install AIO-Gym with "
            "`pip install 'aiogym[rl]'`"
        ) from exc
    algorithm = config.algorithm
    return PPO(
        n_steps=int(algorithm["n_steps"]),
        batch_size=int(algorithm["batch_size"]),
        gae_lambda=float(algorithm["gae_lambda"]),
        clip_range=float(algorithm["clip_range"]),
        n_epochs=int(algorithm["n_epochs"]),
        ent_coef=float(algorithm["ent_coef"]),
        vf_coef=float(algorithm["vf_coef"]),
        max_grad_norm=float(algorithm["max_grad_norm"]),
        **_common_kwargs(config, env, verbose),
    )


def _materialize_policy_kwargs(value) -> dict:
    try:
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch is required; install AIO-Gym with "
            "`pip install 'aiogym[rl]'`"
        ) from exc
    activation_classes = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    return {
        "activation_fn": activation_classes[str(value["activation_fn"])],
        "net_arch": {
            str(name): [int(size) for size in layers]
            for name, layers in value["net_arch"].items()
        },
    }


def _td3_action_noise(algorithm, env):
    if algorithm["action_noise"] == "none":
        return None
    try:
        from stable_baselines3.common.noise import NormalActionNoise
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "stable-baselines3 is required; install AIO-Gym with "
            "`pip install 'aiogym[rl]'`"
        ) from exc
    shape = tuple(getattr(env.action_space, "shape", ()) or ())
    if not shape or any(int(size) <= 0 for size in shape):
        raise ValueError(
            "TD3 requires a finite-dimensional action-space shape"
        )
    sigma = float(algorithm["action_noise_sigma"])
    return NormalActionNoise(
        mean=np.zeros(shape, dtype=np.float32),
        sigma=np.full(shape, sigma, dtype=np.float32),
    )


_REGISTRY = {
    "sac": AlgorithmAdapter("sac", True, _build_sac),
    "td3": AlgorithmAdapter("td3", True, _build_td3),
    "ppo": AlgorithmAdapter("ppo", False, _build_ppo),
}


def get_algorithm_adapter(algorithm_id: str) -> AlgorithmAdapter:
    key = str(algorithm_id).lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown RL algorithm {algorithm_id!r}; available: "
            + ", ".join(sorted(_REGISTRY))
        ) from exc


def list_algorithm_adapters() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def load_algorithm_checkpoint(
    algorithm_id: str,
    path,
    *,
    env,
    device: str = "cpu",
):
    """Load an SB3 checkpoint and attach the current vector environment."""

    try:
        from stable_baselines3 import PPO, SAC, TD3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "stable-baselines3 is required; install AIO-Gym with "
            "`pip install 'aiogym[rl]'`"
        ) from exc
    classes = {"sac": SAC, "td3": TD3, "ppo": PPO}
    key = str(algorithm_id).lower()
    if key not in classes:
        get_algorithm_adapter(key)
        raise ValueError(
            f"{key} uses its native state_dict/load_state_dict contract"
        )
    return classes[key].load(path, env=env, device=device)


__all__ = [
    "AlgorithmAdapter",
    "get_algorithm_adapter",
    "load_algorithm_checkpoint",
    "list_algorithm_adapters",
]
