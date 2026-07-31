"""Private SAC, TD3, and PPO construction adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
    return cls(
        batch_size=int(algorithm["batch_size"]),
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        buffer_size=int(config.replay["capacity"]),
        learning_starts=int(config.replay["learning_starts"]),
        tau=float(algorithm["tau"]),
        **_common_kwargs(config, env, verbose),
    )


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
        **_common_kwargs(config, env, verbose),
    )


def _build_lagrangian_sac(
    config: RLTrainingConfig,
    env,
    *,
    verbose: int = 0,
):
    del verbose
    from .constrained import LagrangianSAC

    algorithm = config.algorithm
    if "cost_limit" not in algorithm:
        raise ValueError(
            "lagrangian_sac requires algorithm.cost_limit"
        )
    return LagrangianSAC(
        observation_dim=int(env.observation_space.shape[0]),
        action_dim=int(env.action_space.shape[0]),
        cost_limit=float(algorithm["cost_limit"]),
        hidden=int(algorithm.get("hidden", 256)),
        gamma=float(algorithm.get("gamma", 0.99)),
        tau=float(algorithm.get("tau", 0.005)),
        entropy_coefficient=float(
            algorithm.get("entropy_coefficient", 0.1)
        ),
        learning_rate=float(algorithm.get("learning_rate", 3e-4)),
        multiplier_learning_rate=float(
            algorithm.get("multiplier_learning_rate", 1e-3)
        ),
        batch_size=int(algorithm.get("batch_size", 256)),
        replay_capacity=int(config.replay.get("capacity", 1_000_000)),
        device=config.device,
        seed=config.training_seed,
        cost_channels=tuple(
            algorithm.get(
                "cost_channels",
                ("soft_safety", "hard_safety"),
            )
        ),
    )


_REGISTRY = {
    "lagrangian_sac": AlgorithmAdapter(
        "lagrangian_sac",
        True,
        _build_lagrangian_sac,
    ),
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
