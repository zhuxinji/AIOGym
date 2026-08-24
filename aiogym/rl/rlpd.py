"""RLPD: persistent prior-data and online-replay learning.

This PyTorch implementation follows the central design of Ball, Smith,
Kostrikov, and Levine, "Efficient Online Reinforcement Learning with Offline
Data" (ICML 2023): https://arxiv.org/abs/2302.02948. The reference code is
https://github.com/ikostrikov/rlpd.

AIO-Gym keeps the paper's symmetric replay sampling, high update-to-data ratio,
LayerNorm critic ensemble, and randomized minimum-Q target. It deliberately
uses AIO-Gym Dataset v2, one environment, and the common ``model.zip`` workflow
instead of the reference JAX runner and checkpoint format.
"""
from __future__ import annotations

import copy
import math
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from aiogym import __version__
from aiogym.core.io import jsonable

from .algorithms import TrainingStep, TrainingStepCallback


RLPD_CHECKPOINT_SCHEMA_VERSION = "aiogym.rlpd.v2"
RLPD_PAPER_URL = "https://arxiv.org/abs/2302.02948"
RLPD_REFERENCE_CODE_URL = "https://github.com/ikostrikov/rlpd"
_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0
_DEFAULTS = {
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "critic_subset": 2,
    "device": "auto",
    "gamma": 0.99,
    "hidden_sizes": [256, 256],
    "initial_alpha": 0.1,
    "learning_rate": 3e-4,
    "learning_starts": 10_000,
    "n_critics": 10,
    "offline_ratio": 0.5,
    "tau": 0.005,
    "utd_ratio": 20,
}


@dataclass(frozen=True)
class RLPDAlgorithmBackend:
    """Adapt the native RLPD model to the common algorithm workflow."""

    id: str = "rlpd"
    behavior_cloning: None = None
    requires_dataset: bool = True

    def effective_kwargs(
        self,
        *,
        steps: int,
        values: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del steps
        requested = dict(values)
        unknown = sorted(set(requested) - set(_DEFAULTS))
        if unknown:
            raise ValueError(f"RLPD algorithm kwargs are unknown: {unknown}")
        resolved = copy.deepcopy(_DEFAULTS)
        resolved.update(copy.deepcopy(requested))
        resolved["batch_size"] = _positive_integer(
            "batch_size", resolved["batch_size"]
        )
        resolved["buffer_size"] = _positive_integer(
            "buffer_size", resolved["buffer_size"]
        )
        resolved["critic_subset"] = _positive_integer(
            "critic_subset", resolved["critic_subset"]
        )
        resolved["learning_starts"] = _nonnegative_integer(
            "learning_starts", resolved["learning_starts"]
        )
        resolved["n_critics"] = _positive_integer(
            "n_critics", resolved["n_critics"]
        )
        if resolved["n_critics"] < 2:
            raise ValueError("RLPD n_critics must be at least 2")
        if resolved["critic_subset"] > resolved["n_critics"]:
            raise ValueError("RLPD critic_subset must not exceed n_critics")
        resolved["utd_ratio"] = _positive_integer(
            "utd_ratio", resolved["utd_ratio"]
        )
        resolved["gamma"] = _bounded_float(
            "gamma", resolved["gamma"], low=0.0, high=1.0, high_inclusive=False
        )
        resolved["tau"] = _bounded_float(
            "tau", resolved["tau"], low=0.0, high=1.0, low_inclusive=False
        )
        resolved["initial_alpha"] = _positive_float(
            "initial_alpha", resolved["initial_alpha"]
        )
        resolved["learning_rate"] = _positive_float(
            "learning_rate", resolved["learning_rate"]
        )
        resolved["offline_ratio"] = _bounded_float(
            "offline_ratio", resolved["offline_ratio"], low=0.0, high=1.0
        )
        hidden_sizes = resolved["hidden_sizes"]
        if not isinstance(hidden_sizes, (list, tuple)) or not hidden_sizes:
            raise TypeError("RLPD hidden_sizes must be a non-empty integer sequence")
        resolved["hidden_sizes"] = [
            _positive_integer("hidden_sizes entry", value) for value in hidden_sizes
        ]
        device = resolved["device"]
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("RLPD device must be 'auto', 'cpu', or 'cuda'")
        serialized = jsonable(resolved)
        if not isinstance(serialized, dict):
            raise TypeError("effective RLPD kwargs must be a mapping")
        return serialized

    def create(
        self,
        *,
        env,
        seed: int,
        algorithm_kwargs: Mapping[str, Any],
    ):
        return _RLPDModel(
            env=env,
            seed=seed,
            config=dict(algorithm_kwargs),
            training=True,
        )

    def learn(
        self,
        model,
        *,
        steps: int,
        dataset,
        on_step: TrainingStepCallback,
    ) -> int:
        if dataset is None:
            raise ValueError("RLPD requires a training Dataset")
        model.load_offline_dataset(dataset)
        initial_steps = model.environment_steps
        observation, _ = model.env.reset(seed=model.seed + initial_steps)
        episode = 0
        for step in range(1, steps + 1):
            total_step = initial_steps + step
            if total_step <= model.config["learning_starts"]:
                action = model.random_action()
            else:
                action = model.action(observation, deterministic=False)
            next_observation, reward, terminated, truncated, _ = model.env.step(
                action
            )
            model.add_online_transition(
                observation,
                action,
                reward,
                next_observation,
                terminated,
            )
            if total_step >= max(1, model.config["learning_starts"]):
                model.update()
            model.environment_steps = total_step
            on_step(
                TrainingStep(
                    step=step,
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
            if terminated or truncated:
                episode += 1
                observation, _ = model.env.reset(
                    seed=model.seed + initial_steps + episode
                )
            else:
                observation = next_observation
        return steps

    def save(self, model, payload: Path) -> None:
        torch, _functional, _Actor, _Critic = _torch_components()
        torch.save(model.training_state(), payload)

    def load(self, payload: Path, *, env=None):
        if env is None:
            raise ValueError("RLPD checkpoint loading requires an environment")
        torch, _functional, _Actor, _Critic = _torch_components()
        try:
            state = torch.load(payload, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError) as error:
            raise ValueError("RLPD payload must be a valid Torch checkpoint") from error
        if not isinstance(state, dict) or state.get("schema_version") != (
            RLPD_CHECKPOINT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"RLPD payload schema must be {RLPD_CHECKPOINT_SCHEMA_VERSION}"
            )
        expected = {
            "schema_version",
            "seed",
            "config",
            "observation_dim",
            "action_dim",
            "action_low",
            "action_high",
            "environment_steps",
            "actor_state_dict",
            "critics_state_dict",
            "target_critics_state_dict",
            "actor_optimizer_state_dict",
            "critic_optimizer_state_dict",
            "log_alpha",
            "alpha_optimizer_state_dict",
            "numpy_rng_state",
            "torch_rng_state",
            "torch_cuda_rng_state",
            "online_replay",
            "offline_samples",
            "online_samples",
            "gradient_updates",
        }
        if set(state) != expected:
            raise ValueError("RLPD payload fields are invalid")
        config = dict(
            self.effective_kwargs(steps=1, values=state["config"])
        )
        model = _RLPDModel(
            env=env,
            seed=_nonnegative_integer("checkpoint seed", state["seed"]),
            config=config,
            training=True,
        )
        model.load_training_state(state)
        return model

    def policy(self, model, *, checkpoint: Path):
        return RLPDCheckpointPolicy(model, checkpoint=checkpoint)

    def runtime_metadata(self) -> Mapping[str, Any]:
        torch, _functional, _Actor, _Critic = _torch_components()
        return {
            "python": platform.python_version(),
            "aiogym": __version__,
            "torch": torch.__version__,
            "implementation": RLPD_CHECKPOINT_SCHEMA_VERSION,
            "paper": RLPD_PAPER_URL,
            "reference_code": RLPD_REFERENCE_CODE_URL,
        }


class RLPDCheckpointPolicy:
    """Expose deterministic RLPD inference through the Policy contract."""

    def __init__(self, model, *, checkpoint: str | Path):
        self.env = model.env
        self.model = model
        self.checkpoint = str(checkpoint)

    def reset(self, seed=None):
        del seed

    def act(self, observation, context):
        del context
        return self.model.action(observation, deterministic=True)

    def metadata(self):
        return {
            "id": "rlpd_checkpoint",
            "kind": "learned_policy",
            "algorithm": "rlpd",
            "checkpoint": self.checkpoint,
        }


class _RLPDModel:
    def __init__(self, *, env, seed: int, config: dict[str, Any], training: bool):
        torch, _functional, Actor, Critic = _torch_components()
        self.env = env
        self.seed = _nonnegative_integer("seed", seed)
        self.config = copy.deepcopy(config)
        self.observation_dim, self.action_dim = _space_dimensions(env)
        self.action_low = np.asarray(env.action_space.low, dtype=np.float32).copy()
        self.action_high = np.asarray(env.action_space.high, dtype=np.float32).copy()
        self.device = _resolve_device(torch, config["device"])
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.actor = Actor(
            self.observation_dim,
            self.action_dim,
            tuple(config["hidden_sizes"]),
        ).to(self.device)
        self.training = bool(training)
        self.offline = None
        self.online = None
        self.offline_samples = 0
        self.online_samples = 0
        self.gradient_updates = 0
        self.environment_steps = 0
        if not self.training:
            return
        self.critics = torch.nn.ModuleList(
            [
                Critic(
                    self.observation_dim,
                    self.action_dim,
                    tuple(config["hidden_sizes"]),
                )
                for _ in range(config["n_critics"])
            ]
        ).to(self.device)
        self.target_critics = copy.deepcopy(self.critics).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config["learning_rate"]
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critics.parameters(), lr=config["learning_rate"]
        )
        self.log_alpha = torch.tensor(
            math.log(config["initial_alpha"]),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config["learning_rate"]
        )
        self.target_entropy = -float(self.action_dim)
        self.online = _OnlineReplay(
            config["buffer_size"], self.observation_dim, self.action_dim
        )

    def load_offline_dataset(self, reader) -> None:
        if not self.training:
            raise RuntimeError("inference-only RLPD model cannot load training data")
        episodes = tuple(reader.iter_episodes())
        observation = np.concatenate(
            [episode.array("observation") for episode in episodes], axis=0
        ).astype(np.float32, copy=False)
        physical_action = np.concatenate(
            [episode.array("commanded_action") for episode in episodes], axis=0
        ).astype(np.float32, copy=False)
        reward = np.concatenate(
            [episode.array("reward") for episode in episodes], axis=0
        ).astype(np.float32, copy=False)
        next_observation = np.concatenate(
            [episode.array("next_observation") for episode in episodes], axis=0
        ).astype(np.float32, copy=False)
        terminated = np.concatenate(
            [episode.array("terminated") for episode in episodes], axis=0
        ).astype(np.bool_, copy=False)
        expected_transitions = reader.transition_count
        expected_observation = (expected_transitions, self.observation_dim)
        expected_action = (expected_transitions, self.action_dim)
        if observation.shape != expected_observation:
            raise ValueError("RLPD Dataset observation shape is incompatible")
        if next_observation.shape != expected_observation:
            raise ValueError("RLPD Dataset next_observation shape is incompatible")
        if physical_action.shape != expected_action:
            raise ValueError("RLPD Dataset commanded_action shape is incompatible")
        if reward.shape != (expected_transitions,):
            raise ValueError("RLPD Dataset reward shape is incompatible")
        if terminated.shape != (expected_transitions,):
            raise ValueError("RLPD Dataset terminated shape is incompatible")
        normalized_action = self.normalize_action(physical_action)
        arrays = {
            "observation": observation,
            "action": normalized_action,
            "reward": reward,
            "next_observation": next_observation,
            "bootstrap_mask": np.logical_not(terminated).astype(np.float32),
        }
        for name, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError(f"RLPD Dataset {name} must be finite")
            value.setflags(write=False)
        self.offline = arrays

    def random_action(self) -> np.ndarray:
        return self.rng.uniform(self.action_low, self.action_high).astype(np.float32)

    def normalize_action(self, action) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        normalized = 2.0 * (value - self.action_low) / (
            self.action_high - self.action_low
        ) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)

    def physical_action(self, normalized) -> np.ndarray:
        value = np.asarray(normalized, dtype=np.float32)
        physical = self.action_low + 0.5 * (value + 1.0) * (
            self.action_high - self.action_low
        )
        return np.clip(physical, self.action_low, self.action_high).astype(
            np.float32, copy=False
        )

    def action(self, observation, *, deterministic: bool) -> np.ndarray:
        torch, _functional, _Actor, _Critic = _torch_components()
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (self.observation_dim,) or not np.isfinite(value).all():
            raise ValueError("RLPD observation is invalid")
        with torch.no_grad():
            tensor = torch.as_tensor(value[None, :], device=self.device)
            normalized = self.actor.act(tensor, deterministic=deterministic)
        return self.physical_action(normalized.cpu().numpy()[0])

    def add_online_transition(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminated,
    ) -> None:
        self.online.add(
            observation=np.asarray(observation, dtype=np.float32),
            action=self.normalize_action(action),
            reward=float(reward),
            next_observation=np.asarray(next_observation, dtype=np.float32),
            bootstrap_mask=0.0 if terminated else 1.0,
        )

    def sample_batch(self) -> dict[str, np.ndarray]:
        if self.offline is None:
            raise RuntimeError("RLPD offline Dataset is not loaded")
        batch_size = self.config["batch_size"]
        offline_count = int(batch_size * self.config["offline_ratio"])
        online_count = batch_size - offline_count
        if online_count and (self.online is None or len(self.online) == 0):
            raise RuntimeError("RLPD online replay is empty")
        offline_indices = self.rng.integers(
            0, len(self.offline["reward"]), size=offline_count
        )
        online = self.online.sample(online_count, self.rng)
        combined = {
            name: np.concatenate(
                [self.offline[name][offline_indices], online[name]], axis=0
            )
            for name in self.offline
        }
        order = self.rng.permutation(batch_size)
        self.offline_samples += offline_count
        self.online_samples += online_count
        return {name: value[order] for name, value in combined.items()}

    def update(self) -> None:
        torch, functional, _Actor, _Critic = _torch_components()
        alpha = self.log_alpha.exp().detach()
        for _ in range(self.config["utd_ratio"]):
            batch = self._torch_batch(self.sample_batch())
            with torch.no_grad():
                next_action, next_log_probability = self.actor.sample(
                    batch["next_observation"]
                )
                selected = self.rng.choice(
                    self.config["n_critics"],
                    size=self.config["critic_subset"],
                    replace=False,
                )
                next_q = torch.stack(
                    [
                        self.target_critics[int(index)](
                            batch["next_observation"], next_action
                        )
                        for index in selected
                    ]
                ).min(dim=0).values
                target = batch["reward"] + self.config["gamma"] * batch[
                    "bootstrap_mask"
                ] * (next_q - alpha * next_log_probability)
            critic_loss = torch.stack(
                [
                    functional.mse_loss(
                        critic(batch["observation"], batch["action"]), target
                    )
                    for critic in self.critics
                ]
            ).mean()
            if not bool(torch.isfinite(critic_loss)):
                raise FloatingPointError("RLPD critic loss became non-finite")
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizer.step()
            self.gradient_updates += 1

        actor_batch = self._torch_batch(self.sample_batch())
        sampled_action, log_probability = self.actor.sample(
            actor_batch["observation"]
        )
        selected = self.rng.choice(
            self.config["n_critics"],
            size=self.config["critic_subset"],
            replace=False,
        )
        policy_q = torch.stack(
            [
                self.critics[int(index)](
                    actor_batch["observation"], sampled_action
                )
                for index in selected
            ]
        ).min(dim=0).values
        actor_loss = (alpha * log_probability - policy_q).mean()
        if not bool(torch.isfinite(actor_loss)):
            raise FloatingPointError("RLPD actor loss became non-finite")
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_probability.detach() + self.target_entropy)
        ).mean()
        if not bool(torch.isfinite(alpha_loss)):
            raise FloatingPointError("RLPD entropy loss became non-finite")
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for critic_parameter, target_parameter in zip(
                self.critics.parameters(), self.target_critics.parameters()
            ):
                target_parameter.mul_(1.0 - self.config["tau"])
                target_parameter.add_(self.config["tau"] * critic_parameter)

    def _torch_batch(self, batch):
        torch, _functional, _Actor, _Critic = _torch_components()
        return {
            name: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for name, value in batch.items()
        }

    def training_state(self) -> dict[str, Any]:
        if not self.training:
            raise RuntimeError("RLPD training state requires a trainable model")
        torch, _functional, _Actor, _Critic = _torch_components()
        return {
            "schema_version": RLPD_CHECKPOINT_SCHEMA_VERSION,
            "seed": self.seed,
            "config": copy.deepcopy(self.config),
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low.copy(),
            "action_high": self.action_high.copy(),
            "environment_steps": self.environment_steps,
            "actor_state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.actor.state_dict().items()
            },
            "critics_state_dict": _cpu_state_dict(self.critics.state_dict()),
            "target_critics_state_dict": _cpu_state_dict(
                self.target_critics.state_dict()
            ),
            "actor_optimizer_state_dict": copy.deepcopy(
                self.actor_optimizer.state_dict()
            ),
            "critic_optimizer_state_dict": copy.deepcopy(
                self.critic_optimizer.state_dict()
            ),
            "log_alpha": self.log_alpha.detach().cpu().clone(),
            "alpha_optimizer_state_dict": copy.deepcopy(
                self.alpha_optimizer.state_dict()
            ),
            "numpy_rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "online_replay": self.online.state(),
            "offline_samples": self.offline_samples,
            "online_samples": self.online_samples,
            "gradient_updates": self.gradient_updates,
        }

    def load_training_state(self, state) -> None:
        if not self.training:
            raise RuntimeError("RLPD training state requires a trainable model")
        torch, _functional, _Actor, _Critic = _torch_components()
        self.validate_checkpoint_dimensions(state)
        self.environment_steps = _nonnegative_integer(
            "checkpoint environment_steps", state["environment_steps"]
        )
        self.actor.load_state_dict(state["actor_state_dict"], strict=True)
        self.critics.load_state_dict(state["critics_state_dict"], strict=True)
        self.target_critics.load_state_dict(
            state["target_critics_state_dict"], strict=True
        )
        self.actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(
            state["critic_optimizer_state_dict"]
        )
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer_state_dict"])
        self.rng.bit_generator.state = copy.deepcopy(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        cuda_state = state["torch_cuda_rng_state"]
        if self.device.type == "cuda":
            if cuda_state is None:
                raise ValueError("RLPD CUDA checkpoint is missing CUDA RNG state")
            torch.cuda.set_rng_state_all(cuda_state)
        self.online.load_state(state["online_replay"])
        self.offline_samples = _nonnegative_integer(
            "checkpoint offline_samples", state["offline_samples"]
        )
        self.online_samples = _nonnegative_integer(
            "checkpoint online_samples", state["online_samples"]
        )
        self.gradient_updates = _nonnegative_integer(
            "checkpoint gradient_updates", state["gradient_updates"]
        )

    def validate_checkpoint_dimensions(self, state) -> None:
        if (
            state["observation_dim"] != self.observation_dim
            or state["action_dim"] != self.action_dim
            or not np.array_equal(state["action_low"], self.action_low)
            or not np.array_equal(state["action_high"], self.action_high)
        ):
            raise ValueError("RLPD checkpoint dimensions or action bounds do not match")


class _OnlineReplay:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int):
        self.capacity = capacity
        self.observation = np.empty((capacity, observation_dim), dtype=np.float32)
        self.action = np.empty((capacity, action_dim), dtype=np.float32)
        self.reward = np.empty(capacity, dtype=np.float32)
        self.next_observation = np.empty(
            (capacity, observation_dim), dtype=np.float32
        )
        self.bootstrap_mask = np.empty(capacity, dtype=np.float32)
        self.size = 0
        self.position = 0

    def __len__(self):
        return self.size

    def add(
        self,
        *,
        observation,
        action,
        reward,
        next_observation,
        bootstrap_mask,
    ) -> None:
        values = (observation, action, reward, next_observation, bootstrap_mask)
        if not all(np.isfinite(value).all() for value in map(np.asarray, values)):
            raise ValueError("RLPD online transition must be finite")
        index = self.position
        self.observation[index] = observation
        self.action[index] = action
        self.reward[index] = reward
        self.next_observation[index] = next_observation
        self.bootstrap_mask[index] = bootstrap_mask
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        indices = rng.integers(0, self.size, size=count)
        return {
            "observation": self.observation[indices],
            "action": self.action[indices],
            "reward": self.reward[indices],
            "next_observation": self.next_observation[indices],
            "bootstrap_mask": self.bootstrap_mask[indices],
        }

    def state(self) -> dict[str, Any]:
        length = self.capacity if self.size == self.capacity else self.size
        return {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "observation": self.observation[:length].copy(),
            "action": self.action[:length].copy(),
            "reward": self.reward[:length].copy(),
            "next_observation": self.next_observation[:length].copy(),
            "bootstrap_mask": self.bootstrap_mask[:length].copy(),
        }

    def load_state(self, state) -> None:
        expected = {
            "capacity",
            "size",
            "position",
            "observation",
            "action",
            "reward",
            "next_observation",
            "bootstrap_mask",
        }
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("RLPD online replay checkpoint fields are invalid")
        capacity = _positive_integer("online replay capacity", state["capacity"])
        size = _nonnegative_integer("online replay size", state["size"])
        position = _nonnegative_integer("online replay position", state["position"])
        if (
            capacity != self.capacity
            or size > capacity
            or position >= capacity
            or (size < capacity and position != size)
        ):
            raise ValueError("RLPD online replay checkpoint bounds are invalid")
        length = capacity if size == capacity else size
        arrays = {
            "observation": (length, self.observation.shape[1]),
            "action": (length, self.action.shape[1]),
            "reward": (length,),
            "next_observation": (length, self.next_observation.shape[1]),
            "bootstrap_mask": (length,),
        }
        for name, shape in arrays.items():
            value = np.asarray(state[name], dtype=np.float32)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"RLPD online replay checkpoint {name} is invalid")
            getattr(self, name)[:length] = value
        self.size = size
        self.position = position


def _cpu_state_dict(state) -> dict[str, Any]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state.items()
    }


@lru_cache(maxsize=1)
def _torch_components():
    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError as error:
        raise RuntimeError("RLPD requires PyTorch; install `aiogym[rl]`") from error

    class Actor(torch.nn.Module):
        def __init__(self, observation_dim, action_dim, hidden_sizes):
            super().__init__()
            layers = []
            input_dim = observation_dim
            for hidden in hidden_sizes:
                layers.extend((torch.nn.Linear(input_dim, hidden), torch.nn.ReLU()))
                input_dim = hidden
            self.trunk = torch.nn.Sequential(*layers)
            self.mean = torch.nn.Linear(input_dim, action_dim)
            self.log_standard_deviation = torch.nn.Linear(input_dim, action_dim)

        def forward(self, observation):
            hidden = self.trunk(observation)
            return self.mean(hidden), self.log_standard_deviation(hidden).clamp(
                _LOG_STD_MIN, _LOG_STD_MAX
            )

        def sample(self, observation):
            mean, log_standard_deviation = self(observation)
            distribution = torch.distributions.Normal(
                mean, log_standard_deviation.exp()
            )
            raw_action = distribution.rsample()
            action = torch.tanh(raw_action)
            log_probability = distribution.log_prob(raw_action).sum(dim=-1)
            log_probability -= torch.log(1.0 - action.square() + 1e-6).sum(
                dim=-1
            )
            return action, log_probability

        def act(self, observation, *, deterministic):
            mean, _log_standard_deviation = self(observation)
            if deterministic:
                return torch.tanh(mean)
            action, _log_probability = self.sample(observation)
            return action

    class Critic(torch.nn.Module):
        def __init__(self, observation_dim, action_dim, hidden_sizes):
            super().__init__()
            layers = []
            input_dim = observation_dim + action_dim
            for hidden in hidden_sizes:
                layers.extend(
                    (
                        torch.nn.Linear(input_dim, hidden),
                        torch.nn.LayerNorm(hidden),
                        torch.nn.ReLU(),
                    )
                )
                input_dim = hidden
            layers.append(torch.nn.Linear(input_dim, 1))
            self.network = torch.nn.Sequential(*layers)

        def forward(self, observation, action):
            return self.network(
                torch.cat((observation, action), dim=-1)
            ).squeeze(dim=-1)

    return torch, functional, Actor, Critic


def _space_dimensions(env) -> tuple[int, int]:
    if len(env.observation_space.shape) != 1 or len(env.action_space.shape) != 1:
        raise ValueError("RLPD requires flat observation and action spaces")
    observation_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    if (
        observation_dim <= 0
        or action_dim <= 0
        or not np.isfinite(action_low).all()
        or not np.isfinite(action_high).all()
        or np.any(action_high <= action_low)
    ):
        raise ValueError("RLPD requires finite, increasing action bounds")
    return observation_dim, action_dim


def _resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RLPD device='cuda' requires CUDA")
    return torch.device(requested)


def _positive_integer(name, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"RLPD {name} must be a positive integer")
    return value


def _nonnegative_integer(name, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"RLPD {name} must be a non-negative integer")
    return value


def _positive_float(name, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"RLPD {name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"RLPD {name} must be finite and positive")
    return number


def _bounded_float(
    name,
    value,
    *,
    low,
    high,
    low_inclusive=True,
    high_inclusive=True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"RLPD {name} must be a number")
    number = float(value)
    low_valid = number >= low if low_inclusive else number > low
    high_valid = number <= high if high_inclusive else number < high
    if not math.isfinite(number) or not low_valid or not high_valid:
        left = "[" if low_inclusive else "("
        right = "]" if high_inclusive else ")"
        raise ValueError(f"RLPD {name} must be in {left}{low}, {high}{right}")
    return number


__all__ = [
    "RLPDAlgorithmBackend",
    "RLPDCheckpointPolicy",
    "RLPD_CHECKPOINT_SCHEMA_VERSION",
]
