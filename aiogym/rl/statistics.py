"""Shared action and observation preprocessing for RL trainers."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class RunningObservationStatistics:
    """Numerically stable running mean/variance with resumable state."""

    shape: tuple[int, ...]
    count: int = 0

    def __post_init__(self) -> None:
        self.shape = tuple(int(value) for value in self.shape)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.m2 = np.zeros(self.shape, dtype=np.float64)
        if self.count != 0:
            raise ValueError("construct non-empty statistics with from_state_dict")

    def update(self, observations) -> None:
        batch = np.asarray(observations, dtype=np.float64)
        if batch.shape == self.shape:
            batch = batch.reshape((1, *self.shape))
        if batch.shape[1:] != self.shape:
            raise ValueError(
                f"observation shape must end in {self.shape}, got {batch.shape}"
            )
        if not np.all(np.isfinite(batch)):
            raise ValueError("observations must be finite")
        for value in batch:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(self.shape, dtype=np.float64)
        return self.m2 / self.count

    def normalize(
        self,
        observation,
        *,
        epsilon: float = 1e-8,
        clip: float | None = 10.0,
    ) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float64)
        normalized = (value - self.mean) / np.sqrt(self.variance + epsilon)
        if clip is not None:
            normalized = np.clip(normalized, -float(clip), float(clip))
        return normalized.astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "count": self.count,
            "mean": self.mean.copy(),
            "m2": self.m2.copy(),
        }

    @classmethod
    def from_state_dict(cls, state) -> "RunningObservationStatistics":
        result = cls(tuple(int(value) for value in state["shape"]))
        result.count = int(state["count"])
        result.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        result.m2 = np.asarray(state["m2"], dtype=np.float64).copy()
        if result.mean.shape != result.shape or result.m2.shape != result.shape:
            raise ValueError("normalization statistic shape mismatch")
        return result


class NormalizedActionWrapper(gym.ActionWrapper):
    """Expose ``[-1, 1]`` while preserving the environment's command bounds."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        source = env.action_space
        if not isinstance(source, spaces.Box):
            raise TypeError("normalized actions require a Box action space")
        self._physical_low = np.asarray(source.low, dtype=np.float64)
        self._physical_high = np.asarray(source.high, dtype=np.float64)
        if not (
            np.all(np.isfinite(self._physical_low))
            and np.all(np.isfinite(self._physical_high))
            and np.all(self._physical_high > self._physical_low)
        ):
            raise ValueError("physical action bounds must be finite and ordered")
        self.action_space = spaces.Box(
            -1.0,
            1.0,
            shape=source.shape,
            dtype=np.float32,
        )
        self._last_policy_action = None
        self._last_physical_action = None

    def __getattr__(self, name):
        wrapped = self.__dict__.get("env")
        if wrapped is None:
            raise AttributeError(name)
        return getattr(wrapped, name)

    def action(self, action):
        policy = np.asarray(action, dtype=np.float64)
        if policy.shape != self.action_space.shape:
            raise ValueError(
                f"policy action shape must be {self.action_space.shape}"
            )
        if not np.all(np.isfinite(policy)):
            raise ValueError("policy action must be finite")
        policy = np.clip(policy, -1.0, 1.0)
        physical = self.denormalize(policy)
        self._last_policy_action = policy.astype(np.float32)
        self._last_physical_action = physical.astype(np.float32)
        return self._last_physical_action

    def reverse_action(self, action):
        return self.normalize(action)

    def normalize(self, physical_action) -> np.ndarray:
        physical = np.asarray(physical_action, dtype=np.float64)
        normalized = (
            2.0
            * (physical - self._physical_low)
            / (self._physical_high - self._physical_low)
            - 1.0
        )
        return normalized.astype(np.float32)

    def denormalize(self, policy_action) -> np.ndarray:
        policy = np.asarray(policy_action, dtype=np.float64)
        physical = self._physical_low + 0.5 * (
            policy + 1.0
        ) * (self._physical_high - self._physical_low)
        return physical.astype(np.float32)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        result_info = dict(info)
        result_info["action_policy_normalized"] = (
            self._last_policy_action.copy()
        )
        result_info.setdefault(
            "action_commanded_physical",
            self._last_physical_action.copy(),
        )
        result_info.setdefault(
            "action_applied_physical",
            np.asarray(
                getattr(self.unwrapped, "last_act", self._last_physical_action),
                dtype=np.float32,
            ),
        )
        return observation, reward, terminated, truncated, result_info


class ObservationNormalizationWrapper(gym.Wrapper):
    """Apply shared running statistics; validation wrappers never update them."""

    def __init__(
        self,
        env: gym.Env,
        statistics: RunningObservationStatistics | None = None,
        *,
        training: bool,
        clip: float = 10.0,
    ) -> None:
        super().__init__(env)
        source = env.observation_space
        if not isinstance(source, spaces.Box):
            raise TypeError("observation normalization requires a Box space")
        self.statistics = statistics or RunningObservationStatistics(source.shape)
        if self.statistics.shape != source.shape:
            raise ValueError("normalizer and observation-space shapes must match")
        self.training = bool(training)
        self.clip = float(clip)
        self.observation_space = spaces.Box(
            -self.clip,
            self.clip,
            source.shape,
            dtype=np.float32,
        )

    def __getattr__(self, name):
        wrapped = self.__dict__.get("env")
        if wrapped is None:
            raise AttributeError(name)
        return getattr(wrapped, name)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._process(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            self._process(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def _process(self, observation):
        if self.training:
            self.statistics.update(observation)
        return self.statistics.normalize(observation, clip=self.clip)

    def set_training(self, training: bool) -> None:
        self.training = bool(training)

    def normalization_state(self) -> dict:
        return copy.deepcopy(self.statistics.state_dict())


__all__ = [
    "NormalizedActionWrapper",
    "ObservationNormalizationWrapper",
    "RunningObservationStatistics",
]
