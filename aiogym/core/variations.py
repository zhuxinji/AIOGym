"""Shared training-episode, physical-disturbance, and channel wrappers."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from typing import Any

import gymnasium as gym
import numpy as np

from .contracts import EpisodeSampler, TrainingDisturbanceSampler
from .specs import EpisodeSpec


_NOISE_DEFAULTS = {"std": 0.01, "bias_std": 0.002}
_DELAY_DEFAULTS = {
    "observation_steps": (0, 2),
    "action_steps": (0, 1),
}
_FAULT_DEFAULTS = {
    "probability": 0.05,
    "severity": (0.2, 0.5),
    "duration_steps": (20, 100),
}


def resolve_noise_option(value: Any) -> dict[str, float] | None:
    if value is False:
        return None
    if value is True:
        return dict(_NOISE_DEFAULTS)
    if isinstance(value, bool):
        raise TypeError("noise must be a boolean, number, or mapping")
    if isinstance(value, (int, float)):
        resolved = {"std": float(value), "bias_std": 0.0}
    elif isinstance(value, Mapping):
        unknown = set(value) - set(_NOISE_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown noise fields: {sorted(unknown)}")
        resolved = {**_NOISE_DEFAULTS, **dict(value)}
        resolved = {name: float(item) for name, item in resolved.items()}
    else:
        raise TypeError("noise must be a boolean, number, or mapping")
    if not all(math.isfinite(item) and item >= 0 for item in resolved.values()):
        raise ValueError("noise std and bias_std must be finite and non-negative")
    return resolved


def resolve_delay_option(value: Any) -> dict[str, tuple[int, int]] | None:
    if value is False:
        return None
    if value is True:
        return dict(_DELAY_DEFAULTS)
    if not isinstance(value, Mapping):
        raise TypeError("delay must be a boolean or mapping")
    unknown = set(value) - set(_DELAY_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown delay fields: {sorted(unknown)}")
    merged = {**_DELAY_DEFAULTS, **dict(value)}
    return {
        name: _integer_range(f"delay {name}", item, minimum=0)
        for name, item in merged.items()
    }


def resolve_fault_option(value: Any) -> dict[str, Any] | None:
    if value is False:
        return None
    if value is True:
        return dict(_FAULT_DEFAULTS)
    if not isinstance(value, Mapping):
        raise TypeError("fault must be a boolean or mapping")
    unknown = set(value) - set(_FAULT_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown fault fields: {sorted(unknown)}")
    merged = {**_FAULT_DEFAULTS, **dict(value)}
    probability = float(merged["probability"])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("fault probability must be within [0, 1]")
    severity = _float_range("fault severity", merged["severity"], 0.0, 1.0)
    duration = _integer_range(
        "fault duration_steps", merged["duration_steps"], minimum=1
    )
    return {
        "probability": probability,
        "severity": severity,
        "duration_steps": duration,
    }


def _integer_range(name: str, value: Any, *, minimum: int) -> tuple[int, int]:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or two-integer range")
    if isinstance(value, int):
        resolved = (value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2 or any(isinstance(item, bool) for item in value):
            raise ValueError(f"{name} must contain exactly two integers")
        if any(not isinstance(item, int) for item in value):
            raise TypeError(f"{name} must contain exactly two integers")
        resolved = (int(value[0]), int(value[1]))
    else:
        raise TypeError(f"{name} must be an integer or two-integer range")
    if resolved[0] < minimum or resolved[1] < resolved[0]:
        raise ValueError(f"{name} must be an ordered range from {minimum}")
    return resolved


def _float_range(
    name: str, value: Any, minimum: float, maximum: float
) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a two-number range")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    resolved = (float(value[0]), float(value[1]))
    if (
        not all(math.isfinite(item) for item in resolved)
        or resolved[0] < minimum
        or resolved[1] > maximum
        or resolved[1] < resolved[0]
    ):
        raise ValueError(f"{name} must be ordered within [{minimum}, {maximum}]")
    return resolved


def _reset_rng(seed: int | None, stream: int, current: np.random.Generator):
    if seed is None:
        return current
    return np.random.default_rng(np.random.SeedSequence([int(seed), stream]))


def _sample_integer(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    return int(rng.integers(bounds[0], bounds[1] + 1))


class EpisodeSamplingWrapper(gym.Wrapper):
    """Resolve scenario-owned tracking conditions and physical disturbances."""

    def __init__(
        self,
        env: gym.Env,
        *,
        episode_sampler: EpisodeSampler | None,
        disturbance_sampler: TrainingDisturbanceSampler | None,
        reward_id: str,
    ) -> None:
        super().__init__(env)
        if episode_sampler is None and disturbance_sampler is None:
            raise ValueError("episode or disturbance sampling must be enabled")
        self._episode_sampler = episode_sampler
        self._disturbance_sampler = disturbance_sampler
        self._reward_id = reward_id
        self._episode_rng = np.random.default_rng()
        self._disturbance_rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options:
            raise ValueError("reset options cannot override a sampled training episode")
        self._episode_rng = _reset_rng(seed, 101, self._episode_rng)
        self._disturbance_rng = _reset_rng(seed, 102, self._disturbance_rng)
        if self._episode_sampler is None:
            episode = self.unwrapped.default_episode
            family = "tracking"
        else:
            episode, family = self._episode_sampler(
                self.unwrapped.model, self._episode_rng, self._reward_id
            )
            if not isinstance(episode, EpisodeSpec):
                raise TypeError("sample_training_episode must return an EpisodeSpec")
            if not isinstance(family, str) or not family.strip():
                raise ValueError("sample_training_episode family must be non-empty")
        if self._disturbance_sampler is not None:
            schedule = self._disturbance_sampler(
                self.unwrapped.model, self._disturbance_rng
            )
            if not isinstance(schedule, Mapping):
                raise TypeError("sample_training_disturbance must return a mapping")
            episode = replace(episode, disturbance_schedule=schedule)
        self.unwrapped.episode_family = family
        return self.env.reset(seed=seed, options={"episode": episode})


class ActionChannelWrapper(gym.Wrapper):
    """Apply episode-sampled command delay and loss-of-effectiveness faults."""

    def __init__(
        self,
        env: gym.Env,
        *,
        delay: Mapping[str, tuple[int, int]] | None,
        fault: Mapping[str, Any] | None,
    ) -> None:
        super().__init__(env)
        self._delay_config = delay
        self._fault_config = fault
        self._delay_rng = np.random.default_rng()
        self._fault_rng = np.random.default_rng()
        self._queue: deque[np.ndarray] = deque()
        self._action_delay = 0
        self._fault: dict[str, Any] | None = None
        self._step_index = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._delay_rng = _reset_rng(seed, 201, self._delay_rng)
        self._fault_rng = _reset_rng(seed, 202, self._fault_rng)
        self._action_delay = (
            0
            if self._delay_config is None
            else _sample_integer(
                self._delay_rng, self._delay_config["action_steps"]
            )
        )
        default_action = np.asarray(
            self.unwrapped.model.default_action(), dtype=np.float32
        )
        self._queue = deque(
            default_action.copy() for _ in range(self._action_delay)
        )
        self._fault = self._sample_fault()
        self._step_index = 0
        resolved = self._variation()
        self.unwrapped.runtime_variation.update(resolved)
        updated = dict(info)
        updated["runtime_variation"] = dict(self.unwrapped.runtime_variation)
        return observation, updated

    def step(self, action):
        commanded = np.asarray(action, dtype=np.float32).reshape(-1)
        if commanded.shape != self.action_space.shape:
            raise ValueError("commanded action shape does not match action space")
        self._queue.append(commanded.copy())
        channel = self._queue.popleft()
        if self._fault_active():
            assert self._fault is not None
            channel = channel.copy()
            channel[self._fault["action_index"]] *= 1.0 - self._fault["severity"]
        observation, reward, terminated, truncated, info = self.env.step(channel)
        updated = dict(info)
        updated["commanded_action"] = commanded.copy()
        updated["channel_action"] = channel.copy()
        updated["runtime_variation"] = dict(self.unwrapped.runtime_variation)
        self._step_index += 1
        return observation, reward, terminated, truncated, updated

    def _sample_fault(self) -> dict[str, Any] | None:
        if self._fault_config is None:
            return None
        if self._fault_rng.random() >= self._fault_config["probability"]:
            return None
        severity = float(
            self._fault_rng.uniform(*self._fault_config["severity"])
        )
        duration = _sample_integer(
            self._fault_rng, self._fault_config["duration_steps"]
        )
        duration = min(duration, int(self.unwrapped.episode_steps))
        latest_start = max(0, int(self.unwrapped.episode_steps) - duration)
        start = int(self._fault_rng.integers(0, latest_start + 1))
        return {
            "kind": "loss-of-effectiveness",
            "action_index": int(
                self._fault_rng.integers(0, self.action_space.shape[0])
            ),
            "severity": severity,
            "start_step": start,
            "duration_steps": duration,
        }

    def _fault_active(self) -> bool:
        if self._fault is None:
            return False
        start = self._fault["start_step"]
        return start <= self._step_index < start + self._fault["duration_steps"]

    def _variation(self) -> dict[str, Any]:
        return {
            "action_delay_steps": self._action_delay,
            "fault": None if self._fault is None else dict(self._fault),
        }


class ObservationChannelWrapper(gym.Wrapper):
    """Apply normalized observation delay, bias, and white measurement noise."""

    def __init__(
        self,
        env: gym.Env,
        *,
        delay: Mapping[str, tuple[int, int]] | None,
        noise: Mapping[str, float] | None,
    ) -> None:
        super().__init__(env)
        self._delay_config = delay
        self._noise_config = noise
        self._delay_rng = np.random.default_rng()
        self._noise_rng = np.random.default_rng()
        self._queue: deque[np.ndarray] = deque()
        self._observation_delay = 0
        self._bias = np.zeros(self.observation_space.shape, dtype=np.float32)
        observation_schema = tuple(self.unwrapped.model.observation_schema())
        if len(observation_schema) != self.observation_space.shape[0]:
            raise ValueError("observation schema must match observation space")
        if noise is None:
            self._noise_mask = np.ones(self.observation_space.shape, dtype=np.float32)
        else:
            missing_kind = [
                index for index, row in enumerate(observation_schema) if "kind" not in row
            ]
            if missing_kind:
                raise ValueError(
                    "observation noise requires kind metadata for schema rows "
                    f"{missing_kind}"
                )
            kinds = [row["kind"] for row in observation_schema]
            if any(
                not isinstance(kind, str)
                or kind not in {"measurement", "reference", "action"}
                for kind in kinds
            ):
                raise ValueError(
                    "observation schema kind must be measurement, reference, or action"
                )
            self._noise_mask = np.asarray(
                [kind != "reference" for kind in kinds], dtype=np.float32
            )
        scale = np.asarray(
            self.observation_space.high - self.observation_space.low,
            dtype=np.float32,
        )
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("observation variation requires finite non-zero bounds")
        self._scale = scale

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._delay_rng = _reset_rng(seed, 301, self._delay_rng)
        self._noise_rng = _reset_rng(seed, 302, self._noise_rng)
        self._observation_delay = (
            0
            if self._delay_config is None
            else _sample_integer(
                self._delay_rng, self._delay_config["observation_steps"]
            )
        )
        initial = np.asarray(observation, dtype=np.float32)
        self._queue = deque(
            initial.copy() for _ in range(self._observation_delay)
        )
        bias_std = 0.0 if self._noise_config is None else self._noise_config["bias_std"]
        self._bias = np.asarray(
            self._noise_rng.normal(0.0, bias_std, size=initial.shape)
            * self._scale
            * self._noise_mask,
            dtype=np.float32,
        )
        self.unwrapped.runtime_variation.update(
            {
                "observation_delay_steps": self._observation_delay,
                "observation_bias": self._bias.tolist(),
            }
        )
        updated = dict(info)
        updated["runtime_variation"] = dict(self.unwrapped.runtime_variation)
        return self._apply(initial), updated

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._queue.append(np.asarray(observation, dtype=np.float32).copy())
        delayed = self._queue.popleft()
        updated = dict(info)
        updated["runtime_variation"] = dict(self.unwrapped.runtime_variation)
        return self._apply(delayed), reward, terminated, truncated, updated

    def _apply(self, observation: np.ndarray) -> np.ndarray:
        std = 0.0 if self._noise_config is None else self._noise_config["std"]
        white = (
            self._noise_rng.normal(0.0, std, size=observation.shape)
            * self._scale
            * self._noise_mask
        )
        varied = observation + self._bias + white
        return np.clip(
            varied, self.observation_space.low, self.observation_space.high
        ).astype(np.float32)


__all__ = [
    "ActionChannelWrapper",
    "EpisodeSamplingWrapper",
    "ObservationChannelWrapper",
    "resolve_delay_option",
    "resolve_fault_option",
    "resolve_noise_option",
]
