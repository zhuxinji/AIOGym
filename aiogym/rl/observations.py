"""Explicit feedforward, history, and recurrent observation contracts."""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


OBSERVATION_CONTRACT_SCHEMA_VERSION = "aiogym.observation_contract.v1"


@dataclass(frozen=True)
class ObservationContract:
    """Policy-facing sensing and temporal-memory declaration."""

    sensing: str = "full_state"
    temporal: str = "single_step"
    history_length: int = 1
    include_action_history: bool = False
    recurrent_state_shape: tuple[int, ...] = ()
    schema_version: str = OBSERVATION_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OBSERVATION_CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported observation contract schema")
        if self.sensing not in {"full_state", "measured_output"}:
            raise ValueError(
                "observation sensing must be one of: "
                "full_state, measured_output"
            )
        if self.temporal not in {"single_step", "history", "recurrent"}:
            raise ValueError(
                "temporal observation must be one of: "
                "single_step, history, recurrent"
            )
        if (
            isinstance(self.history_length, bool)
            or not isinstance(self.history_length, int)
            or self.history_length <= 0
        ):
            raise ValueError("history_length must be a positive integer")
        if not isinstance(self.include_action_history, bool):
            raise TypeError("include_action_history must be a boolean")
        shape = tuple(int(value) for value in self.recurrent_state_shape)
        if any(value <= 0 for value in shape):
            raise ValueError("recurrent_state_shape values must be positive")
        object.__setattr__(self, "recurrent_state_shape", shape)
        if self.temporal == "single_step" and (
            self.history_length != 1
            or self.include_action_history
            or shape
        ):
            raise ValueError(
                "single_step observation cannot declare history or "
                "recurrent state"
            )
        if self.temporal == "history":
            if self.history_length < 2:
                raise ValueError(
                    "history observation requires history_length >= 2"
                )
            if shape:
                raise ValueError(
                    "history observation cannot declare recurrent state"
                )
        if self.temporal == "recurrent":
            if not shape:
                raise ValueError(
                    "recurrent observation requires recurrent_state_shape"
                )
            if self.history_length != 1 or self.include_action_history:
                raise ValueError(
                    "recurrent observation uses network state, not frame history"
                )

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "sensing": self.sensing,
            "temporal": self.temporal,
            "history_length": self.history_length,
            "include_action_history": self.include_action_history,
            "recurrent_state_shape": list(self.recurrent_state_shape),
        }

    @classmethod
    def from_mapping(cls, value) -> "ObservationContract":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("observation contract must be a mapping")
        data = dict(value)
        unknown = sorted(
            set(data)
            - {
                "schema_version",
                "sensing",
                "temporal",
                "history_length",
                "include_action_history",
                "recurrent_state_shape",
            }
        )
        if unknown:
            raise ValueError(
                "unknown observation contract fields: "
                + ", ".join(unknown)
            )
        return cls(
            sensing=data.get("sensing", "full_state"),
            temporal=data.get("temporal", "single_step"),
            history_length=data.get("history_length", 1),
            include_action_history=data.get(
                "include_action_history",
                False,
            ),
            recurrent_state_shape=tuple(
                data.get("recurrent_state_shape", ())
            ),
            schema_version=data.get(
                "schema_version",
                OBSERVATION_CONTRACT_SCHEMA_VERSION,
            ),
        )


class HistoryObservationWrapper(gym.Wrapper):
    """Expose a fixed oldest-to-newest observation/action history."""

    def __init__(self, env: gym.Env, contract: ObservationContract) -> None:
        super().__init__(env)
        if contract.temporal != "history":
            raise ValueError(
                "HistoryObservationWrapper requires temporal='history'"
            )
        _validate_sensing(env, contract)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("history observations require a Box space")
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("action history requires a Box action space")
        self.observation_contract = contract
        self._observations = deque(maxlen=contract.history_length)
        self._actions = deque(maxlen=contract.history_length - 1)
        lows = [env.observation_space.low] * contract.history_length
        highs = [env.observation_space.high] * contract.history_length
        if contract.include_action_history:
            lows.extend(
                [env.action_space.low] * (contract.history_length - 1)
            )
            highs.extend(
                [env.action_space.high] * (contract.history_length - 1)
            )
        self.observation_space = spaces.Box(
            np.concatenate(lows).astype(np.float32),
            np.concatenate(highs).astype(np.float32),
            dtype=np.float32,
        )

    def __getattr__(self, name):
        wrapped = self.__dict__.get("env")
        if wrapped is None:
            raise AttributeError(name)
        return getattr(wrapped, name)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        value = np.asarray(observation, dtype=np.float32).reshape(-1)
        self._observations.clear()
        self._actions.clear()
        for _ in range(self.observation_contract.history_length):
            self._observations.append(value.copy())
        zero_action = np.clip(
            np.zeros(self.action_space_of_env.shape, dtype=np.float32),
            self.action_space_of_env.low,
            self.action_space_of_env.high,
        )
        for _ in range(self.observation_contract.history_length - 1):
            self._actions.append(zero_action.copy())
        return self._stack(), info

    @property
    def action_space_of_env(self):
        return self.env.action_space

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._observations.append(
            np.asarray(observation, dtype=np.float32).reshape(-1)
        )
        self._actions.append(
            np.asarray(action, dtype=np.float32).reshape(-1)
        )
        return (
            self._stack(),
            reward,
            terminated,
            truncated,
            info,
        )

    def _stack(self):
        pieces = list(self._observations)
        if self.observation_contract.include_action_history:
            pieces.extend(self._actions)
        return np.concatenate(pieces).astype(np.float32)


@dataclass(frozen=True)
class RecurrentStateContract:
    """Shape and episode-boundary reset semantics for recurrent policies."""

    hidden_shape: tuple[int, ...]
    dtype: str = "float32"

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.hidden_shape)
        if not shape or any(value <= 0 for value in shape):
            raise ValueError("recurrent hidden_shape must contain positive values")
        dtype = np.dtype(self.dtype)
        if dtype.kind != "f":
            raise TypeError("recurrent state dtype must be floating point")
        object.__setattr__(self, "hidden_shape", shape)
        object.__setattr__(self, "dtype", dtype.name)

    def initial_state(self, batch_size: int = 1) -> np.ndarray:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("recurrent batch_size must be positive")
        return np.zeros(
            (batch_size, *self.hidden_shape),
            dtype=self.dtype,
        )

    def reset_where(self, state, episode_starts) -> np.ndarray:
        value = np.asarray(state, dtype=self.dtype).copy()
        starts = np.asarray(episode_starts, dtype=np.bool_).reshape(-1)
        expected = (starts.size, *self.hidden_shape)
        if value.shape != expected:
            raise ValueError(
                f"recurrent state shape must be {expected}, got {value.shape}"
            )
        value[starts] = 0.0
        return value


def wrap_observation_contract(
    env: gym.Env,
    contract: ObservationContract | dict,
):
    resolved = ObservationContract.from_mapping(contract)
    _validate_sensing(env, resolved)
    if resolved.temporal == "history":
        return HistoryObservationWrapper(env, resolved)
    env.observation_contract = resolved
    return env


def _validate_sensing(env, contract) -> None:
    source = str(
        getattr(env.unwrapped, "observation_mode", "full_state")
    )
    if source != contract.sensing:
        raise ValueError(
            "environment observation_mode does not match observation "
            f"contract ({source!r} != {contract.sensing!r})"
        )


__all__ = [
    "OBSERVATION_CONTRACT_SCHEMA_VERSION",
    "HistoryObservationWrapper",
    "ObservationContract",
    "RecurrentStateContract",
    "wrap_observation_contract",
]
