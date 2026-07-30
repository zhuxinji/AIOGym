"""Small dependency-free replay buffer with complete resumable state."""
from __future__ import annotations

import copy

import numpy as np


class ReplayBuffer:
    """Ring buffer used by unified trainers and checkpoint smoke tests."""

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0
        self._rng = np.random.default_rng(int(seed))
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        *,
        observation,
        action,
        reward,
        next_observation,
        terminated,
        truncated,
        bootstrap_mask=None,
    ) -> None:
        row = {
            "observation": np.asarray(observation, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": np.asarray(reward, dtype=np.float32),
            "next_observation": np.asarray(next_observation, dtype=np.float32),
            "terminated": np.asarray(terminated, dtype=np.bool_),
            "truncated": np.asarray(truncated, dtype=np.bool_),
            "bootstrap_mask": np.asarray(
                float(not bool(terminated))
                if bootstrap_mask is None
                else bootstrap_mask,
                dtype=np.float32,
            ),
        }
        if not self._arrays:
            self._arrays = {
                name: np.empty((self.capacity, *value.shape), dtype=value.dtype)
                for name, value in row.items()
            }
        for name, value in row.items():
            if value.shape != self._arrays[name].shape[1:]:
                raise ValueError(f"replay {name} shape changed")
            self._arrays[name][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        indexes = self._rng.integers(0, self.size, size=int(batch_size))
        return {
            name: values[indexes].copy()
            for name, values in self._arrays.items()
        }

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "arrays": {
                name: value.copy() for name, value in self._arrays.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state) -> "ReplayBuffer":
        result = cls(int(state["capacity"]))
        result.size = int(state["size"])
        result.position = int(state["position"])
        if not 0 <= result.size <= result.capacity:
            raise ValueError("invalid replay size")
        if not 0 <= result.position < result.capacity:
            raise ValueError("invalid replay position")
        result._arrays = {
            str(name): np.asarray(value).copy()
            for name, value in state["arrays"].items()
        }
        for value in result._arrays.values():
            if value.shape[0] != result.capacity:
                raise ValueError("replay array capacity mismatch")
        result._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        return result


__all__ = ["ReplayBuffer"]
