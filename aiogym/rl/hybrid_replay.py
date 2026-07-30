"""Canonical offline/online replay mixing for RLPD."""
from __future__ import annotations

import copy

import numpy as np

from .dataset_replay import DatasetReplay
from .replay import ReplayBuffer


_CORE_FIELDS = (
    "observation",
    "action",
    "reward",
    "next_observation",
    "terminated",
    "truncated",
    "bootstrap_mask",
)


class RLPDBatchSampler:
    """Mix immutable prior data and online replay with explicit accounting."""

    def __init__(
        self,
        offline: DatasetReplay,
        online: ReplayBuffer,
        *,
        offline_fraction: float = 0.5,
        canonical: bool = True,
    ) -> None:
        if not isinstance(offline, DatasetReplay):
            raise TypeError("offline replay must be DatasetReplay")
        if not isinstance(online, ReplayBuffer):
            raise TypeError("online replay must be ReplayBuffer")
        self.offline = offline
        self.online = online
        self.offline_fraction = float(offline_fraction)
        self.canonical = bool(canonical)
        if not 0.0 <= self.offline_fraction <= 1.0:
            raise ValueError("offline_fraction must be in [0, 1]")
        if self.canonical and self.offline_fraction != 0.5:
            raise ValueError("canonical RLPD requires offline_fraction=0.5")
        self.offline_samples = 0
        self.online_samples = 0

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if len(self.online) == 0:
            offline_count = int(batch_size)
            online_count = 0
        else:
            offline_count = int(round(batch_size * self.offline_fraction))
            offline_count = min(batch_size, max(0, offline_count))
            online_count = batch_size - offline_count
        parts = []
        if offline_count:
            offline = self.offline.sample(offline_count)
            parts.append(
                {
                    "observation": offline["observation"],
                    "action": offline["action_policy_normalized"],
                    "reward": offline["reward_scalar"],
                    "next_observation": offline["next_observation"],
                    "terminated": offline["terminated"],
                    "truncated": offline["truncated"],
                    "bootstrap_mask": offline["bootstrap_mask"],
                    "source": offline["source"],
                    "collector_id": offline["collector_id"],
                    "difficulty_tag": offline["difficulty_tag"],
                }
            )
        if online_count:
            online = self.online.sample(online_count)
            online["source"] = np.full(
                online_count,
                "online",
                dtype="<U16",
            )
            online["collector_id"] = np.full(
                online_count,
                "online-policy",
                dtype="<U32",
            )
            online["difficulty_tag"] = np.full(
                online_count,
                "online",
                dtype="<U32",
            )
            parts.append(online)
        self.offline_samples += offline_count
        self.online_samples += online_count
        return {
            name: np.concatenate([part[name] for part in parts], axis=0)
            for name in (*_CORE_FIELDS, "source", "collector_id", "difficulty_tag")
        }

    def accounting(self) -> dict[str, int | float]:
        total = self.offline_samples + self.online_samples
        return {
            "offline_dataset_transitions": len(self.offline),
            "online_replay_transitions": len(self.online),
            "offline_samples": self.offline_samples,
            "online_samples": self.online_samples,
            "sampled_offline_fraction": (
                self.offline_samples / total if total else 0.0
            ),
        }

    def state_dict(self) -> dict:
        return {
            "offline": self.offline.state_dict(),
            "online": self.online.state_dict(),
            "offline_fraction": self.offline_fraction,
            "canonical": self.canonical,
            "offline_samples": self.offline_samples,
            "online_samples": self.online_samples,
        }

    @classmethod
    def from_state_dict(cls, state) -> "RLPDBatchSampler":
        result = cls(
            DatasetReplay.from_state_dict(state["offline"]),
            ReplayBuffer.from_state_dict(state["online"]),
            offline_fraction=float(state["offline_fraction"]),
            canonical=bool(state["canonical"]),
        )
        result.offline_samples = int(state["offline_samples"])
        result.online_samples = int(state["online_samples"])
        return result

    def copy_state(self) -> dict:
        return copy.deepcopy(self.state_dict())


__all__ = ["RLPDBatchSampler"]
