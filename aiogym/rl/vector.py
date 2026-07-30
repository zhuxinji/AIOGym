"""Vector-rollout transition handling, including autoreset terminals."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VectorTransitionBatch:
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    reset_observation: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    bootstrap_mask: np.ndarray


def vector_transition_batch(
    observation,
    action,
    reward,
    next_observation,
    terminated,
    truncated,
    infos,
) -> VectorTransitionBatch:
    """Replace autoreset observations with the true terminal observations."""

    obs = np.asarray(observation).copy()
    actions = np.asarray(action).copy()
    rewards = np.asarray(reward, dtype=np.float64).reshape(-1)
    reset_obs = np.asarray(next_observation).copy()
    storage_next = reset_obs.copy()
    terms = np.asarray(terminated, dtype=np.bool_).reshape(-1)
    truncs = np.asarray(truncated, dtype=np.bool_).reshape(-1)
    if rewards.shape != terms.shape or terms.shape != truncs.shape:
        raise ValueError("reward/terminated/truncated vector lengths must match")
    if obs.shape[0] != rewards.size or storage_next.shape[0] != rewards.size:
        raise ValueError("observation rows must match vector transition count")

    for index, info in enumerate(_info_rows(infos, rewards.size)):
        if not (terms[index] or truncs[index]):
            continue
        terminal = info.get("terminal_observation")
        if terminal is None:
            terminal = info.get("final_observation")
        if terminal is None:
            raise ValueError(
                "autoreset transition is missing terminal_observation/"
                "final_observation"
            )
        terminal_array = np.asarray(terminal, dtype=storage_next.dtype)
        if terminal_array.shape != storage_next[index].shape:
            raise ValueError("terminal observation shape does not match vector output")
        storage_next[index] = terminal_array

    return VectorTransitionBatch(
        observation=obs,
        action=actions,
        reward=rewards,
        next_observation=storage_next,
        reset_observation=reset_obs,
        terminated=terms,
        truncated=truncs,
        bootstrap_mask=np.logical_not(terms).astype(np.float32),
    )


def _info_rows(infos: Any, count: int) -> list[dict[str, Any]]:
    if isinstance(infos, (list, tuple)):
        if len(infos) != count:
            raise ValueError("info rows must match vector transition count")
        return [dict(value or {}) for value in infos]
    if not isinstance(infos, dict):
        raise TypeError("infos must be a sequence or Gymnasium vector-info mapping")
    rows = [dict() for _ in range(count)]
    for key, values in infos.items():
        if key.startswith("_"):
            continue
        mask = infos.get(f"_{key}")
        if mask is None:
            mask = np.ones(count, dtype=np.bool_)
        for index, include in enumerate(np.asarray(mask).reshape(-1)):
            if include:
                rows[index][key] = values[index]
    return rows


__all__ = ["VectorTransitionBatch", "vector_transition_batch"]
