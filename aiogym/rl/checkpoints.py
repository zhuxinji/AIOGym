"""Atomic, resumable training-state checkpoints."""
from __future__ import annotations

import copy
import os
import pickle
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import RLTrainingConfig


TRAINING_CHECKPOINT_SCHEMA_VERSION = "aiogym.rl_checkpoint.v1"


@dataclass(frozen=True)
class TrainingCheckpoint:
    config: RLTrainingConfig
    transition_count: int
    update_count: int
    algorithm_state: Mapping[str, Any]
    replay_state: Mapping[str, Any] | None
    normalization_state: Mapping[str, Any] | None
    coordinator_state: Mapping[str, Any]
    curriculum_state: Mapping[str, Any] | None
    best_validation: Mapping[str, Any] | None
    rng_state: Mapping[str, Any]
    resume_mode: str = "restart_episode"
    last_committed_episode_index: int | None = None
    next_episode_index: int | None = None
    partial_episodes_discarded: int = 0
    n_envs: int = 1
    vector_backend: str = "single"
    code_commit: str | None = None
    schema_version: str = TRAINING_CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported training checkpoint: {self.schema_version!r}"
            )
        if self.transition_count < 0 or self.update_count < 0:
            raise ValueError("checkpoint counters must be non-negative")
        if not isinstance(self.config, RLTrainingConfig):
            raise TypeError("checkpoint config must be RLTrainingConfig")
        if self.resume_mode != "restart_episode":
            raise ValueError("resume_mode must be 'restart_episode'")
        if self.partial_episodes_discarded < 0:
            raise ValueError(
                "partial_episodes_discarded must be non-negative"
            )
        if self.n_envs <= 0:
            raise ValueError("checkpoint n_envs must be positive")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.as_dict(),
            "config_hash": self.config.config_hash,
            "transition_count": int(self.transition_count),
            "update_count": int(self.update_count),
            "algorithm_state": copy.deepcopy(dict(self.algorithm_state)),
            "replay_state": copy.deepcopy(self.replay_state),
            "normalization_state": copy.deepcopy(self.normalization_state),
            "coordinator_state": copy.deepcopy(dict(self.coordinator_state)),
            "curriculum_state": copy.deepcopy(self.curriculum_state),
            "best_validation": copy.deepcopy(self.best_validation),
            "rng_state": copy.deepcopy(dict(self.rng_state)),
            "resume_mode": self.resume_mode,
            "last_committed_episode_index": (
                self.last_committed_episode_index
            ),
            "next_episode_index": self.next_episode_index,
            "partial_episodes_discarded": int(
                self.partial_episodes_discarded
            ),
            "n_envs": int(self.n_envs),
            "vector_backend": self.vector_backend,
            "code_commit": self.code_commit,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TrainingCheckpoint":
        data = dict(payload)
        config = RLTrainingConfig.from_mapping(data.pop("config"))
        expected_hash = data.pop("config_hash")
        if config.config_hash != expected_hash:
            raise ValueError("checkpoint config hash does not match its payload")
        data.setdefault("resume_mode", "restart_episode")
        data.setdefault("last_committed_episode_index", None)
        data.setdefault(
            "next_episode_index",
            dict(data.get("coordinator_state") or {}).get(
                "next_episode_index"
            ),
        )
        data.setdefault("partial_episodes_discarded", 0)
        data.setdefault("n_envs", int(config.n_envs))
        data.setdefault("vector_backend", "single")
        return cls(config=config, **data)


def capture_rng_state() -> dict[str, Any]:
    """Capture process RNGs without importing optional Torch eagerly."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
    }
    try:
        import torch
    except ModuleNotFoundError:
        return state
    state["torch_cpu"] = torch.get_rng_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    if "torch_cpu" not in state:
        return
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "checkpoint contains Torch RNG state but Torch is unavailable"
        ) from exc
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class CheckpointManager:
    """Persist a complete training checkpoint using atomic replacement."""

    @staticmethod
    def save(path: str | Path, checkpoint: TrainingCheckpoint) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                pickle.dump(
                    checkpoint.payload(),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
        return target

    @staticmethod
    def load(
        path: str | Path,
        *,
        expected_config: RLTrainingConfig | None = None,
        allow_runtime_changes: bool = False,
    ) -> TrainingCheckpoint:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("training checkpoint payload must be a mapping")
        checkpoint = TrainingCheckpoint.from_payload(payload)
        if expected_config is not None:
            if allow_runtime_changes:
                validate_resume_config(
                    checkpoint.config,
                    expected_config,
                )
            elif (
                checkpoint.config.config_hash
                != expected_config.config_hash
            ):
                raise ValueError(
                    "resume config does not match checkpoint config"
                )
        return checkpoint


def validate_resume_config(
    previous: RLTrainingConfig,
    current: RLTrainingConfig,
) -> None:
    """Allow runtime scaling while preserving every identity contract."""

    old = previous.as_dict()
    new = current.as_dict()
    for field in (
        "n_envs",
        "total_transitions",
        "device",
        "resume_checkpoint",
    ):
        old.pop(field, None)
        new.pop(field, None)
    for payload in (old, new):
        algorithm = dict(payload.get("algorithm") or {})
        algorithm.pop("rollout_vector_steps", None)
        payload["algorithm"] = algorithm
    if old != new:
        raise ValueError(
            "resume training contract changes Track/distribution, algorithm, "
            "observation/action, replay, dataset, or seed contract"
        )


__all__ = [
    "TRAINING_CHECKPOINT_SCHEMA_VERSION",
    "CheckpointManager",
    "TrainingCheckpoint",
    "capture_rng_state",
    "restore_rng_state",
    "validate_resume_config",
]
