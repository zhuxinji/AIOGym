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

from aiogym._internal.serialization import file_sha256, stable_json_hash

from .config import RLTrainingConfig


TRAINING_CHECKPOINT_SCHEMA_VERSION = "aiogym.rl_checkpoint.v2"
LEGACY_TRAINING_CHECKPOINT_SCHEMA_VERSION = "aiogym.rl_checkpoint.v1"


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
    validation_state: Mapping[str, Any] | None = None
    selected_checkpoint: Mapping[str, Any] | None = None
    legacy_partial_validation_state: bool = False
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
        if not isinstance(self.legacy_partial_validation_state, bool):
            raise TypeError(
                "legacy_partial_validation_state must be boolean"
            )

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
            "validation_state": copy.deepcopy(self.validation_state),
            "selected_checkpoint": copy.deepcopy(
                self.selected_checkpoint
            ),
            "legacy_partial_validation_state": bool(
                self.legacy_partial_validation_state
            ),
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
        source_schema = data.get("schema_version")
        if source_schema == LEGACY_TRAINING_CHECKPOINT_SCHEMA_VERSION:
            data["schema_version"] = TRAINING_CHECKPOINT_SCHEMA_VERSION
            data.setdefault("validation_state", None)
            data.setdefault("selected_checkpoint", None)
            data["legacy_partial_validation_state"] = True
        elif source_schema != TRAINING_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported training checkpoint: {source_schema!r}"
            )
        raw_config = data.pop("config")
        config = RLTrainingConfig.from_mapping(raw_config)
        expected_hash = data.pop("config_hash")
        if config.config_hash != expected_hash:
            legacy_hash = stable_json_hash(raw_config, ensure_ascii=False)
            if (
                raw_config.get("schema_version")
                != "aiogym.rl_training_config.v2"
                or legacy_hash != expected_hash
            ):
                raise ValueError(
                    "checkpoint config hash does not match its payload"
                )
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
        data.setdefault("validation_state", None)
        data.setdefault("selected_checkpoint", None)
        data.setdefault("legacy_partial_validation_state", False)
        return cls(config=config, **data)

    def require_exact_validation_resume(self) -> None:
        if self.legacy_partial_validation_state:
            raise ValueError(
                "legacy v1 checkpoint has only partial validation state; "
                "exact resume is unavailable"
            )
        if self.validation_state is None:
            raise ValueError(
                "checkpoint is missing validation selector state"
            )
        if self.selected_checkpoint is not None:
            validate_selected_checkpoint(self.selected_checkpoint)


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

    if previous.algorithm_id != current.algorithm_id:
        raise ValueError("resume cannot change algorithm_id")
    if current.algorithm_id == "bc":
        raise ValueError("BC resume is not implemented")
    if previous.n_envs != current.n_envs:
        raise ValueError(
            f"{current.algorithm_id.upper()} resume requires unchanged n_envs"
        )
    if current.total_transitions < previous.total_transitions:
        raise ValueError("resume total budget cannot decrease")

    old = previous.as_dict()
    new = current.as_dict()
    for field in ("budget", "device", "resume_checkpoint"):
        old.pop(field, None)
        new.pop(field, None)
    for payload in (old, new):
        output = dict(payload.get("output") or {})
        for name in ("artifact_dir", "directory", "name"):
            output.pop(name, None)
        payload["output"] = output
    if old != new:
        raise ValueError(
            "resume training contract changes Track/distribution, algorithm, "
            "observation/action, replay, dataset, or seed contract"
        )


def selected_checkpoint_manifest(
    path: str | Path,
    selection_record: Mapping[str, Any],
) -> dict[str, Any]:
    record = copy.deepcopy(dict(selection_record))
    target = Path(path).resolve()
    return {
        "path": str(target),
        "sha256": file_sha256(target),
        "checkpoint_id": str(record["checkpoint_id"]),
        "step": int(record["step"]),
        "selection_record": record,
    }


def validate_selected_checkpoint(
    manifest: Mapping[str, Any],
) -> Path:
    if not isinstance(manifest, Mapping):
        raise TypeError("selected_checkpoint must be a mapping")
    payload = dict(manifest)
    required = {
        "path",
        "sha256",
        "checkpoint_id",
        "step",
        "selection_record",
    }
    if set(payload) != required:
        raise ValueError(
            "selected_checkpoint fields do not match the v2 schema"
        )
    record = payload["selection_record"]
    if not isinstance(record, Mapping):
        raise TypeError("selected checkpoint record must be a mapping")
    if record.get("checkpoint_id") != payload["checkpoint_id"]:
        raise ValueError("selected checkpoint ID does not match its record")
    if record.get("step") != payload["step"]:
        raise ValueError("selected checkpoint step does not match its record")
    target = Path(payload["path"])
    if not target.is_file():
        raise FileNotFoundError(
            f"selected checkpoint does not exist: {target}"
        )
    actual_hash = file_sha256(target)
    if actual_hash != payload["sha256"]:
        raise ValueError("selected checkpoint SHA256 does not match")
    return target


__all__ = [
    "TRAINING_CHECKPOINT_SCHEMA_VERSION",
    "LEGACY_TRAINING_CHECKPOINT_SCHEMA_VERSION",
    "CheckpointManager",
    "TrainingCheckpoint",
    "capture_rng_state",
    "restore_rng_state",
    "file_sha256",
    "selected_checkpoint_manifest",
    "validate_selected_checkpoint",
    "validate_resume_config",
]
