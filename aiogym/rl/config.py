"""Canonical configuration for all reinforcement-learning trainers."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


RL_TRAINING_CONFIG_SCHEMA_VERSION = "aiogym.rl_training_config.v2"
_ALGORITHMS = frozenset({"bc", "ppo", "rlpd", "sac", "td3"})


def list_algorithms() -> tuple[str, ...]:
    """Return algorithms supported by the stable config-first runner."""

    return tuple(sorted(_ALGORITHMS))


@dataclass(frozen=True)
class RLTrainingConfig:
    """Immutable, hashable input shared by online and hybrid trainers."""

    track_id: str
    algorithm_id: str
    training_seed: int
    total_transitions: int
    n_envs: int
    device: str = "cpu"
    algorithm: Mapping[str, Any] = field(default_factory=dict)
    replay: Mapping[str, Any] = field(default_factory=dict)
    evaluation: Mapping[str, Any] = field(default_factory=dict)
    checkpointing: Mapping[str, Any] = field(default_factory=dict)
    output: Mapping[str, Any] = field(default_factory=dict)
    validation_seeds: tuple[int, ...] = (5000,)
    resume_mode: str = "restart_episode"
    resume_checkpoint: str | None = None
    dataset_id: str | None = None
    dataset_path: str | None = None
    dataset_hash: str | None = None
    curriculum_id: str | None = None
    schema_version: str = RL_TRAINING_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RL_TRAINING_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported RL training config: {self.schema_version!r}"
            )
        if not isinstance(self.track_id, str) or not self.track_id:
            raise ValueError("track_id must be a non-empty string")
        algorithm_id = str(self.algorithm_id).lower()
        if algorithm_id not in _ALGORITHMS:
            raise ValueError(
                "algorithm_id must be one of: " + ", ".join(sorted(_ALGORITHMS))
            )
        object.__setattr__(self, "algorithm_id", algorithm_id)
        for name in ("training_seed", "total_transitions", "n_envs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.training_seed < 0:
            raise ValueError("training_seed must be non-negative")
        if self.total_transitions <= 0:
            raise ValueError("total_transitions must be positive")
        if self.n_envs <= 0:
            raise ValueError("n_envs must be positive")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        for name in (
            "algorithm",
            "replay",
            "evaluation",
            "checkpointing",
            "output",
        ):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            plain = _json_mapping(name, value)
            object.__setattr__(self, name, _freeze_json(plain))
        seeds = tuple(self.validation_seeds)
        if (
            not seeds
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < 0
                for seed in seeds
            )
            or len(set(seeds)) != len(seeds)
        ):
            raise ValueError(
                "validation_seeds must be unique non-negative integers"
            )
        object.__setattr__(self, "validation_seeds", seeds)
        if self.resume_mode not in {
            "restart_episode",
            "exact_single_process",
        }:
            raise ValueError(
                "resume_mode must be restart_episode or exact_single_process"
            )
        if self.resume_mode == "exact_single_process" and self.n_envs != 1:
            raise ValueError(
                "exact_single_process resume requires n_envs=1"
            )
        for name in (
            "resume_checkpoint",
            "dataset_id",
            "dataset_path",
            "curriculum_id",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be null or a non-empty string")
        if self.dataset_hash is not None and (
            not isinstance(self.dataset_hash, str)
            or len(self.dataset_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.dataset_hash
            )
        ):
            raise ValueError(
                "dataset_hash must be null or a lowercase SHA-256 digest"
            )

    @property
    def utd_ratio(self) -> float:
        """Optimizer updates per newly collected environment transition."""

        default = (
            0.0
            if self.algorithm_id in {"bc", "ppo"}
            else 5.0
            if self.algorithm_id == "rlpd"
            else 1.0
        )
        value = float(self.algorithm.get("utd_ratio", default))
        if value < 0.0:
            raise ValueError("algorithm.utd_ratio must be non-negative")
        if self.algorithm_id in {"bc", "ppo"} and value != 0.0:
            raise ValueError(
                f"{self.algorithm_id.upper()} does not use an UTD ratio"
            )
        return value

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "track_id": self.track_id,
            "algorithm_id": self.algorithm_id,
            "training_seed": self.training_seed,
            "total_transitions": self.total_transitions,
            "n_envs": self.n_envs,
            "device": self.device,
            "algorithm": _thaw_json(self.algorithm),
            "replay": _thaw_json(self.replay),
            "evaluation": _thaw_json(self.evaluation),
            "checkpointing": _thaw_json(self.checkpointing),
            "output": _thaw_json(self.output),
            "validation_seeds": list(self.validation_seeds),
            "resume_mode": self.resume_mode,
            "resume_checkpoint": self.resume_checkpoint,
            "dataset_id": self.dataset_id,
            "dataset_path": self.dataset_path,
            "dataset_hash": self.dataset_hash,
            "curriculum_id": self.curriculum_id,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RLTrainingConfig":
        data = dict(value)
        unknown = sorted(
            set(data)
            - {
                "schema_version",
                "track_id",
                "algorithm_id",
                "training_seed",
                "total_transitions",
                "n_envs",
                "device",
                "algorithm",
                "replay",
                "evaluation",
                "checkpointing",
                "output",
                "validation_seeds",
                "resume_mode",
                "resume_checkpoint",
                "dataset_id",
                "dataset_path",
                "dataset_hash",
                "curriculum_id",
            }
        )
        if unknown:
            raise ValueError(
                "unknown RL training config field(s): " + ", ".join(unknown)
            )
        return cls(**data)

    @classmethod
    def load(cls, path: str | Path) -> "RLTrainingConfig":
        with Path(path).open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, Mapping):
            raise TypeError("RL training config file must contain a mapping")
        return cls.from_mapping(value)


def _json_mapping(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            _thaw_json(value),
            sort_keys=True,
            allow_nan=False,
        )
        plain = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-compatible values") from exc
    if not isinstance(plain, dict):
        raise TypeError(f"{name} must be a mapping")
    return plain


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType(
            {str(name): _freeze_json(item) for name, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value):
    if isinstance(value, Mapping):
        return {str(name): _thaw_json(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "RL_TRAINING_CONFIG_SCHEMA_VERSION",
    "RLTrainingConfig",
    "list_algorithms",
]
