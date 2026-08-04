"""Canonical configuration for all reinforcement-learning trainers."""
from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field, replace as dataclass_replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from aiogym._internal.validation import seed_sequence
from aiogym._internal.serialization import canonical_json_bytes, stable_json_hash


RL_TRAINING_CONFIG_SCHEMA_VERSION = "aiogym.rl_training_config.v3"
LEGACY_RL_TRAINING_CONFIG_SCHEMA_VERSION = "aiogym.rl_training_config.v2"
_ALGORITHMS = frozenset({"bc", "ppo", "rlpd", "sac", "td3"})
_ALGORITHM_FIELDS = {
    "bc": frozenset({"batch_size", "hidden", "learning_rate"}),
    "sac": frozenset(
        {
            "batch_size",
            "gamma",
            "learning_rate",
            "policy",
            "policy_kwargs",
            "rollout_vector_steps",
            "subproc_start_method",
            "tau",
            "ent_coef",
            "target_entropy",
            "tensorboard_log",
            "torch_threads",
            "utd_ratio",
            "vector_backend",
            "verbose",
        }
    ),
    "td3": frozenset(
        {
            "batch_size",
            "gamma",
            "learning_rate",
            "policy",
            "policy_kwargs",
            "action_noise",
            "action_noise_sigma",
            "policy_delay",
            "target_policy_noise",
            "target_noise_clip",
            "rollout_vector_steps",
            "subproc_start_method",
            "tau",
            "tensorboard_log",
            "torch_threads",
            "utd_ratio",
            "vector_backend",
            "verbose",
        }
    ),
    "ppo": frozenset(
        {
            "batch_size",
            "gamma",
            "learning_rate",
            "n_steps",
            "policy",
            "policy_kwargs",
            "gae_lambda",
            "clip_range",
            "n_epochs",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "subproc_start_method",
            "tensorboard_log",
            "torch_threads",
            "vector_backend",
            "verbose",
        }
    ),
    "rlpd": frozenset(
        {
            "batch_size",
            "bc_steps",
            "n_critics",
            "offline_fraction",
            "pretrain_updates",
            "utd_ratio",
        }
    ),
}
_REPLAY_FIELDS = {
    "bc": frozenset(),
    "ppo": frozenset(),
    "rlpd": frozenset({"capacity", "schema"}),
    "sac": frozenset({"capacity", "learning_starts", "schema"}),
    "td3": frozenset({"capacity", "learning_starts", "schema"}),
}
_WORKFLOW_FIELDS = {
    "evaluation": frozenset({"every_transitions"}),
    "checkpointing": frozenset({"every_transitions"}),
    "output": frozenset(
        {
            "artifact_dir",
            "directory",
            "name",
            "onnx",
            "rollout_steps",
            "save_rollout",
            "strict_export",
        }
    ),
}
_ALGORITHM_DEFAULTS = {
    "bc": {
        "batch_size": 256,
        "hidden": 64,
        "learning_rate": 1e-3,
    },
    "sac": {
        "batch_size": 256,
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "policy": "MlpPolicy",
        "policy_kwargs": {
            "activation_fn": "relu",
            "net_arch": {"pi": [256, 256], "qf": [256, 256]},
        },
        "rollout_vector_steps": 1,
        "subproc_start_method": "spawn",
        "tau": 0.005,
        "ent_coef": "auto",
        "target_entropy": "auto",
        "tensorboard_log": None,
        "torch_threads": 2,
        "utd_ratio": 1.0,
        "vector_backend": "subproc",
        "verbose": 1,
    },
    "td3": {
        "batch_size": 256,
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "policy": "MlpPolicy",
        "policy_kwargs": {
            "activation_fn": "relu",
            "net_arch": {"pi": [256, 256], "qf": [256, 256]},
        },
        "action_noise": "normal",
        "action_noise_sigma": 0.1,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "rollout_vector_steps": 1,
        "subproc_start_method": "spawn",
        "tau": 0.005,
        "tensorboard_log": None,
        "torch_threads": 2,
        "utd_ratio": 1.0,
        "vector_backend": "subproc",
        "verbose": 1,
    },
    "ppo": {
        "batch_size": 256,
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "policy": "MlpPolicy",
        "policy_kwargs": {
            "activation_fn": "tanh",
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        },
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "n_epochs": 10,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "subproc_start_method": "spawn",
        "tensorboard_log": None,
        "torch_threads": 2,
        "vector_backend": "subproc",
        "verbose": 1,
    },
    "rlpd": {
        "batch_size": 256,
        "bc_steps": 4000,
        "n_critics": 5,
        "offline_fraction": 0.5,
        "pretrain_updates": 5000,
        "utd_ratio": 5.0,
    },
}
_REPLAY_DEFAULTS = {
    "bc": {},
    "ppo": {},
    "rlpd": {"capacity": 1_000_000},
    "sac": {"capacity": 300_000, "learning_starts": 100},
    "td3": {"capacity": 300_000, "learning_starts": 100},
}
_EVALUATION_DEFAULTS = {
    "bc": {},
    "ppo": {"every_transitions": 10_000},
    "rlpd": {"every_transitions": 2_500},
    "sac": {"every_transitions": 10_000},
    "td3": {"every_transitions": 10_000},
}
_OUTPUT_DEFAULTS = {
    "bc": {},
    "ppo": {"onnx": False, "rollout_steps": None, "save_rollout": True, "strict_export": False},
    "rlpd": {"onnx": False, "rollout_steps": None, "save_rollout": False, "strict_export": False},
    "sac": {"onnx": False, "rollout_steps": None, "save_rollout": True, "strict_export": False},
    "td3": {"onnx": False, "rollout_steps": None, "save_rollout": True, "strict_export": False},
}


def list_algorithms() -> tuple[str, ...]:
    """Return algorithms supported by the stable config-first runner."""

    return tuple(sorted(_ALGORITHMS))


def resolve_training_defaults(config: "RLTrainingConfig") -> "RLTrainingConfig":
    """Materialize backend defaults before hashing or execution."""

    if not isinstance(config, RLTrainingConfig):
        raise TypeError("config must be an RLTrainingConfig")
    algorithm_id = config.algorithm_id
    return dataclass_replace(
        config,
        algorithm={
            **_ALGORITHM_DEFAULTS[algorithm_id],
            **dict(config.algorithm),
        },
        replay={
            **_REPLAY_DEFAULTS[algorithm_id],
            **dict(config.replay),
        },
        evaluation={
            **_EVALUATION_DEFAULTS[algorithm_id],
            **dict(config.evaluation),
        },
        output={
            **_OUTPUT_DEFAULTS[algorithm_id],
            **dict(config.output),
        },
    )


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
    migration_metadata: Mapping[str, Any] | None = None
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
        _reject_unknown_mapping_fields(
            f"algorithm for {algorithm_id}",
            self.algorithm,
            _ALGORITHM_FIELDS[algorithm_id],
        )
        _reject_unknown_mapping_fields(
            f"replay for {algorithm_id}",
            self.replay,
            _REPLAY_FIELDS[algorithm_id],
        )
        if algorithm_id in {"sac", "td3", "ppo"}:
            policy_kwargs = self.algorithm.get("policy_kwargs")
            if policy_kwargs is not None:
                _validate_policy_kwargs(algorithm_id, policy_kwargs)
        if algorithm_id == "td3":
            action_noise = self.algorithm.get("action_noise")
            if action_noise is not None and action_noise not in {
                "normal",
                "none",
            }:
                raise ValueError(
                    "algorithm.action_noise must be one of: normal, none"
                )
        for name, allowed in _WORKFLOW_FIELDS.items():
            _reject_unknown_mapping_fields(
                name,
                getattr(self, name),
                allowed,
            )
        start_method = self.algorithm.get("subproc_start_method")
        if start_method is not None and start_method not in {
            "spawn",
            "forkserver",
            "fork",
        }:
            raise ValueError(
                "algorithm.subproc_start_method must be one of: "
                "spawn, forkserver, fork"
            )
        utd_ratio = self.utd_ratio
        if self.algorithm_id == "rlpd" and (
            utd_ratio <= 0.0 or not utd_ratio.is_integer()
        ):
            raise ValueError(
                "RLPD algorithm.utd_ratio must be a positive integer"
            )
        seeds = seed_sequence(
            "validation_seeds", self.validation_seeds
        )
        object.__setattr__(self, "validation_seeds", seeds)
        if self.resume_mode != "restart_episode":
            raise ValueError("resume_mode must be 'restart_episode'")
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
        if self.migration_metadata is not None:
            if not isinstance(self.migration_metadata, Mapping):
                raise TypeError("migration_metadata must be a mapping")
            object.__setattr__(
                self,
                "migration_metadata",
                _freeze_json(
                    _json_mapping(
                        "migration_metadata", self.migration_metadata
                    )
                ),
            )

    @property
    def budget_unit(self) -> str:
        return (
            "optimizer_updates"
            if self.algorithm_id == "bc"
            else "environment_transitions"
        )

    @property
    def budget_value(self) -> int:
        return int(self.total_transitions)

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
        raw_value = self.algorithm.get("utd_ratio", default)
        if isinstance(raw_value, bool) or not isinstance(
            raw_value, (int, float)
        ):
            raise TypeError("algorithm.utd_ratio must be a number")
        value = float(raw_value)
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
            "budget": {
                "unit": self.budget_unit,
                "value": self.budget_value,
            },
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
            "migration_metadata": (
                None
                if self.migration_metadata is None
                else _thaw_json(self.migration_metadata)
            ),
        }

    def canonical_json(self) -> str:
        return canonical_json_bytes(
            self.as_dict(), ensure_ascii=False
        ).decode("utf-8")

    @property
    def config_hash(self) -> str:
        return stable_json_hash(self.as_dict(), ensure_ascii=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RLTrainingConfig":
        data = dict(value)
        source_schema = data.get("schema_version")
        if source_schema in {
            None,
            LEGACY_RL_TRAINING_CONFIG_SCHEMA_VERSION,
        }:
            if "total_transitions" not in data:
                raise ValueError(
                    "v2 RL training config requires total_transitions"
                )
            algorithm_id = str(data.get("algorithm_id", "")).lower()
            unit = (
                "optimizer_updates"
                if algorithm_id == "bc"
                else "environment_transitions"
            )
            data["schema_version"] = RL_TRAINING_CONFIG_SCHEMA_VERSION
            data["migration_metadata"] = {
                "source_schema_version": (
                    source_schema or "unspecified-v2-shape"
                ),
                "legacy_budget_field": "total_transitions",
                "resolved_budget_unit": unit,
            }
            warnings.warn(
                "migrating v2 total_transitions to explicit v3 budget",
                DeprecationWarning,
                stacklevel=2,
            )
        elif source_schema != RL_TRAINING_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported RL training config: {source_schema!r}"
            )
        if source_schema == RL_TRAINING_CONFIG_SCHEMA_VERSION:
            if "total_transitions" in data:
                raise ValueError(
                    "v3 RL training config uses budget, not total_transitions"
                )
            budget = data.pop("budget", None)
            if not isinstance(budget, Mapping):
                raise TypeError("v3 RL training config budget must be a mapping")
            if set(budget) != {"unit", "value"}:
                raise ValueError("budget requires exactly unit and value")
            algorithm_id = str(data.get("algorithm_id", "")).lower()
            expected_unit = (
                "optimizer_updates"
                if algorithm_id == "bc"
                else "environment_transitions"
            )
            if budget["unit"] != expected_unit:
                raise ValueError(
                    f"{algorithm_id.upper()} budget unit must be "
                    f"{expected_unit}"
                )
            value = budget["value"]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("budget value must be an integer")
            if value <= 0:
                raise ValueError("budget value must be positive")
            data["total_transitions"] = value
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
                "migration_metadata",
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


def _reject_unknown_mapping_fields(
    name: str,
    value: Mapping[str, Any],
    allowed: frozenset[str],
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"unknown {name} field(s): " + ", ".join(unknown)
        )


def _validate_policy_kwargs(
    algorithm_id: str,
    value: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("algorithm.policy_kwargs must be a mapping")
    _reject_unknown_mapping_fields(
        "algorithm.policy_kwargs",
        value,
        frozenset({"activation_fn", "net_arch"}),
    )
    activation = value.get("activation_fn")
    if activation not in {"relu", "tanh", "elu"}:
        raise ValueError(
            "algorithm.policy_kwargs.activation_fn must be one of: "
            "elu, relu, tanh"
        )
    net_arch = value.get("net_arch")
    if not isinstance(net_arch, Mapping):
        raise TypeError(
            "algorithm.policy_kwargs.net_arch must be a mapping"
        )
    expected = (
        frozenset({"pi", "vf"})
        if algorithm_id == "ppo"
        else frozenset({"pi", "qf"})
    )
    if set(net_arch) != expected:
        raise ValueError(
            "algorithm.policy_kwargs.net_arch requires exactly: "
            + ", ".join(sorted(expected))
        )
    for name, layers in net_arch.items():
        if not isinstance(layers, (list, tuple)) or not layers:
            raise ValueError(
                f"algorithm.policy_kwargs.net_arch.{name} must be a "
                "non-empty sequence"
            )
        if any(
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            for size in layers
        ):
            raise ValueError(
                f"algorithm.policy_kwargs.net_arch.{name} layer sizes "
                "must be positive integers"
            )


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
