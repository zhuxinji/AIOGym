"""Episode-oriented Dataset v2 schema."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


DATASET_SCHEMA_VERSION = "aiogym.dataset.v2"
EPISODE_DATA_SCHEMA_VERSION = "aiogym.dataset.episode.v2"

_REQUIRED_METADATA = frozenset(
    {
        "episode_id",
        "split",
        "track_id",
        "distribution_id",
        "distribution_hash",
        "episode_spec_id",
        "resolved_hash",
        "base_seed",
        "component_seeds",
        "scenario",
        "goal",
        "action_mode",
        "collector_id",
        "policy_id",
        "collector_quality_tag",
        "plant_parameters",
        "initial_state",
        "reference_schedule",
        "disturbance_schedule",
        "sensor_model",
        "actuator_model",
        "termination_reason",
        "summary",
    }
)
_ARRAY_FIELDS = (
    "observation",
    "true_state",
    "reference",
    "measured_disturbance",
    "action_policy_normalized",
    "action_commanded_physical",
    "action_applied_physical",
    "reward_scalar",
    "next_observation",
    "next_true_state",
    "terminated",
    "truncated",
    "bootstrap_mask",
    "step_index",
    "physical_time",
)


@dataclass(frozen=True, init=False)
class DatasetEpisode:
    """One validated episode with immutable transition arrays."""

    _metadata_json: str
    _arrays: Mapping[str, np.ndarray]
    _reward_term_names: tuple[str, ...]
    _cost_channel_names: tuple[str, ...]

    def __init__(
        self,
        *,
        metadata: Mapping[str, Any],
        observation,
        true_state,
        reference,
        action_policy_normalized,
        action_commanded_physical,
        action_applied_physical,
        reward_scalar,
        next_observation,
        next_true_state,
        terminated,
        truncated,
        bootstrap_mask,
        step_index,
        physical_time,
        measured_disturbance=None,
        reward_terms: Mapping[str, Any] | None = None,
        cost_channels: Mapping[str, Any] | None = None,
    ) -> None:
        normalized_metadata = _validated_metadata(metadata)
        arrays = {
            "observation": _matrix("observation", observation),
            "true_state": _matrix("true_state", true_state),
            "reference": _matrix("reference", reference),
            "measured_disturbance": _matrix(
                "measured_disturbance",
                measured_disturbance,
                rows_hint=len(observation),
                allow_empty_width=True,
            ),
            "action_policy_normalized": _matrix(
                "action_policy_normalized",
                action_policy_normalized,
            ),
            "action_commanded_physical": _matrix(
                "action_commanded_physical",
                action_commanded_physical,
            ),
            "action_applied_physical": _matrix(
                "action_applied_physical",
                action_applied_physical,
            ),
            "reward_scalar": _vector(
                "reward_scalar",
                reward_scalar,
                dtype=np.float64,
            ),
            "next_observation": _matrix(
                "next_observation",
                next_observation,
            ),
            "next_true_state": _matrix(
                "next_true_state",
                next_true_state,
            ),
            "terminated": _vector(
                "terminated",
                terminated,
                dtype=np.bool_,
            ),
            "truncated": _vector(
                "truncated",
                truncated,
                dtype=np.bool_,
            ),
            "bootstrap_mask": _vector(
                "bootstrap_mask",
                bootstrap_mask,
                dtype=np.float32,
            ),
            "step_index": _vector(
                "step_index",
                step_index,
                dtype=np.int64,
            ),
            "physical_time": _vector(
                "physical_time",
                physical_time,
                dtype=np.float64,
            ),
        }
        length = arrays["reward_scalar"].shape[0]
        if length <= 0:
            raise ValueError("DatasetEpisode must contain at least one transition")
        for name, array in arrays.items():
            if array.shape[0] != length:
                raise ValueError(
                    f"{name} has {array.shape[0]} rows; expected {length}"
                )
        if arrays["observation"].shape != arrays["next_observation"].shape:
            raise ValueError(
                "observation and next_observation shapes must match"
            )
        if arrays["true_state"].shape != arrays["next_true_state"].shape:
            raise ValueError("true_state and next_true_state shapes must match")
        action_shape = arrays["action_policy_normalized"].shape
        for name in (
            "action_commanded_physical",
            "action_applied_physical",
        ):
            if arrays[name].shape != action_shape:
                raise ValueError(
                    f"{name} shape must match action_policy_normalized"
                )
        if not np.all(
            np.logical_or(
                arrays["bootstrap_mask"] == 0.0,
                arrays["bootstrap_mask"] == 1.0,
            )
        ):
            raise ValueError("bootstrap_mask values must be 0 or 1")
        if np.any(
            np.logical_and(
                arrays["terminated"],
                arrays["bootstrap_mask"] != 0.0,
            )
        ):
            raise ValueError(
                "terminated transitions must have bootstrap_mask=0"
            )
        if np.any(
            np.logical_and(arrays["terminated"], arrays["truncated"])
        ):
            raise ValueError(
                "a transition cannot be both terminated and truncated"
            )
        done = np.logical_or(arrays["terminated"], arrays["truncated"])
        if np.any(done[:-1]):
            raise ValueError(
                "terminated/truncated may only be true on the final transition"
            )
        expected_steps = np.arange(length, dtype=np.int64)
        if not np.array_equal(arrays["step_index"], expected_steps):
            raise ValueError("step_index must be contiguous and start at zero")
        if np.any(np.diff(arrays["physical_time"]) <= 0.0):
            raise ValueError("physical_time must be strictly increasing")
        initial_state = np.asarray(
            normalized_metadata["initial_state"],
            dtype=np.float64,
        ).reshape(-1)
        if (
            initial_state.shape[0] != arrays["true_state"].shape[1]
            or not np.allclose(
                initial_state,
                arrays["true_state"][0],
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise ValueError(
                "metadata initial_state must match the first true_state"
            )

        reward_names, reward_matrix = _channels(
            "reward_terms",
            reward_terms,
            length,
        )
        cost_names, cost_matrix = _channels(
            "cost_channels",
            cost_channels,
            length,
        )
        arrays["reward_terms"] = reward_matrix
        arrays["cost_channels"] = cost_matrix
        for array in arrays.values():
            array.setflags(write=False)
        object.__setattr__(
            self,
            "_metadata_json",
            _canonical_json(normalized_metadata),
        )
        object.__setattr__(self, "_arrays", arrays)
        object.__setattr__(self, "_reward_term_names", reward_names)
        object.__setattr__(self, "_cost_channel_names", cost_names)

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self._metadata_json)

    @property
    def episode_id(self) -> str:
        return str(self.metadata["episode_id"])

    @property
    def split(self) -> str:
        return str(self.metadata["split"])

    @property
    def transition_count(self) -> int:
        return int(self._arrays["reward_scalar"].shape[0])

    @property
    def reward_term_names(self) -> tuple[str, ...]:
        return self._reward_term_names

    @property
    def cost_channel_names(self) -> tuple[str, ...]:
        return self._cost_channel_names

    @property
    def reward_terms(self) -> dict[str, np.ndarray]:
        matrix = self._arrays["reward_terms"]
        return {
            name: matrix[:, index]
            for index, name in enumerate(self.reward_term_names)
        }

    @property
    def cost_channels(self) -> dict[str, np.ndarray]:
        matrix = self._arrays["cost_channels"]
        return {
            name: matrix[:, index]
            for index, name in enumerate(self.cost_channel_names)
        }

    @property
    def content_hash(self) -> str:
        digest = hashlib.sha256(self._metadata_json.encode("utf-8"))
        for name in sorted(self._arrays):
            array = np.ascontiguousarray(self._arrays[name])
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        digest.update(_canonical_json(list(self.reward_term_names)).encode())
        digest.update(_canonical_json(list(self.cost_channel_names)).encode())
        return digest.hexdigest()

    def array(self, name: str) -> np.ndarray:
        try:
            return self._arrays[str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown episode array {name!r}") from exc

    def arrays(self) -> dict[str, np.ndarray]:
        return dict(self._arrays)

    def storage_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": EPISODE_DATA_SCHEMA_VERSION,
            "metadata": self.metadata,
            "reward_term_names": list(self.reward_term_names),
            "cost_channel_names": list(self.cost_channel_names),
            "transition_count": self.transition_count,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_storage(
        cls,
        storage_metadata: Mapping[str, Any],
        arrays: Mapping[str, Any],
    ) -> "DatasetEpisode":
        if (
            storage_metadata.get("schema_version")
            != EPISODE_DATA_SCHEMA_VERSION
        ):
            raise ValueError("unsupported stored episode schema")
        reward_names = tuple(storage_metadata.get("reward_term_names", ()))
        cost_names = tuple(storage_metadata.get("cost_channel_names", ()))
        reward_matrix = np.asarray(arrays["reward_terms"])
        cost_matrix = np.asarray(arrays["cost_channels"])
        reward_terms = {
            name: reward_matrix[:, index]
            for index, name in enumerate(reward_names)
        }
        cost_channels = {
            name: cost_matrix[:, index]
            for index, name in enumerate(cost_names)
        }
        episode = cls(
            metadata=storage_metadata["metadata"],
            observation=arrays["observation"],
            true_state=arrays["true_state"],
            reference=arrays["reference"],
            measured_disturbance=arrays["measured_disturbance"],
            action_policy_normalized=arrays[
                "action_policy_normalized"
            ],
            action_commanded_physical=arrays[
                "action_commanded_physical"
            ],
            action_applied_physical=arrays[
                "action_applied_physical"
            ],
            reward_scalar=arrays["reward_scalar"],
            reward_terms=reward_terms,
            cost_channels=cost_channels,
            next_observation=arrays["next_observation"],
            next_true_state=arrays["next_true_state"],
            terminated=arrays["terminated"],
            truncated=arrays["truncated"],
            bootstrap_mask=arrays["bootstrap_mask"],
            step_index=arrays["step_index"],
            physical_time=arrays["physical_time"],
        )
        expected_hash = storage_metadata.get("content_hash")
        if expected_hash is not None and episode.content_hash != expected_hash:
            raise ValueError("stored episode content hash mismatch")
        return episode


def _validated_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("episode metadata must be a mapping")
    data = copy.deepcopy(dict(metadata))
    missing = sorted(_REQUIRED_METADATA - set(data))
    if missing:
        raise ValueError(
            "episode metadata is missing fields: " + ", ".join(missing)
        )
    for name in (
        "episode_id",
        "split",
        "track_id",
        "distribution_id",
        "distribution_hash",
        "episode_spec_id",
        "resolved_hash",
        "scenario",
        "goal",
        "action_mode",
        "collector_id",
        "policy_id",
        "collector_quality_tag",
        "termination_reason",
    ):
        if not isinstance(data[name], str) or not data[name]:
            raise ValueError(f"episode metadata {name} must be non-empty")
    if data["split"] not in {"training", "validation", "test"}:
        raise ValueError(
            "episode metadata split must be training, validation, or test"
        )
    if data["goal"] not in {"regulation", "economic"}:
        raise ValueError("episode metadata goal is invalid")
    if data["action_mode"] not in {"actuator", "setpoint"}:
        raise ValueError("episode metadata action_mode is invalid")
    if (
        isinstance(data["base_seed"], bool)
        or not isinstance(data["base_seed"], int)
        or data["base_seed"] < 0
    ):
        raise ValueError("episode metadata base_seed must be non-negative")
    for name in (
        "component_seeds",
        "plant_parameters",
        "sensor_model",
        "actuator_model",
        "summary",
    ):
        if not isinstance(data[name], Mapping):
            raise TypeError(f"episode metadata {name} must be a mapping")
    for name in (
        "initial_state",
        "reference_schedule",
        "disturbance_schedule",
    ):
        if (
            isinstance(data[name], (str, bytes))
            or not isinstance(data[name], Sequence)
        ):
            raise TypeError(f"episode metadata {name} must be a sequence")
    _canonical_json(data)
    return data


def _matrix(
    name: str,
    value,
    *,
    rows_hint: int | None = None,
    allow_empty_width: bool = False,
) -> np.ndarray:
    if value is None and allow_empty_width:
        if rows_hint is None:
            raise ValueError(f"{name} requires rows_hint")
        return np.empty((int(rows_hint), 0), dtype=np.float32)
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if array.ndim == 1 and allow_empty_width and array.size == 0:
        return np.empty((int(rows_hint or 0), 0), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not allow_empty_width and array.shape[1] <= 0:
        raise ValueError(f"{name} must have at least one column")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _vector(name: str, value, *, dtype) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a vector") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.issubdtype(array.dtype, np.number) and not np.all(
        np.isfinite(array)
    ):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _channels(name, channels, length):
    source = dict(channels or {})
    names = tuple(sorted(str(key) for key in source))
    if len(set(names)) != len(names):
        raise ValueError(f"{name} names must be unique")
    columns = []
    for channel_name in names:
        if not channel_name:
            raise ValueError(f"{name} names must be non-empty")
        column = _vector(
            f"{name}.{channel_name}",
            source[channel_name],
            dtype=np.float32,
        )
        if column.shape[0] != length:
            raise ValueError(
                f"{name}.{channel_name} has {column.shape[0]} rows; "
                f"expected {length}"
            )
        columns.append(column)
    matrix = (
        np.column_stack(columns).astype(np.float32, copy=False)
        if columns
        else np.empty((length, 0), dtype=np.float32)
    )
    return names, np.ascontiguousarray(matrix)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _jsonable(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    raise TypeError(
        f"metadata contains non-JSON value {type(value).__name__}"
    )


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "EPISODE_DATA_SCHEMA_VERSION",
    "DatasetEpisode",
]
