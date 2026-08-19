"""Small episode-oriented NumPy dataset format."""
from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aiogym.core.io import write_json


DATASET_SCHEMA_VERSION = "aiogym.dataset.v2"
REQUIRED_ARRAYS = (
    "observation",
    "action",
    "commanded_action",
    "channel_action",
    "applied_action",
    "reward",
    "next_observation",
    "terminated",
    "truncated",
    "step_index",
    "physical_time",
    "reference",
    "disturbance",
    "transition_reference",
    "transition_disturbance",
    "minimum_safety_margin",
)


@dataclass(frozen=True)
class DatasetEpisode:
    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]

    @property
    def episode_id(self) -> str:
        return str(self.metadata["episode_id"])

    @property
    def transition_count(self) -> int:
        return int(self.arrays["reward"].shape[0])

    def array(self, name: str) -> np.ndarray:
        try:
            return self.arrays[name]
        except KeyError as error:
            raise KeyError(f"unknown dataset array {name!r}") from error


class DatasetWriter:
    """Write a new dataset directory. Existing non-empty directories are rejected."""

    def __init__(
        self,
        output: str | Path,
        *,
        environment: Mapping[str, Any],
        policy: Mapping[str, Any],
        base_seed: int,
    ) -> None:
        self.path = Path(output)
        self.metadata_path = self.path / "metadata.json"
        if self.path.exists() and any(self.path.iterdir()):
            raise FileExistsError(
                f"refusing to create dataset in non-empty directory: {self.path}"
            )
        self.path.mkdir(parents=True, exist_ok=True)
        self.metadata: dict[str, Any] = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "environment": dict(environment),
            "policy": dict(policy),
            "base_seed": _nonnegative_int("base_seed", base_seed),
            "episode_count": 0,
            "transition_count": 0,
            "episodes": [],
        }
        write_json(self.metadata_path, self.metadata)

    def append(
        self,
        index: int,
        seed: int,
        arrays: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        expected = len(self.metadata["episodes"])
        if isinstance(index, bool) or not isinstance(index, int) or index != expected:
            raise ValueError(f"episode index must be contiguous; expected {expected}")
        normalized = _validated_arrays(arrays)
        episode_id = f"episode-{index:06d}"
        filename = f"{episode_id}.npz"
        target = self.path / filename
        if target.exists():
            raise FileExistsError(f"episode file already exists: {target}")
        np.savez_compressed(target, **normalized)
        record = {
            "episode_id": episode_id,
            "episode_index": index,
            "file": filename,
            "seed": _nonnegative_int("seed", seed),
            "transitions": int(normalized["reward"].shape[0]),
            **({} if metadata is None else dict(metadata)),
        }
        self.metadata["episodes"].append(record)
        self.metadata["episode_count"] = len(self.metadata["episodes"])
        self.metadata["transition_count"] = sum(
            row["transitions"] for row in self.metadata["episodes"]
        )
        write_json(self.metadata_path, self.metadata, overwrite=True)
        return record


class DatasetReader:
    """Read the current dataset format directly."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata = _read_metadata(self.path / "metadata.json")
        self._records = tuple(self.metadata["episodes"])
        self._by_id = {row["episode_id"]: row for row in self._records}

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, episode: int | str) -> DatasetEpisode:
        return self.load_episode(episode)

    @property
    def transition_count(self) -> int:
        return int(self.metadata["transition_count"])

    def load_episode(self, episode: int | str) -> DatasetEpisode:
        record = self._record(episode)
        path = self.path / record["file"]
        if not path.is_file():
            raise FileNotFoundError(f"dataset episode file is missing: {path}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays = _validated_arrays(arrays)
        if int(arrays["reward"].shape[0]) != record["transitions"]:
            raise ValueError(
                f"dataset episode transition count is inconsistent: {record['episode_id']}"
            )
        for array in arrays.values():
            array.setflags(write=False)
        episode_metadata = {
            key: value for key, value in record.items() if key != "file"
        }
        return DatasetEpisode(metadata=episode_metadata, arrays=arrays)

    def iter_episodes(self) -> Iterator[DatasetEpisode]:
        for index in range(len(self)):
            yield self.load_episode(index)

    def _record(self, episode: int | str) -> Mapping[str, Any]:
        if isinstance(episode, bool):
            raise TypeError("episode must be an integer index or episode id")
        if isinstance(episode, int):
            return self._records[episode]
        try:
            return self._by_id[str(episode)]
        except KeyError as error:
            raise KeyError(f"unknown episode {episode!r}") from error


def _validated_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    normalized = {str(name): np.asarray(value) for name, value in arrays.items()}
    missing = set(REQUIRED_ARRAYS) - set(normalized)
    if missing:
        raise ValueError(f"dataset episode arrays missing: {sorted(missing)}")
    reward = normalized["reward"]
    if reward.ndim == 0:
        raise ValueError("dataset reward array must have a transition dimension")
    length = int(reward.shape[0])
    if length <= 0:
        raise ValueError("dataset episodes must contain transitions")
    for name, array in normalized.items():
        if array.ndim == 0 or array.shape[0] != length:
            raise ValueError(f"dataset array {name!r} has inconsistent length")
        if array.dtype == object:
            raise TypeError(f"dataset array {name!r} must not use object dtype")
    if not np.array_equal(normalized["step_index"], np.arange(length)):
        raise ValueError("dataset step_index must be contiguous from zero")
    if np.any(np.diff(normalized["physical_time"]) <= 0):
        raise ValueError("dataset physical_time must be strictly increasing")
    return normalized


def _read_metadata(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant {value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("dataset metadata must contain a JSON object")
    if payload.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset schema; expected {DATASET_SCHEMA_VERSION}")
    for field in (
        "environment",
        "policy",
        "base_seed",
        "episode_count",
        "transition_count",
        "episodes",
    ):
        if field not in payload:
            raise ValueError(f"dataset metadata is missing {field!r}")
    if not isinstance(payload["environment"], dict):
        raise ValueError("dataset environment metadata must be an object")
    if not isinstance(payload["policy"], dict):
        raise ValueError("dataset policy metadata must be an object")
    episodes = payload["episodes"]
    if not isinstance(episodes, list):
        raise ValueError("dataset episodes must be a list")
    transition_count = 0
    for index, record in enumerate(episodes):
        expected_id = f"episode-{index:06d}"
        if not isinstance(record, dict):
            raise ValueError("dataset episode records must be objects")
        if (
            record.get("episode_id") != expected_id
            or record.get("episode_index") != index
            or record.get("file") != f"{expected_id}.npz"
        ):
            raise ValueError(f"dataset episode record {index} is inconsistent")
        transitions = record.get("transitions")
        if (
            isinstance(transitions, bool)
            or not isinstance(transitions, int)
            or transitions <= 0
        ):
            raise ValueError("dataset episode transitions must be positive")
        transition_count += transitions
    if payload["episode_count"] != len(episodes):
        raise ValueError("dataset episode_count is inconsistent")
    if payload["transition_count"] != transition_count:
        raise ValueError("dataset transition_count is inconsistent")
    return payload


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DatasetEpisode",
    "DatasetReader",
    "DatasetWriter",
    "REQUIRED_ARRAYS",
]
