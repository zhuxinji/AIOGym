"""Episode-oriented NumPy Dataset v3 reader and atomic writer."""
from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aiogym.core import file_sha256, stable_hash, write_json


DATASET_SCHEMA_VERSION = "aiogym.dataset.v3"
EPISODE_SCHEMA_VERSION = "aiogym.dataset.episode.v3"
REQUIRED_ARRAYS = (
    "observation",
    "action",
    "reward",
    "next_observation",
    "terminated",
    "truncated",
    "step_index",
    "physical_time",
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
            raise KeyError(f"unknown Dataset v3 array {name!r}") from error


class DatasetWriter:
    def __init__(
        self,
        output: str | Path,
        *,
        dataset_id: str,
        task_id: str,
        task_hash: str,
        plant_id: str,
        plant_hash: str,
        preset: str,
        policy: Mapping[str, Any],
        base_seed: int,
        observation_schema: Mapping[str, Any],
        action_schema: Mapping[str, Any],
        resume: bool = False,
        legacy_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(output)
        self.manifest_path = self.path / "manifest.json"
        identity = {
            "dataset_id": dataset_id,
            "task_id": task_id,
            "task_hash": task_hash,
            "plant_id": plant_id,
            "plant_hash": plant_hash,
            "preset": preset,
            "policy": dict(policy),
            "base_seed": int(base_seed),
            "observation_schema": dict(observation_schema),
            "action_schema": dict(action_schema),
        }
        collection_hash = stable_hash(identity)
        if self.manifest_path.exists():
            if not resume:
                raise FileExistsError(f"dataset already exists: {self.path}")
            self.manifest = _read_manifest(self.manifest_path)
            if self.manifest["collection_hash"] != collection_hash:
                raise ValueError("resume Dataset v3 identity does not match manifest")
        else:
            if self.path.exists() and any(self.path.iterdir()):
                raise FileExistsError(
                    f"refusing to create Dataset v3 in non-empty directory: {self.path}"
                )
            self.path.mkdir(parents=True, exist_ok=True)
            self.manifest = {
                "schema_version": DATASET_SCHEMA_VERSION,
                **identity,
                "collection_hash": collection_hash,
                "episode_count": 0,
                "transition_count": 0,
                "episodes": [],
            }
            if legacy_metadata is not None:
                self.manifest["legacy_metadata"] = dict(legacy_metadata)
            write_json(self.manifest_path, self.manifest)

    def append(self, index: int, seed: int, arrays: Mapping[str, Any], metadata=None):
        if index != len(self.manifest["episodes"]):
            raise ValueError(
                f"episode index must be contiguous; expected {len(self.manifest['episodes'])}"
            )
        normalized = _validated_arrays(arrays)
        episode_id = f"episode-{index:06d}"
        filename = f"{episode_id}.npz"
        target = self.path / filename
        if target.exists():
            raise FileExistsError(f"episode file already exists: {target}")
        episode_metadata = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "episode_id": episode_id,
            "episode_index": index,
            "seed": int(seed),
            **dict(metadata or {}),
        }
        content_hash = episode_content_hash(episode_metadata, normalized)
        _write_npz(target, normalized)
        record = {
            **episode_metadata,
            "file": filename,
            "file_sha256": file_sha256(target),
            "content_hash": content_hash,
            "transition_count": int(normalized["reward"].shape[0]),
            "array_shapes": {
                name: list(value.shape) for name, value in normalized.items()
            },
        }
        previous = dict(self.manifest)
        self.manifest["episodes"] = [*self.manifest["episodes"], record]
        self.manifest["episode_count"] = len(self.manifest["episodes"])
        self.manifest["transition_count"] = sum(
            row["transition_count"] for row in self.manifest["episodes"]
        )
        try:
            write_json(self.manifest_path, self.manifest, overwrite=True)
        except Exception:
            self.manifest = previous
            target.unlink(missing_ok=True)
            raise
        return record


class DatasetReader:
    def __init__(self, path: str | Path, *, verify_checksums: bool = False):
        self.path = Path(path)
        self.manifest = _read_manifest(self.path / "manifest.json")
        self._records = tuple(self.manifest["episodes"])
        self._by_id = {row["episode_id"]: row for row in self._records}
        self._verified: set[str] = set()
        if verify_checksums:
            report = self.validate()
            if not report["ok"]:
                raise ValueError("Dataset v3 integrity failed: " + "; ".join(report["errors"]))

    def __len__(self):
        return len(self._records)

    @property
    def transition_count(self):
        return int(self.manifest["transition_count"])

    def load_episode(self, episode: int | str) -> DatasetEpisode:
        record = self._record(episode)
        path = self.path / record["file"]
        if file_sha256(path) != record["file_sha256"]:
            raise ValueError(f"episode checksum mismatch: {record['episode_id']}")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays = _validated_arrays(arrays)
        metadata = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "file",
                "file_sha256",
                "content_hash",
                "array_shapes",
                "transition_count",
            }
        }
        if episode_content_hash(metadata, arrays) != record["content_hash"]:
            raise ValueError(f"episode content hash mismatch: {record['episode_id']}")
        for array in arrays.values():
            array.setflags(write=False)
        return DatasetEpisode(metadata=metadata, arrays=arrays)

    def iter_episodes(self):
        for index in range(len(self)):
            yield self.load_episode(index)

    def validate(self):
        errors = []
        transitions = 0
        for index, record in enumerate(self._records):
            if record["episode_index"] != index:
                errors.append(f"non-contiguous episode index at {index}")
            try:
                episode = self.load_episode(index)
            except Exception as error:
                errors.append(str(error))
                continue
            transitions += episode.transition_count
        if transitions != self.transition_count:
            errors.append("manifest transition_count mismatch")
        if len(self) != int(self.manifest["episode_count"]):
            errors.append("manifest episode_count mismatch")
        return {
            "ok": not errors,
            "dataset_id": self.manifest["dataset_id"],
            "episode_count": len(self),
            "transition_count": transitions,
            "errors": errors,
        }

    def _record(self, episode):
        if isinstance(episode, int) and not isinstance(episode, bool):
            return self._records[episode]
        try:
            return self._by_id[str(episode)]
        except KeyError as error:
            raise KeyError(f"unknown episode {episode!r}") from error


def episode_content_hash(metadata, arrays):
    digest = hashlib.sha256(stable_hash(metadata).encode("ascii"))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validated_arrays(arrays):
    normalized = {str(name): np.asarray(value) for name, value in arrays.items()}
    missing = set(REQUIRED_ARRAYS) - set(normalized)
    if missing:
        raise ValueError(f"Dataset v3 episode arrays missing: {sorted(missing)}")
    length = int(normalized["reward"].shape[0])
    if length <= 0:
        raise ValueError("Dataset v3 episodes must contain transitions")
    for name, array in normalized.items():
        if array.shape[0] != length:
            raise ValueError(f"Dataset v3 array {name!r} has inconsistent length")
        if array.dtype == object:
            raise TypeError(f"Dataset v3 array {name!r} must not use object dtype")
    if not np.array_equal(normalized["step_index"], np.arange(length)):
        raise ValueError("Dataset v3 step_index must be contiguous from zero")
    if np.any(np.diff(normalized["physical_time"]) <= 0):
        raise ValueError("Dataset v3 physical_time must be strictly increasing")
    return normalized


def _write_npz(target: Path, arrays):
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            raise FileExistsError(f"episode file already exists: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest(path):
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported Dataset schema; expected aiogym.dataset.v3")
    return payload


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DatasetEpisode",
    "DatasetReader",
    "DatasetWriter",
    "episode_content_hash",
]
