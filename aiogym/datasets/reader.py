"""Streaming metadata and random-access reader for Dataset v2."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import numpy as np

from .schema import DatasetEpisode
from .writer import file_sha256, load_manifest


class DatasetReader:
    """Read episode metadata without arrays and load shards on demand."""

    def __init__(
        self,
        path: str | Path,
        *,
        verify_checksums: bool = False,
    ) -> None:
        self.path = Path(path)
        self.manifest = load_manifest(self.path)
        self._records = tuple(self.manifest["episodes"])
        self._by_id = {
            record["episode_id"]: record for record in self._records
        }
        self._counts = np.asarray(
            [record["transition_count"] for record in self._records],
            dtype=np.int64,
        )
        self._cumulative = np.cumsum(self._counts)
        self._verify_checksums = bool(verify_checksums)
        self._verified_shards: set[str] = set()
        if verify_checksums:
            report = self.validate_integrity(load_episodes=False)
            if not report["ok"]:
                raise ValueError(
                    "dataset checksum validation failed: "
                    + "; ".join(report["errors"])
                )

    def __len__(self) -> int:
        return len(self._records)

    @property
    def transition_count(self) -> int:
        return int(self.manifest["transition_count"])

    @property
    def episode_ids(self) -> tuple[str, ...]:
        return tuple(record["episode_id"] for record in self._records)

    def metadata_records(self) -> tuple[dict, ...]:
        """Return manifest episode rows without loading any shard arrays."""

        return tuple(copy.deepcopy(record) for record in self._records)

    def load_episode(self, episode: str | int) -> DatasetEpisode:
        record = self._record(episode)
        shard = self.path / record["shard"]
        if self._verify_checksums:
            self._verify_shard(record)
        with np.load(shard, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        return DatasetEpisode.from_storage(record["episode"], arrays)

    def iter_episodes(
        self,
        episode_ids: Iterable[str] | None = None,
    ):
        selected = self.episode_ids if episode_ids is None else episode_ids
        for episode_id in selected:
            yield self.load_episode(episode_id)

    def random_batch(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator | None = None,
        fields: tuple[str, ...] = (
            "observation",
            "action_policy_normalized",
            "reward_scalar",
            "next_observation",
            "bootstrap_mask",
        ),
    ) -> dict[str, np.ndarray]:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if self.transition_count <= 0:
            raise ValueError("cannot sample an empty dataset")
        generator = rng or np.random.default_rng()
        global_indices = generator.integers(
            0,
            self.transition_count,
            size=batch_size,
        )
        episode_indices = np.searchsorted(
            self._cumulative,
            global_indices,
            side="right",
        )
        starts = np.concatenate(
            (np.asarray([0], dtype=np.int64), self._cumulative[:-1])
        )
        local_indices = global_indices - starts[episode_indices]
        cache = {}
        rows = {field: [] for field in fields}
        episode_ids = []
        for episode_index, local_index in zip(
            episode_indices,
            local_indices,
        ):
            index = int(episode_index)
            if index not in cache:
                cache[index] = self.load_episode(index)
            episode = cache[index]
            for field in fields:
                rows[field].append(episode.array(field)[int(local_index)])
            episode_ids.append(episode.episode_id)
        result = {
            field: np.asarray(values) for field, values in rows.items()
        }
        result["episode_id"] = np.asarray(episode_ids, dtype=np.str_)
        result["transition_index"] = np.asarray(
            local_indices,
            dtype=np.int64,
        )
        return result

    def validate_integrity(self, *, load_episodes: bool = True) -> dict:
        errors = []
        seen_ids = set()
        seen_hashes = set()
        transitions = 0
        for record in self._records:
            episode_id = record["episode_id"]
            if episode_id in seen_ids:
                errors.append(f"duplicate episode_id {episode_id}")
            seen_ids.add(episode_id)
            if record["content_hash"] in seen_hashes:
                errors.append(f"duplicate content hash for {episode_id}")
            seen_hashes.add(record["content_hash"])
            shard = self.path / record["shard"]
            if not shard.is_file():
                errors.append(f"missing shard for {episode_id}")
                continue
            if file_sha256(shard) != record["shard_sha256"]:
                errors.append(f"checksum mismatch for {episode_id}")
                continue
            self._verified_shards.add(str(record["shard"]))
            transitions += int(record["transition_count"])
            if load_episodes:
                try:
                    episode = self.load_episode(episode_id)
                except Exception as exc:
                    errors.append(f"invalid episode {episode_id}: {exc}")
                    continue
                if episode.content_hash != record["content_hash"]:
                    errors.append(f"content hash mismatch for {episode_id}")
        if transitions != int(self.manifest["transition_count"]):
            errors.append("manifest transition_count mismatch")
        if len(self._records) != int(self.manifest["episode_count"]):
            errors.append("manifest episode_count mismatch")
        return {
            "ok": not errors,
            "dataset_id": self.manifest["dataset_id"],
            "episode_count": len(self._records),
            "transition_count": transitions,
            "errors": errors,
        }

    def _verify_shard(self, record) -> None:
        identity = str(record["shard"])
        if identity in self._verified_shards:
            return
        shard = self.path / identity
        actual_checksum = file_sha256(shard)
        if actual_checksum != record["shard_sha256"]:
            raise ValueError(
                f"shard checksum mismatch for {record['episode_id']!r}"
            )
        self._verified_shards.add(identity)

    def _record(self, episode):
        if isinstance(episode, int) and not isinstance(episode, bool):
            try:
                return self._records[episode]
            except IndexError as exc:
                raise IndexError("episode index out of range") from exc
        try:
            return self._by_id[str(episode)]
        except KeyError as exc:
            raise KeyError(f"unknown episode_id {episode!r}") from exc


def validate_dataset(path: str | Path) -> dict:
    try:
        reader = DatasetReader(path)
    except Exception as exc:
        return {
            "ok": False,
            "episode_count": 0,
            "transition_count": 0,
            "errors": [str(exc)],
        }
    return reader.validate_integrity(load_episodes=True)


__all__ = ["DatasetReader", "validate_dataset"]
