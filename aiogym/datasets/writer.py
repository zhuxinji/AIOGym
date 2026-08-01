"""Atomic streaming writer for the core compressed-NPZ dataset backend."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .schema import DATASET_SCHEMA_VERSION, DatasetEpisode


MANIFEST_FILENAME = "manifest.json"
_SHARD_PATTERN = re.compile(r"part-(\d{8})\.npz$")


class DatasetWriter:
    """Append complete episodes using atomic shard and manifest replacement."""

    def __init__(
        self,
        path: str | Path,
        *,
        dataset_id: str,
        split: str,
        resume: bool = False,
        collection_metadata: dict | None = None,
    ) -> None:
        self.path = Path(path)
        self.manifest_path = self.path / MANIFEST_FILENAME
        self.shards_path = self.path / "shards"
        self.reports_path = self.path / "reports"
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        if split not in {"training", "validation", "test"}:
            raise ValueError("split must be training, validation, or test")
        self.path.mkdir(parents=True, exist_ok=True)
        self.shards_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)

        if self.manifest_path.exists():
            if not resume:
                raise FileExistsError(
                    f"dataset manifest already exists: {self.manifest_path}"
                )
            manifest = load_manifest(self.path)
            if manifest["dataset_id"] != dataset_id:
                raise ValueError("resume dataset_id does not match manifest")
            if manifest["split"] != split:
                raise ValueError("resume split does not match manifest")
            existing_collection = manifest.get("collection")
            if (
                collection_metadata is not None
                and existing_collection is not None
                and existing_collection.get("config_hash")
                != collection_metadata.get("config_hash")
            ):
                raise ValueError(
                    "resume collection config hash does not match manifest"
                )
            self._manifest = manifest
        else:
            now = _utc_now()
            self._manifest = {
                "schema_version": DATASET_SCHEMA_VERSION,
                "backend": "compressed_npz_episode_shards_v1",
                "dataset_id": dataset_id,
                "split": split,
                "created_at": now,
                "updated_at": now,
                "episode_count": 0,
                "transition_count": 0,
                "episodes": [],
            }
            if collection_metadata is not None:
                self._manifest["collection"] = copy.deepcopy(
                    collection_metadata
                )
            self._write_manifest()
        self._reconcile_shards()
        self._episode_ids = {
            record["episode_id"]
            for record in self._manifest["episodes"]
        }
        self._content_hashes = {
            record["content_hash"]
            for record in self._manifest["episodes"]
        }
        self._next_shard_index = _next_shard_index(
            self.shards_path,
            self._manifest["episodes"],
        )

    @property
    def manifest(self) -> dict:
        return copy.deepcopy(self._manifest)

    def append_episode(self, episode: DatasetEpisode) -> dict:
        if not isinstance(episode, DatasetEpisode):
            raise TypeError("episode must be a DatasetEpisode")
        if episode.split != self._manifest["split"]:
            raise ValueError(
                f"episode split {episode.split!r} does not match dataset "
                f"split {self._manifest['split']!r}"
            )
        if episode.episode_id in self._episode_ids:
            raise ValueError(
                f"duplicate episode_id {episode.episode_id!r}"
            )
        if episode.content_hash in self._content_hashes:
            raise ValueError(
                "duplicate episode content hash "
                f"{episode.content_hash}"
            )
        metadata = episode.metadata
        episode_index = metadata.get("episode_index")
        if episode_index is not None:
            episode_index = int(episode_index)
            committed = [
                int(record["episode_index"])
                for record in self._manifest["episodes"]
                if record.get("episode_index") is not None
            ]
            expected = max(committed, default=-1) + 1
            if episode_index != expected:
                raise ValueError(
                    "episode_index must follow canonical ordering; "
                    f"expected {expected}, got {episode_index}"
                )

        shard_name = f"part-{self._next_shard_index:08d}.npz"
        shard_path = self.shards_path / shard_name
        temporary = self.shards_path / (
            f".{shard_name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **episode.arrays())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, shard_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        checksum = file_sha256(shard_path)
        storage_metadata = episode.storage_metadata()
        record = {
            "episode_id": episode.episode_id,
            "episode_spec_id": metadata["episode_spec_id"],
            "resolved_hash": metadata["resolved_hash"],
            "distribution_id": metadata["distribution_id"],
            "distribution_hash": metadata["distribution_hash"],
            "collector_id": metadata["collector_id"],
            "collector_quality_tag": metadata[
                "collector_quality_tag"
            ],
            "termination_reason": metadata["termination_reason"],
            "difficulty_tags": list(
                metadata.get("difficulty_tags", ())
            ),
            "transition_count": episode.transition_count,
            "content_hash": episode.content_hash,
            "shard": f"shards/{shard_name}",
            "shard_sha256": checksum,
            "shard_bytes": shard_path.stat().st_size,
            "episode": storage_metadata,
        }
        if episode_index is not None:
            record["episode_index"] = episode_index
        previous_manifest = copy.deepcopy(self._manifest)
        self._manifest["episodes"].append(record)
        self._manifest["episode_count"] += 1
        self._manifest["transition_count"] += episode.transition_count
        self._manifest["updated_at"] = _utc_now()
        try:
            self._write_manifest()
        except Exception as manifest_error:
            self._manifest = previous_manifest
            try:
                shard_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                manifest_error.add_note(
                    "failed to remove uncommitted dataset shard "
                    f"{shard_path}: {cleanup_error}"
                )
            raise
        self._episode_ids.add(episode.episode_id)
        self._content_hashes.add(episode.content_hash)
        self._next_shard_index += 1
        return copy.deepcopy(record)

    def update_collection_metadata(self, **values) -> None:
        """Atomically update collection accounting in the manifest."""

        collection = self._manifest.setdefault("collection", {})
        collection.update(copy.deepcopy(values))
        self._manifest["updated_at"] = _utc_now()
        self._write_manifest()

    def close(self) -> None:
        self._write_manifest()

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()

    def _write_manifest(self) -> None:
        _atomic_write_json(
            self.manifest_path,
            manifest_with_hash(self._manifest),
        )

    def _reconcile_shards(self) -> None:
        """Quarantine uncommitted files and fail on committed corruption."""

        referenced: set[Path] = set()
        for record in self._manifest["episodes"]:
            relative = Path(record["shard"])
            shard = self.path / relative
            if shard.parent != self.shards_path:
                raise ValueError(
                    f"dataset manifest has invalid shard path: {relative}"
                )
            if not shard.is_file():
                raise FileNotFoundError(
                    f"dataset manifest referenced shard is missing: {shard}"
                )
            if file_sha256(shard) != record["shard_sha256"]:
                raise ValueError(
                    f"dataset referenced shard checksum mismatch: {shard}"
                )
            referenced.add(shard.resolve())

        candidates = list(self.shards_path.glob("part-*.npz"))
        candidates.extend(self.shards_path.glob(".*.tmp"))
        orphaned = [
            path
            for path in candidates
            if path.resolve() not in referenced
        ]
        if not orphaned:
            return
        quarantine = self.shards_path / ".orphaned"
        quarantine.mkdir(parents=True, exist_ok=True)
        recovered_at = _utc_now()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        events = []
        for source in sorted(orphaned):
            destination = quarantine / (
                f"{source.name}.{stamp}.{uuid.uuid4().hex}.orphan"
            )
            os.replace(source, destination)
            events.append(
                {
                    "kind": "orphan_shard_quarantined",
                    "source": str(source.relative_to(self.path)),
                    "quarantined": str(destination.relative_to(self.path)),
                    "recovered_at": recovered_at,
                }
            )
        collection = self._manifest.setdefault("collection", {})
        collection.setdefault("recovery_events", []).extend(events)
        self._manifest["updated_at"] = recovered_at
        self._write_manifest()


def manifest_with_hash(manifest: dict) -> dict:
    payload = copy.deepcopy(dict(manifest))
    payload.pop("manifest_hash", None)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    payload["manifest_hash"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return payload


def load_manifest(path: str | Path) -> dict:
    manifest_path = Path(path) / MANIFEST_FILENAME
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported dataset manifest schema")
    expected = manifest.get("manifest_hash")
    actual = manifest_with_hash(manifest)["manifest_hash"]
    if expected != actual:
        raise ValueError("dataset manifest hash mismatch")
    manifest.pop("manifest_hash", None)
    return manifest


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _next_shard_index(shards_path: Path, episodes=()) -> int:
    indices = []
    for path in shards_path.glob("part-*.npz"):
        match = _SHARD_PATTERN.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    for record in episodes:
        match = _SHARD_PATTERN.match(Path(record["shard"]).name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "MANIFEST_FILENAME",
    "DatasetWriter",
    "file_sha256",
    "load_manifest",
    "manifest_with_hash",
]
