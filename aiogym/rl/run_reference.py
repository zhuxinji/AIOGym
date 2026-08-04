"""Resolve self-describing training runs for evaluation entry points."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from aiogym.benchmarks.tracks.registry import load_track
from aiogym.controllers.checkpoints import (
    SUPPORTED_POLICY_ALGORITHMS,
    checkpoint_sha256,
)

from .runner import RUN_RESULT_SCHEMA_VERSION


@dataclass(frozen=True)
class RunReference:
    manifest_path: Path
    track_id: str
    track_hash: str
    algorithm_id: str
    policy_path: Path
    policy_sha256: str
    output_dir: Path
    payload: Mapping[str, Any]


def load_run_reference(source: str | Path) -> RunReference:
    manifest_path = _resolve_manifest_path(Path(source))
    with manifest_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("run-result manifest must contain a JSON object")
    if payload.get("schema_version") != RUN_RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported run-result schema: {payload.get('schema_version')!r}"
        )
    track_id = payload.get("track_id")
    if not isinstance(track_id, str) or not track_id:
        raise ValueError("run-result requires canonical track_id")
    track = load_track(track_id)
    if track_id != track.id:
        raise ValueError("run-result track_id must be canonical")
    track_hash = payload.get("track_hash")
    if track_hash != track.track_hash:
        raise ValueError("run-result Track hash does not match current Track")
    algorithm_id = str(payload.get("algorithm_id", "")).lower()
    if algorithm_id not in SUPPORTED_POLICY_ALGORITHMS:
        raise ValueError(
            "run-result algorithm_id must be one of: "
            + ", ".join(SUPPORTED_POLICY_ALGORITHMS)
        )
    digest = payload.get("policy_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("run-result policy_sha256 must be a lowercase SHA-256")
    policy_value = payload.get("policy_path")
    if not isinstance(policy_value, str) or not policy_value:
        raise ValueError("run-result requires policy_path")
    policy_path = _resolve_policy_path(
        policy_value,
        manifest_path=manifest_path,
        expected_digest=digest,
    )
    output_value = payload.get("output_dir")
    if not isinstance(output_value, str) or not output_value:
        raise ValueError("run-result requires output_dir")
    return RunReference(
        manifest_path=manifest_path,
        track_id=track.id,
        track_hash=track.track_hash,
        algorithm_id=algorithm_id,
        policy_path=policy_path,
        policy_sha256=digest,
        output_dir=Path(output_value),
        payload=MappingProxyType(dict(payload)),
    )


def _resolve_manifest_path(source: Path) -> Path:
    if source.is_dir():
        candidates = tuple(sorted(source.glob("*.run-result.json")))
        if len(candidates) != 1:
            names = ", ".join(path.name for path in candidates) or "none"
            raise ValueError(
                f"run directory must contain exactly one *.run-result.json; "
                f"candidates: {names}"
            )
        return candidates[0]
    if source.is_file():
        return source
    stem_candidate = Path(f"{source}.run-result.json")
    if stem_candidate.is_file():
        return stem_candidate
    raise FileNotFoundError(f"run-result manifest not found: {source}")


def _resolve_policy_path(
    raw_value: str,
    *,
    manifest_path: Path,
    expected_digest: str,
) -> Path:
    raw = Path(raw_value)
    candidates = (raw, manifest_path.parent / raw, manifest_path.parent / raw.name)
    existing = []
    seen = set()
    for candidate in candidates:
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        existing.append(resolved)
    if not existing:
        raise FileNotFoundError(
            f"run-result checkpoint does not exist: {raw_value}"
        )
    matching = [
        path for path in existing if checkpoint_sha256(path) == expected_digest
    ]
    if not matching:
        raise ValueError("run-result checkpoint SHA-256 mismatch")
    if len(matching) != 1:
        raise ValueError(
            "run-result checkpoint path is ambiguous: "
            + ", ".join(map(str, matching))
        )
    return matching[0]


__all__ = ["RunReference", "load_run_reference"]
