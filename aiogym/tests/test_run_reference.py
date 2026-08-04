"""Tests for self-describing training run references."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aiogym.benchmarks.tracks.registry import load_track
from aiogym.rl.run_reference import load_run_reference
from aiogym.rl.runner import RUN_RESULT_SCHEMA_VERSION


TRACK_ID = "quadruple-regulation-generalist-v1"


def _write_manifest(
    directory: Path,
    *,
    policy_value: str | None = None,
    policy_bytes: bytes = b"policy",
    overrides: dict | None = None,
) -> tuple[Path, Path, dict]:
    directory.mkdir(parents=True, exist_ok=True)
    policy = directory / "policy.pt"
    policy.write_bytes(policy_bytes)
    track = load_track(TRACK_ID)
    payload = {
        "schema_version": RUN_RESULT_SCHEMA_VERSION,
        "track_id": track.id,
        "track_hash": track.track_hash,
        "algorithm_id": "bc",
        "policy_path": policy_value or str(policy),
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "output_dir": str(directory),
    }
    payload.update(overrides or {})
    manifest = directory / "training.run-result.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, policy, payload


@pytest.mark.parametrize("source_kind", ["file", "stem", "directory"])
def test_run_reference_accepts_all_three_source_forms(tmp_path, source_kind):
    manifest, policy, _ = _write_manifest(tmp_path / "run")
    source = {
        "file": manifest,
        "stem": Path(str(manifest).removesuffix(".run-result.json")),
        "directory": manifest.parent,
    }[source_kind]

    reference = load_run_reference(source)

    assert reference.manifest_path == manifest
    assert reference.track_id == TRACK_ID
    assert reference.algorithm_id == "bc"
    assert reference.policy_path == policy.resolve()


@pytest.mark.parametrize("manifest_count", [0, 2])
def test_run_directory_requires_exactly_one_manifest(tmp_path, manifest_count):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for index in range(manifest_count):
        (run_dir / f"{index}.run-result.json").write_text("{}")

    with pytest.raises(ValueError, match="exactly one"):
        load_run_reference(run_dir)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": "aiogym.run_result.v0"}, "schema"),
        ({"track_id": "quadruple"}, "canonical"),
        ({"track_hash": "0" * 64}, "Track hash"),
        ({"algorithm_id": "unknown"}, "algorithm_id"),
        ({"policy_sha256": "ABC"}, "lowercase SHA-256"),
    ],
)
def test_run_reference_rejects_invalid_identity(tmp_path, overrides, message):
    manifest, _, _ = _write_manifest(
        tmp_path / "run",
        overrides=overrides,
    )

    with pytest.raises((KeyError, ValueError), match=message):
        load_run_reference(manifest)


def test_run_reference_rejects_missing_or_changed_checkpoint(tmp_path):
    manifest, policy, _ = _write_manifest(tmp_path / "run")
    policy.unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        load_run_reference(manifest)

    policy.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_run_reference(manifest)


@pytest.mark.parametrize("resolution", ["cwd", "manifest", "basename"])
def test_relative_checkpoint_resolution_is_compatibility_aware(
    tmp_path,
    monkeypatch,
    resolution,
):
    manifest_dir = tmp_path / "manifest"
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    if resolution == "cwd":
        manifest, policy, payload = _write_manifest(
            manifest_dir,
            policy_value="policy.pt",
        )
        policy.unlink()
        policy = working_dir / "policy.pt"
        policy.write_bytes(b"policy")
    elif resolution == "manifest":
        manifest, original, payload = _write_manifest(
            manifest_dir,
            policy_value="nested/policy.pt",
        )
        original.unlink()
        policy = manifest_dir / "nested" / "policy.pt"
        policy.parent.mkdir()
        policy.write_bytes(b"policy")
    else:
        manifest, policy, payload = _write_manifest(
            manifest_dir,
            policy_value="old/location/policy.pt",
        )
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.chdir(working_dir)

    assert load_run_reference(manifest).policy_path == policy.resolve()


def test_relative_checkpoint_resolution_rejects_ambiguity(
    tmp_path,
    monkeypatch,
):
    manifest_dir = tmp_path / "manifest"
    working_dir = tmp_path / "working"
    working_dir.mkdir()
    manifest, _, _ = _write_manifest(
        manifest_dir,
        policy_value="policy.pt",
    )
    (working_dir / "policy.pt").write_bytes(b"policy")
    monkeypatch.chdir(working_dir)

    with pytest.raises(ValueError, match="ambiguous"):
        load_run_reference(manifest)
