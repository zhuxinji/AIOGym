"""Atomic ownership records for one named training run."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from aiogym._internal.serialization import file_sha256, stable_json_hash


RUN_CLAIM_SCHEMA_VERSION = "aiogym.run_claim.v1"
SEED_SWEEP_CLAIM_SCHEMA_VERSION = "aiogym.seed_sweep_claim.v1"


@dataclass(frozen=True)
class RunClaimIdentity:
    run_name: str
    config_hash: str
    track_hash: str
    algorithm_id: str
    training_seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "config_hash": self.config_hash,
            "track_hash": self.track_hash,
            "algorithm_id": self.algorithm_id,
            "training_seed": self.training_seed,
        }


@dataclass(frozen=True)
class SeedSweepIdentity:
    base_name: str
    config_hash: str
    track_id: str
    track_hash: str
    algorithm_id: str
    seeds: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_name": self.base_name,
            "config_hash": self.config_hash,
            "track_id": self.track_id,
            "track_hash": self.track_hash,
            "algorithm_id": self.algorithm_id,
            "seeds": list(self.seeds),
        }

    @property
    def sweep_id(self) -> str:
        return _mapping_hash(self.as_dict())


class RunClaim:
    def __init__(self, path: Path, identity: RunClaimIdentity) -> None:
        self.path = path
        self.identity = identity

    @classmethod
    def acquire(
        cls,
        path: str | Path,
        identity: RunClaimIdentity,
        *,
        occupied_paths: Sequence[Path] = (),
        overwrite: bool = False,
        resume: bool = False,
        previous_artifact: Path | None = None,
    ) -> "RunClaim":
        if overwrite and resume:
            raise ValueError("overwrite and resume are mutually exclusive")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        claim = cls(target, identity)
        occupied = tuple(path for path in occupied_paths if path.exists())
        if not overwrite and not resume and occupied:
            raise FileExistsError(
                "run output already exists: " + ", ".join(map(str, occupied))
            )
        if target.exists():
            if not overwrite and not resume:
                raise FileExistsError(f"run claim already exists: {target}")
            claim._replace_existing(
                overwrite=overwrite,
                resume=resume,
                previous_artifact=previous_artifact,
            )
        else:
            state = claim._new_state(mode="resume" if resume else "fresh")
            _exclusive_json(target, state)
        return claim

    def complete(self, *, run_result_hash: str, artifact_hash: str | None) -> None:
        state = self._read_current()
        state.update(
            {
                "status": "complete",
                "completed_at": _utc_now(),
                "run_result_hash": run_result_hash,
                "artifact_hash": artifact_hash,
                "failure": None,
            }
        )
        _atomic_json(self.path, state)

    def fail(self, exc: BaseException) -> None:
        state = self._read_current()
        state.update(
            {
                "status": "failed",
                "completed_at": _utc_now(),
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_json(self.path, state)

    def state(self) -> dict[str, Any]:
        return self._read_current(require_running=False)

    def _replace_existing(
        self,
        *,
        overwrite: bool,
        resume: bool,
        previous_artifact: Path | None,
    ) -> None:
        guard = self.path.with_name(f".{self.path.name}.transition")
        descriptor = os.open(guard, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            previous = _read_json(self.path)
            if previous.get("status") == "running":
                raise RuntimeError("cannot replace a running run claim")
            _validate_stable_identity(previous, self.identity)
            if resume and previous.get("status") == "complete":
                raise RuntimeError("cannot resume a complete run")
            audit = list(previous.get("overwrite_audit") or ())
            if overwrite:
                audit.append(
                    {
                        "at": _utc_now(),
                        "previous_claim_hash": _mapping_hash(previous),
                        "previous_artifact_hash": (
                            _file_hash(previous_artifact)
                            if previous_artifact is not None
                            and previous_artifact.is_file()
                            else None
                        ),
                    }
                )
            state = self._new_state(
                mode="overwrite" if overwrite else "resume",
            )
            state["overwrite_audit"] = audit
            if resume:
                state["resumed_claim_hash"] = _mapping_hash(previous)
            _atomic_json(self.path, state)
        finally:
            guard.unlink(missing_ok=True)

    def _new_state(self, *, mode: str) -> dict[str, Any]:
        return {
            "schema_version": RUN_CLAIM_SCHEMA_VERSION,
            **self.identity.as_dict(),
            "status": "running",
            "mode": mode,
            "claimed_at": _utc_now(),
            "completed_at": None,
            "run_result_hash": None,
            "artifact_hash": None,
            "failure": None,
            "overwrite_audit": [],
        }

    def _read_current(self, *, require_running: bool = True) -> dict[str, Any]:
        state = _read_json(self.path)
        if state.get("schema_version") != RUN_CLAIM_SCHEMA_VERSION:
            raise ValueError("unsupported run claim schema")
        for name, value in self.identity.as_dict().items():
            if state.get(name) != value:
                raise ValueError(f"run claim identity mismatch for {name}")
        if require_running and state.get("status") != "running":
            raise RuntimeError("run claim is not running")
        return state


class SeedSweepClaim:
    def __init__(self, path: Path, identity: SeedSweepIdentity) -> None:
        self.path = path
        self.identity = identity

    @classmethod
    def acquire(
        cls,
        path: str | Path,
        identity: SeedSweepIdentity,
        *,
        summary_path: str | Path,
        overwrite: bool = False,
    ) -> "SeedSweepClaim":
        target = Path(path)
        summary = Path(summary_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        claim = cls(target, identity)
        if target.exists():
            if not overwrite:
                raise FileExistsError(
                    f"seed sweep claim already exists: {target}"
                )
            claim._replace_existing(summary)
            return claim
        if summary.exists():
            if not overwrite:
                raise FileExistsError(
                    f"seed sweep summary already exists: {summary}"
                )
            existing_summary = _read_json(summary)
            if existing_summary.get("sweep_id") != identity.sweep_id:
                raise ValueError("seed sweep summary identity mismatch")
        _exclusive_json(target, claim._new_state())
        return claim

    def record_child(self, *, seed: int, result_hash: str) -> None:
        state = self._read_current()
        completed = list(state.get("completed_seeds") or ())
        if seed in completed:
            raise ValueError(f"seed sweep child already recorded: {seed}")
        completed.append(int(seed))
        results = list(state.get("child_results") or ())
        results.append({"seed": int(seed), "result_hash": str(result_hash)})
        state.update({"completed_seeds": completed, "child_results": results})
        _atomic_json(self.path, state)

    def complete(self, *, summary_hash: str) -> None:
        state = self._read_current()
        state.update(
            {
                "status": "complete",
                "completed_at": _utc_now(),
                "summary_hash": str(summary_hash),
                "failure": None,
            }
        )
        _atomic_json(self.path, state)

    def fail(self, exc: BaseException) -> None:
        state = self._read_current()
        state.update(
            {
                "status": "failed",
                "completed_at": _utc_now(),
                "failure": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        _atomic_json(self.path, state)

    def state(self) -> dict[str, Any]:
        return self._read_current(require_running=False)

    def _replace_existing(self, summary_path: Path) -> None:
        guard = self.path.with_name(f".{self.path.name}.transition")
        descriptor = os.open(
            guard,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        os.close(descriptor)
        try:
            previous = _read_json(self.path)
            if previous.get("status") == "running":
                raise RuntimeError(
                    "cannot replace a running seed sweep claim"
                )
            self._validate_identity(previous)
            if summary_path.exists():
                summary = _read_json(summary_path)
                if summary.get("sweep_id") != self.identity.sweep_id:
                    raise ValueError("seed sweep summary identity mismatch")
            state = self._new_state()
            state["overwrite_audit"] = [
                *list(previous.get("overwrite_audit") or ()),
                {
                    "at": _utc_now(),
                    "previous_claim_hash": _mapping_hash(previous),
                    "previous_summary_hash": (
                        _file_hash(summary_path)
                        if summary_path.is_file()
                        else None
                    ),
                },
            ]
            _atomic_json(self.path, state)
        finally:
            guard.unlink(missing_ok=True)

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema_version": SEED_SWEEP_CLAIM_SCHEMA_VERSION,
            "sweep_id": self.identity.sweep_id,
            **self.identity.as_dict(),
            "status": "running",
            "claimed_at": _utc_now(),
            "completed_at": None,
            "completed_seeds": [],
            "child_results": [],
            "summary_hash": None,
            "failure": None,
            "overwrite_audit": [],
        }

    def _read_current(self, *, require_running: bool = True) -> dict[str, Any]:
        state = _read_json(self.path)
        if state.get("schema_version") != SEED_SWEEP_CLAIM_SCHEMA_VERSION:
            raise ValueError("unsupported seed sweep claim schema")
        self._validate_identity(state)
        if require_running and state.get("status") != "running":
            raise RuntimeError("seed sweep claim is not running")
        return state

    def _validate_identity(self, state: Mapping[str, Any]) -> None:
        expected = {
            "sweep_id": self.identity.sweep_id,
            **self.identity.as_dict(),
        }
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(
                    f"seed sweep claim identity mismatch for {name}"
                )


def _validate_stable_identity(state: Mapping[str, Any], identity) -> None:
    for name in ("run_name", "track_hash", "algorithm_id", "training_seed"):
        if state.get(name) != getattr(identity, name):
            raise ValueError(f"run claim identity mismatch for {name}")


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("run claim must contain a JSON object")
    return value


def _mapping_hash(value: Mapping[str, Any]) -> str:
    return stable_json_hash(dict(value))


def _file_hash(path: Path) -> str:
    return file_sha256(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "RUN_CLAIM_SCHEMA_VERSION",
    "SEED_SWEEP_CLAIM_SCHEMA_VERSION",
    "RunClaim",
    "RunClaimIdentity",
    "SeedSweepClaim",
    "SeedSweepIdentity",
]
