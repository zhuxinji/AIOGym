"""One-shot locked final-test protocol."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from aiogym.benchmarks.evaluation import _evaluate_policy_on_track_split
from aiogym.benchmarks.tracks.registry import load_track
from aiogym.benchmarks.tracks.schema import TrackSpec
from aiogym.evaluation.statistics import build_final_statistical_report
from aiogym.evaluation.provenance import code_commit, package_version
from aiogym._internal.serialization import file_sha256, stable_json_hash
from aiogym._internal.validation import seed_sequence


FINAL_TEST_LOCK_SCHEMA_VERSION = "aiogym.final_test_lock.v2"


class FinalTestLock:
    """Consume a predeclared test evaluation exactly once."""

    def __init__(
        self,
        path: str | Path,
        *,
        track,
        config_hash: str,
        checkpoint_ids: Mapping[str, str],
        base_seeds: Sequence[int],
    ) -> None:
        self.path = Path(path)
        self.track = (
            track if isinstance(track, TrackSpec) else load_track(track)
        )
        self.config_hash = _non_empty("config_hash", config_hash)
        self.checkpoint_ids = {
            str(name): _non_empty("checkpoint_id", checkpoint_id)
            for name, checkpoint_id in sorted(checkpoint_ids.items())
        }
        if not self.checkpoint_ids:
            raise ValueError("final test requires checkpoints")
        self.base_seeds = seed_sequence("final test base seeds", base_seeds)
        if self.path.exists():
            state = self._read()
            self._validate_identity(state)
        else:
            self._write(
                {
                    **self._identity(),
                    "status": "locked",
                    "attempted_at": None,
                    "completed_at": None,
                    "artifact_path": None,
                    "artifact_sha256": None,
                    "report_hash": None,
                    "package_version": package_version(),
                    "code_commit": code_commit(),
                    "failure_stage": None,
                    "failure": None,
                }
            )

    def run_and_commit(
        self,
        controllers: Mapping[str, Any],
        *,
        artifact_path: str | Path,
        artifact_builder,
        baseline: str | None = None,
        bootstrap_repetitions: int = 2000,
        _evaluate_test_fn=None,
        _artifact_commit_fn=None,
    ) -> dict[str, Any]:
        """Evaluate and atomically publish before completing the lock."""

        output = Path(artifact_path)
        if output.exists():
            raise FileExistsError(f"final-test artifact exists: {output}")
        if not callable(artifact_builder):
            raise TypeError("artifact_builder must be callable")
        state = self._read()
        self._validate_identity(state)
        if state["status"] != "locked":
            raise RuntimeError(
                f"final test lock is already consumed ({state['status']})"
            )
        if set(controllers) != set(self.checkpoint_ids):
            raise ValueError(
                "final-test controllers must exactly match locked checkpoints"
            )
        self._claim_attempt()
        state = self._read()
        self._validate_identity(state)
        if state["status"] != "locked":
            raise RuntimeError(
                f"final test lock is already consumed ({state['status']})"
            )
        state.update(
            {
                "status": "running",
                "attempted_at": _utc_now(),
            }
        )
        self._write(state)
        try:
            evaluate_test = (
                _evaluate_policy_on_track_split
                if _evaluate_test_fn is None
                else _evaluate_test_fn
            )
            evaluations = {
                algorithm: evaluate_test(
                    controller,
                    self.track,
                    split="test",
                    base_seeds=self.base_seeds,
                    include_episodes=True,
                )
                for algorithm, controller in controllers.items()
            }
            report = build_final_statistical_report(
                evaluations,
                baseline=baseline,
                bootstrap_repetitions=bootstrap_repetitions,
            )
            report.update(
                {
                    "locked_config_hash": self.config_hash,
                    "checkpoint_ids": dict(self.checkpoint_ids),
                    "final_test_attempted_at": state["attempted_at"],
                }
            )
        except BaseException as exc:
            state.update(
                {
                    "status": "failed_evaluation",
                    "failure_stage": "evaluation",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
            self._write(state)
            raise
        state.update({
            "status": "artifact_pending",
            "report_hash": stable_json_hash(report),
            "artifact_path": str(output.resolve()),
            "failure_stage": None,
            "failure": None,
        })
        self._write(state)
        evaluation_result = {
            "evaluations": evaluations,
            "statistical_report": report,
        }
        try:
            artifact = artifact_builder(evaluation_result, dict(state))
            if not isinstance(artifact, Mapping):
                raise TypeError("artifact_builder must return a mapping")
            commit = (
                self._commit_artifact
                if _artifact_commit_fn is None
                else _artifact_commit_fn
            )
            artifact_sha256 = commit(output, dict(artifact))
        except BaseException as exc:
            state.update(
                {
                    "status": "failed_artifact_commit",
                    "failure_stage": "artifact_commit",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
            self._write(state)
            raise
        state.update(
            {
                "status": "complete",
                "completed_at": _utc_now(),
                "artifact_sha256": artifact_sha256,
                "failure_stage": None,
                "failure": None,
            }
        )
        self._write(state)
        self._validate_complete_artifact(state)
        return {
            **evaluation_result,
            "lock": dict(state),
        }

    def state(self) -> dict[str, Any]:
        state = self._read()
        self._validate_complete_artifact(state)
        return state

    def _identity(self) -> dict[str, Any]:
        return {
            "schema_version": FINAL_TEST_LOCK_SCHEMA_VERSION,
            "track_id": self.track.id,
            "track_hash": self.track.track_hash,
            "test_seed_namespace": self.track.seed_namespace("test"),
            "config_hash": self.config_hash,
            "checkpoint_ids": dict(self.checkpoint_ids),
            "base_seeds": list(self.base_seeds),
        }

    def _validate_identity(self, state) -> None:
        expected = self._identity()
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(
                    f"final test lock identity mismatch for {name}"
                )

    def _read(self) -> dict[str, Any]:
        with self.path.open(encoding="utf-8") as stream:
            state = json.load(stream)
        if state.get("schema_version") != FINAL_TEST_LOCK_SCHEMA_VERSION:
            raise ValueError("unsupported final test lock schema")
        return state

    def _validate_complete_artifact(self, state) -> None:
        if state.get("status") != "complete":
            return
        artifact_path = Path(str(state.get("artifact_path", "")))
        if not artifact_path.is_file():
            raise ValueError("complete final test artifact is missing")
        actual = file_sha256(artifact_path)
        if actual != state.get("artifact_sha256"):
            raise ValueError("complete final test artifact hash mismatch")

    def _claim_attempt(self) -> None:
        claim_path = self.path.with_name(f".{self.path.name}.claimed")
        try:
            descriptor = os.open(
                claim_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            status = self._read().get("status", "claimed")
            raise RuntimeError(
                f"final test lock is already consumed ({status})"
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    **self._identity(),
                    "claimed_at": _utc_now(),
                },
                stream,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _write(self, state) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(state, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @staticmethod
    def _commit_artifact(path: Path, artifact: Mapping[str, Any]) -> str:
        if path.exists():
            raise FileExistsError(f"final-test artifact exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                dict(artifact),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists():
                raise FileExistsError(
                    f"final-test artifact exists: {path}"
                )
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return hashlib.sha256(payload).hexdigest()


def _non_empty(name: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["FINAL_TEST_LOCK_SCHEMA_VERSION", "FinalTestLock"]
