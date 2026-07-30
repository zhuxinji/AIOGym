"""One-shot locked final-test protocol."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from aiogym.benchmarks import TrackSpec, evaluate_policy_on_track, load_track
from aiogym.evaluation.statistics import build_final_statistical_report


FINAL_TEST_LOCK_SCHEMA_VERSION = "aiogym.final_test_lock.v1"


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
        self.base_seeds = tuple(int(seed) for seed in base_seeds)
        if not self.base_seeds:
            raise ValueError("final test requires base seeds")
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
                    "report_hash": None,
                    "failure": None,
                }
            )

    def run(
        self,
        controllers: Mapping[str, Any],
        *,
        baseline: str | None = None,
        bootstrap_repetitions: int = 2000,
        evaluate_fn=evaluate_policy_on_track,
    ) -> dict[str, Any]:
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
            evaluations = {
                algorithm: evaluate_fn(
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
                    "status": "failed",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
            self._write(state)
            raise
        canonical = json.dumps(
            report,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        state.update(
            {
                "status": "complete",
                "completed_at": _utc_now(),
                "report_hash": hashlib.sha256(
                    canonical.encode("utf-8")
                ).hexdigest(),
                "failure": None,
            }
        )
        self._write(state)
        return {
            "evaluations": evaluations,
            "statistical_report": report,
            "lock": dict(state),
        }

    def state(self) -> dict[str, Any]:
        return self._read()

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
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _non_empty(name: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["FINAL_TEST_LOCK_SCHEMA_VERSION", "FinalTestLock"]
