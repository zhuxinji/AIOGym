"""Explicit one-shot final-test command."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from aiogym.benchmarks.tracks.registry import load_track
from aiogym.controllers.checkpoints import (
    SUPPORTED_POLICY_ALGORITHMS,
    learned_policy_spec_for_track,
    load_policy_checkpoint,
)
from aiogym.rl.final_test import FinalTestLock


FINAL_TEST_CONFIG_SCHEMA_VERSION = "aiogym.final_test.v1"
_FIELDS = frozenset(
    {
        "schema_version",
        "track_id",
        "lock_path",
        "base_seeds",
        "checkpoints",
        "output",
        "baseline",
        "bootstrap_repetitions",
    }
)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Consume a locked benchmark test split exactly once.",
    )
    parser.add_argument("--config", required=True)
    return parser


def main(argv=None, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    declaration = _load_config(args.config)
    track = load_track(declaration["track_id"])
    controllers = {}
    checkpoint_ids = {}
    for name, checkpoint in sorted(
        declaration["checkpoints"].items()
    ):
        policy_spec = learned_policy_spec_for_track(
            checkpoint["path"],
            checkpoint["algorithm_id"],
            checkpoint["sha256"],
            track,
        )
        controllers[name] = load_policy_checkpoint(policy_spec)
        checkpoint_ids[name] = policy_spec.sha256
    config_hash = _stable_hash(declaration)
    lock = FinalTestLock(
        declaration["lock_path"],
        track=track,
        config_hash=config_hash,
        checkpoint_ids=checkpoint_ids,
        base_seeds=declaration["base_seeds"],
    )
    output = Path(declaration["output"])
    if output.exists():
        raise FileExistsError(f"final-test artifact exists: {output}")
    result = lock.run(
        controllers,
        baseline=declaration.get("baseline"),
        bootstrap_repetitions=int(
            declaration.get("bootstrap_repetitions", 2000)
        ),
    )
    artifact = {
        "schema_version": "aiogym.final_test_artifact.v1",
        "track_id": track.id,
        "track_hash": track.track_hash,
        "config_hash": config_hash,
        "checkpoint_sha256": checkpoint_ids,
        **result,
    }
    _atomic_json(output, artifact)
    print(
        json.dumps(
            {
                "track_id": track.id,
                "output": str(output),
                "lock_status": result["lock"]["status"],
                "report_hash": result["lock"]["report_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


def _load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise TypeError("final-test config must be a mapping")
    unknown = set(data) - _FIELDS
    if unknown:
        raise ValueError(
            "unknown final-test config fields: "
            + ", ".join(sorted(unknown))
        )
    required = _FIELDS - {"baseline", "bootstrap_repetitions"}
    missing = required - set(data)
    if missing:
        raise ValueError(
            "final-test config is missing fields: "
            + ", ".join(sorted(missing))
        )
    if data["schema_version"] != FINAL_TEST_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported final-test config schema")
    for field in ("track_id", "lock_path", "output"):
        if not isinstance(data[field], str) or not data[field]:
            raise ValueError(f"{field} must be a non-empty string")
    seeds = data["base_seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("base_seeds must be a non-empty list")
    if any(
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        for seed in seeds
    ):
        raise ValueError("base_seeds must contain non-negative integers")
    checkpoints = data["checkpoints"]
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise ValueError("checkpoints must be a non-empty mapping")
    for name, row in checkpoints.items():
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint names must be non-empty")
        if not isinstance(row, Mapping):
            raise TypeError(f"checkpoint {name!r} must be a mapping")
        if set(row) != {"path", "algorithm_id", "sha256"}:
            raise ValueError(
                f"checkpoint {name!r} requires path, algorithm_id, sha256"
            )
        path = Path(str(row["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        algorithm = str(row["algorithm_id"]).lower()
        if algorithm not in SUPPORTED_POLICY_ALGORITHMS:
            raise ValueError(
                f"checkpoint {name!r} has unsupported algorithm_id"
            )
        digest = str(row["sha256"])
        if len(digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in digest
        ):
            raise ValueError(
                f"checkpoint {name!r} sha256 must be a lowercase digest"
            )
    repetitions = int(data.get("bootstrap_repetitions", 2000))
    if repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    return dict(data)


def _stable_hash(value) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


__all__ = [
    "FINAL_TEST_CONFIG_SCHEMA_VERSION",
    "build_parser",
    "main",
]
