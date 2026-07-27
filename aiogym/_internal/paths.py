"""Shared paths for generated AIO-Gym run artifacts."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


RUNS_DIR_ENV = "AIOGYM_RUNS_DIR"


def runs_dir() -> Path:
    """Return the configured root for generated runs.

    Relative values remain relative to the caller's working directory. An empty
    environment variable is treated like an unset value.
    """

    configured = os.environ.get(RUNS_DIR_ENV)
    return Path(configured).expanduser() if configured else Path("runs")


def run_path(*parts: str) -> Path:
    """Build a generated-output path below :func:`runs_dir`."""

    return runs_dir().joinpath(*parts)


def artifact_run_id(now: datetime | None = None) -> str:
    """Return a sortable UTC identifier for a non-overwriting artifact run."""

    stamp = now or datetime.now(timezone.utc)
    return stamp.strftime("%Y%m%dT%H%M%S%fZ")


def timestamped_artifact_path(
    base: str | Path, run_id: str | None = None
) -> Path:
    """Return a timestamped sibling of an artifact directory."""

    path = Path(base)
    return path.with_name(f"{path.name}_{run_id or artifact_run_id()}")
