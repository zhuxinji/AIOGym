"""JSON conversion and writing shared by benchmark orchestration modules."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JSON_WRITE_CLAIM_SCHEMA_VERSION = "aiogym.json_write_claim.v1"


def jsonable(value):
    if hasattr(value, "metadata") and callable(value.metadata):
        return jsonable(value.metadata())
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist") and callable(value.tolist):
        return jsonable(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        try:
            return jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(jsonable(data), stream, indent=2)
        stream.write("\n")


def write_json_artifact(
    path: str | Path,
    data: Any,
    *,
    overwrite: bool = False,
) -> Path:
    """Durably commit one JSON artifact without silent replacement."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    claim = target.with_name(f".{target.name}.write-claim")
    try:
        descriptor = os.open(
            claim,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"artifact write claim already exists: {claim}"
        ) from exc
    temporary: Path | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "schema_version": JSON_WRITE_CLAIM_SCHEMA_VERSION,
                    "pid": os.getpid(),
                    "target": str(target),
                    "claimed_at": datetime.now(timezone.utc).isoformat(),
                    "overwrite": bool(overwrite),
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(f"artifact already exists: {target}")
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(temporary_descriptor, "w", encoding="utf-8") as stream:
            json.dump(jsonable(data), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
        _fsync_directory(target.parent)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        claim.unlink(missing_ok=True)


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory sync; unsupported platforms safely degrade."""

    descriptor = None
    try:
        descriptor = os.open(directory, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)
