"""Canonical serialization, hashing, and atomic artifact writes."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


def deep_freeze(value: Any) -> Any:
    """Defensively copy a strict-JSON value into recursively immutable containers."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    # Reuse the canonical serializer's scalar validation, including finite floats.
    jsonable(value)
    return value


def deep_thaw(value: Any) -> Any:
    """Return a fully independent mutable strict-JSON representation."""

    if isinstance(value, Mapping):
        return {key: deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [deep_thaw(item) for item in value]
    return value


def jsonable(value: Any) -> JSONValue:
    """Return a strict-JSON representation, rejecting non-finite numbers."""

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must not contain NaN or Infinity")
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            jsonable(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> Path:
    """Atomically write canonical JSON without silently replacing artifacts."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    return _atomic_write(target, canonical_json_bytes(value), overwrite=overwrite)


def write_text(path: str | Path, value: str, *, overwrite: bool = False) -> Path:
    if not isinstance(value, str):
        raise TypeError("text artifact value must be a string")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    return _atomic_write(target, value.encode("utf-8"), overwrite=overwrite)


def _atomic_write(target: Path, payload: bytes, *, overwrite: bool) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = [
    "JSONValue",
    "canonical_json_bytes",
    "deep_freeze",
    "deep_thaw",
    "file_sha256",
    "jsonable",
    "stable_hash",
    "write_json",
    "write_text",
]
