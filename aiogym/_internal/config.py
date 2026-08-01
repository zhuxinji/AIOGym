"""Shared config parsing helpers for public environment and benchmark APIs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .validation import nonnegative_int, positive_int, seed_sequence


def resolve_auto_events(
    auto_events: bool | None = None,
    *,
    default: bool | None = None,
) -> bool | None:
    """Validate and resolve the automatic-event flag."""

    if auto_events is not None and not isinstance(auto_events, bool):
        raise TypeError("auto_events must be a boolean")
    if auto_events is not None:
        return auto_events
    return default


def load_config(config: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    with Path(config).open() as stream:
        return json.load(stream)


def as_list(value) -> list:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def parse_seed_list(
    raw: str | None,
    seed: int,
    episodes: int,
    *,
    option: str = "--seed-list",
) -> list[int]:
    """Parse explicit seeds or generate one deterministic seed per episode."""

    if raw:
        return list(seed_sequence(option, parse_csv_ints(raw)))
    count = positive_int("episodes", episodes)
    start = nonnegative_int("seed", seed)
    return list(seed_sequence(option, range(start, start + count)))


def parse_csv_ints(raw: str, *, option: str = "value") -> tuple[int, ...]:
    """Parse comma-separated integer syntax without applying semantics."""

    if not isinstance(raw, str):
        raise TypeError(f"{option} must be a comma-separated string")
    parts = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not parts:
        raise ValueError(f"{option} must contain at least one integer")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{option} must contain only integers") from exc
