"""Shared config parsing helpers for public environment and benchmark APIs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


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
        seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
        if not seeds:
            raise ValueError(f"{option} must contain at least one integer seed")
        return seeds
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    return [seed + i for i in range(episodes)]
