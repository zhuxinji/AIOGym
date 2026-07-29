"""Loading and discovery for built-in and custom benchmark tracks."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .schema import TrackSpec


BUILTIN_TRACK_DIR = Path(__file__).with_name("builtin")
DEFAULT_BENCHMARK_TRACK_ID = "quadruple-regulation-generalist-v1"


def list_tracks() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in BUILTIN_TRACK_DIR.glob("*.json")))


def load_track(
    source: str | Path | Mapping[str, Any],
    *,
    validate_policy_contract: bool = True,
) -> TrackSpec:
    if isinstance(source, Mapping):
        declaration = dict(source)
    else:
        path = Path(source)
        if isinstance(source, str) and "/" not in source and not path.suffix:
            path = BUILTIN_TRACK_DIR / f"{source}.json"
        if not path.is_file():
            if isinstance(source, str) and "/" not in source:
                available = ", ".join(list_tracks()) or "none"
                raise FileNotFoundError(
                    f"unknown Track ID {source!r}; "
                    f"available tracks: {available}"
                )
            raise FileNotFoundError(f"benchmark track not found: {source}")
        with path.open(encoding="utf-8") as stream:
            declaration = json.load(stream)
    track = TrackSpec(declaration)
    if validate_policy_contract:
        track.validate_policy_contract()
    return track


__all__ = [
    "BUILTIN_TRACK_DIR",
    "DEFAULT_BENCHMARK_TRACK_ID",
    "list_tracks",
    "load_track",
]
