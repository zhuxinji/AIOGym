"""Loading and discovery for built-in and custom benchmark tracks."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .schema import TrackSpec


BUILTIN_TRACK_DIR = Path(__file__).with_name("builtin")
DEFAULT_BENCHMARK_TRACK_ID = "quadruple-regulation-generalist-v2"
TRACK_SELECTORS = {
    "quadruple": "quadruple-regulation-generalist-v2",
    "cascade": "cascade-regulation-generalist-v2",
    "cascade:economic": "cascade-economic-specialist-v1",
    "cascade:recovery": "cascade-recovery-diagnostic-v1",
    "cascade-recirculating": (
        "cascade-recirculating-regulation-generalist-v2"
    ),
    "cascade-recirculating:recovery": (
        "cascade-recirculating-recovery-diagnostic-v1"
    ),
}


def list_tracks() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in BUILTIN_TRACK_DIR.glob("*.json")))


def _validate_track_selectors(selectors: Mapping[str, str]) -> None:
    if not isinstance(selectors, Mapping):
        raise TypeError("Track selectors must be a mapping")
    canonical_ids = list_tracks()
    values = []
    for selector, track_id in selectors.items():
        if not isinstance(selector, str) or not selector:
            raise ValueError("Track selectors must be non-empty strings")
        if not isinstance(track_id, str) or track_id not in canonical_ids:
            raise ValueError(
                f"Track selector {selector!r} points to unknown Track "
                f"{track_id!r}"
            )
        values.append(track_id)
    if len(set(values)) != len(values):
        raise ValueError("each canonical Track may have only one selector")


def list_track_selectors() -> tuple[str, ...]:
    return tuple(TRACK_SELECTORS)


def resolve_track_selector(value: str) -> str:
    """Return one canonical built-in Track ID."""

    if not isinstance(value, str) or not value:
        raise ValueError("Track selector must be a non-empty string")
    if value in TRACK_SELECTORS:
        return TRACK_SELECTORS[value]
    if value in list_tracks():
        return value
    selectors = ", ".join(list_track_selectors())
    raise FileNotFoundError(
        f"unknown Track ID or selector {value!r}; available selectors: "
        f"{selectors}; use 'aiogym list tracks --ids' for canonical IDs"
    )


def preferred_track_selector(track_id: str) -> str | None:
    for selector, canonical_id in TRACK_SELECTORS.items():
        if canonical_id == track_id:
            return selector
    return None


def iter_track_catalog(*, include_all: bool = False) -> tuple[dict[str, Any], ...]:
    rows = []
    selected = {track_id: selector for selector, track_id in TRACK_SELECTORS.items()}
    track_ids = list_tracks() if include_all else tuple(selected)
    for track_id in track_ids:
        selector = selected.get(track_id)
        track = load_track(track_id, validate_policy_contract=False)
        rows.append(
            {
                "selector": selector or "-",
                "track_id": track.id,
                "scenario": track.scenario,
                "goal": track.goal,
                "policy_scope": track.policy_scope,
                "reward_spec_id": track.reward_spec_id,
                "description": str(
                    track.declaration.get("description", "")
                ),
            }
        )
    return tuple(rows)


def load_track(
    source: str | Path | Mapping[str, Any],
    *,
    validate_policy_contract: bool = True,
) -> TrackSpec:
    official = False
    if isinstance(source, Mapping):
        declaration = dict(source)
    else:
        path = Path(source)
        if path.is_file():
            pass
        elif isinstance(source, str):
            track_id = resolve_track_selector(source)
            path = BUILTIN_TRACK_DIR / f"{track_id}.json"
            official = True
        if not path.is_file():
            raise FileNotFoundError(f"benchmark track not found: {source}")
        try:
            official = (
                official
                or path.resolve().parent == BUILTIN_TRACK_DIR.resolve()
            )
        except OSError:
            pass
        with path.open(encoding="utf-8") as stream:
            declaration = json.load(stream)
    track = TrackSpec(declaration, official=official)
    if validate_policy_contract:
        track.validate_policy_contract()
    return track


_validate_track_selectors(TRACK_SELECTORS)


__all__ = [
    "BUILTIN_TRACK_DIR",
    "DEFAULT_BENCHMARK_TRACK_ID",
    "TRACK_SELECTORS",
    "iter_track_catalog",
    "list_track_selectors",
    "list_tracks",
    "load_track",
    "preferred_track_selector",
    "resolve_track_selector",
]
