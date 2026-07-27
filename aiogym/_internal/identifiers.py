"""Canonical public resource IDs and internal storage mappings."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_SCENARIO_CANONICAL_BY_INTERNAL = {
    "cascade_recirculating": "cascade-recirculating",
}
_SCENARIO_INTERNAL_BY_ALIAS = {
    canonical: internal
    for internal, canonical in _SCENARIO_CANONICAL_BY_INTERNAL.items()
}


def canonical_scenario_id(value: str) -> str:
    """Return the public canonical ID for a scenario registration key."""

    return _SCENARIO_CANONICAL_BY_INTERNAL.get(value, value)


def internal_scenario_id(value: str) -> str:
    """Map a canonical public ID to its internal registration key."""

    return _SCENARIO_INTERNAL_BY_ALIAS.get(value, value)


def canonical_scenario_ids(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(canonical_scenario_id(value) for value in values))


def require_canonical_scenario_id(value: str) -> str:
    """Reject internal storage names at public API boundaries."""

    if value in _SCENARIO_CANONICAL_BY_INTERNAL:
        canonical = _SCENARIO_CANONICAL_BY_INTERNAL[value]
        raise ValueError(f"scenario ID {value!r} is not canonical; use {canonical!r}")
    return value


def scenario_catalog_text(values: Sequence[str]) -> str:
    """Render canonical IDs and their accepted aliases for error messages."""

    rows = []
    for internal in sorted(values, key=canonical_scenario_id):
        canonical = canonical_scenario_id(internal)
        rows.append(canonical)
    return ", ".join(rows)


def canonical_task_id(value: str) -> str:
    """Canonicalize the scenario prefix of a ``scenario/task`` ID."""

    parts = value.split("/", 1)
    if len(parts) != 2:
        return value
    return f"{canonical_scenario_id(internal_scenario_id(parts[0]))}/{parts[1]}"


def suite_catalog_text(canonical_ids: Sequence[str]) -> str:
    return ", ".join(sorted(canonical_ids))


def canonicalize_artifact_ids(value: Any, *, field: str | None = None) -> Any:
    """Canonicalize scenario-bearing fields in newly written artifacts."""

    if isinstance(value, Mapping):
        return {
            key: canonicalize_artifact_ids(item, field=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [canonicalize_artifact_ids(item, field=field) for item in value]
    if isinstance(value, tuple):
        return [canonicalize_artifact_ids(item, field=field) for item in value]
    if isinstance(value, str):
        if field == "scenario":
            return canonical_scenario_id(internal_scenario_id(value))
        if field == "scenarios":
            return canonical_scenario_id(internal_scenario_id(value))
        if field == "task" and "/" in value:
            return canonical_task_id(value)
    return value


__all__ = [
    "canonical_scenario_id",
    "canonical_scenario_ids",
    "canonical_task_id",
    "canonicalize_artifact_ids",
    "internal_scenario_id",
    "require_canonical_scenario_id",
    "scenario_catalog_text",
    "suite_catalog_text",
]
