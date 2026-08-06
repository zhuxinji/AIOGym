"""One-release task aliases for the AIO-Gym 0.3 three-tank merge."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
import warnings

from .specs import OperatingCondition, PlantConfig


LEGACY_TASK_ALIASES = {
    "cascade/regulation": {
        "task": "three_tank/regulation",
        "plant": "open-cascade-v1",
        "condition": "continuous-benchmark",
    },
    "cascade/economic": {
        "task": "three_tank/economic",
        "plant": "open-cascade-v1",
        "condition": "continuous-benchmark",
    },
    "cascade_recirculating/regulation": {
        "task": "three_tank/regulation",
        "plant": "recirculating-h1-v1",
        "condition": "commissioning",
    },
}


def resolve_legacy_request(task, plant=None, condition=None):
    alias = LEGACY_TASK_ALIASES.get(task)
    if alias is None:
        return task, plant, condition
    _check_explicit("plant", plant, alias["plant"])
    _check_explicit("condition", condition, alias["condition"])
    warnings.warn(
        f"task {task!r} is deprecated; use {alias['task']!r} with "
        f"plant={alias['plant']!r} and condition={alias['condition']!r}",
        DeprecationWarning,
        stacklevel=3,
    )
    return (
        alias["task"],
        alias["plant"] if plant is None else plant,
        alias["condition"] if condition is None else condition,
    )


def resolve_condition_alias(condition=None, preset=None):
    if condition is not None and preset is not None:
        raise TypeError("condition and deprecated preset cannot both be provided")
    if preset is not None:
        warnings.warn(
            "preset= is deprecated; use condition=",
            DeprecationWarning,
            stacklevel=3,
        )
        return preset
    return condition


def _check_explicit(kind, value, expected):
    if value is None:
        return
    identifier = _identifier(value)
    if identifier != expected:
        raise ValueError(
            f"legacy alias requires {kind} {expected!r}; got {identifier!r}"
        )


def _identifier(value):
    if isinstance(value, (PlantConfig, OperatingCondition)):
        return value.id
    if isinstance(value, Mapping):
        return str(value.get("id", ""))
    path = Path(value)
    if path.is_file():
        return str(json.loads(path.read_text(encoding="utf-8")).get("id", ""))
    return str(value)


__all__ = [
    "LEGACY_TASK_ALIASES",
    "resolve_condition_alias",
    "resolve_legacy_request",
]
