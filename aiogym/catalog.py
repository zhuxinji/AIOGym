"""Public discovery helpers for registered and bundled AIO-Gym resources."""
from __future__ import annotations

from ._internal.identifiers import canonical_scenario_ids
from .controllers.registry import _controller_ids
from .models.cases import list_cases as _list_cases
from .models.registry import MODELS


def list_scenarios() -> tuple[str, ...]:
    """Return canonical IDs for all currently registered process scenarios."""

    return canonical_scenario_ids(tuple(MODELS))


def list_cases(scenario: str | None = None) -> tuple[str, ...]:
    """Return canonical ``scenario/name`` IDs for bundled Case v2 specs."""

    return _list_cases(scenario)


def list_controllers() -> tuple[str, ...]:
    """Return canonical IDs for all currently registered controllers."""

    return _controller_ids()


__all__ = [
    "list_cases",
    "list_controllers",
    "list_scenarios",
]
