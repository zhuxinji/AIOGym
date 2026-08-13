"""Built-in Scenario registrations."""
from __future__ import annotations

from aiogym.core.registry import list_scenarios, register_scenario

from .quadruple import SCENARIO as QUADRUPLE
from .three_tank import SCENARIO as THREE_TANK


BUILTIN_SCENARIOS = (
    QUADRUPLE,
    THREE_TANK,
)


def register_builtin_scenarios() -> None:
    registered = set(list_scenarios())
    for scenario in BUILTIN_SCENARIOS:
        if scenario.id not in registered:
            register_scenario(scenario)


register_builtin_scenarios()

__all__ = ["BUILTIN_SCENARIOS", "register_builtin_scenarios"]
