"""Built-in Scenario registrations."""
from __future__ import annotations

from aiogym.core.registry import list_scenarios, register_scenario

from .cstr import SCENARIO as CSTR
from .crystallization import SCENARIO as CRYSTALLIZATION
from .extraction import SCENARIO as EXTRACTION
from .heater import SCENARIO as HEATER
from .hvac import SCENARIO as HVAC
from .quadruple import SCENARIO as QUADRUPLE
from .three_tank import SCENARIO as THREE_TANK


BUILTIN_SCENARIOS = (
    CSTR,
    CRYSTALLIZATION,
    EXTRACTION,
    HEATER,
    HVAC,
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
