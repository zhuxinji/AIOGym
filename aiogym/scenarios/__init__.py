"""Built-in Scenario catalog."""
from __future__ import annotations

from .cascade import SCENARIO as CASCADE
from .cstr import SCENARIO as CSTR
from .crystallization import SCENARIO as CRYSTALLIZATION
from .extraction import SCENARIO as EXTRACTION
from .heater import SCENARIO as HEATER
from .hvac import SCENARIO as HVAC
from .quadruple import SCENARIO as QUADRUPLE
from .three_tank import SCENARIO as THREE_TANK


BUILTIN_SCENARIOS = {
    scenario.id: scenario
    for scenario in (
        CASCADE,
        CSTR,
        CRYSTALLIZATION,
        EXTRACTION,
        HEATER,
        HVAC,
        QUADRUPLE,
        THREE_TANK,
    )
}

__all__ = ["BUILTIN_SCENARIOS"]
