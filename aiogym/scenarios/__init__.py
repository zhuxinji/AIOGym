"""Built-in vertical ScenarioPlugin registrations."""
from __future__ import annotations

from aiogym.core import list_scenarios, register_scenario

from .cascade import PLUGIN as CASCADE
from .cascade_recirculating import PLUGIN as CASCADE_RECIRCULATING
from .quadruple import PLUGIN as QUADRUPLE
from .three_tank import PLUGIN as THREE_TANK


BUILTIN_PLUGINS = (
    QUADRUPLE,
    CASCADE,
    CASCADE_RECIRCULATING,
    THREE_TANK,
)


def register_builtin_scenarios() -> None:
    registered = set(list_scenarios())
    for plugin in BUILTIN_PLUGINS:
        if plugin.id not in registered:
            register_scenario(plugin)


register_builtin_scenarios()

__all__ = ["BUILTIN_PLUGINS", "register_builtin_scenarios"]
