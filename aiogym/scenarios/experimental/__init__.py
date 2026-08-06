from aiogym.scenarios._legacy import build_plugin


PLUGINS = tuple(
    build_plugin(scenario, economic=scenario in {"cstr", "heater"})
    for scenario in ("crystallization", "cstr", "extraction", "heater", "hvac")
)

__all__ = ["PLUGINS"]
