"""The single Scenario/Task registry."""
from __future__ import annotations

from collections.abc import Iterable

from .contracts import ScenarioPlugin
from .specs import TaskSpec


_SCENARIOS: dict[str, ScenarioPlugin] = {}


def register_scenario(plugin: ScenarioPlugin, *, replace: bool = False) -> None:
    if not isinstance(plugin, ScenarioPlugin):
        raise TypeError("plugin must be a ScenarioPlugin")
    if plugin.id in _SCENARIOS and not replace:
        raise ValueError(f"scenario {plugin.id!r} is already registered")
    _SCENARIOS[plugin.id] = plugin


def unregister_scenario(scenario: str) -> None:
    _SCENARIOS.pop(scenario, None)


def get_scenario(scenario: str) -> ScenarioPlugin:
    try:
        return _SCENARIOS[scenario]
    except KeyError as error:
        available = ", ".join(list_scenarios()) or "<none>"
        raise KeyError(f"unknown scenario {scenario!r}; available: {available}") from error


def get_task(task_id: str) -> TaskSpec:
    if not isinstance(task_id, str) or task_id.count("/") != 1:
        raise ValueError("task id must use '<scenario>/<objective>'")
    scenario, objective = task_id.split("/", 1)
    plugin = get_scenario(scenario)
    try:
        return plugin.tasks[objective]
    except KeyError as error:
        raise KeyError(f"unknown task {task_id!r}") from error


def list_scenarios() -> tuple[str, ...]:
    return tuple(sorted(_SCENARIOS))


def list_tasks(*, scenario: str | None = None) -> tuple[str, ...]:
    plugins: Iterable[ScenarioPlugin]
    if scenario is None:
        plugins = _SCENARIOS.values()
    else:
        plugins = (get_scenario(scenario),)
    return tuple(sorted(task.id for plugin in plugins for task in plugin.tasks.values()))


__all__ = [
    "get_scenario",
    "get_task",
    "list_scenarios",
    "list_tasks",
    "register_scenario",
    "unregister_scenario",
]
