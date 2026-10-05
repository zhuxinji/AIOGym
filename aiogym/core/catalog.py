"""Resolve built-in names or explicit Scenario definitions."""
from __future__ import annotations

from .contracts import Scenario
from .information import parameter_information, variable_information
from .specs import Benchmark, Reward

BUILTIN_SCENARIOS: dict[str, Scenario] = {}


def get_scenario(scenario: str | Scenario) -> Scenario:
    if isinstance(scenario, Scenario):
        return scenario
    if not isinstance(scenario, str):
        raise TypeError("scenario must be a built-in name or Scenario")

    try:
        return BUILTIN_SCENARIOS[scenario]
    except KeyError as error:
        available = ", ".join(list_scenarios())
        raise KeyError(f"unknown scenario {scenario!r}; available: {available}") from error


def get_reward(scenario: str | Scenario, reward: str) -> Reward:
    definition = get_scenario(scenario)
    try:
        return definition.rewards[reward]
    except KeyError as error:
        available = ", ".join(sorted(definition.rewards))
        raise KeyError(
            f"unknown reward {reward!r} for scenario {definition.id!r}; available: {available}"
        ) from error


def get_benchmark(scenario: str | Scenario, benchmark: str) -> Benchmark:
    definition = get_scenario(scenario)
    try:
        return definition.benchmarks[benchmark]
    except KeyError as error:
        available = ", ".join(sorted(definition.benchmarks))
        raise KeyError(
            f"unknown benchmark {benchmark!r} for scenario {definition.id!r}; "
            f"available: {available}"
        ) from error


def list_scenarios() -> tuple[str, ...]:
    return tuple(sorted(BUILTIN_SCENARIOS))


def list_rewards(*, scenario: str | Scenario) -> tuple[str, ...]:
    return tuple(sorted(get_scenario(scenario).rewards))


def list_benchmarks(*, scenario: str | Scenario) -> tuple[str, ...]:
    return tuple(sorted(get_scenario(scenario).benchmarks))


def list_parameters(*, scenario: str | Scenario) -> tuple[dict[str, object], ...]:
    model = get_scenario(scenario).make_model(None)
    return tuple(
        {"name": name, **row, "default": row["value"]}
        for name, row in parameter_information(model).items()
    )


def list_variables(*, scenario: str | Scenario, category: str):
    model = get_scenario(scenario).make_model(None)
    return variable_information(model, category)


__all__ = [
    "get_scenario",
    "get_reward",
    "get_benchmark",
    "list_scenarios",
    "list_benchmarks",
    "list_parameters",
    "list_rewards",
    "list_variables",
]
