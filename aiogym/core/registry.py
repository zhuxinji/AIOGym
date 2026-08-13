"""The single Scenario/Reward registry."""
from __future__ import annotations

from .contracts import Scenario
from .specs import Benchmark, Reward


_SCENARIOS: dict[str, Scenario] = {}


def register_scenario(definition: Scenario, *, replace: bool = False) -> None:
    if not isinstance(definition, Scenario):
        raise TypeError("scenario must be a Scenario")
    if definition.id in _SCENARIOS and not replace:
        raise ValueError(f"scenario {definition.id!r} is already registered")
    _SCENARIOS[definition.id] = definition


def unregister_scenario(scenario: str) -> None:
    if scenario in _SCENARIOS:
        del _SCENARIOS[scenario]


def get_scenario(scenario: str) -> Scenario:
    try:
        return _SCENARIOS[scenario]
    except KeyError as error:
        available = ", ".join(list_scenarios())
        raise KeyError(f"unknown scenario {scenario!r}; available: {available}") from error


def get_reward(scenario: str, reward: str) -> Reward:
    definition = get_scenario(scenario)
    try:
        return definition.rewards[reward]
    except KeyError as error:
        available = ", ".join(sorted(definition.rewards))
        raise KeyError(
            f"unknown reward {reward!r} for scenario {scenario!r}; available: {available}"
        ) from error


def get_benchmark(scenario: str, benchmark: str) -> Benchmark:
    definition = get_scenario(scenario)
    try:
        return definition.benchmarks[benchmark]
    except KeyError as error:
        available = ", ".join(sorted(definition.benchmarks))
        raise KeyError(
            f"unknown benchmark {benchmark!r} for scenario {scenario!r}; "
            f"available: {available}"
        ) from error


def list_scenarios() -> tuple[str, ...]:
    return tuple(sorted(_SCENARIOS))


def list_rewards(*, scenario: str) -> tuple[str, ...]:
    return tuple(sorted(get_scenario(scenario).rewards))


def list_benchmarks(*, scenario: str) -> tuple[str, ...]:
    return tuple(sorted(get_scenario(scenario).benchmarks))


def list_parameters(*, scenario: str) -> tuple[dict[str, object], ...]:
    model = get_scenario(scenario).make_model(None)
    defaults = model.resolved_parameters
    units = model.parameter_units
    missing_units = sorted(set(defaults) - set(units))
    unknown_units = sorted(set(units) - set(defaults))
    if missing_units or unknown_units:
        raise ValueError(
            f"parameter unit metadata mismatch for scenario {scenario!r}; "
            f"missing: {missing_units}; unknown: {unknown_units}"
        )
    invalid_units = sorted(
        name for name, unit in units.items() if not isinstance(unit, str) or not unit
    )
    if invalid_units:
        raise ValueError(
            f"parameter units must be non-empty strings for scenario {scenario!r}: "
            f"{invalid_units}"
        )
    return tuple(
        {"name": name, "default": defaults[name], "unit": units[name]}
        for name in sorted(defaults)
    )


__all__ = [
    "get_scenario",
    "get_reward",
    "get_benchmark",
    "list_scenarios",
    "list_benchmarks",
    "list_parameters",
    "list_rewards",
    "register_scenario",
    "unregister_scenario",
]
