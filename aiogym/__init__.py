"""Small public API for AIO-Gym 0.34.0."""

from __future__ import annotations


__version__ = "0.34.0"

from .core.contracts import Policy, Scenario
from .controllers.base import make_controller
from .controllers.policies import FunctionPolicy
from .rl.algorithms import list_algorithms
from .workflows import (
    DatasetReader, collect, evaluate, load_policy,
    plot_training_curve, train,
)
from .workflows._metadata import environment_metadata
from .core import catalog as _catalog
from .scenarios import BUILTIN_SCENARIOS as _builtin_scenarios

_catalog.BUILTIN_SCENARIOS = _builtin_scenarios


def list_scenarios():
    from .core.catalog import list_scenarios as implementation

    return implementation()


def list_rewards(scenario: str | Scenario):
    from .core.information import reward_information
    from .core.catalog import get_scenario

    return reward_information(get_scenario(scenario))


def list_benchmarks(scenario: str | Scenario):
    from .core.information import benchmark_information
    from .core.catalog import get_scenario

    return benchmark_information(get_scenario(scenario))


def list_parameters(scenario: str | Scenario):
    from .core.catalog import list_parameters as implementation

    return implementation(scenario=scenario)


def list_states(scenario: str | Scenario):
    """List physical state definitions, keyed by their canonical names."""
    from .core.catalog import list_variables

    return list_variables(scenario=scenario, category="state")


def list_actions(scenario: str | Scenario):
    """List action definitions and physical interpretations."""
    from .core.catalog import list_variables

    return list_variables(scenario=scenario, category="action")


def list_observations(scenario: str | Scenario):
    """List observation definitions, sources and normalization rules."""
    from .core.catalog import list_variables

    return list_variables(scenario=scenario, category="observation")


def list_outputs(scenario: str | Scenario):
    """List controlled output definitions."""
    from .core.catalog import list_variables

    return list_variables(scenario=scenario, category="output")


def list_safety_rules(scenario: str | Scenario):
    """List documented safety conditions and effects; None means not provided."""
    from .core.io import jsonable
    from .core.catalog import get_scenario

    model = get_scenario(scenario).make_model(None)
    method = getattr(model, "safety_metadata", None)
    return None if method is None else jsonable(method())


def list_info(scenario: str | Scenario, *, parameters=None, reward=None, benchmark=None):
    """List a complete configuration template without resetting or running an episode."""
    env = make_env(scenario, parameters=parameters, reward=reward, benchmark=benchmark)
    try:
        return env.describe()
    finally:
        env.close()


def make_env(
    scenario: str | Scenario,
    *,
    reward=None,
    parameters=None,
    initial_state=None,
    benchmark=None,
    randomize=False,
    boundary_probability=0.0,
    disturbance=False,
    disturbance_schedule=None,
    noise=False,
    delay=False,
    fault=False,
):
    """Create an environment from a built-in name or a Scenario definition."""
    from .core.env import make_env as implementation

    return implementation(
        scenario,
        reward=reward,
        parameters=parameters,
        initial_state=initial_state,
        benchmark=benchmark,
        randomize=randomize,
        boundary_probability=boundary_probability,
        disturbance=disturbance,
        disturbance_schedule=disturbance_schedule,
        noise=noise,
        delay=delay,
        fault=fault,
    )


__all__ = [
    "__version__",
    "DatasetReader",
    "FunctionPolicy",
    "Policy",
    "Scenario",
    "collect",
    "environment_metadata",
    "evaluate",
    "list_algorithms",
    "list_actions",
    "list_info",
    "list_observations",
    "list_outputs",
    "list_safety_rules",
    "list_scenarios",
    "list_states",
    "list_benchmarks",
    "list_parameters",
    "list_rewards",
    "load_policy",
    "make_controller",
    "make_env",
    "plot_training_curve",
    "train",
]
