"""Small public API for AIO-Gym 0.33.1."""

from __future__ import annotations


__version__ = "0.33.1"

from .core.contracts import Policy
from .rl import AlgorithmBackend, SB3AlgorithmBackend, TrainingStep
from .controllers.base import make_controller
from .rl.algorithms import list_algorithms, register_algorithm, register_sb3_algorithm
from .workflows import (
    DatasetReader, collect, compare_policies, evaluate, load_policy,
    plot_training_curve, train,
)


def _register():
    import aiogym.scenarios  # noqa: F401


def list_scenarios():
    _register()
    from .core.registry import list_scenarios as implementation

    return implementation()


def list_rewards(scenario: str):
    _register()
    from .core.registry import list_rewards as implementation

    return implementation(scenario=scenario)


def list_benchmarks(scenario: str):
    _register()
    from .core.registry import list_benchmarks as implementation

    return implementation(scenario=scenario)


def list_parameters(scenario: str):
    _register()
    from .core.registry import list_parameters as implementation

    return implementation(scenario=scenario)


def make_env(
    scenario: str,
    *,
    reward=None,
    parameters=None,
    heater=None,
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
    """Create a default, user-initialized, randomized, or Benchmark environment."""
    _register()
    from .core.env import make_env as implementation

    return implementation(
        scenario,
        reward=reward,
        parameters=parameters,
        heater=heater,
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
    "AlgorithmBackend",
    "DatasetReader",
    "Policy",
    "SB3AlgorithmBackend",
    "TrainingStep",
    "collect",
    "compare_policies",
    "evaluate",
    "list_algorithms",
    "list_scenarios",
    "list_benchmarks",
    "list_parameters",
    "list_rewards",
    "load_policy",
    "make_controller",
    "make_env",
    "plot_training_curve",
    "register_algorithm",
    "register_sb3_algorithm",
    "train",
]
