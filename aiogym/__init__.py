"""Small public API for AIO-Gym 0.7."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


__version__ = "0.7.0"

from .workflows.dataset import DatasetReader


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
    benchmark=None,
    randomize=False,
    noise=False,
    delay=False,
    fault=False,
):
    """Create a default, randomized training, or fixed benchmark environment."""
    _register()
    from .core.env import make_env as implementation

    return implementation(
        scenario,
        reward=reward,
        parameters=parameters,
        benchmark=benchmark,
        randomize=randomize,
        noise=noise,
        delay=delay,
        fault=fault,
    )


def make_controller(controller_id: str, *, env, config=None):
    from .controllers.base import make_controller as implementation

    return implementation(controller_id, env=env, config=config)


def collect(
    *,
    env,
    output: str | Path,
    policy="random",
    episodes: int = 1,
    seed: int = 0,
    max_steps: int | None = None,
):
    from .workflows import collect as implementation

    return implementation(
        env=env,
        output=output,
        policy=policy,
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
    )


def compare_policies(
    *,
    env,
    policies: Mapping[str, Any],
    seeds: Sequence[int],
    max_steps: int | None = None,
    output: str | Path | None = None,
):
    from .workflows import compare_policies as implementation

    return implementation(
        env=env,
        policies=policies,
        seeds=seeds,
        max_steps=max_steps,
        output=output,
    )


def train(
    *,
    env,
    algorithm: str,
    steps: int,
    output: str | Path,
    seed: int = 0,
    algorithm_kwargs: Mapping[str, Any] | None = None,
    record_every: int = 500,
):
    from .workflows import train as implementation

    return implementation(
        env=env,
        algorithm=algorithm,
        steps=steps,
        output=output,
        seed=seed,
        algorithm_kwargs=algorithm_kwargs,
        record_every=record_every,
    )


def plot_training_curve(
    curve: Mapping[str, Any] | str | Path,
    *,
    output: str | Path,
):
    from .workflows import plot_training_curve as implementation

    return implementation(curve, output=output)


def load_policy(checkpoint: str | Path, *, algorithm: str, env=None):
    from .workflows import load_policy as implementation

    return implementation(checkpoint, algorithm=algorithm, env=env)


def evaluate(
    *,
    env,
    policy,
    seeds: Sequence[int],
    max_steps: int | None = None,
    output: str | Path | None = None,
):
    from .workflows import evaluate as implementation

    return implementation(
        env=env,
        policy=policy,
        seeds=seeds,
        max_steps=max_steps,
        output=output,
    )


__all__ = [
    "__version__",
    "DatasetReader",
    "collect",
    "compare_policies",
    "evaluate",
    "list_scenarios",
    "list_benchmarks",
    "list_parameters",
    "list_rewards",
    "load_policy",
    "make_controller",
    "make_env",
    "plot_training_curve",
    "train",
]
