"""Small public API for AIO-Gym 0.8."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


__version__ = "0.8.0"

from .workflows.algorithms import AlgorithmBackend, TrainingStep
from .workflows._sb3_runtime import SB3AlgorithmBackend
from .core.contracts import Policy
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


def list_algorithms():
    from .workflows.algorithms import list_algorithms as implementation

    return implementation()


def register_algorithm(backend: AlgorithmBackend) -> None:
    from .workflows.algorithms import register_algorithm as implementation

    implementation(backend)


def register_sb3_algorithm(
    algorithm: str,
    model_class: type,
    *,
    behavior_cloning=None,
) -> None:
    from .workflows.algorithms import register_sb3_algorithm as implementation

    implementation(
        algorithm,
        model_class,
        behavior_cloning=behavior_cloning,
    )


def make_env(
    scenario: str,
    *,
    reward=None,
    parameters=None,
    benchmark=None,
    randomize=False,
    disturbance=False,
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
        disturbance=disturbance,
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
    evaluation_env=None,
    evaluate_every: int | None = None,
    evaluation_seed: int = 0,
    demonstrations: str | Path | None = None,
    behavior_cloning_epochs: int | None = None,
    behavior_cloning_batch_size: int = 256,
    behavior_cloning_learning_rate: float = 3e-4,
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
        evaluation_env=evaluation_env,
        evaluate_every=evaluate_every,
        evaluation_seed=evaluation_seed,
        demonstrations=demonstrations,
        behavior_cloning_epochs=behavior_cloning_epochs,
        behavior_cloning_batch_size=behavior_cloning_batch_size,
        behavior_cloning_learning_rate=behavior_cloning_learning_rate,
    )


def plot_training_curve(
    curve: Mapping[str, Any] | str | Path,
    *,
    output: str | Path,
):
    from .workflows import plot_training_curve as implementation

    return implementation(curve, output=output)


def load_policy(checkpoint: str | Path, *, env):
    from .workflows import load_policy as implementation

    return implementation(checkpoint, env=env)


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
