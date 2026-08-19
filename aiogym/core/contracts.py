"""Minimal cross-package contracts for scenarios and policies."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .io import jsonable
from .specs import Benchmark, EpisodeSpec, Reward


@runtime_checkable
class ProcessModel(Protocol):
    scenario: str
    dt_micro: float
    parameter_units: Mapping[str, str]

    @property
    def resolved_parameters(self) -> Mapping[str, Any]: ...

    def initial_state(self) -> Sequence[float]: ...

    def dynamics(
        self,
        state: Sequence[float],
        action: Sequence[float],
        disturbances: Mapping[str, float] | None = None,
    ) -> Sequence[float]: ...

    def outputs(self, state: Sequence[float]) -> Sequence[float]: ...

    def action_schema(self) -> Sequence[Mapping[str, Any]]: ...

    def state_schema(self) -> Sequence[Mapping[str, Any]]: ...

    def default_action(self) -> Sequence[float]: ...

    def default_setpoint_vector(self) -> Sequence[float]: ...

    def default_disturbances(self) -> Mapping[str, float]: ...

    def output_schema(self) -> Sequence[Mapping[str, Any]]: ...

    def observation_schema(self) -> Sequence[Mapping[str, Any]]: ...

    def action_slew_limits(self) -> Sequence[float] | None: ...

    def action_dim(self) -> int: ...

    def action_vector(self, action: Sequence[float]) -> Sequence[float]: ...

    def controlled_output_scales(self) -> Sequence[float]: ...

    def measurement(
        self,
        state: Sequence[float],
        disturbances: Mapping[str, float] | None = None,
    ) -> Mapping[str, Any]: ...

    def measurement_from_observation(
        self,
        observation: Sequence[float],
        disturbances: Mapping[str, float] | None = None,
    ) -> Mapping[str, Any]: ...

    def tracking_steady_state_action(
        self,
        reference: Sequence[float],
        disturbances: Mapping[str, float] | None = None,
    ) -> Sequence[float] | None: ...

    def tracking_steady_state_state(
        self,
        reference: Sequence[float],
        disturbances: Mapping[str, float] | None = None,
    ) -> Sequence[float] | None: ...

    def clamp_state(self, state: Sequence[float]) -> Sequence[float]: ...

    def observation(
        self,
        state: Sequence[float],
        reference: Sequence[float],
        previous_action: Sequence[float],
        disturbances: Mapping[str, float],
    ) -> Sequence[float]: ...

    def constraint_costs(
        self, state: Sequence[float], disturbances: Mapping[str, float]
    ) -> Mapping[str, float]: ...

    def safety_margins(
        self, state: Sequence[float], disturbances: Mapping[str, float]
    ) -> Mapping[str, float]: ...

    def step_info(
        self,
        state: Sequence[float],
        action: Sequence[float] | None,
        disturbances: Mapping[str, float],
    ) -> Mapping[str, Any]: ...


@runtime_checkable
class Policy(Protocol):
    env: Any | None

    def reset(self, seed: int | None = None) -> None: ...

    def act(
        self, observation: np.ndarray, context: Mapping[str, Any]
    ) -> np.ndarray: ...

    def metadata(self) -> Mapping[str, Any]: ...


def validate_policy(policy: Any) -> Policy:
    if not isinstance(policy, Policy):
        raise TypeError(
            "policy must provide env, reset(), act(), and metadata()"
        )
    return policy


def policy_metadata(policy: Any) -> dict[str, Any]:
    resolved = validate_policy(policy)
    metadata = resolved.metadata()
    if not isinstance(metadata, Mapping):
        raise TypeError("policy metadata() must return a mapping")
    serialized = jsonable(dict(metadata))
    if not isinstance(serialized, dict):
        raise TypeError("policy metadata() must return a mapping")
    return serialized


EpisodeSampler = Callable[
    [ProcessModel, np.random.Generator, str], tuple[EpisodeSpec, str]
]
TrainingDisturbanceSampler = Callable[
    [ProcessModel, np.random.Generator], Mapping[int, Mapping[str, float]]
]
ControllerConfig = Callable[[str, str], Mapping[str, Any]]


@dataclass(frozen=True)
class Scenario:
    id: str
    make_model: Callable[[Mapping[str, Any] | None], ProcessModel]
    control_dt: float
    make_default_episode: Callable[[ProcessModel], EpisodeSpec]
    sample_training_episode: EpisodeSampler
    sample_training_disturbance: TrainingDisturbanceSampler
    benchmarks: Mapping[str, Benchmark]
    rewards: Mapping[str, Reward]
    default_reward: str
    controller_config: ControllerConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("scenario id must be non-empty")
        control_dt = float(self.control_dt)
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("scenario control_dt must be finite and positive")
        if not callable(self.make_default_episode):
            raise TypeError("scenario make_default_episode must be callable")
        if not callable(self.sample_training_episode):
            raise TypeError("scenario sample_training_episode must be callable")
        if not callable(self.sample_training_disturbance):
            raise TypeError("scenario sample_training_disturbance must be callable")
        if not self.benchmarks:
            raise ValueError("scenario must declare at least one benchmark")
        for benchmark_id, benchmark in self.benchmarks.items():
            if benchmark_id != benchmark.id:
                raise ValueError("scenario benchmark keys must match Benchmark.id")
        if not self.rewards:
            raise ValueError("scenario must declare at least one reward")
        if self.default_reward not in self.rewards:
            raise ValueError("default_reward must be declared in rewards")
        for reward_id, reward in self.rewards.items():
            if reward_id != reward.id:
                raise ValueError("scenario reward keys must match Reward.id")
        unknown_benchmark_rewards = sorted(
            {
                benchmark.reward_id
                for benchmark in self.benchmarks.values()
                if benchmark.reward_id not in self.rewards
            }
        )
        if unknown_benchmark_rewards:
            raise ValueError(
                "benchmark reward_id values must be declared in rewards: "
                f"{unknown_benchmark_rewards}"
            )
        model = self.make_model(None)
        if model.scenario != self.id:
            raise ValueError("scenario model does not belong to scenario id")
        if not isinstance(self.make_default_episode(model), EpisodeSpec):
            raise TypeError("make_default_episode must return EpisodeSpec")
        for benchmark in self.benchmarks.values():
            benchmark.make_episode(model, 0)
        object.__setattr__(self, "control_dt", control_dt)
        object.__setattr__(self, "benchmarks", MappingProxyType(dict(self.benchmarks)))
        object.__setattr__(self, "rewards", MappingProxyType(dict(self.rewards)))


__all__ = ["Policy", "ProcessModel", "Scenario"]
