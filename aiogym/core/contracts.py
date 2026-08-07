"""Minimal cross-package contracts for scenarios and policies."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .specs import OperatingCondition, PlantConfig, ResolvedPlant, TaskSpec


@runtime_checkable
class ProcessModel(Protocol):
    scenario: str

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


@runtime_checkable
class Policy(Protocol):
    def reset(self, seed: int | None = None) -> None: ...

    def act(
        self, observation: np.ndarray, context: Mapping[str, Any]
    ) -> np.ndarray: ...

    def metadata(self) -> Mapping[str, Any]: ...


@runtime_checkable
class StudyProvider(Protocol):
    def checks(self, context: "StudyContext") -> Sequence[Any]: ...

    def dynamic_requirements(self, context: "StudyContext") -> Mapping[str, Any]: ...

    def steady_check(
        self,
        context: "StudyContext",
        disturbances: Mapping[str, float],
    ) -> Any: ...

    def default_robustness(self, context: "StudyContext") -> tuple[int, int]: ...

    def sample_disturbances(
        self,
        context: "StudyContext",
        rng: np.random.Generator,
    ) -> dict[str, float]: ...


@dataclass(frozen=True)
class StudyContext:
    plant: ResolvedPlant
    condition: OperatingCondition
    model: ProcessModel


@dataclass(frozen=True)
class ScenarioPlugin:
    id: str
    make_model: Callable[[ResolvedPlant], ProcessModel]
    default_plant: str | Callable[[], Mapping[str, Any] | PlantConfig]
    resolve_plant: Callable[[PlantConfig], ResolvedPlant]
    tasks: Mapping[str, TaskSpec]
    built_in_plants: Mapping[
        str, Callable[[], Mapping[str, Any] | PlantConfig]
    ] = field(default_factory=dict)
    controller_defaults: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    resolve_controller_profile: Callable[..., Mapping[str, Any]] | None = None
    study_provider: StudyProvider | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("scenario plugin id must be non-empty")
        for objective, task in self.tasks.items():
            if task.scenario != self.id:
                raise ValueError(f"task {task.id!r} does not belong to scenario {self.id!r}")
            if objective != task.objective:
                raise ValueError("scenario task keys must match TaskSpec.objective")


__all__ = [
    "Policy",
    "ProcessModel",
    "ScenarioPlugin",
    "StudyContext",
    "StudyProvider",
]
