"""Small immutable specifications shared by AIO-Gym core workflows."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from .io import stable_hash


PLANT_SCHEMA_VERSION = "aiogym.plant.v2"
MetricDirection = Literal["minimize", "maximize"]
RewardFunction = Callable[
    [Sequence[float], Sequence[float], Sequence[float], Mapping[str, Any]],
    float | tuple[float, Mapping[str, float]],
]


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class PlantConfig:
    id: str
    scenario: str
    plant: Mapping[str, Any]
    description: str = ""
    conditions: Mapping[str, Any] = field(default_factory=dict)
    default_condition: str = ""
    # Transitional input for the v0.2 reader. New declarations use conditions.
    operating_point: Mapping[str, Any] = field(default_factory=dict)
    study: Mapping[str, Any] = field(default_factory=dict)
    references: tuple[Any, ...] = ()
    schema_version: str = PLANT_SCHEMA_VERSION
    plant_hash: str = field(init=False)
    condition_hashes: Mapping[str, str] = field(init=False)
    study_hash: str = field(init=False)
    config_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != PLANT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {PLANT_SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("plant id must be a non-empty string")
        if not isinstance(self.scenario, str) or not self.scenario.strip():
            raise ValueError("plant scenario must be a non-empty string")
        object.__setattr__(self, "plant", _mapping(self.plant))
        conditions = {
            str(name): (
                value
                if isinstance(value, OperatingCondition)
                else OperatingCondition.from_mapping(value, default_id=str(name))
            )
            for name, value in self.conditions.items()
        }
        if self.default_condition and self.default_condition not in conditions:
            raise ValueError("default_condition must be declared in conditions")
        object.__setattr__(self, "conditions", MappingProxyType(conditions))
        object.__setattr__(self, "operating_point", _mapping(self.operating_point))
        object.__setattr__(self, "study", _mapping(self.study))
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(
            self,
            "plant_hash",
            stable_hash({"scenario": self.scenario, "plant": dict(self.plant)}),
        )
        object.__setattr__(
            self,
            "condition_hashes",
            MappingProxyType(
                {name: condition.condition_hash for name, condition in conditions.items()}
            ),
        )
        object.__setattr__(self, "study_hash", stable_hash(dict(self.study)))
        object.__setattr__(
            self, "config_hash", stable_hash(self.as_dict(include_hash=False))
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PlantConfig":
        allowed = {
            "schema_version",
            "id",
            "scenario",
            "description",
            "plant",
            "conditions",
            "default_condition",
            "operating_point",
            "study",
            "references",
            "plant_hash",
            "condition_hashes",
            "study_hash",
            "config_hash",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown PlantConfig fields: {sorted(unknown)}")
        schema = str(value.get("schema_version", PLANT_SCHEMA_VERSION))
        if schema not in {"aiogym.plant.v1", PLANT_SCHEMA_VERSION}:
            raise ValueError(f"unsupported PlantConfig schema_version {schema!r}")
        config = cls(
            schema_version=PLANT_SCHEMA_VERSION,
            id=str(value.get("id", "")),
            scenario=str(value.get("scenario", "")),
            description=str(value.get("description", "")),
            plant=dict(value.get("plant", {})),
            conditions=dict(value.get("conditions", {})),
            default_condition=str(value.get("default_condition", "")),
            operating_point=dict(value.get("operating_point", {})),
            study=dict(value.get("study", {})),
            references=tuple(value.get("references", ())),
        )
        declared = value.get("plant_hash")
        if declared is not None and str(declared) != config.plant_hash:
            raise ValueError("declared plant_hash does not match resolved PlantConfig")
        for field_name in ("study_hash", "config_hash"):
            declared_hash = value.get(field_name)
            if declared_hash is not None and str(declared_hash) != getattr(
                config, field_name
            ):
                raise ValueError(
                    f"declared {field_name} does not match resolved PlantConfig"
                )
        return config

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "id": self.id,
            "scenario": self.scenario,
            "description": self.description,
            "plant": dict(self.plant),
            "conditions": {
                name: condition.as_dict(include_hash=False)
                for name, condition in sorted(self.conditions.items())
            },
            "default_condition": self.default_condition,
            "study": dict(self.study),
            "references": list(self.references),
        }
        if self.operating_point:
            payload["operating_point"] = dict(self.operating_point)
        if include_hash:
            payload["plant_hash"] = self.plant_hash
            payload["condition_hashes"] = dict(self.condition_hashes)
            payload["study_hash"] = self.study_hash
            payload["config_hash"] = self.config_hash
        return payload


@dataclass(frozen=True)
class OperatingCondition:
    id: str
    initial_state: tuple[float, ...]
    reference: tuple[float, ...]
    control_dt: float
    horizon: int
    disturbances: Mapping[str, float] = field(default_factory=dict)
    reference_schedule: Mapping[int, tuple[float, ...]] = field(default_factory=dict)
    disturbance_schedule: Mapping[int, Mapping[str, float]] = field(
        default_factory=dict
    )
    observation: str = "state-reference-disturbance"
    condition_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("condition id must be a non-empty string")
        initial_state = _finite_tuple("initial_state", self.initial_state)
        reference = _finite_tuple("reference", self.reference)
        control_dt = float(self.control_dt)
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        if isinstance(self.horizon, bool) or int(self.horizon) <= 0:
            raise ValueError("horizon must be a positive integer")
        disturbances = _finite_mapping("disturbances", self.disturbances)
        reference_schedule = {}
        for raw_step, values in self.reference_schedule.items():
            step = _schedule_step(raw_step, int(self.horizon))
            resolved = _finite_tuple(f"reference_schedule[{step}]", values)
            if len(resolved) != len(reference):
                raise ValueError("scheduled references must match reference length")
            reference_schedule[step] = resolved
        disturbance_schedule = {}
        for raw_step, values in self.disturbance_schedule.items():
            step = _schedule_step(raw_step, int(self.horizon))
            disturbance_schedule[step] = _finite_mapping(
                f"disturbance_schedule[{step}]", values
            )
        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "control_dt", control_dt)
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "disturbances", MappingProxyType(disturbances))
        object.__setattr__(
            self, "reference_schedule", MappingProxyType(reference_schedule)
        )
        object.__setattr__(
            self,
            "disturbance_schedule",
            MappingProxyType(
                {
                    step: MappingProxyType(values)
                    for step, values in disturbance_schedule.items()
                }
            ),
        )
        object.__setattr__(
            self, "condition_hash", stable_hash(self.as_dict(include_hash=False))
        )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, default_id: str = ""
    ) -> "OperatingCondition":
        allowed = {
            "id",
            "initial_state",
            "reference",
            "control_dt",
            "horizon",
            "disturbances",
            "reference_schedule",
            "disturbance_schedule",
            "observation",
            "condition_hash",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown OperatingCondition fields: {sorted(unknown)}")
        condition = cls(
            id=str(value.get("id", default_id)),
            initial_state=tuple(value.get("initial_state", ())),
            reference=tuple(value.get("reference", ())),
            control_dt=float(value.get("control_dt", 0.0)),
            horizon=value.get("horizon", 0),
            disturbances=dict(value.get("disturbances", {})),
            reference_schedule={
                int(step): tuple(values)
                for step, values in value.get("reference_schedule", {}).items()
            },
            disturbance_schedule={
                int(step): dict(values)
                for step, values in value.get("disturbance_schedule", {}).items()
            },
            observation=str(
                value.get("observation", "state-reference-disturbance")
            ),
        )
        declared = value.get("condition_hash")
        if declared is not None and str(declared) != condition.condition_hash:
            raise ValueError("declared condition_hash does not match condition")
        return condition

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "initial_state": list(self.initial_state),
            "reference": list(self.reference),
            "control_dt": self.control_dt,
            "horizon": self.horizon,
            "disturbances": dict(self.disturbances),
            "reference_schedule": {
                str(step): list(values)
                for step, values in sorted(self.reference_schedule.items())
            },
            "disturbance_schedule": {
                str(step): dict(values)
                for step, values in sorted(self.disturbance_schedule.items())
            },
            "observation": self.observation,
        }
        if include_hash:
            payload["condition_hash"] = self.condition_hash
        return payload


def _finite_tuple(name: str, values: Sequence[float]) -> tuple[float, ...]:
    resolved = tuple(float(value) for value in values)
    if not resolved or not all(math.isfinite(value) for value in resolved):
        raise ValueError(f"{name} must contain finite numeric values")
    return resolved


def _finite_mapping(name: str, values: Mapping[str, float]) -> dict[str, float]:
    resolved = {str(key): float(value) for key, value in values.items()}
    if not all(math.isfinite(value) for value in resolved.values()):
        raise ValueError(f"{name} values must be finite")
    return resolved


def _schedule_step(value: Any, horizon: int) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError("schedule keys must be integers")
    step = int(value)
    if step < 0 or step >= horizon:
        raise ValueError("schedule keys must be within [0, horizon)")
    return step


@dataclass(frozen=True)
class ResolvedPlant:
    config: PlantConfig
    parameters: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _mapping(self.parameters))
        object.__setattr__(self, "provenance", _mapping(self.provenance))

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def scenario(self) -> str:
        return self.config.scenario

    @property
    def plant_hash(self) -> str:
        return self.config.plant_hash


@dataclass(frozen=True)
class TaskSpec:
    id: str
    scenario: str
    objective: str
    reward: RewardFunction
    metrics: tuple[str, ...]
    primary_metric: str
    metric_direction: MetricDirection
    revision: int = 1
    reward_id: str = "reward-v1"
    metric_suite_id: str = "metrics-v1"
    required_capabilities: tuple[str, ...] = ()
    reward_term_names: tuple[str, ...] = ()
    task_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.id != f"{self.scenario}/{self.objective}":
            raise ValueError("task id must use '<scenario>/<objective>'")
        if self.metric_direction not in {"minimize", "maximize"}:
            raise ValueError("metric_direction must be minimize or maximize")
        if self.primary_metric not in self.metrics:
            raise ValueError("primary_metric must appear in metrics")
        if isinstance(self.revision, bool) or int(self.revision) <= 0:
            raise ValueError("revision must be a positive integer")
        if not self.reward_id or not self.metric_suite_id:
            raise ValueError("reward_id and metric_suite_id must be non-empty")
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "required_capabilities", tuple(self.required_capabilities))
        object.__setattr__(self, "reward_term_names", tuple(self.reward_term_names))
        object.__setattr__(self, "task_hash", stable_hash(self.identity()))

    def identity(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "revision": int(self.revision),
            "reward_id": self.reward_id,
            "metric_suite_id": self.metric_suite_id,
            "reward_term_names": list(self.reward_term_names),
            "metrics": list(self.metrics),
            "primary_metric": self.primary_metric,
            "metric_direction": self.metric_direction,
            "required_capabilities": list(self.required_capabilities),
        }


@dataclass(frozen=True)
class EnvironmentIdentity:
    task_id: str
    task_hash: str
    plant_id: str
    plant_hash: str
    condition_id: str
    condition_hash: str
    interface_hash: str
    integrator_id: str = "rk4-v1"
    env_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "task_hash",
            "plant_id",
            "plant_hash",
            "condition_id",
            "condition_hash",
            "interface_hash",
            "integrator_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(
            self,
            "env_hash",
            stable_hash(
                {
                    "task_hash": self.task_hash,
                    "plant_hash": self.plant_hash,
                    "condition_hash": self.condition_hash,
                    "interface_hash": self.interface_hash,
                    "integrator_id": self.integrator_id,
                }
            ),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "task_hash": self.task_hash,
            "plant_id": self.plant_id,
            "plant_hash": self.plant_hash,
            "condition_id": self.condition_id,
            "condition_hash": self.condition_hash,
            "interface_hash": self.interface_hash,
            "integrator_id": self.integrator_id,
            "env_hash": self.env_hash,
        }


@dataclass(frozen=True)
class RunResult:
    workflow: str
    output: Path | None
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", None if self.output is None else Path(self.output))
        object.__setattr__(self, "manifest", _mapping(self.manifest))
        object.__setattr__(self, "metrics", _mapping(self.metrics))


@dataclass(frozen=True)
class CheckResult:
    name: str
    category: str
    passed: bool
    summary: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.category:
            raise ValueError("check name and category must be non-empty")
        object.__setattr__(self, "metrics", _mapping(self.metrics))
        object.__setattr__(self, "warnings", tuple(self.warnings))


__all__ = [
    "MetricDirection",
    "CheckResult",
    "EnvironmentIdentity",
    "OperatingCondition",
    "PLANT_SCHEMA_VERSION",
    "PlantConfig",
    "ResolvedPlant",
    "RewardFunction",
    "RunResult",
    "TaskSpec",
]
