"""Small immutable specifications shared by AIO-Gym core workflows."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from .io import stable_hash


PLANT_SCHEMA_VERSION = "aiogym.plant.v1"
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
    operating_point: Mapping[str, Any] = field(default_factory=dict)
    study: Mapping[str, Any] = field(default_factory=dict)
    references: tuple[Any, ...] = ()
    schema_version: str = PLANT_SCHEMA_VERSION
    plant_hash: str = field(init=False)

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
        object.__setattr__(self, "operating_point", _mapping(self.operating_point))
        object.__setattr__(self, "study", _mapping(self.study))
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "plant_hash", stable_hash(self.as_dict(include_hash=False)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PlantConfig":
        allowed = {
            "schema_version",
            "id",
            "scenario",
            "description",
            "plant",
            "operating_point",
            "study",
            "references",
            "plant_hash",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown PlantConfig fields: {sorted(unknown)}")
        config = cls(
            schema_version=str(value.get("schema_version", PLANT_SCHEMA_VERSION)),
            id=str(value.get("id", "")),
            scenario=str(value.get("scenario", "")),
            description=str(value.get("description", "")),
            plant=dict(value.get("plant", {})),
            operating_point=dict(value.get("operating_point", {})),
            study=dict(value.get("study", {})),
            references=tuple(value.get("references", ())),
        )
        declared = value.get("plant_hash")
        if declared is not None and str(declared) != config.plant_hash:
            raise ValueError("declared plant_hash does not match resolved PlantConfig")
        return config

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "id": self.id,
            "scenario": self.scenario,
            "description": self.description,
            "plant": dict(self.plant),
            "operating_point": dict(self.operating_point),
            "study": dict(self.study),
            "references": list(self.references),
        }
        if include_hash:
            payload["plant_hash"] = self.plant_hash
        return payload


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
class PresetSpec:
    id: str
    config: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("preset id must be non-empty")
        object.__setattr__(self, "config", _mapping(self.config))


@dataclass(frozen=True)
class TaskSpec:
    id: str
    scenario: str
    objective: str
    reward: RewardFunction
    metrics: tuple[str, ...]
    primary_metric: str
    metric_direction: MetricDirection
    horizon: int
    control_dt: float
    presets: Mapping[str, PresetSpec] = field(default_factory=dict)
    default_preset: str = "default"
    reference: tuple[float, ...] = ()
    task_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if self.id != f"{self.scenario}/{self.objective}":
            raise ValueError("task id must use '<scenario>/<objective>'")
        if self.metric_direction not in {"minimize", "maximize"}:
            raise ValueError("metric_direction must be minimize or maximize")
        if self.primary_metric not in self.metrics:
            raise ValueError("primary_metric must appear in metrics")
        if isinstance(self.horizon, bool) or int(self.horizon) <= 0:
            raise ValueError("horizon must be a positive integer")
        if float(self.control_dt) <= 0:
            raise ValueError("control_dt must be positive")
        presets = dict(self.presets)
        if not presets:
            presets = {"default": PresetSpec("default")}
        if self.default_preset not in presets:
            raise ValueError("default_preset must be declared in presets")
        object.__setattr__(self, "presets", MappingProxyType(presets))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "reference", tuple(float(v) for v in self.reference))
        object.__setattr__(self, "task_hash", stable_hash(self.identity()))

    def identity(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "metrics": list(self.metrics),
            "primary_metric": self.primary_metric,
            "metric_direction": self.metric_direction,
            "horizon": int(self.horizon),
            "control_dt": float(self.control_dt),
            "presets": {
                name: dict(preset.config) for name, preset in sorted(self.presets.items())
            },
            "default_preset": self.default_preset,
            "reference": list(self.reference),
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


__all__ = [
    "MetricDirection",
    "PLANT_SCHEMA_VERSION",
    "PlantConfig",
    "PresetSpec",
    "ResolvedPlant",
    "RewardFunction",
    "RunResult",
    "TaskSpec",
]
