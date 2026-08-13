"""Small immutable specifications shared by AIO-Gym core workflows."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Literal

from .io import deep_freeze, deep_thaw


MetricDirection = Literal["minimize", "maximize"]
RewardFunction = Callable[
    [Sequence[float], Sequence[float], Sequence[float], Mapping[str, Any]],
    float | tuple[float, Mapping[str, float]],
]
EpisodeMetricFunction = Callable[[Any, Any], Mapping[str, float]]
EpisodeFactory = Callable[[Any], "EpisodeSpec"]


@dataclass(frozen=True)
class EpisodeSpec:
    """One fully resolved process episode.

    This is an internal runtime value. Public environment selection is expressed
    through ``benchmark=`` or ``randomize=True`` rather than named episodes.
    """

    initial_state: tuple[float, ...]
    reference: tuple[float, ...]
    horizon: int
    disturbances: Mapping[str, float] = field(default_factory=dict)
    reference_schedule: Mapping[int, tuple[float, ...]] = field(default_factory=dict)
    disturbance_schedule: Mapping[int, Mapping[str, float]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        initial_state = _finite_tuple("initial_state", self.initial_state)
        reference = _finite_tuple("reference", self.reference)
        if isinstance(self.horizon, bool) or int(self.horizon) != self.horizon:
            raise ValueError("horizon must be a positive integer")
        horizon = int(self.horizon)
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        disturbances = _finite_mapping("disturbances", self.disturbances)
        reference_schedule: dict[int, tuple[float, ...]] = {}
        for raw_step, values in self.reference_schedule.items():
            step = _schedule_step(raw_step, horizon)
            resolved = _finite_tuple(f"reference_schedule[{step}]", values)
            if len(resolved) != len(reference):
                raise ValueError("scheduled references must match reference length")
            reference_schedule[step] = resolved
        disturbance_schedule: dict[int, dict[str, float]] = {}
        for raw_step, values in self.disturbance_schedule.items():
            step = _schedule_step(raw_step, horizon)
            disturbance_schedule[step] = _finite_mapping(
                f"disturbance_schedule[{step}]", values
            )
        object.__setattr__(self, "initial_state", initial_state)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "horizon", horizon)
        object.__setattr__(self, "disturbances", deep_freeze(disturbances))
        object.__setattr__(
            self, "reference_schedule", deep_freeze(reference_schedule)
        )
        object.__setattr__(
            self, "disturbance_schedule", deep_freeze(disturbance_schedule)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "initial_state": list(self.initial_state),
            "reference": list(self.reference),
            "horizon": self.horizon,
            "disturbances": deep_thaw(self.disturbances),
            "reference_schedule": {
                str(step): list(values)
                for step, values in sorted(self.reference_schedule.items())
            },
            "disturbance_schedule": {
                str(step): deep_thaw(values)
                for step, values in sorted(self.disturbance_schedule.items())
            },
        }


@dataclass(frozen=True)
class Benchmark:
    """A fixed evaluation protocol owned by one scenario."""

    id: str
    make_episode: EpisodeFactory
    metric_function: EpisodeMetricFunction
    ranking_metrics: tuple[tuple[str, MetricDirection], ...]
    measurement_noise: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or "/" in self.id:
            raise ValueError("benchmark id must be a non-empty local name")
        if not callable(self.make_episode):
            raise TypeError("benchmark make_episode must be callable")
        if not callable(self.metric_function):
            raise TypeError("benchmark metric_function must be callable")
        if not self.ranking_metrics:
            raise ValueError("benchmark must declare at least one ranking metric")
        names: set[str] = set()
        for name, direction in self.ranking_metrics:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("ranking metric names must be non-empty strings")
            if name in names:
                raise ValueError("benchmark ranking metric names must be unique")
            if direction not in {"minimize", "maximize"}:
                raise ValueError("ranking metric direction must be minimize or maximize")
            names.add(name)
        if self.measurement_noise is None:
            noise = None
        elif not isinstance(self.measurement_noise, Mapping):
            raise TypeError("benchmark measurement_noise must be a mapping or None")
        else:
            noise = {str(name): float(value) for name, value in self.measurement_noise.items()}
            if set(noise) != {"std", "bias_std"}:
                raise ValueError(
                    "benchmark measurement_noise must contain exactly std and bias_std"
                )
            if not all(math.isfinite(value) and value >= 0.0 for value in noise.values()):
                raise ValueError(
                    "benchmark measurement_noise values must be finite and non-negative"
                )
            noise = deep_freeze(noise)
        object.__setattr__(self, "measurement_noise", noise)


@dataclass(frozen=True)
class Reward:
    id: str
    function: RewardFunction
    episode_metric_function: EpisodeMetricFunction | None = None
    primary_metric: str | None = None
    metric_direction: MetricDirection = "minimize"
    safety_violation_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip() or "/" in self.id:
            raise ValueError("reward id must be a non-empty local name")
        if not callable(self.function):
            raise TypeError("reward function must be callable")
        if self.metric_direction not in {"minimize", "maximize"}:
            raise ValueError("metric_direction must be minimize or maximize")
        if self.episode_metric_function is not None and not callable(
            self.episode_metric_function
        ):
            raise TypeError("episode_metric_function must be callable")
        if self.primary_metric is not None and (
            not isinstance(self.primary_metric, str)
            or not self.primary_metric.strip()
        ):
            raise ValueError("primary_metric must be a non-empty string or None")
        if self.episode_metric_function is None and self.primary_metric is not None:
            raise ValueError("primary_metric requires an episode_metric_function")
        penalty = float(self.safety_violation_penalty)
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError("safety_violation_penalty must be finite and non-negative")
        object.__setattr__(self, "safety_violation_penalty", penalty)


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


__all__ = [
    "Benchmark",
    "EpisodeMetricFunction",
    "EpisodeSpec",
    "MetricDirection",
    "Reward",
    "RewardFunction",
]
