"""Goal-level evaluation semantics independent of reward scalarization."""
from __future__ import annotations

from dataclasses import dataclass

from .metric_catalog import (
    SCORECARD_METRICS,
    metric_direction,
)


GOAL_NAMES = ("regulation", "economic")
DEFAULT_GOAL_PRIMARY_METRICS = {
    "regulation": "regulation_cost",
    "economic": "profit",
}


@dataclass(frozen=True)
class GoalSpec:
    """Evaluation goal plus the metric used to rank a declared track."""

    name: str
    source: str
    primary_metric: str
    direction: str
    metrics: tuple[str, ...] = SCORECARD_METRICS

    def __post_init__(self) -> None:
        if self.name not in GOAL_NAMES:
            raise ValueError("goal must be one of: economic, regulation")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("goal source must be a non-empty string")
        if not isinstance(self.primary_metric, str) or not self.primary_metric:
            raise ValueError("goal primary_metric must be a non-empty string")
        if self.primary_metric not in SCORECARD_METRICS:
            raise ValueError(
                "goal primary_metric must be a scorecard metric; "
                "training return is not a cross-reward ranking metric"
            )
        expected_direction = metric_direction(self.primary_metric)
        if self.direction != expected_direction:
            raise ValueError(
                f"goal direction for {self.primary_metric!r} must be "
                f"{expected_direction!r}"
            )

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "metrics": list(self.metrics),
        }


def goal_spec(
    name: str,
    *,
    source: str = "explicit",
    primary_metric: str | None = None,
) -> GoalSpec:
    """Build a validated regulation or economic GoalSpec."""

    if name not in GOAL_NAMES:
        raise ValueError("goal must be one of: economic, regulation")
    metric = primary_metric or DEFAULT_GOAL_PRIMARY_METRICS[name]
    return GoalSpec(
        name=name,
        source=str(source),
        primary_metric=metric,
        direction=metric_direction(metric),
    )


def resolve_goal(
    *,
    explicit: str | GoalSpec | None = None,
    source: str = "explicit",
    primary_metric: str | None = None,
) -> GoalSpec:
    """Resolve a GoalSpec from a canonical goal selection."""

    if isinstance(explicit, GoalSpec):
        if primary_metric is not None and primary_metric != explicit.primary_metric:
            raise ValueError("explicit GoalSpec conflicts with primary_metric")
        return explicit
    if explicit is not None:
        return goal_spec(
            str(explicit),
            source=source,
            primary_metric=primary_metric,
        )
    raise ValueError("no goal was resolved")


__all__ = [
    "DEFAULT_GOAL_PRIMARY_METRICS",
    "GOAL_NAMES",
    "GoalSpec",
    "goal_spec",
    "resolve_goal",
]
