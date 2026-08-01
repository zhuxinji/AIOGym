"""Versioned reward specifications and auditable reward results."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Sequence


GoalName = Literal["regulation", "economic"]


def _frozen_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        resolved = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("reward metadata keys must be strings")
            resolved[key] = _frozen_json_value(item)
        return MappingProxyType(resolved)
    if isinstance(value, (list, tuple)):
        return tuple(_frozen_json_value(item) for item in value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("reward metadata numbers must be finite")
        return value
    raise TypeError(
        "reward metadata must contain only JSON-compatible values"
    )


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _canonical_json_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, tuple):
        return [_canonical_json_value(item) for item in value]
    return value


def _validated_weights(
    name: str,
    values: Mapping[str, float],
) -> Mapping[str, float]:
    resolved = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name} values must be finite and non-negative")
        resolved[key] = number
    return MappingProxyType(resolved)


@dataclass(frozen=True)
class RewardSpec:
    """Immutable identity and configuration for one reward definition."""

    id: str
    version: str
    goal: GoalName
    term_weights: Mapping[str, float] = field(default_factory=dict)
    cost_weights: Mapping[str, float] = field(default_factory=dict)
    terminal_failure_cost_rate: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("reward spec id must not be empty")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("reward spec version must not be empty")
        if self.goal not in {"regulation", "economic"}:
            raise ValueError("reward spec goal must be one of: economic, regulation")
        terminal_rate = float(self.terminal_failure_cost_rate)
        if not math.isfinite(terminal_rate) or terminal_rate < 0:
            raise ValueError(
                "terminal_failure_cost_rate must be finite and non-negative"
            )
        object.__setattr__(self, "terminal_failure_cost_rate", terminal_rate)
        object.__setattr__(
            self,
            "term_weights",
            _validated_weights("term_weights", self.term_weights),
        )
        object.__setattr__(
            self,
            "cost_weights",
            _validated_weights("cost_weights", self.cost_weights),
        )
        object.__setattr__(
            self,
            "metadata",
            _frozen_json_value(self.metadata),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible reward identity."""

        return {
            "id": self.id,
            "version": self.version,
            "goal": self.goal,
            "term_weights": dict(sorted(self.term_weights.items())),
            "cost_weights": dict(sorted(self.cost_weights.items())),
            "terminal_failure_cost_rate": self.terminal_failure_cost_rate,
            "metadata": _canonical_json_value(self.metadata),
        }

    @property
    def spec_hash(self) -> str:
        """SHA-256 identity of the complete canonical reward definition."""

        canonical = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def canonical(self) -> bool:
        """Whether the registered spec is reproducible without a custom override."""

        return bool(self.metadata.get("canonical", False))


@dataclass(frozen=True)
class StageRewardContext:
    """Read-only transition context supplied to a custom stage reward."""

    model: Any
    setpoint: tuple[float, ...]
    disturbance: Mapping[str, Any]
    previous_action: Any
    goal: GoalName
    base_reward: float
    terminated: bool
    info: Mapping[str, Any]
    reward_spec_id: str = ""
    reward_spec_is_canonical: bool = False


StageRewardOverride = Callable[
    [Sequence[float], Any, Sequence[float], StageRewardContext],
    float,
]


@dataclass(frozen=True)
class RewardResult:
    """Reward decomposition for one state transition."""

    scalar_reward: float
    goal_reward: float
    reward_terms: Mapping[str, float]
    costs: Mapping[str, float]
    applied_cost_penalties: Mapping[str, float]
    terminal_failure_cost: float
    terminated: bool
    termination_reason: str | None
    info: Mapping[str, Any]
