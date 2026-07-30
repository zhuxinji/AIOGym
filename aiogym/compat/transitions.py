"""Minimal reader model for migrating ``aiogym.transition.v1`` rows.

This module intentionally provides no collection, training, or Dataset v2
conversion helpers. New data must be collected and consumed through
``aiogym.datasets``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from aiogym._internal.validation import nonnegative_int


TRANSITION_SCHEMA_VERSION = "aiogym.transition.v1"


@dataclass(frozen=True)
class Transition:
    """One validated legacy transition used only as migration input."""

    obs: Sequence[float]
    state: Sequence[float]
    action: Sequence[float]
    reward: float
    next_obs: Sequence[float]
    next_state: Sequence[float]
    terminated: bool
    truncated: bool
    setpoint: Sequence[float] = ()
    disturbance: Mapping[str, Any] = field(default_factory=dict)
    info: Mapping[str, Any] = field(default_factory=dict)
    episode: int = 0
    step: int = 0
    schema_version: str = TRANSITION_SCHEMA_VERSION

    def __post_init__(self):
        for name in (
            "obs",
            "state",
            "action",
            "next_obs",
            "next_state",
            "setpoint",
        ):
            object.__setattr__(
                self,
                name,
                _finite_vector(name, getattr(self, name)),
            )
        if len(self.obs) != len(self.next_obs):
            raise ValueError("obs and next_obs must have the same length")
        if len(self.state) != len(self.next_state):
            raise ValueError("state and next_state must have the same length")
        reward = float(self.reward)
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))
        object.__setattr__(
            self,
            "episode",
            nonnegative_int("episode", self.episode),
        )
        object.__setattr__(self, "step", nonnegative_int("step", self.step))
        object.__setattr__(
            self,
            "disturbance",
            copy.deepcopy(dict(self.disturbance)),
        )
        object.__setattr__(self, "info", copy.deepcopy(dict(self.info)))
        if self.schema_version != TRANSITION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported transition schema: {self.schema_version!r}"
            )

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "episode": self.episode,
            "step": self.step,
            "obs": list(self.obs),
            "state": list(self.state),
            "action": list(self.action),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "next_obs": list(self.next_obs),
            "next_state": list(self.next_state),
            "setpoint": list(self.setpoint),
            "disturbance": _plain(self.disturbance),
            "info": _plain(self.info),
        }

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "Transition":
        data = {
            key: value
            for key, value in row.items()
            if key in _TRANSITION_FIELDS
        }
        setpoint = data.get("setpoint", ())
        if isinstance(setpoint, Mapping):
            data["setpoint"] = setpoint.get("y_sp", ())
        data.setdefault("episode", 0)
        data.setdefault("step", 0)
        data.setdefault("schema_version", TRANSITION_SCHEMA_VERSION)
        return cls(**data)


class TransitionDataset:
    """Validated legacy rows accepted by the one-way migration adapter."""

    def __init__(
        self,
        transitions: Iterable[Transition | Mapping[str, Any]] = (),
    ):
        self._items: list[Transition] = []
        self._dims: tuple[int, int, int] | None = None
        for transition in transitions:
            self.append(transition)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def append(self, transition: Transition | Mapping[str, Any]) -> None:
        item = (
            transition
            if isinstance(transition, Transition)
            else Transition.from_mapping(transition)
        )
        dims = (len(item.obs), len(item.state), len(item.action))
        if self._dims is None:
            self._dims = dims
        elif dims != self._dims:
            raise ValueError(
                "legacy transition dimensions must match: "
                f"expected obs/state/action={self._dims}, got {dims}"
            )
        self._items.append(item)

    def to_rows(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items]

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
    ) -> "TransitionDataset":
        return cls(Transition.from_mapping(row) for row in rows)


def _finite_vector(
    name: str,
    values: Sequence[float],
) -> tuple[float, ...]:
    try:
        vector = tuple(
            float(value) for value in np.asarray(values).reshape(-1)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if not all(np.isfinite(value) for value in vector):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _plain(value):
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


_TRANSITION_FIELDS = {
    "schema_version",
    "episode",
    "step",
    "obs",
    "state",
    "action",
    "reward",
    "terminated",
    "truncated",
    "next_obs",
    "next_state",
    "setpoint",
    "disturbance",
    "info",
}


__all__ = [
    "TRANSITION_SCHEMA_VERSION",
    "Transition",
    "TransitionDataset",
]
