"""Internal contracts and built-in implementations for the training workflow."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from aiogym.core.contracts import Policy
from aiogym.core.io import jsonable


@dataclass(frozen=True)
class TrainingStep:
    """One environment transition reported by an algorithm backend."""

    step: int
    reward: float
    terminated: bool
    truncated: bool


TrainingStepCallback = Callable[[TrainingStep], None]


class BehaviorCloningHook(Protocol):
    def __call__(
        self,
        model: Any,
        observations: np.ndarray,
        actions: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
        source: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class AlgorithmBackend(Protocol):
    """Adapter implemented once by each trainable algorithm family."""

    id: str
    behavior_cloning: BehaviorCloningHook | None
    requires_dataset: bool

    def effective_kwargs(
        self,
        *,
        steps: int,
        values: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def create(
        self,
        *,
        env: Any,
        seed: int,
        algorithm_kwargs: Mapping[str, Any],
    ) -> Any: ...

    def learn(
        self,
        model: Any,
        *,
        steps: int,
        dataset: Any | None,
        on_step: TrainingStepCallback,
    ) -> int: ...

    def save(self, model: Any, payload: Path) -> None: ...

    def load(self, payload: Path, *, env: Any | None) -> Any: ...

    def policy(self, model: Any, *, checkpoint: Path) -> Policy: ...

    def runtime_metadata(self) -> Mapping[str, Any]: ...


def list_algorithms() -> tuple[str, ...]:
    """Return the built-in algorithm names supported by Python and the CLI."""

    return tuple(sorted(_BUILTIN_BACKENDS))


def get_algorithm(algorithm: str) -> AlgorithmBackend:
    """Resolve one built-in algorithm name."""

    if not isinstance(algorithm, str):
        raise TypeError("algorithm must be a built-in algorithm name (string)")
    try:
        return _BUILTIN_BACKENDS[algorithm.lower()]
    except KeyError as error:
        raise ValueError(
            f"algorithm must be one of {', '.join(list_algorithms())}"
        ) from error


def resolve_algorithm_kwargs(
    backend: AlgorithmBackend,
    *,
    steps: int,
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that backend-resolved arguments remain artifact-safe JSON."""

    resolved = backend.effective_kwargs(steps=steps, values=values)
    if not isinstance(resolved, Mapping):
        raise TypeError("algorithm backend effective_kwargs must return a mapping")
    return jsonable(dict(resolved))


# Keep the implementations below their shared contracts to avoid circular imports.
from .sb3 import built_in_backends
from .rlpd import RLPDAlgorithmBackend


_BUILTIN_BACKENDS = {
    backend.id: backend
    for backend in (*built_in_backends(), RLPDAlgorithmBackend())
}


__all__ = [
    "AlgorithmBackend",
    "TrainingStep",
    "TrainingStepCallback",
    "get_algorithm",
    "list_algorithms",
]
