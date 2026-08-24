"""Public algorithm-backend contract used by the training workflow."""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from aiogym.core.contracts import Policy
from aiogym.core.io import jsonable


_ALGORITHM_ID = re.compile(r"[a-z][a-z0-9_-]*")


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


@runtime_checkable
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

_BACKENDS: dict[str, AlgorithmBackend] = {}
_BUILTINS_REGISTERED = False
_INSTALLED_BACKENDS_REGISTERED = False


def register_algorithm(backend: AlgorithmBackend) -> None:
    """Register one backend in the current Python process."""

    _register_builtins()
    _register_installed_backends()
    _register_backend(backend)


def register_sb3_algorithm(
    algorithm: str,
    model_class: type,
    *,
    behavior_cloning: BehaviorCloningHook | None = None,
) -> None:
    """Register one SB3 ``BaseAlgorithm`` subclass in the current process."""

    from .sb3 import SB3AlgorithmBackend

    backend = SB3AlgorithmBackend(
        id=algorithm,
        model_class=model_class,
        behavior_cloning=behavior_cloning,
    )
    register_algorithm(backend)


def list_algorithms() -> tuple[str, ...]:
    """Return all algorithm ids registered in the current Python process."""

    _register_builtins()
    _register_installed_backends()
    return tuple(sorted(_BACKENDS))


def get_algorithm(algorithm: str) -> AlgorithmBackend:
    """Resolve one registered backend or raise a precise unknown-id error."""

    _register_builtins()
    _register_installed_backends()
    key = _normalize_lookup_id(algorithm)
    try:
        return _BACKENDS[key]
    except KeyError as error:
        raise ValueError(
            f"algorithm must be one of {', '.join(sorted(_BACKENDS))}"
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
    serialized = jsonable(dict(resolved))
    if not isinstance(serialized, dict):
        raise TypeError("algorithm backend effective_kwargs must return a mapping")
    return serialized


def _register_builtins() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from .sb3 import built_in_backends
    from .rlpd import RLPDAlgorithmBackend

    for backend in (*built_in_backends(), RLPDAlgorithmBackend()):
        _register_backend(backend)
    _BUILTINS_REGISTERED = True


def _register_installed_backends() -> None:
    global _INSTALLED_BACKENDS_REGISTERED
    if _INSTALLED_BACKENDS_REGISTERED:
        return
    for entry_point in entry_points(group="aiogym.algorithms"):
        backend = entry_point.load()
        if not isinstance(backend, AlgorithmBackend):
            raise TypeError(
                "aiogym.algorithms entry point must expose one complete "
                f"AlgorithmBackend instance: {entry_point.name}"
            )
        if entry_point.name != backend.id:
            raise ValueError(
                "aiogym.algorithms entry-point name must equal backend id: "
                f"{entry_point.name} != {backend.id}"
            )
        _register_backend(backend)
    _INSTALLED_BACKENDS_REGISTERED = True


def _register_backend(backend: AlgorithmBackend) -> None:
    if not isinstance(backend, AlgorithmBackend):
        raise TypeError(
            "backend must implement the complete AIO-Gym AlgorithmBackend contract"
        )
    algorithm_id = backend.id
    if (
        not isinstance(algorithm_id, str)
        or _ALGORITHM_ID.fullmatch(algorithm_id) is None
    ):
        raise ValueError(
            "algorithm backend id must match [a-z][a-z0-9_-]*"
        )
    if backend.behavior_cloning is not None and not callable(
        backend.behavior_cloning
    ):
        raise TypeError("backend behavior_cloning must be callable or None")
    if not isinstance(backend.requires_dataset, bool):
        raise TypeError("backend requires_dataset must be bool")
    if algorithm_id in _BACKENDS:
        raise ValueError(f"algorithm backend is already registered: {algorithm_id}")
    _BACKENDS[algorithm_id] = backend


def _normalize_lookup_id(algorithm: str) -> str:
    if not isinstance(algorithm, str) or not algorithm.strip():
        raise TypeError("algorithm must be a non-empty string")
    return algorithm.lower()


__all__ = [
    "AlgorithmBackend",
    "TrainingStep",
    "TrainingStepCallback",
    "get_algorithm",
    "list_algorithms",
    "register_algorithm",
    "register_sb3_algorithm",
]
