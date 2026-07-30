"""Deterministic, namespaced random-seed derivation for episode generation."""
from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass

import numpy as np


SEED_SPLITS = ("training", "validation", "test")
SEED_COMPONENTS = (
    "plant",
    "initial_state",
    "reference",
    "disturbance",
    "sensor",
    "actuator",
    "exploration",
    "policy",
)


def seed_namespace(scope_id: str, split: str) -> str:
    """Return the canonical v1 namespace for one dataset or track split."""

    scope = _non_empty_string("scope_id", scope_id)
    selected_split = str(split)
    if selected_split not in SEED_SPLITS:
        raise ValueError(
            "seed split must be one of: " + ", ".join(SEED_SPLITS)
        )
    return f"aiogym:{scope}:{selected_split}:episodes:v1"


@dataclass(frozen=True, init=False)
class SeedTree:
    """Immutable component seeds derived from one episode identity.

    Only the namespace, base seed, and global episode index define episode
    content. ``worker_index`` is retained as placement metadata and is
    deliberately excluded from seed derivation.
    """

    base_seed: int
    namespace: str
    worker_index: int
    episode_index: int
    _component_seeds: tuple[tuple[str, int], ...]

    def __init__(
        self,
        base_seed: int,
        namespace: str,
        *,
        worker_index: int = 0,
        episode_index: int = 0,
    ) -> None:
        base = _non_negative_integer("base_seed", base_seed)
        worker = _non_negative_integer("worker_index", worker_index)
        episode = _non_negative_integer("episode_index", episode_index)
        selected_namespace = _non_empty_string("namespace", namespace)

        namespace_entropy = np.frombuffer(
            hashlib.sha256(selected_namespace.encode("utf-8")).digest(),
            dtype=np.uint32,
        )
        root = np.random.SeedSequence(
            [
                base,
                episode,
                *(int(value) for value in namespace_entropy),
            ]
        )
        children = root.spawn(len(SEED_COMPONENTS))
        component_seeds = tuple(
            (
                component,
                int(child.generate_state(1, dtype=np.uint32)[0]),
            )
            for component, child in zip(SEED_COMPONENTS, children)
        )

        object.__setattr__(self, "base_seed", base)
        object.__setattr__(self, "namespace", selected_namespace)
        object.__setattr__(self, "worker_index", worker)
        object.__setattr__(self, "episode_index", episode)
        object.__setattr__(self, "_component_seeds", component_seeds)

    @classmethod
    def for_split(
        cls,
        base_seed: int,
        scope_id: str,
        split: str,
        *,
        worker_index: int = 0,
        episode_index: int = 0,
    ) -> "SeedTree":
        """Build a tree in a canonical training/validation/test namespace."""

        return cls(
            base_seed,
            seed_namespace(scope_id, split),
            worker_index=worker_index,
            episode_index=episode_index,
        )

    @property
    def component_seeds(self) -> dict[str, int]:
        return dict(self._component_seeds)

    @property
    def namespace_hash(self) -> str:
        return hashlib.sha256(self.namespace.encode("utf-8")).hexdigest()

    def seed(self, component: str) -> int:
        """Return one named component seed."""

        try:
            return self.component_seeds[str(component)]
        except KeyError as exc:
            raise KeyError(
                f"unknown seed component {component!r}; expected one of: "
                + ", ".join(SEED_COMPONENTS)
            ) from exc

    def generator(self, component: str) -> np.random.Generator:
        """Create a local generator for one component stream."""

        return np.random.default_rng(self.seed(component))

    def metadata(self) -> dict:
        return deepcopy(
            {
                "base_seed": self.base_seed,
                "namespace": self.namespace,
                "namespace_hash": self.namespace_hash,
                "worker_index": self.worker_index,
                "episode_index": self.episode_index,
                "component_seeds": self.component_seeds,
            }
        )


def _non_empty_string(name: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _non_negative_integer(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result
