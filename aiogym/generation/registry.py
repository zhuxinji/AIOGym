"""Registry for versioned programmatic training distributions."""
from __future__ import annotations

from collections.abc import Callable

from .specs import DistributionSpec


DistributionFactory = Callable[[], DistributionSpec]
SamplerFactory = Callable[[DistributionSpec, str], object]
_REGISTRY: dict[str, tuple[DistributionFactory, SamplerFactory]] = {}
_BUILTINS_REGISTERED = False


def register_distribution(
    distribution_id: str,
    distribution_factory: DistributionFactory,
    sampler_factory: SamplerFactory,
    *,
    replace: bool = False,
) -> None:
    key = str(distribution_id)
    if not key:
        raise ValueError("distribution_id must be non-empty")
    if not callable(distribution_factory) or not callable(sampler_factory):
        raise TypeError("distribution and sampler factories must be callable")
    if key in _REGISTRY and not replace:
        raise ValueError(f"distribution {key!r} is already registered")
    distribution = distribution_factory()
    if not isinstance(distribution, DistributionSpec):
        raise TypeError("distribution factory must return DistributionSpec")
    if distribution.distribution_id != key:
        raise ValueError("registered distribution identity mismatch")
    _REGISTRY[key] = (distribution_factory, sampler_factory)


def list_distributions() -> tuple[str, ...]:
    _ensure_builtin_distributions()
    return tuple(sorted(_REGISTRY))


def load_distribution(distribution_id: str) -> DistributionSpec:
    _ensure_builtin_distributions()
    try:
        distribution_factory, _ = _REGISTRY[str(distribution_id)]
    except KeyError as exc:
        available = ", ".join(list_distributions())
        raise FileNotFoundError(
            f"unknown training distribution ID {distribution_id!r}; "
            f"available: {available}"
        ) from exc
    distribution = distribution_factory()
    if distribution.distribution_id != distribution_id:
        raise RuntimeError("training distribution registry identity mismatch")
    return distribution


def sampler_factory_for(distribution_id: str) -> SamplerFactory:
    _ensure_builtin_distributions()
    try:
        return _REGISTRY[str(distribution_id)][1]
    except KeyError as exc:
        raise FileNotFoundError(
            f"no sampler registered for distribution {distribution_id!r}"
        ) from exc


def _ensure_builtin_distributions() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from .quadruple import (
        QuadrupleTrainingSampler,
        quadruple_training_distribution,
    )
    from .cascade import (
        CascadeTrainingSampler,
        cascade_training_distribution,
    )
    from .cascade_recirculating import (
        CascadeRecirculatingTrainingSampler,
        cascade_recirculating_training_distribution,
    )

    for level in ("L0", "L1", "L2", "L3", "L4"):
        distribution_id = (
            f"quadruple-regulation-training-{level.lower()}-v1"
        )
        register_distribution(
            distribution_id,
            lambda level=level: quadruple_training_distribution(level),
            lambda distribution, split: QuadrupleTrainingSampler(
                distribution,
                split=split,
            ),
        )
    families = (
        (
            "cascade-regulation-training",
            cascade_training_distribution,
            CascadeTrainingSampler,
        ),
        (
            "cascade-recirculating-regulation-training",
            cascade_recirculating_training_distribution,
            CascadeRecirculatingTrainingSampler,
        ),
    )
    for prefix, distribution_builder, sampler_type in families:
        for level in ("L0", "L1", "L2"):
            distribution_id = f"{prefix}-{level.lower()}-v1"
            register_distribution(
                distribution_id,
                lambda level=level, builder=distribution_builder: builder(
                    level
                ),
                lambda distribution, split, sampler=sampler_type: sampler(
                    distribution,
                    split=split,
                ),
            )
    _BUILTINS_REGISTERED = True


__all__ = [
    "DistributionFactory",
    "SamplerFactory",
    "list_distributions",
    "load_distribution",
    "register_distribution",
    "sampler_factory_for",
]
