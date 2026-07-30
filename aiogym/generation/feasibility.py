"""Shared bounded-rejection helpers for physics-aware episode samplers."""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, TypeVar

import numpy as np


T = TypeVar("T")


class FeasibilitySamplingError(RuntimeError):
    """Raised when a bounded sampler cannot produce a feasible candidate."""


def rejection_sample(
    component: str,
    candidate_factory: Callable[[], T],
    feasibility: Callable[[T], bool | Mapping[str, Any]],
    *,
    max_attempts: int = 64,
) -> T:
    """Return the first feasible candidate or report the rejected component."""

    if not isinstance(max_attempts, int) or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    last_reason = "feasibility predicate returned false"
    for _ in range(max_attempts):
        candidate = candidate_factory()
        result = feasibility(candidate)
        if isinstance(result, Mapping):
            if bool(result.get("feasible", False)):
                return candidate
            reasons = result.get(
                "infeasible_reasons",
                result.get("reasons", "unspecified infeasibility"),
            )
            last_reason = repr(reasons)
        elif bool(result):
            return candidate
        else:
            last_reason = "feasibility predicate returned false"
    raise FeasibilitySamplingError(
        f"failed to sample feasible {component!r} after "
        f"{max_attempts} attempts; last rejection: {last_reason}"
    )


def require_finite_vector(
    component: str,
    values: Sequence[float],
    *,
    expected_length: int | None = None,
) -> list[float]:
    """Resolve a sequence to finite floats and optionally check its length."""

    resolved = [float(value) for value in values]
    if expected_length is not None and len(resolved) != expected_length:
        raise ValueError(
            f"{component} must contain {expected_length} values, "
            f"got {len(resolved)}"
        )
    if any(not math.isfinite(value) for value in resolved):
        raise ValueError(f"{component} must contain only finite values")
    return resolved


def correlated_equilibrium_perturbation(
    model,
    equilibrium: Sequence[float],
    rng: np.random.Generator,
    *,
    level_std: float,
    temperature_std: float,
    level_margin: float,
    temperature_margin: float = 1.0,
) -> list[float]:
    """Perturb a three-tank equilibrium with shared physical modes.

    The state ordering of both cascade models is level/temperature interleaved.
    Shared inventory and thermal modes create correlation across tanks, while
    smaller local modes retain useful state diversity.
    """

    state = np.asarray(
        require_finite_vector(
            "equilibrium",
            equilibrium,
            expected_length=6,
        ),
        dtype=np.float64,
    )
    if level_std <= 0.0 and temperature_std <= 0.0:
        return state.tolist()

    level_common = float(rng.normal(0.0, level_std))
    level_local = rng.normal(0.0, 0.35 * level_std, size=3)
    thermal_common = float(rng.normal(0.0, temperature_std))
    thermal_gradient = float(rng.normal(0.0, 0.35 * temperature_std))
    thermal_local = rng.normal(0.0, 0.20 * temperature_std, size=3)

    candidate = state.copy()
    for index in range(3):
        candidate[2 * index] += level_common + float(level_local[index])
        candidate[2 * index + 1] += (
            thermal_common
            + (index - 1) * thermal_gradient
            + float(thermal_local[index])
        )

    schema = model.state_schema()
    for index, row in enumerate(schema):
        bounds = row.get("bounds")
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            continue
        lower, upper = bounds
        margin = level_margin if index % 2 == 0 else temperature_margin
        if lower is not None:
            candidate[index] = max(
                candidate[index],
                float(lower) + float(margin),
            )
        if upper is not None:
            candidate[index] = min(
                candidate[index],
                float(upper) - float(margin),
            )
    return require_finite_vector(
        "correlated initial state",
        candidate.tolist(),
        expected_length=6,
    )


def initial_disturbance_schedule(
    model,
    environment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolve every model disturbance at step zero."""

    values = model.runtime_env(
        {
            **model.disturbance_defaults(),
            **dict(environment or {}),
        }
    )
    return [
        {
            "at_step": 0,
            "name": str(name),
            "value": deepcopy(value),
            "kind": "initial",
        }
        for name, value in sorted(values.items())
    ]


def bounded_disturbance_value(
    model,
    name: str,
    value: float,
) -> float:
    """Clip one numeric disturbance to the model-declared bounds."""

    rows = {
        str(row["name"]): row
        for row in model.disturbance_schema()
        if row.get("kind") != "setpoint"
    }
    if name not in rows:
        raise KeyError(f"unknown disturbance {name!r}")
    resolved = float(value)
    bounds = rows[name].get("bounds")
    if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
        lower, upper = bounds
        if lower is not None:
            resolved = max(resolved, float(lower))
        if upper is not None:
            resolved = min(resolved, float(upper))
    if not math.isfinite(resolved):
        raise ValueError(f"disturbance {name!r} must be finite")
    return resolved


__all__ = [
    "FeasibilitySamplingError",
    "bounded_disturbance_value",
    "correlated_equilibrium_perturbation",
    "initial_disturbance_schedule",
    "rejection_sample",
    "require_finite_vector",
]
