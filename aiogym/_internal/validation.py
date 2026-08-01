"""Small reusable validators for public configuration and runtime options."""
from __future__ import annotations

import math
from numbers import Integral
from collections.abc import Iterable


def positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def positive_float(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def nonnegative_float(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def seed_sequence(
    name: str,
    values: Iterable[int],
    *,
    require_unique: bool = True,
    nonempty: bool = True,
) -> tuple[int, ...]:
    """Normalize one ordered sequence of non-negative integer seeds."""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of integers")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of integers") from exc
    if nonempty and not raw:
        raise ValueError(f"{name} must not be empty")
    resolved = tuple(nonnegative_int(name, value) for value in raw)
    if require_unique and len(set(resolved)) != len(resolved):
        raise ValueError(f"{name} must contain unique seeds")
    return resolved
