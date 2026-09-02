"""Small numeric primitives shared by the dependency-free SVG renderers."""

from __future__ import annotations

import math

import numpy as np


def plot_range(
    values,
    *,
    include_zero: bool,
    padding_fraction: float,
    floor_zero: bool = False,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return 0.0, 1.0
    if not np.isfinite(array).all():
        raise ValueError("plot values must be finite")
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if include_zero:
        minimum = min(0.0, minimum)
        maximum = max(0.0, maximum)
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-15):
        padding = max(1.0, abs(minimum) * 0.05)
    else:
        padding = float(padding_fraction) * (maximum - minimum)
    lower = minimum - padding
    if floor_zero and include_zero and minimum >= 0.0:
        lower = 0.0
    return lower, maximum + padding


def map_value(value, source_min, source_max, target_min, target_max):
    return target_min + (float(value) - source_min) / (
        source_max - source_min
    ) * (target_max - target_min)


__all__ = ["map_value", "plot_range"]
