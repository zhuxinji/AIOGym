"""Numerical control helpers shared by rewards and evaluation."""
from __future__ import annotations

import math

import numpy as np


def normalized_tracking_errors(model, y, y_sp) -> list[float]:
    """Return signed output errors divided by model-declared output scales."""

    errors = []
    scales = list(model.controlled_output_scales())
    for index, (value, setpoint) in enumerate(zip(y, y_sp)):
        scale = scales[index] if index < len(scales) else 1.0
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("controlled-output scales must be finite and positive")
        error = (float(value) - float(setpoint)) / scale
        if not math.isfinite(error):
            raise ValueError("normalized tracking errors must be finite")
        errors.append(error)
    return errors


def normalized_tracking_error_sum(model, y, y_sp) -> float:
    return float(sum(abs(error) for error in normalized_tracking_errors(model, y, y_sp)))


def normalized_action(model, action) -> np.ndarray:
    """Normalize a canonical action using only ``action_schema().bounds``.

    A missing or invalid bound uses the documented identity fallback ``(0, 1)``.
    Values are intentionally not clipped so invalid/extrapolated commands remain
    visible to reward and scorecard consumers.
    """

    values = np.asarray(model.action_vector(action), dtype=np.float64)
    rows = list(model.action_schema())
    if values.ndim != 1:
        raise ValueError("canonical action must be a one-dimensional vector")
    if values.size != len(rows):
        raise ValueError(
            "canonical action and action schema dimensions differ: "
            f"{values.size} != {len(rows)}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("canonical action values must be finite")
    normalized = []
    for value, row in zip(values, rows):
        bounds = row.get("bounds") if isinstance(row, dict) else None
        low, high = 0.0, 1.0
        if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            candidate_low, candidate_high = bounds
            if candidate_low is not None and candidate_high is not None:
                candidate_low = float(candidate_low)
                candidate_high = float(candidate_high)
                if (
                    math.isfinite(candidate_low)
                    and math.isfinite(candidate_high)
                    and candidate_high > candidate_low
                ):
                    low, high = candidate_low, candidate_high
        normalized.append((float(value) - low) / (high - low))
    result = np.asarray(normalized, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("normalized action values must be finite")
    return result


__all__ = [
    "normalized_action",
    "normalized_tracking_error_sum",
    "normalized_tracking_errors",
]
