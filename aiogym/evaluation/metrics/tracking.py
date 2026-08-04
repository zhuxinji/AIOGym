"""Setpoint-tracking metrics for evaluation."""
from __future__ import annotations

from aiogym._internal.control_math import (
    normalized_tracking_error_sum,
    normalized_tracking_errors,
)


def tracking_step_metrics(info, setpoint, time_sec: float, dt: float, env):
    y = list(info["y"])
    y_sp = list(setpoint.get("y_sp") or env.model.default_setpoint_vector())
    raw_errors = raw_tracking_errors(y, y_sp)
    normalized_errors = normalized_tracking_errors(env.model, y, y_sp)
    normalized_abs_errors = [abs(err) for err in normalized_errors]
    normalized_error_cost = sum(err * err for err in normalized_errors)
    move_cost = float(info.get("tracking_move_cost", 0.0) or 0.0)
    cost = float(info.get("tracking_cost", normalized_error_cost + move_cost) or 0.0)
    normalized_tol = 0.02
    output_rows = list(env.model.controlled_output_schema())
    raw_by_output = {}
    for i, error in enumerate(raw_errors):
        row = output_rows[i] if i < len(output_rows) else {}
        name = str(row.get("name") or f"y{i}")
        unit = str(row.get("unit") or "")
        raw_by_output[name] = {
            "unit": unit,
            "mse_integral": float(error * error * dt),
            "iae": float(abs(error) * dt),
            "ise": float(error * error * dt),
            "itae": float(time_sec * abs(error) * dt),
            "overshoot": float(max(0.0, error)),
            "settled": abs(normalized_errors[i]) <= normalized_tol,
        }
    return {
        "tracking_cost": float(cost),
        "tracking_return": float(-cost),
        "tracking_error_cost": float(info.get("tracking_error_cost", normalized_error_cost) or 0.0),
        "tracking_move_cost": float(move_cost),
        "tracking_iae": float(sum(normalized_abs_errors) * dt),
        "tracking_mse": float(
            normalized_error_cost / max(len(normalized_errors), 1) * dt
        ),
        "tracking_ise": float(normalized_error_cost * dt),
        "tracking_itae": float(time_sec * sum(normalized_abs_errors) * dt),
        "tracking_overshoot": float(max(0.0, max(normalized_errors, default=0.0))),
        "tracking_settled": all(
            abs(error) <= normalized_tol for error in normalized_errors
        ),
        "tracking_raw_by_output": raw_by_output,
    }


def raw_tracking_errors(y, y_sp):
    """Return signed controlled-output errors without range normalization."""

    return [float(value) - float(setpoint) for value, setpoint in zip(y, y_sp)]


__all__ = [
    "normalized_tracking_error_sum",
    "normalized_tracking_errors",
    "raw_tracking_errors",
    "tracking_step_metrics",
]
