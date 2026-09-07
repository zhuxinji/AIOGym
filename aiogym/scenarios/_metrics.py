"""Reward-owned episode metric implementations for built-in scenarios."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def regulation_episode_metrics(
    env,
    episode,
    *,
    output_scale=None,
    output_indices=None,
    settling_tolerance=None,
) -> dict[str, float]:
    trace = _episode_trace(
        env,
        episode,
        output_scale=output_scale,
        output_indices=output_indices,
    )
    metrics = dict(trace["metrics"])
    matrix = np.asarray(trace["errors"], dtype=float)
    if matrix.size:
        absolute = np.abs(matrix)
        tolerance = _normalized_settling_tolerance(
            settling_tolerance,
            trace["output_scale"],
        )
        metrics.update(
            {
                "tracking_iae": float(np.sum(absolute) * trace["dt"]),
                "tracking_ise": float(np.sum(matrix**2) * trace["dt"]),
                "tracking_itae": float(
                    sum(
                        (index + 1) * trace["dt"] * float(np.sum(row)) * trace["dt"]
                        for index, row in enumerate(absolute)
                    )
                ),
                "overshoot": float(np.max(np.maximum(matrix, 0.0))),
                "final_error": float(np.max(absolute[-1])),
                "settling_time": _settling_time(
                    absolute,
                    trace["dt"],
                    tolerance=tolerance,
                ),
            }
        )
        event_steps = sorted(
            int(step)
            for step in episode.reset_info["episode_spec"]["disturbance_schedule"]
        )
        if event_steps:
            first = min(event_steps)
            last = max(event_steps)
            metrics["disturbance_iae"] = float(np.sum(absolute[first:]) * trace["dt"])
            metrics["disturbance_ise"] = float(
                np.sum(matrix[first:] ** 2) * trace["dt"]
            )
            metrics["recovery_time"] = _recovery_time(
                absolute,
                trace["dt"],
                start_step=last,
                tolerance=tolerance,
            )
    return metrics


def _episode_trace(
    env,
    episode,
    *,
    output_scale=None,
    output_indices=None,
) -> dict[str, Any]:
    transitions = episode.transitions
    if not transitions:
        raise ValueError("episode metrics require at least one transition")
    first_output, _ = _output_reference(transitions[0])
    selected = None
    if output_indices is not None:
        selected = np.asarray(output_indices)
        if selected.ndim != 1 or selected.size == 0:
            raise ValueError(
                "output_indices must be a non-empty one-dimensional sequence"
            )
        if selected.dtype.kind not in "iu":
            raise TypeError("output_indices must contain integers")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("output_indices must not contain duplicates")
        if np.any(selected < 0) or np.any(selected >= len(first_output)):
            raise ValueError("output_indices contains an index outside the outputs")
    dt = float(env.unwrapped.control_dt)
    hours_per_step = dt * {"s": 1 / 3600, "h": 1}[getattr(env.unwrapped.model, "time_unit", "s")]
    metrics = {
        "return": float(episode.episode_return),
        "constraint_violations": 0.0,
        "termination": float(bool(transitions and transitions[-1].terminated)),
        "energy": 0.0,
        "constraint_violation_cost": 0.0,
    }
    errors = []
    scale = _output_scale(env, len(first_output), output_scale=output_scale)
    selected_scale = scale if selected is None else scale[selected]
    minimum_safety_margin = math.inf
    first_violation_step = None
    for transition in transitions:
        info = transition.info
        constraints = info["constraint_costs"]
        violated = any(float(value) > 0.0 for value in constraints.values())
        metrics["constraint_violations"] += float(violated)
        metrics["constraint_violation_cost"] += sum(
            max(0.0, float(value)) for value in constraints.values()
        )
        minimum_safety_margin = min(
            minimum_safety_margin, float(info["minimum_safety_margin"])
        )
        if violated and first_violation_step is None:
            first_violation_step = transition.step_index + 1
        metrics["energy"] += float(info["energy_kw"]) * hours_per_step
        output, reference = _output_reference(transition)
        normalized_error = (output - reference) / scale
        if selected is not None:
            normalized_error = normalized_error[selected]
        errors.append(normalized_error)
    step_count = len(transitions)
    metrics.update(
        {
            "unsafe_rate": float(metrics["constraint_violations"] / step_count),
            "safe_completion": float(
                not transitions[-1].terminated and transitions[-1].truncated
            ),
            "minimum_safety_margin": float(minimum_safety_margin),
            "time_to_violation": float(
                (step_count if first_violation_step is None else first_violation_step)
                * dt
            ),
        }
    )
    return {
        "dt": dt,
        "metrics": metrics,
        "errors": errors,
        "output_scale": selected_scale,
    }


def _output_reference(transition):
    info = transition.info
    reference = info["transition_reference"]
    output = info["y"]
    output = np.asarray(output, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    if output.shape != reference.shape:
        raise ValueError("output and transition reference shapes must match")
    return output, reference


def _output_scale(env, size, *, output_scale=None):
    source = (
        env.unwrapped.model.output_scales() if output_scale is None else output_scale
    )
    values = np.asarray(source, dtype=float).reshape(-1)
    if values.shape != (size,) or not np.all(values > 0):
        raise ValueError("controlled output scales must be positive and match outputs")
    return values


def _normalized_settling_tolerance(settling_tolerance, output_scale):
    if settling_tolerance is None:
        return 0.02
    scale = np.asarray(output_scale, dtype=float).reshape(-1)
    tolerance = np.asarray(settling_tolerance, dtype=float).reshape(-1)
    if tolerance.shape != scale.shape:
        raise ValueError(
            "settling_tolerance must match the selected controlled outputs"
        )
    if not np.all(np.isfinite(tolerance)) or np.any(tolerance <= 0.0):
        raise ValueError("settling_tolerance values must be finite and positive")
    return tolerance / scale


def _settling_time(absolute_errors, dt, tolerance=0.02):
    unsettled = np.any(absolute_errors > tolerance, axis=1)
    indexes = np.flatnonzero(unsettled)
    return 0.0 if indexes.size == 0 else float((indexes[-1] + 1) * dt)


def _recovery_time(absolute_errors, dt, *, start_step, tolerance=0.02):
    remaining = np.asarray(absolute_errors[start_step:], dtype=float)
    if remaining.size == 0:
        return 0.0
    unsettled = np.any(remaining > tolerance, axis=1)
    indexes = np.flatnonzero(unsettled)
    return 0.0 if indexes.size == 0 else float((indexes[-1] + 1) * dt)


__all__ = ["regulation_episode_metrics"]
