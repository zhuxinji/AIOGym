"""Reward-owned episode metric implementations for built-in scenarios."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


TANK3_OUTPUT_INDICES = (2, 5)
TANK3_TRACKING_TOLERANCES = (0.01, 0.5)
TANK3_TRACKING_WEIGHTS = (1.0, 1.0)


def regulation_episode_metrics(env, episode) -> dict[str, float]:
    trace = _episode_trace(env, episode)
    metrics = dict(trace["metrics"])
    matrix = np.asarray(trace["errors"], dtype=float)
    if matrix.size:
        absolute = np.abs(matrix)
        metrics.update(
            {
                "tracking_iae": float(np.sum(absolute) * trace["dt"]),
                "tracking_ise": float(np.sum(matrix**2) * trace["dt"]),
                "tracking_itae": float(
                    sum(
                        (index + 1)
                        * trace["dt"]
                        * float(np.sum(row))
                        * trace["dt"]
                        for index, row in enumerate(absolute)
                    )
                ),
                "overshoot": float(np.max(np.maximum(matrix, 0.0))),
                "final_error": float(np.max(absolute[-1])),
                "settling_time": _settling_time(absolute, trace["dt"]),
            }
        )
        event_steps = sorted(
            int(step)
            for step in episode.reset_info["episode_spec"][
                "disturbance_schedule"
            ]
        )
        if event_steps:
            first = min(event_steps)
            last = max(event_steps)
            metrics["disturbance_iae"] = float(
                np.sum(absolute[first:]) * trace["dt"]
            )
            metrics["recovery_time"] = _recovery_time(
                absolute,
                trace["dt"],
                start_step=last,
            )
    return metrics


def economic_episode_metrics(env, episode) -> dict[str, float]:
    trace = _episode_trace(env, episode)
    metrics = dict(trace["metrics"])
    metrics.update(
        {
            "economic_objective": float(episode.episode_return),
            "net_economic_value": float(episode.episode_return),
            "product_value": float(trace["product_value"]),
            "energy_cost": float(trace["energy_cost"]),
            "production_volume_m3": float(trace["production_volume"]),
            "energy_kwh": float(metrics["energy"]),
        }
    )
    return metrics


def tank3_regulation_episode_metrics(env, episode) -> dict[str, float]:
    trace = _episode_trace(env, episode)
    raw_matrix = np.asarray(trace["raw_errors"], dtype=float)
    action_matrix = np.asarray(trace["applied_actions"], dtype=float)
    previous_action_matrix = np.asarray(
        trace["previous_applied_actions"], dtype=float
    )
    if raw_matrix.size == 0 or action_matrix.size == 0:
        raise ValueError("tank3-regulation requires at least one complete transition")

    indices = np.asarray(TANK3_OUTPUT_INDICES, dtype=int)
    tolerances = np.asarray(TANK3_TRACKING_TOLERANCES, dtype=float)
    weights = np.asarray(TANK3_TRACKING_WEIGHTS, dtype=float)
    selected = raw_matrix[:, indices]
    normalized = selected / tolerances
    absolute = np.abs(normalized)
    weighted = absolute * weights
    slew_limits = np.asarray(env.unwrapped.model.action_slew_limits(), dtype=float)
    action_delta = np.abs(action_matrix - previous_action_matrix)
    slew_ratio = action_delta / slew_limits
    heater = action_matrix[:, 4]
    metrics = dict(trace["metrics"])
    metrics.update(
        {
            "tank3_tracking_iae": float(np.sum(weighted) * trace["dt"]),
            "tank3_level_iae": float(np.sum(np.abs(selected[:, 0])) * trace["dt"]),
            "tank3_temperature_iae": float(
                np.sum(np.abs(selected[:, 1])) * trace["dt"]
            ),
            "upstream_level_iae": float(
                np.sum(np.abs(raw_matrix[:, :2])) * trace["dt"]
            ),
            "upstream_level_max_error": float(
                np.max(np.abs(raw_matrix[:, :2]))
            ),
            "upstream_temperature_deviation_iae": float(
                np.sum(np.abs(raw_matrix[:, 3:5])) * trace["dt"]
            ),
            "final_tank3_level_error_m": float(abs(selected[-1, 0])),
            "final_tank3_temperature_error_degC": float(abs(selected[-1, 1])),
            "action_slew_violation_count": float(
                np.sum(action_delta > slew_limits + 1e-6)
            ),
            "maximum_action_slew_ratio": float(np.max(slew_ratio)),
            "heater_low_saturation_fraction": float(np.mean(heater <= 1e-6)),
            "heater_high_saturation_fraction": float(
                np.mean(heater >= 1.0 - 1e-6)
            ),
            "heater_total_variation": float(np.sum(np.abs(action_delta[:, 4]))),
            "outlet_valve_total_variation": float(
                np.sum(np.abs(action_delta[:, 3]))
            ),
            "overshoot": float(np.max(np.maximum(normalized, 0.0))),
            "final_error": float(np.max(absolute[-1])),
            "settling_time": _settling_time(
                absolute, trace["dt"], tolerance=1.0
            ),
        }
    )
    return metrics


def _episode_trace(env, episode) -> dict[str, Any]:
    transitions = episode.transitions
    if not transitions:
        raise ValueError("episode metrics require at least one transition")
    dt = float(env.unwrapped.control_dt)
    metrics = {
        "return": float(episode.episode_return),
        "constraint_violations": 0.0,
        "termination": float(bool(transitions and transitions[-1].terminated)),
        "energy": 0.0,
        "constraint_violation_cost": 0.0,
    }
    errors = []
    raw_errors = []
    applied_actions = []
    previous_applied_actions = []
    product_value = 0.0
    energy_cost = 0.0
    production_volume = 0.0
    minimum_safety_margin = math.inf
    first_violation_step = None
    for transition in transitions:
        info = transition.info
        applied_action = info["applied_action"]
        previous_applied_action = info["previous_applied_action"]
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
        metrics["energy"] += float(info["energy_kw"]) * dt / 3600.0
        reward_terms = info["reward_terms"]
        applied_actions.append(np.asarray(applied_action, dtype=float))
        previous_applied_actions.append(
            np.asarray(previous_applied_action, dtype=float)
        )
        if "product_value" in reward_terms:
            product_value += float(reward_terms["product_value"])
        if "energy_cost" in reward_terms:
            energy_cost -= float(reward_terms["energy_cost"])
        if "product_flow_m3s" in info:
            production_volume += float(info["product_flow_m3s"]) * dt
        output, reference = _output_reference(transition)
        scale = _output_scale(env, len(reference))
        errors.append((output - reference) / scale)
        raw_errors.append(output - reference)
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
        "raw_errors": raw_errors,
        "applied_actions": applied_actions,
        "previous_applied_actions": previous_applied_actions,
        "product_value": product_value,
        "energy_cost": energy_cost,
        "production_volume": production_volume,
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


def _output_scale(env, size):
    values = np.asarray(
        env.unwrapped.model.controlled_output_scales(), dtype=float
    ).reshape(-1)
    if values.shape != (size,) or not np.all(values > 0):
        raise ValueError("controlled output scales must be positive and match outputs")
    return values


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


__all__ = [
    "TANK3_OUTPUT_INDICES",
    "TANK3_TRACKING_TOLERANCES",
    "TANK3_TRACKING_WEIGHTS",
    "economic_episode_metrics",
    "regulation_episode_metrics",
    "tank3_regulation_episode_metrics",
]
