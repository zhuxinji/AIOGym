"""Reward-independent transition quantities."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Mapping, Sequence

import numpy as np

from .._internal.control_math import (
    normalized_action,
    normalized_tracking_error_sum,
    normalized_tracking_errors,
)


RAW_COST_CHANNELS = (
    "service_shortfall",
    "soft_safety",
    "hard_safety",
    "protection_intervention",
)


@dataclass(frozen=True)
class TransitionTerms:
    """Raw transition quantities shared by every reward definition."""

    state: tuple[float, ...]
    action: Any
    next_state: tuple[float, ...]
    previous_action: Any
    terminated: bool
    termination_reason: str | None
    info: Mapping[str, Any]


def transition_terms(
    model,
    state: Sequence[float],
    action,
    next_state: Sequence[float],
    *,
    setpoint: Sequence[float],
    disturbance: Mapping[str, Any],
    previous_action,
    terminate_on_runaway: bool,
    dt: float,
    economic_config: Mapping[str, Any] | None,
) -> TransitionTerms:
    """Evaluate trajectory-derived terms without changing environment state."""

    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")

    x = model.state_vector(state)
    x_next = model.state_vector(next_state)
    action = model.action_vector(action)
    previous_action = model.action_vector(previous_action)
    env = dict(disturbance)
    y_sp = [float(value) for value in setpoint]
    out = model.outputs(x_next)
    levels = list(out["levels"])
    temps = list(out["temps"])
    y = list(out["y"])

    track = normalized_tracking_error_sum(model, y, y_sp)
    (
        tracking_weighted_error_cost,
        tracking_error_cost,
        tracking_move_cost,
        tracking_cost,
    ) = tracking_cost_terms(
        model,
        y,
        y_sp,
        action,
        previous_action,
        [1.0] * len(y),
        0.0,
    )

    cons_info = dict(model.common_constraint_info(levels, temps))
    cons_info.update(model.process_constraint_info(x_next, levels, temps, env))
    hard_termination_resolver = getattr(model, "hard_termination_reasons", None)
    hard_termination_reasons = (
        tuple(
            str(reason)
            for reason in hard_termination_resolver(x_next, levels, temps, env)
        )
        if callable(hard_termination_resolver)
        else ()
    )
    runaway = bool(model.runaway_state(levels, temps)) or bool(
        hard_termination_reasons
    )
    process_extra = process_info(model, x_next, levels, temps, env, action)
    constraint = constraint_penalty(model, cons_info)

    action_energy_kw = float(model.action_energy_kw(action, x_next, env))
    terminated = bool(hard_termination_reasons) or bool(
        terminate_on_runaway and runaway
    )
    termination_reason = (
        hard_termination_reasons[0]
        if hard_termination_reasons
        else "runaway"
        if terminated
        else None
    )
    info = {
        "track": track,
        "constraint": constraint,
        "tracking_cost": tracking_cost,
        "tracking_return": -tracking_cost,
        "tracking_error_cost": tracking_error_cost,
        "tracking_move_cost": tracking_move_cost,
        "energy_kw": action_energy_kw,
        "runaway": runaway,
        "cons_info": cons_info,
        "cons_violated": any(value > 0 for value in cons_info.values()),
        "levels": levels,
        "temps": temps,
        "y": y,
        "y_sp": y_sp,
        "safety_events": list(hard_termination_reasons),
    }
    if process_extra:
        info.update(process_extra)
    if termination_reason is not None:
        info["termination_reason"] = termination_reason

    return TransitionTerms(
        state=tuple(x),
        action=action,
        next_state=tuple(x_next),
        previous_action=previous_action,
        terminated=terminated,
        termination_reason=termination_reason,
        info=info,
    )


def raw_cost_channels(
    model,
    terms: TransitionTerms,
    *,
    disturbance: Mapping[str, Any],
    dt: float,
) -> dict[str, float]:
    """Return reward-independent service, safety, and protection channels."""

    dt = float(dt)
    production = 0.0
    production_resolver = getattr(model, "production", None)
    if callable(production_resolver):
        production = float(
            production_resolver(terms.next_state, terms.action, disturbance)
        )
    service_shortfall = 0.0
    shortfall_resolver = getattr(model, "product_flow_shortfall", None)
    if callable(shortfall_resolver):
        service_shortfall = float(shortfall_resolver(production))

    info = terms.info
    protection_active = any(
        _active_indicator(info.get(name))
        for name in (
            "protection_events",
            "hardware_interlocks_active",
            "passive_safety_events",
            "heater_interlocked",
            "temperature_trip_active",
            "low_level_interlock_active",
        )
    )
    costs = {
        "service_shortfall": max(0.0, service_shortfall) * dt,
        "soft_safety": max(0.0, float(info["constraint"])) * dt,
        "hard_safety": 1.0
        if bool(info["runaway"]) or bool(info["safety_events"])
        else 0.0,
        "protection_intervention": dt if protection_active else 0.0,
    }
    if not all(np.isfinite(value) and value >= 0.0 for value in costs.values()):
        raise ValueError("raw reward cost channels must be finite and non-negative")
    return costs


def _active_indicator(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_active_indicator(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_active_indicator(item) for item in value)
    return bool(value)


def tracking_cost_terms(model, y, y_sp, action, previous_action, q_y, r_move):
    errors = normalized_tracking_errors(model, y, y_sp)
    squared_errors = [error * error for error in errors]
    error_cost = float(sum(squared_errors))
    weighted_error_cost = 0.0
    for i, squared_error in enumerate(squared_errors):
        weight = float(q_y[i]) if i < len(q_y) else 1.0
        weighted_error_cost += weight * squared_error

    u = normalized_action(model, action)
    u_previous = normalized_action(model, previous_action)
    move_cost = 0.0
    if u.shape == u_previous.shape:
        move_cost = float(np.sum((u - u_previous) ** 2))

    tracking_cost = weighted_error_cost + float(r_move) * move_cost
    return (
        float(weighted_error_cost),
        float(error_cost),
        float(move_cost),
        float(tracking_cost),
    )


def process_info(model, state, levels, temps, disturbance, action):
    resolver = model.process_info
    args = (state, levels, temps, disturbance)
    try:
        inspect.signature(resolver).bind(*args, action)
    except (TypeError, ValueError):
        return resolver(*args)
    return resolver(*args, action)


def constraint_penalty(model, cons_info):
    height_max = list(model.height_max)
    level_scale = 0.1 * max(max(height_max), 1e-9) if height_max else 0.1
    scales = {"temp_high": 10.0, "temp_trip": 10.0}
    scales.update(model.constraint_penalty_scales())
    total = 0.0
    for key, value in cons_info.items():
        violation = max(0.0, float(value))
        scale = level_scale if key.startswith("level_") else scales.get(key, 1.0)
        total += violation / max(scale, 1e-9)
    return float(total)
