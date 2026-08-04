"""Canonical, physical-time-consistent reward definitions."""
from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .._internal.control_math import (
    normalized_action,
    normalized_tracking_errors,
)
from .scalarizers import fixed_penalty_scalarizer
from .specs import (
    RewardResult,
    RewardSpec,
    StageRewardContext,
    StageRewardOverride,
)
from .terms import raw_cost_channels, transition_terms


def canonical_stage_reward(
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
    reward_override: StageRewardOverride | None,
    reward_spec: RewardSpec,
    remaining_physical_time: float,
) -> RewardResult:
    """Evaluate one canonical regulation or economic reward transition."""

    dt = _finite_nonnegative("dt", dt, positive=True)
    remaining_physical_time = _finite_nonnegative(
        "remaining_physical_time",
        remaining_physical_time,
    )
    terms = transition_terms(
        model,
        state,
        action,
        next_state,
        setpoint=setpoint,
        disturbance=disturbance,
        previous_action=previous_action,
        terminate_on_runaway=terminate_on_runaway,
        dt=dt,
        economic_config=economic_config,
    )
    costs = raw_cost_channels(
        model,
        terms,
        disturbance=disturbance,
        dt=dt,
    )
    info = dict(terms.info)
    config = dict(economic_config or getattr(model, "economic_config", {}))

    if reward_spec.goal == "regulation":
        reward_terms, regulation_info = _regulation_reward_terms(
            model,
            terms,
            setpoint=setpoint,
            dt=dt,
            spec=reward_spec,
        )
        info.update(regulation_info)
    else:
        reward_terms, economic_info = _economic_reward_terms(
            model,
            terms,
            disturbance=disturbance,
            dt=dt,
            config=config,
            spec=reward_spec,
        )
        info.update(economic_info)

    goal_reward = float(sum(reward_terms.values()))
    applied_cost_penalties = _applied_cost_penalties(
        reward_spec,
        costs,
        economic_config=config,
    )
    reward_spec_is_canonical = bool(
        reward_spec.canonical and reward_override is None
    )
    info.update(
        {
            "goal": reward_spec.goal,
            "reward_spec_id": reward_spec.id,
            "reward_spec_version": reward_spec.version,
            "reward_spec_is_canonical": reward_spec_is_canonical,
            "goal_reward": goal_reward,
            "reward_terms": dict(reward_terms),
            "costs": dict(costs),
            "applied_cost_penalties": dict(applied_cost_penalties),
        }
    )

    if reward_override is not None:
        context = StageRewardContext(
            model=model,
            setpoint=tuple(float(value) for value in setpoint),
            disturbance=copy.deepcopy(dict(disturbance)),
            previous_action=copy.deepcopy(terms.previous_action),
            goal=reward_spec.goal,
            base_reward=goal_reward,
            terminated=terms.terminated,
            info=copy.deepcopy(info),
            reward_spec_id=reward_spec.id,
            reward_spec_is_canonical=False,
        )
        goal_reward = float(
            reward_override(
                terms.state,
                copy.deepcopy(terms.action),
                terms.next_state,
                context,
            )
        )
        if not math.isfinite(goal_reward):
            raise ValueError("custom_stage_reward must return a finite scalar")
        reward_terms = {"custom_stage_reward": goal_reward}
        reward_spec_is_canonical = False
        info["custom_stage_reward"] = True

    terminal_failure_cost = (
        remaining_physical_time * reward_spec.terminal_failure_cost_rate
        if terms.terminated
        else 0.0
    )
    scalar_reward = fixed_penalty_scalarizer(
        goal_reward,
        applied_cost_penalties,
        terminal_failure_cost=terminal_failure_cost,
    )
    info.update(
        {
            "reward_spec_is_canonical": reward_spec_is_canonical,
            "goal_reward": goal_reward,
            "reward_terms": dict(reward_terms),
            "costs": dict(costs),
            "applied_cost_penalties": dict(applied_cost_penalties),
            "terminal_failure_cost": terminal_failure_cost,
            "remaining_physical_time": remaining_physical_time,
        }
    )
    return RewardResult(
        scalar_reward=scalar_reward,
        goal_reward=goal_reward,
        reward_terms=dict(reward_terms),
        costs=dict(costs),
        applied_cost_penalties=dict(applied_cost_penalties),
        terminal_failure_cost=terminal_failure_cost,
        terminated=terms.terminated,
        termination_reason=terms.termination_reason,
        info=info,
    )


def _regulation_reward_terms(model, terms, *, setpoint, dt, spec):
    errors = np.asarray(
        normalized_tracking_errors(model, terms.info["y"], setpoint),
        dtype=np.float64,
    )
    output_weights = _output_weights(spec, errors.size)
    error_rate = (
        float(np.mean(output_weights * errors**2)) if errors.size else 0.0
    )

    action = normalized_action(model, terms.action)
    previous_action = normalized_action(model, terms.previous_action)
    if action.shape != previous_action.shape:
        raise ValueError("current and previous actions must have the same shape")
    slew_rate = (
        float(np.mean(((action - previous_action) / dt) ** 2))
        if action.size
        else 0.0
    )
    effort_rate = float(np.mean(action**2)) if action.size else 0.0

    error_cost = dt * float(spec.term_weights["tracking_error"]) * error_rate
    slew_cost = dt * float(spec.term_weights["slew"]) * slew_rate
    effort_cost = dt * float(spec.term_weights["effort"]) * effort_rate
    regulation_cost = error_cost + slew_cost + effort_cost
    return (
        {
            "tracking_error": -float(error_cost),
            "slew": -float(slew_cost),
            "effort": -float(effort_cost),
        },
        {
            "regulation_cost": float(regulation_cost),
            "regulation_error_cost": float(error_cost),
            "regulation_slew_cost": float(slew_cost),
            "regulation_effort_cost": float(effort_cost),
            "regulation_error_rate": float(error_rate),
            "regulation_slew_rate": float(slew_rate),
            "regulation_effort_rate": float(effort_rate),
        },
    )


def _economic_reward_terms(
    model,
    terms,
    *,
    disturbance,
    dt,
    config,
    spec,
):
    production_rate = 0.0
    if config.get("value") == "production":
        resolver = getattr(model, "production", None)
        if callable(resolver):
            production_rate = float(
                resolver(terms.next_state, terms.action, disturbance)
            )
    product_value = (
        dt
        * float(spec.term_weights["product_value"])
        * float(config.get("w_value", 0.0))
        * production_rate
    )
    energy_cost = (
        dt
        * float(spec.term_weights["energy_cost"])
        * float(config.get("w_energy", 0.0))
        * float(terms.info["energy_kw"])
    )
    material_cost = (
        dt
        * float(spec.term_weights["material_cost"])
        * _declared_cost_rate(model, terms, disturbance, "material")
    )
    waste_cost = (
        dt
        * float(spec.term_weights["waste_cost"])
        * _declared_cost_rate(model, terms, disturbance, "waste")
    )
    reward_terms = {
        "product_value": float(product_value),
        "energy_cost": -float(energy_cost),
        "material_cost": -float(material_cost),
        "waste_cost": -float(waste_cost),
    }
    profit = float(sum(reward_terms.values()))
    return reward_terms, {
        "production": float(production_rate * dt),
        "profit": profit,
        "profit_rate": profit / dt,
        "product_value": float(product_value),
        "energy_cost": float(energy_cost),
        "material_cost": float(material_cost),
        "waste_cost": float(waste_cost),
    }


def _declared_cost_rate(model, terms, disturbance, name):
    resolver = getattr(model, f"{name}_cost_rate", None)
    if callable(resolver):
        value = resolver(terms.next_state, terms.action, disturbance)
    else:
        value = terms.info.get(f"{name}_cost_rate", 0.0)
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name}_cost_rate must be finite and non-negative")
    return value


def _applied_cost_penalties(spec, costs, *, economic_config):
    penalties = {}
    for name, raw_cost in costs.items():
        multiplier = float(spec.cost_weights.get(name, 0.0))
        if spec.goal == "economic" and name == "service_shortfall":
            multiplier *= float(economic_config.get("w_product_shortfall", 0.0))
        if spec.goal == "economic" and name == "soft_safety":
            multiplier *= float(economic_config.get("w_viol", 0.0))
        penalty = multiplier * float(raw_cost)
        if not math.isfinite(penalty) or penalty < 0.0:
            raise ValueError("applied reward cost penalties must be non-negative")
        penalties[name] = float(penalty)
    return penalties


def _output_weights(spec, size):
    configured = spec.metadata.get("output_weights", 1.0)
    if isinstance(configured, (int, float)):
        weights = np.full(size, float(configured), dtype=np.float64)
    else:
        weights = np.asarray(configured, dtype=np.float64).reshape(-1)
        if weights.size == 1:
            weights = np.full(size, float(weights[0]), dtype=np.float64)
    if weights.size != size:
        raise ValueError(
            f"reward spec output_weights has length {weights.size}; expected {size}"
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("reward spec output_weights must be finite and non-negative")
    return weights


def _finite_nonnegative(name, value, *, positive=False):
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or (positive and value == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return value
