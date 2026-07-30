"""Neutral per-episode accumulator initialization."""
from __future__ import annotations


_ZERO_METRICS = (
    "return",
    "track",
    "constraint",
    "profit",
    "product_value",
    "energy_cost",
    "material_cost",
    "waste_cost",
    "production",
    "energy_kwh",
    "runtime_seconds",
    "runtime_seconds_per_step",
    "tracking_cost",
    "tracking_return",
    "tracking_error_cost",
    "tracking_move_cost",
    "tracking_mse",
    "tracking_iae",
    "tracking_ise",
    "tracking_itae",
    "tracking_overshoot",
    "tracking_settling_time",
    "constraint_violation_count",
    "constraint_violation_duration",
    "constraint_violation_severity",
    "action_violation_count",
    "action_violation_duration",
    "action_violation_severity",
    "runaway_count",
    "runaway_duration",
    "safety_margin_min",
    "service_shortfall_cost",
    "service_shortfall_count",
    "service_shortfall_duration",
    "soft_safety_cost",
    "hard_safety_cost",
    "protection_intervention_cost",
    "state_violation_count",
    "state_violation_duration",
    "state_violation_severity",
    "command_violation_count",
    "command_violation_duration",
    "command_violation_severity",
    "protection_intervention_count",
    "protection_intervention_duration",
    "shield_intervention_count",
    "shield_intervention_duration",
    "shield_intervention_magnitude",
    "actuator_intervention_count",
    "actuator_intervention_duration",
    "actuator_intervention_magnitude",
    "hard_termination_count",
    "initial_safety_debt_count",
    "initial_safety_debt_duration",
    "initial_safety_debt_severity",
    "controller_created_state_violation_count",
    "controller_created_state_violation_duration",
    "controller_created_state_violation_severity",
    "regulation_cost",
    "regulation_error_cost",
    "regulation_slew_cost",
    "regulation_effort_cost",
    "controller_solve_count",
    "controller_solver_success_count",
    "controller_solver_limited_count",
    "controller_solver_failure_count",
    "controller_fallback_count",
    "controller_degraded_count",
)


def empty_episode_totals(episode: int, seed: int) -> dict:
    totals = {name: 0.0 for name in _ZERO_METRICS}
    totals.update(
        {
            "episode": int(episode),
            "seed": int(seed),
            "service_availability": 1.0,
            "tracking_raw_by_output": {},
        }
    )
    return totals


__all__ = ["empty_episode_totals"]
