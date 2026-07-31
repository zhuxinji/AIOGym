"""Separated state, command, protection, and hard-termination metrics."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def action_bound_metrics(action, env):
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    arr = np.asarray(action, dtype=np.float64)
    below = np.maximum(low - arr, 0.0)
    above = np.maximum(arr - high, 0.0)
    severity = float(np.sum(below + above))
    return {
        "violated": severity > 0.0,
        "severity": severity,
    }


def safety_step_metrics(info, bound_metrics, dt: float):
    """Measure safety channels without deciding leaderboard eligibility."""

    cons = dict(info.get("cons_info", {}))
    state_severity = float(
        sum(max(0.0, float(value)) for value in cons.values())
    )
    state_violated = (
        bool(info.get("cons_violated", False)) or state_severity > 0.0
    )
    command_severity = float(bound_metrics.get("severity", 0.0))
    command_violated = bool(bound_metrics.get("violated", False))
    runaway = bool(info.get("runaway", False))
    hard_termination = bool(info.get("termination_reason"))
    protection_active = _protection_active(info)
    shield_active = bool(info.get("shield_intervened", False))
    shield_magnitude = float(
        info.get("shield_intervention_magnitude", 0.0)
    )
    actuator_active = bool(info.get("actuator_intervened", False))
    actuator_magnitude = float(
        info.get("actuator_command_applied_l1", 0.0)
    )
    worst = max(
        state_severity,
        command_severity,
        1.0 if runaway or hard_termination else 0.0,
    )
    return {
        "state_violation_count": 1.0 if state_violated else 0.0,
        "state_violation_duration": dt if state_violated else 0.0,
        "state_violation_severity": state_severity,
        "command_violation_count": 1.0 if command_violated else 0.0,
        "command_violation_duration": dt if command_violated else 0.0,
        "command_violation_severity": command_severity,
        "protection_intervention_count": 1.0 if protection_active else 0.0,
        "protection_intervention_duration": dt if protection_active else 0.0,
        "shield_intervention_count": 1.0 if shield_active else 0.0,
        "shield_intervention_duration": dt if shield_active else 0.0,
        "shield_intervention_magnitude": shield_magnitude,
        "actuator_intervention_count": 1.0 if actuator_active else 0.0,
        "actuator_intervention_duration": dt if actuator_active else 0.0,
        "actuator_intervention_magnitude": actuator_magnitude,
        "hard_termination_count": 1.0 if hard_termination else 0.0,
        "runaway_count": 1.0 if runaway else 0.0,
        "runaway_duration": dt if runaway else 0.0,
        "safety_margin_min": -float(worst) if worst > 0.0 else 0.0,
        # Stable aggregate names used by Track gates and artifact reports.
        "constraint_violation_count": 1.0 if state_violated else 0.0,
        "constraint_violation_duration": dt if state_violated else 0.0,
        "constraint_violation_severity": state_severity,
        "action_violation_count": 1.0 if command_violated else 0.0,
        "action_violation_duration": dt if command_violated else 0.0,
        "action_violation_severity": command_severity,
    }


class SafetyDebtTracker:
    """Separate recovery-case initial debt from controller-created violations."""

    def __init__(self, *, initial_safety_debt: bool = False) -> None:
        self.initial_safety_debt = bool(initial_safety_debt)
        self._debt_active = self.initial_safety_debt

    def observe(self, metrics: Mapping[str, float]) -> dict[str, float]:
        row = dict(metrics)
        violated = float(row.get("state_violation_count", 0.0)) > 0.0
        if self._debt_active and not violated:
            self._debt_active = False
        debt_violation = bool(self._debt_active and violated)
        controller_created = bool(violated and not debt_violation)
        row.update(
            {
                "initial_safety_debt_count": (
                    float(row["state_violation_count"])
                    if debt_violation
                    else 0.0
                ),
                "initial_safety_debt_duration": (
                    float(row["state_violation_duration"])
                    if debt_violation
                    else 0.0
                ),
                "initial_safety_debt_severity": (
                    float(row["state_violation_severity"])
                    if debt_violation
                    else 0.0
                ),
                "controller_created_state_violation_count": (
                    float(row["state_violation_count"])
                    if controller_created
                    else 0.0
                ),
                "controller_created_state_violation_duration": (
                    float(row["state_violation_duration"])
                    if controller_created
                    else 0.0
                ),
                "controller_created_state_violation_severity": (
                    float(row["state_violation_severity"])
                    if controller_created
                    else 0.0
                ),
            }
        )
        return row


def _protection_active(info):
    costs = info.get("costs", {})
    if isinstance(costs, Mapping) and float(
        costs.get("protection_intervention", 0.0)
    ) > 0.0:
        return True
    return any(
        _active(info.get(name))
        for name in (
            "protection_events",
            "hardware_interlocks_active",
            "passive_safety_events",
            "heater_interlocked",
            "temperature_trip_active",
            "low_level_interlock_active",
        )
    )


def _active(value):
    if isinstance(value, Mapping):
        return any(_active(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_active(item) for item in value)
    return bool(value)


__all__ = [
    "SafetyDebtTracker",
    "action_bound_metrics",
    "safety_step_metrics",
]
