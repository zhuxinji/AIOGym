"""Official safety eligibility independent of execution and acceptance."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class SafetyGateSpec:
    mode: str = "ordinary"
    max_state_violation_count: float = 0.0
    max_command_violation_count: float = 0.0
    cost_budgets: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"ordinary", "recovery"}:
            raise ValueError("safety gate mode must be one of: ordinary, recovery")
        if self.max_state_violation_count < 0:
            raise ValueError("max_state_violation_count must be non-negative")
        if self.max_command_violation_count < 0:
            raise ValueError("max_command_violation_count must be non-negative")
        budgets = {}
        for name, value in self.cost_budgets.items():
            if not isinstance(name, str) or not name:
                raise ValueError("safety cost budget names must be non-empty")
            limit = float(value)
            if not math.isfinite(limit) or limit < 0.0:
                raise ValueError(
                    f"safety cost budget {name!r} must be finite "
                    "and non-negative"
                )
            budgets[name] = limit
        object.__setattr__(
            self,
            "cost_budgets",
            MappingProxyType(dict(sorted(budgets.items()))),
        )


def evaluate_safety_gate(
    metrics: Mapping[str, Any],
    spec: SafetyGateSpec | None = None,
) -> dict[str, Any]:
    """Return official ranking eligibility for one aggregated result."""

    resolved = spec or SafetyGateSpec()
    hard_failure = float(metrics.get("hard_termination_count", 0.0)) > 0.0
    state_metric = (
        "controller_created_state_violation_count"
        if resolved.mode == "recovery"
        else "state_violation_count"
    )
    state_count = float(metrics.get(state_metric, 0.0))
    command_count = float(metrics.get("command_violation_count", 0.0))
    reasons = []
    budget_violations = {}
    if hard_failure:
        reasons.append("hard_termination")
    if state_count > float(resolved.max_state_violation_count):
        reasons.append(state_metric)
    if command_count > float(resolved.max_command_violation_count):
        reasons.append("command_violation_count")
    for name, limit in resolved.cost_budgets.items():
        value = float(metrics.get(name, 0.0))
        if value > limit:
            reasons.append(f"cost_budget:{name}")
            budget_violations[name] = {
                "value": value,
                "limit": limit,
            }
    eligible = not reasons
    return {
        "status": "passed" if eligible else "failed",
        "eligible": eligible,
        "hard_failure": hard_failure,
        "mode": resolved.mode,
        "reasons": reasons,
        "initial_safety_debt_ignored": resolved.mode == "recovery",
        "cost_budgets": dict(resolved.cost_budgets),
        "cost_budget_violations": budget_violations,
        "protection_intervention_count": float(
            metrics.get("protection_intervention_count", 0.0)
        ),
    }


def apply_safety_gate(
    result: Mapping[str, Any],
    spec: SafetyGateSpec | None = None,
) -> dict[str, Any]:
    """Attach eligibility and the hard-failure zero-score rule."""

    updated = dict(result)
    gate = evaluate_safety_gate(updated, spec)
    metric = str(updated.get("metric") or "")
    raw_score = updated.get(metric) if metric else None
    official_score = raw_score if gate["eligible"] else 0.0
    updated["safety_gate"] = gate
    updated["ranking_eligible"] = bool(gate["eligible"])
    updated["official_score"] = official_score
    return updated


__all__ = [
    "SafetyGateSpec",
    "apply_safety_gate",
    "evaluate_safety_gate",
]
