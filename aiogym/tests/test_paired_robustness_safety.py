from __future__ import annotations

import pytest

from aiogym.evaluation.metric_catalog import (
    METRIC_DIRECTIONS,
)
from aiogym.evaluation.metrics.robustness import (
    directional_degradation,
    paired_robustness_summary,
    paired_seed_metadata,
)
from aiogym.evaluation.results import paired_robustness_report
from aiogym.evaluation.metrics.safety import (
    SafetyDebtTracker,
    safety_step_metrics,
)
from aiogym.evaluation.safety_gate import SafetyGateSpec, apply_safety_gate


def test_robustness_requires_complete_pairs():
    with pytest.raises(ValueError, match="complete nominal/shifted pairs"):
        paired_robustness_summary(
            [{"seed": 3, "cost": 1.0}, {"seed": 4, "cost": 2.0}],
            [{"seed": 3, "cost": 1.5}],
            ["cost"],
            {"cost": "minimize"},
        )


def test_robustness_pairs_share_base_randomness():
    pairs = paired_seed_metadata([11, 12], pair_id="plant-shift")
    for seed in (11, 12):
        members = [row for row in pairs if row["base_seed"] == seed]
        assert {row["pair_role"] for row in members} == {
            "nominal",
            "shifted",
        }
        assert {row["environment_seed"] for row in members} == {seed}
        assert {row["pair_id"] for row in members} == {"plant-shift"}


def test_cost_degradation_direction():
    assert directional_degradation(10.0, 13.0, "minimize") == 3.0
    assert directional_degradation(10.0, 7.0, "minimize") == -3.0
    summary = paired_robustness_summary(
        [
            {"seed": 1, "cost": 10.0},
            {"seed": 2, "cost": 10.0},
            {"seed": 3, "cost": 10.0},
        ],
        [
            {"seed": 1, "cost": 11.0},
            {"seed": 2, "cost": 12.0},
            {"seed": 3, "cost": 14.0},
        ],
        ["cost"],
        {"cost": "minimize"},
    )
    stats = summary["metrics"]["cost"]
    assert stats["median"] == 2.0
    assert stats["p90"] == pytest.approx(3.6)
    assert stats["worst"] == 4.0
    assert stats["cvar"] == 4.0


def test_utility_degradation_direction():
    assert directional_degradation(10.0, 7.0, "maximize") == 3.0
    assert directional_degradation(10.0, 13.0, "maximize") == -3.0


def test_hard_safety_failure_sets_ineligible():
    result = apply_safety_gate(
        {
            "metric": "profit",
            "profit": 250.0,
            "hard_termination_count": 1.0,
        }
    )
    assert result["ranking_eligible"] is False
    assert result["official_score"] == 0.0
    assert result["safety_gate"]["hard_failure"] is True


def test_initial_recovery_debt_is_not_counted_as_controller_created():
    tracker = SafetyDebtTracker(initial_safety_debt=True)
    inherited = tracker.observe(_state_safety_metrics(2.0))
    cleared = tracker.observe(_state_safety_metrics(0.0))
    created = tracker.observe(_state_safety_metrics(1.0))

    assert inherited["initial_safety_debt_count"] == 1.0
    assert inherited["controller_created_state_violation_count"] == 0.0
    assert cleared["initial_safety_debt_count"] == 0.0
    assert created["initial_safety_debt_count"] == 0.0
    assert created["controller_created_state_violation_count"] == 1.0

    gated = apply_safety_gate(
        {
            "metric": "regulation_cost",
            "regulation_cost": 4.0,
            **inherited,
        },
        SafetyGateSpec(mode="recovery"),
    )
    assert gated["ranking_eligible"] is True


def test_protection_intervention_is_reported_separately():
    metrics = safety_step_metrics(
        {
            "costs": {"protection_intervention": 0.5},
            "cons_info": {},
        },
        {"violated": False, "severity": 0.0},
        0.5,
    )
    assert metrics["protection_intervention_count"] == 1.0
    assert metrics["protection_intervention_duration"] == 0.5
    assert metrics["state_violation_count"] == 0.0
    assert metrics["command_violation_count"] == 0.0
    assert metrics["hard_termination_count"] == 0.0


def test_paired_results_are_included_in_evaluation_report():
    nominal_metrics = {
        metric: 1.0 for metric in METRIC_DIRECTIONS
    }
    shifted_metrics = {
        metric: 2.0 for metric in METRIC_DIRECTIONS
    }
    common = {
        "pair_id": "shift-a",
        "base_case_id": "case-a",
        "controller_name": "PID",
    }
    report = paired_robustness_report(
        [
            {
                **common,
                "pair_role": "nominal",
                "episode_metrics": [{"seed": 9, **nominal_metrics}],
            },
            {
                **common,
                "pair_role": "shifted",
                "episode_metrics": [{"seed": 9, **shifted_metrics}],
            },
        ]
    )
    assert report[0]["pair_count"] == 1
    profit = report[0]["metrics"]["profit"]
    assert profit["direction"] == METRIC_DIRECTIONS["profit"]
    assert profit["worst"] == -1.0


def _state_safety_metrics(severity: float) -> dict[str, float]:
    return safety_step_metrics(
        {
            "cons_violated": severity > 0.0,
            "cons_info": {"temperature": severity},
        },
        {"violated": False, "severity": 0.0},
        0.5,
    )
