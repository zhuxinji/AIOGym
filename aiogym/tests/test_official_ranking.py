from __future__ import annotations

import math

import pytest

from aiogym.benchmarks.ranking import (
    fixed_anchor_score,
    weighted_geometric_mean,
)
from aiogym.evaluation.safety_gate import (
    SafetyGateSpec,
    apply_safety_gate,
)
from aiogym.rl.hpo import _validation_utility
from aiogym.rl.validation import EligibilityAwareSelector


def test_fixed_anchor_golden_values():
    assert fixed_anchor_score(
        5.0,
        bad_utility=0.0,
        reference_utility=10.0,
    ) == 50.0
    assert fixed_anchor_score(
        15.0,
        bad_utility=0.0,
        reference_utility=10.0,
    ) == 150.0
    assert fixed_anchor_score(
        -1.0,
        bad_utility=0.0,
        reference_utility=10.0,
    ) == 0.0


def test_weighted_geometric_mean_golden_values():
    assert weighted_geometric_mean([100.0, 25.0], [1.0, 1.0]) == (
        pytest.approx(50.0)
    )
    expected = math.exp(0.25 * math.log(100.0) + 0.75 * math.log(25.0))
    assert weighted_geometric_mean([100.0, 25.0], [1.0, 3.0]) == (
        pytest.approx(expected)
    )


def test_every_gate_failure_has_zero_official_score():
    result = apply_safety_gate(
        {
            "metric": "profit",
            "profit": 25.0,
            "state_violation_count": 1.0,
        },
        SafetyGateSpec(max_state_violation_count=0.0),
    )
    assert result["ranking_eligible"] is False
    assert result["official_score"] == 0.0


def test_validation_selector_and_hpo_use_official_score():
    selector = EligibilityAwareSelector()
    lower_raw_better_official = _evaluation(
        raw=1.0,
        official=80.0,
    )
    higher_raw_worse_official = _evaluation(
        raw=2.0,
        official=40.0,
    )
    assert selector.consider(
        "official-best",
        lower_raw_better_official,
        step=1,
    )
    assert not selector.consider(
        "raw-only-best",
        higher_raw_worse_official,
        step=2,
    )
    assert selector.best.checkpoint_id == "official-best"
    assert _validation_utility(lower_raw_better_official) == 80.0
    assert _validation_utility(higher_raw_worse_official) == 40.0


def _evaluation(*, raw, official):
    return {
        "split": "validation",
        "results": [
            {
                "ranking_eligible": True,
                "safety_gate": {"reasons": []},
            }
        ],
        "aggregate": {
            "metric": "regulation_cost_rate",
            "metric_direction": "minimize",
            "metric_value": raw,
            "case_values": [raw],
            "official_score": official,
            "ranking_eligible": True,
        },
    }
