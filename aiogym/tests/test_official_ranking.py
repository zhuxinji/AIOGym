from __future__ import annotations

import math

import pytest

from aiogym.benchmarks.ranking import (
    fixed_anchor_margin,
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
    assert fixed_anchor_margin(
        -1.0,
        bad_utility=0.0,
        reference_utility=10.0,
    ) == -10.0


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


def test_selector_uses_unclipped_margin_when_one_case_score_is_zero():
    selector = EligibilityAwareSelector()
    commissioning_favored = _evaluation(
        raw=1.0,
        official=8e-6,
        case_values=[0.014, 0.009],
        anchor_margins=[60.0, -4000.0],
    )
    bottleneck_favored = _evaluation(
        raw=1.1,
        official=6e-6,
        case_values=[0.018, 0.003],
        anchor_margins=[50.0, -1200.0],
    )
    assert selector.consider("commissioning", commissioning_favored, step=1)
    assert selector.consider("bottleneck", bottleneck_favored, step=2)
    assert selector.best.checkpoint_id == "bottleneck"
    assert selector.best.worst_anchor_margin == -1200.0


def test_selector_returns_to_official_score_after_all_cases_clear_bad_anchor():
    selector = EligibilityAwareSelector()
    better_worst_margin = _evaluation(
        raw=1.0,
        official=40.0,
        case_values=[1.0, 1.0],
        anchor_margins=[40.0, 40.0],
    )
    better_official = _evaluation(
        raw=1.1,
        official=50.0,
        case_values=[1.1, 1.1],
        anchor_margins=[100.0, 25.0],
    )
    assert selector.consider("balanced", better_worst_margin, step=1)
    assert selector.consider("official", better_official, step=2)
    assert selector.best.checkpoint_id == "official"


def _evaluation(
    *,
    raw,
    official,
    case_values=None,
    anchor_margins=None,
):
    values = list(case_values or [raw])
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
            "case_values": values,
            "case_anchor_margins": list(anchor_margins or ()),
            "official_score": official,
            "ranking_eligible": True,
        },
    }
