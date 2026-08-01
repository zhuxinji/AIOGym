"""Phase-F acceptance tests for validation, HPO, and final statistics."""
from __future__ import annotations

import copy
import numpy as np
import pytest

from aiogym import load_track
from aiogym.evaluation.statistics import build_final_statistical_report
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.final_test import FinalTestLock
from aiogym.rl.hpo import apply_hpo_parameters
from aiogym.rl.validation import (
    CompleteValidationCallback,
    EligibilityAwareSelector,
    ValidationEpisodePlan,
)


def test_checkpoint_selection_never_reads_test():
    track = load_track("quadruple-regulation-generalist-v1")
    visited = []

    def evaluate(controller, resolved_track, **kwargs):
        visited.append("split" not in kwargs)
        assert kwargs["episode_plan"].plan_hash
        return _validation_evaluation(
            track,
            values=[1.0, 2.0],
            eligible=True,
        )

    callback = CompleteValidationCallback(
        track,
        base_seeds=[10, 11],
        evaluate_fn=evaluate,
    )
    callback.evaluate(object(), checkpoint_id="checkpoint-1", step=100)
    assert visited == [True]
    with pytest.raises(ValueError, match="validation results only"):
        callback.selector.consider(
            "illegal",
            {"split": "test", "aggregate": {}},
            step=200,
        )


def test_ineligible_checkpoint_cannot_be_best():
    track = load_track("quadruple-regulation-generalist-v1")
    selector = EligibilityAwareSelector()
    assert selector.consider(
        "eligible",
        _validation_evaluation(track, values=[2.0, 2.2], eligible=True),
        step=10,
    )
    assert not selector.consider(
        "unsafe-but-low-cost",
        _validation_evaluation(track, values=[0.1, 0.1], eligible=False),
        step=20,
    )
    assert selector.best.checkpoint_id == "eligible"
    assert selector.records[-1].rejection_reasons == ("hard_termination",)


def test_resume_preserves_pre_resume_global_best_and_records():
    track = load_track("quadruple-regulation-generalist-v1")
    selector = EligibilityAwareSelector()
    assert selector.consider(
        "step-100",
        _validation_evaluation(track, values=[1.0, 1.1], eligible=True),
        step=100,
    )
    restored = EligibilityAwareSelector.from_state_dict(
        selector.state_dict()
    )
    assert not restored.consider(
        "step-200",
        _validation_evaluation(track, values=[2.0, 2.1], eligible=True),
        step=200,
    )
    assert [record.checkpoint_id for record in restored.records] == [
        "step-100",
        "step-200",
    ]
    assert restored.best.checkpoint_id == "step-100"


def test_selector_rejects_tampered_global_best_and_nonfinite_values():
    track = load_track("quadruple-regulation-generalist-v1")
    selector = EligibilityAwareSelector()
    selector.consider(
        "best",
        _validation_evaluation(track, values=[1.0, 1.1], eligible=True),
        step=1,
    )
    selector.consider(
        "worse",
        _validation_evaluation(track, values=[2.0, 2.1], eligible=True),
        step=2,
    )
    tampered = copy.deepcopy(selector.state_dict())
    tampered["best_checkpoint_id"] = "worse"
    with pytest.raises(ValueError, match="not the best"):
        EligibilityAwareSelector.from_state_dict(tampered)
    tampered = copy.deepcopy(selector.state_dict())
    tampered["records"][0]["official_score"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        EligibilityAwareSelector.from_state_dict(tampered)


def test_validation_callback_state_round_trip():
    track = load_track("quadruple-regulation-generalist-v1")

    def evaluate(controller, resolved_track, **kwargs):
        return _validation_evaluation(
            track, values=[1.0, 1.1], eligible=True
        )

    callback = CompleteValidationCallback(
        track,
        base_seeds=[10, 11],
        evaluate_fn=evaluate,
    )
    callback.evaluate(object(), checkpoint_id="step-10", step=10)
    restored = CompleteValidationCallback(
        track,
        base_seeds=[10, 11],
        evaluate_fn=evaluate,
    )
    restored.load_state_dict(callback.state_dict())
    assert restored.selector.best.checkpoint_id == "step-10"
    assert len(restored.history) == 1


def test_hpo_does_not_override_reward_or_track():
    track = load_track("quadruple-regulation-generalist-v1")
    config = RLTrainingConfig(
        track_id=track.id,
        algorithm_id="sac",
        training_seed=1,
        total_transitions=100,
        n_envs=2,
        algorithm={"learning_rate": 3e-4, "utd_ratio": 1.0},
    )
    candidate = apply_hpo_parameters(
        config,
        track,
        {"learning_rate": 1e-4, "tau": 0.01},
    )
    assert candidate.track_id == track.id
    assert candidate.algorithm["learning_rate"] == 1e-4
    assert track.reward_spec_id == "regulation-v1"
    with pytest.raises(ValueError, match="benchmark-owned"):
        apply_hpo_parameters(
            config,
            track,
            {"reward_spec_id": "economic-v1"},
        )
    with pytest.raises(ValueError, match="benchmark-owned"):
        apply_hpo_parameters(
            config,
            track,
            {"track_id": "another-track"},
        )


def test_common_validation_episode_specs_are_shared_across_algorithms():
    track = load_track("quadruple-regulation-generalist-v1")
    sac_plan = ValidationEpisodePlan(track, base_seeds=[101, 102, 103])
    ppo_plan = ValidationEpisodePlan(track, base_seeds=[101, 102, 103])
    assert sac_plan.plan_hash == ppo_plan.plan_hash
    assert sac_plan.metadata()["episodes"] == ppo_plan.metadata()["episodes"]
    for case in track.resolved_cases("validation"):
        sac_specs = sac_plan.episode_specs(case.case_id)
        ppo_specs = ppo_plan.episode_specs(case.case_id)
        assert [spec.resolved_hash for spec in sac_specs] == [
            spec.resolved_hash for spec in ppo_specs
        ]


def test_final_report_contains_per_seed_per_case_matrix():
    track = load_track("quadruple-regulation-generalist-v1")
    seeds = [7, 8, 9]
    sac = _test_evaluation(
        track,
        seeds,
        case_values={
            "minimum-phase": [10.0, 20.0, 30.0],
            "nonminimum-phase": [20.0, 30.0, 40.0],
        },
    )
    pid = _test_evaluation(
        track,
        seeds,
        case_values={
            "minimum-phase": [20.0, 30.0, 40.0],
            "nonminimum-phase": [30.0, 40.0, 50.0],
        },
    )
    report = build_final_statistical_report(
        {"sac": sac, "pid": pid},
        baseline="pid",
        bootstrap_repetitions=50,
        bootstrap_seed=4,
    )
    assert report["seeds"] == seeds
    assert report["case_ids"] == ["minimum-phase", "nonminimum-phase"]
    assert np.asarray(
        report["per_seed_per_case_matrix"]["sac"]
    ).shape == (3, 2)
    assert report["summaries"]["sac"]["iqm_bootstrap"]["lower"] <= (
        report["summaries"]["sac"]["iqm"]
    )
    assert report["summaries"]["sac"][
        "probability_of_improvement_vs_baseline"
    ] == 1.0
    assert "profiles" in report["performance_profile"]


def test_final_test_lock_is_consumed_once(tmp_path):
    track = load_track("quadruple-regulation-generalist-v1")
    lock = FinalTestLock(
        tmp_path / "final-test-lock.json",
        track=track,
        config_hash="config-hash",
        checkpoint_ids={"sac": "checkpoint-hash"},
        base_seeds=[5, 6],
    )

    def evaluate(controller, resolved_track, **kwargs):
        assert kwargs["split"] == "test"
        return _test_evaluation(
            track,
            [5, 6],
            case_values={
                "minimum-phase": [10.0, 12.0],
                "nonminimum-phase": [20.0, 22.0],
            },
        )

    output = lock.run_and_commit(
        {"sac": object()},
        artifact_path=tmp_path / "final-test.json",
        artifact_builder=lambda result, state: {
            "result": result,
            "lock": state,
        },
        bootstrap_repetitions=20,
        _evaluate_test_fn=evaluate,
    )
    assert output["lock"]["status"] == "complete"
    with pytest.raises(RuntimeError, match="already consumed"):
        lock.run_and_commit(
            {"sac": object()},
            artifact_path=tmp_path / "another-final-test.json",
            artifact_builder=lambda result, state: {
                "result": result,
                "lock": state,
            },
            bootstrap_repetitions=20,
            _evaluate_test_fn=evaluate,
        )


def _validation_evaluation(track, *, values, eligible):
    results = [
        {
            "case_id": f"case-{index}",
            "ranking_eligible": eligible,
            "safety_gate": {
                "reasons": [] if eligible else ["hard_termination"]
            },
        }
        for index, _ in enumerate(values)
    ]
    return {
        "track_id": track.id,
        "track_hash": track.track_hash,
        "split": "validation",
        "results": results,
        "aggregate": {
            "metric": "regulation_cost_rate",
            "metric_direction": "minimize",
            "metric_value": float(np.mean(values)),
            "case_values": list(values),
            "case_count": len(values),
            "ranking_eligible": eligible,
        },
    }


def _test_evaluation(track, seeds, *, case_values):
    results = []
    for case_id, values in case_values.items():
        results.append(
            {
                "case_id": case_id,
                "case_horizon_seconds": 10.0,
                "ranking_eligible": True,
                "episode_metrics": [
                    {
                        "seed": seed,
                        "regulation_cost": value,
                    }
                    for seed, value in zip(seeds, values)
                ],
            }
        )
    rates = [
        float(np.mean(values)) / 10.0
        for values in case_values.values()
    ]
    return {
        "track_id": track.id,
        "track_hash": track.track_hash,
        "split": "test",
        "base_seeds": list(seeds),
        "results": results,
        "aggregate": {
            "metric": "regulation_cost_rate",
            "metric_direction": "minimize",
            "metric_value": float(np.mean(rates)),
            "case_values": rates,
            "case_count": len(rates),
            "ranking_eligible": True,
        },
    }
