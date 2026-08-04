from __future__ import annotations

import math

import pytest

from aiogym.rl.runner import RunResult, _training_seed_statistics


TRACK_ID = "quadruple-regulation-generalist-v2"
TRACK_HASH = "a" * 64
PLAN_HASH = "b" * 64


def _result(
    seed,
    values,
    *,
    direction="minimize",
    eligible=True,
    case_ids=("minimum-phase", "nonminimum-phase"),
    track_hash=TRACK_HASH,
    plan_hash=PLAN_HASH,
):
    metric_value = sum(values) / len(values)
    metric = "cost" if direction == "minimize" else "profit"
    results = [
        {
            "case_id": case_id,
            "resolved_case_hash": str(index + 1) * 64,
        }
        for index, case_id in enumerate(case_ids)
    ]
    return RunResult(
        config_hash="c" * 64,
        track_id=TRACK_ID,
        track_hash=track_hash,
        algorithm_id="sac",
        training_seed=seed,
        output_dir="runs",
        policy_path=f"seed-{seed}.zip",
        artifact_dir=f"seed-{seed}-artifacts",
        resolved_config_path=f"seed-{seed}.resolved.json",
        validation_plan_hash=plan_hash,
        policy_sha256="d" * 64,
        backend={
            "final_step": 100,
            "selected_checkpoint_step": seed * 10,
        },
        validation={
            "track_id": TRACK_ID,
            "track_hash": track_hash,
            "split": "validation",
            "episode_plan_hash": plan_hash,
            "results": results,
            "aggregate": {
                "metric": metric,
                "metric_direction": direction,
                "metric_value": metric_value,
                "case_values": list(values),
                "official_score": 100.0 - metric_value,
                "ranking_eligible": eligible,
            },
        },
    )


def test_training_seed_statistics_builds_ordered_case_matrix():
    rows = (
        _result(1, (1.0, 2.0)),
        _result(2, (2.0, 4.0), eligible=False),
        _result(3, (3.0, 6.0)),
    )

    statistics = _training_seed_statistics(rows)

    assert statistics["unit"] == "independent_training_seed"
    assert statistics["training_seeds"] == [1, 2, 3]
    assert statistics["case_ids"] == [
        "minimum-phase",
        "nonminimum-phase",
    ]
    assert statistics["case_matrix"] == [
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ]
    assert statistics["run_metric_values"] == [1.5, 3.0, 4.5]
    assert statistics["mean"] == pytest.approx(3.0)
    assert statistics["median"] == pytest.approx(3.0)
    assert statistics["std"] == pytest.approx(1.5)
    assert math.isfinite(statistics["iqm"])
    assert all(
        math.isfinite(statistics["iqm_bootstrap"][name])
        for name in ("estimate", "lower", "upper")
    )
    assert statistics["ranking_eligibility_rate"] == pytest.approx(2 / 3)
    assert statistics["selected_checkpoint_steps"] == [10, 20, 30]
    assert statistics["worst_training_seed"] == {
        "training_seed": 3,
        "metric_value": 4.5,
    }


def test_worst_training_seed_respects_maximize_direction():
    rows = (
        _result(1, (1.0, 2.0), direction="maximize"),
        _result(2, (3.0, 4.0), direction="maximize"),
        _result(3, (5.0, 6.0), direction="maximize"),
    )

    statistics = _training_seed_statistics(rows)

    assert statistics["worst_training_seed"] == {
        "training_seed": 1,
        "metric_value": 1.5,
    }


def test_training_seed_statistics_rejects_case_order_mismatch():
    with pytest.raises(ValueError, match="ordered Cases"):
        _training_seed_statistics(
            (
                _result(1, (1.0, 2.0)),
                _result(
                    2,
                    (2.0, 1.0),
                    case_ids=("nonminimum-phase", "minimum-phase"),
                ),
            )
        )


def test_training_seed_statistics_rejects_track_hash_mismatch():
    with pytest.raises(ValueError, match="Track/hash"):
        _training_seed_statistics(
            (
                _result(1, (1.0, 2.0)),
                _result(2, (2.0, 3.0), track_hash="e" * 64),
            )
        )


def test_training_seed_statistics_rejects_plan_hash_mismatch():
    with pytest.raises(ValueError, match="validation plan/hash"):
        _training_seed_statistics(
            (
                _result(1, (1.0, 2.0)),
                _result(2, (2.0, 3.0), plan_hash="f" * 64),
            )
        )


def test_empty_legacy_sweep_remains_compatible():
    empty = _result(1, (1.0, 2.0))
    object.__setattr__(empty, "validation", {})

    assert _training_seed_statistics((empty,)) == {}
