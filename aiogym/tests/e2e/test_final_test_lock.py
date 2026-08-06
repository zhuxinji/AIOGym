from __future__ import annotations

import hashlib

import pytest

pytestmark = pytest.mark.e2e

from aiogym import load_track
from aiogym.rl.final_test import FinalTestLock


@pytest.mark.parametrize(
    "track_id",
    (
        "quadruple-regulation-generalist-v1",
        "cascade-regulation-generalist-v1",
        "cascade-recirculating-regulation-generalist-v2",
    ),
)
def test_final_test_builds_statistics_and_consumes_lock_once(
    tmp_path,
    track_id,
):
    track = load_track(track_id)
    lock = FinalTestLock(
        tmp_path / "lock.json",
        track=track,
        config_hash="e2e-config",
        checkpoint_ids={"sac": "e2e-checkpoint"},
        base_seeds=[7201],
    )
    visited = []

    def evaluate(controller, resolved_track, **kwargs):
        assert controller is not None
        assert resolved_track.track_hash == track.track_hash
        visited.append(kwargs["split"])
        results = []
        for index, case in enumerate(track.resolved_cases("test")):
            results.append(
                {
                    "case_id": case.case_id,
                    "case_horizon_seconds": 10.0,
                    "ranking_eligible": True,
                    "episode_metrics": [
                        {
                            "seed": 7201,
                            "regulation_cost": float(index + 1),
                        }
                    ],
                }
            )
        return {
            "track_id": track.id,
            "track_hash": track.track_hash,
            "split": "test",
            "base_seeds": [7201],
            "results": results,
            "aggregate": {
                "metric": "regulation_cost_rate",
                "metric_direction": "minimize",
                "metric_value": 0.3,
                "ranking_eligible": True,
            },
        }

    artifact_path = tmp_path / "final-test.json"
    result = lock.run_and_commit(
        {"sac": object()},
        artifact_path=artifact_path,
        artifact_builder=lambda evaluation, state: {
            "schema_version": "aiogym.final_test_artifact.v2",
            "lock_at_commit": state,
            **evaluation,
        },
        bootstrap_repetitions=10,
        _evaluate_test_fn=evaluate,
    )
    report = result["statistical_report"]

    assert visited == ["test"]
    assert result["lock"]["status"] == "complete"
    assert result["lock"]["report_hash"]
    assert artifact_path.is_file()
    assert result["lock"]["artifact_sha256"] == hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    assert report["split"] == "test"
    assert report["seeds"] == [7201]
    case_count = len(track.resolved_cases("test"))
    assert len(report["case_ids"]) == case_count
    assert len(report["per_seed_per_case_matrix"]["sac"][0]) == case_count
    with pytest.raises(RuntimeError, match="already consumed"):
        lock.run_and_commit(
            {"sac": object()},
            artifact_path=tmp_path / "second.json",
            artifact_builder=lambda evaluation, state: evaluation,
            bootstrap_repetitions=10,
            _evaluate_test_fn=evaluate,
        )
