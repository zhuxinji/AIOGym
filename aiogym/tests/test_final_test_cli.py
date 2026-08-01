from __future__ import annotations

import hashlib
import json

import pytest

from aiogym.cli.final_test import main
from aiogym.rl.final_test import FinalTestLock


TRACK_ID = "quadruple-regulation-generalist-v1"


def test_final_test_cli_hashes_checkpoint_and_writes_lock_artifact(
    tmp_path,
    monkeypatch,
    capsys,
):
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(b"checkpoint").hexdigest()
    output = tmp_path / "final.json"
    lock_path = tmp_path / "lock.json"
    config = {
        "schema_version": "aiogym.final_test.v1",
        "track_id": TRACK_ID,
        "lock_path": str(lock_path),
        "base_seeds": [5],
        "checkpoints": {
            "sac": {
                "path": str(checkpoint),
                "algorithm_id": "sac",
                "sha256": digest,
            }
        },
        "output": str(output),
        "bootstrap_repetitions": 10,
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    loaded = []

    def load_policy(spec):
        loaded.append(spec)
        return object()

    monkeypatch.setattr(
        "aiogym.cli.final_test.load_policy_checkpoint",
        load_policy,
    )

    def run(lock, controllers, *, artifact_path, artifact_builder, **kwargs):
        assert set(controllers) == {"sac"}
        state = lock.state()
        state.update({"status": "artifact_pending", "report_hash": "a" * 64})
        evaluation = {
            "evaluations": {"sac": {"split": "test"}},
            "statistical_report": {"score": 1.0},
        }
        artifact = artifact_builder(evaluation, state)
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        state.update({"status": "complete", "artifact_sha256": "b" * 64})
        return {**evaluation, "lock": state}

    monkeypatch.setattr(FinalTestLock, "run_and_commit", run)
    assert main(["--config", str(config_path)]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert loaded[0].algorithm_id == "sac"
    assert loaded[0].sha256 == digest
    assert loaded[0].scenario == "quadruple"
    assert artifact["checkpoint_sha256"] == {"sac": digest}
    assert artifact["lock_at_commit"]["report_hash"] == "a" * 64
    assert json.loads(capsys.readouterr().out)["lock_status"] == "complete"


def test_final_test_lock_identity_changes_and_failures_consume_lock(
    tmp_path,
):
    lock_path = tmp_path / "lock.json"
    lock = FinalTestLock(
        lock_path,
        track=TRACK_ID,
        config_hash="config",
        checkpoint_ids={"sac": "hash-a"},
        base_seeds=[1],
    )

    def fail(*args, **kwargs):
        raise RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        lock.run_and_commit(
            {"sac": object()},
            artifact_path=tmp_path / "final.json",
            artifact_builder=lambda result, state: result,
            _evaluate_test_fn=fail,
        )
    assert lock.state()["status"] == "failed_evaluation"
    with pytest.raises(RuntimeError, match="already consumed"):
        lock.run_and_commit(
            {"sac": object()},
            artifact_path=tmp_path / "other.json",
            artifact_builder=lambda result, state: result,
            _evaluate_test_fn=fail,
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        FinalTestLock(
            lock_path,
            track=TRACK_ID,
            config_hash="config",
            checkpoint_ids={"sac": "hash-b"},
            base_seeds=[1],
        )


def test_final_test_lock_defaults_to_private_test_evaluator_once(
    tmp_path,
    monkeypatch,
):
    calls = []

    def evaluate_test(controller, track, **kwargs):
        calls.append((controller, track, kwargs))
        return {"split": "test"}

    monkeypatch.setattr(
        "aiogym.rl.final_test._evaluate_policy_on_track_split",
        evaluate_test,
    )
    monkeypatch.setattr(
        "aiogym.rl.final_test.build_final_statistical_report",
        lambda evaluations, **kwargs: {
            "split": "test",
            "algorithms": sorted(evaluations),
        },
    )
    lock = FinalTestLock(
        tmp_path / "lock.json",
        track=TRACK_ID,
        config_hash="config",
        checkpoint_ids={"sac": "checkpoint"},
        base_seeds=[17],
    )
    controller = object()

    output = tmp_path / "final.json"
    result = lock.run_and_commit(
        {"sac": controller},
        artifact_path=output,
        artifact_builder=lambda evaluation, state: {
            **evaluation,
            "lock_at_commit": state,
        },
        bootstrap_repetitions=10,
    )

    assert len(calls) == 1
    assert calls[0][0] is controller
    assert calls[0][2] == {
        "split": "test",
        "base_seeds": (17,),
        "include_episodes": True,
    }
    assert result["evaluations"]["sac"]["split"] == "test"
    assert result["lock"]["status"] == "complete"
    assert result["lock"]["config_hash"] == "config"
    assert result["lock"]["checkpoint_ids"] == {"sac": "checkpoint"}
    assert result["lock"]["test_seed_namespace"]
    assert result["lock"]["report_hash"]
    assert result["lock"]["artifact_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()


def test_final_test_hash_failure_does_not_create_or_consume_lock(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    lock_path = tmp_path / "lock.json"
    config = {
        "schema_version": "aiogym.final_test.v1",
        "track_id": TRACK_ID,
        "lock_path": str(lock_path),
        "base_seeds": [5],
        "checkpoints": {
            "bc": {
                "path": str(checkpoint),
                "algorithm_id": "bc",
                "sha256": "0" * 64,
            }
        },
        "output": str(tmp_path / "final.json"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        main(["--config", str(config_path)])
    assert not lock_path.exists()


def test_artifact_commit_failure_has_a_distinct_consumed_state(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "aiogym.rl.final_test.build_final_statistical_report",
        lambda evaluations, **kwargs: {"split": "test"},
    )
    lock = FinalTestLock(
        tmp_path / "lock.json",
        track=TRACK_ID,
        config_hash="config",
        checkpoint_ids={"sac": "checkpoint"},
        base_seeds=[1],
    )

    def fail_commit(path, artifact):
        raise OSError("injected artifact failure")

    with pytest.raises(OSError, match="injected artifact failure"):
        lock.run_and_commit(
            {"sac": object()},
            artifact_path=tmp_path / "final.json",
            artifact_builder=lambda evaluation, state: evaluation,
            _evaluate_test_fn=lambda *args, **kwargs: {"split": "test"},
            _artifact_commit_fn=fail_commit,
        )

    state = lock.state()
    assert state["status"] == "failed_artifact_commit"
    assert state["failure_stage"] == "artifact_commit"
    assert state["artifact_sha256"] is None


def test_complete_lock_rejects_missing_or_modified_artifact(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "aiogym.rl.final_test.build_final_statistical_report",
        lambda evaluations, **kwargs: {"split": "test"},
    )

    def complete(name):
        output = tmp_path / f"{name}.json"
        lock = FinalTestLock(
            tmp_path / f"{name}.lock.json",
            track=TRACK_ID,
            config_hash=name,
            checkpoint_ids={"sac": "checkpoint"},
            base_seeds=[1],
        )
        lock.run_and_commit(
            {"sac": object()},
            artifact_path=output,
            artifact_builder=lambda evaluation, state: evaluation,
            _evaluate_test_fn=lambda *args, **kwargs: {"split": "test"},
        )
        return lock, output

    missing_lock, missing = complete("missing")
    missing.unlink()
    with pytest.raises(ValueError, match="artifact is missing"):
        missing_lock.state()

    modified_lock, modified = complete("modified")
    modified.write_bytes(modified.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        modified_lock.state()
