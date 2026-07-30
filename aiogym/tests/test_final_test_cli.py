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
    monkeypatch.setattr(
        "aiogym.cli.final_test.make_controller",
        lambda *args, **kwargs: object(),
    )

    def run(lock, controllers, **kwargs):
        assert set(controllers) == {"sac"}
        state = lock.state()
        state.update(
            {"status": "complete", "report_hash": "a" * 64}
        )
        return {
            "evaluations": {"sac": {"split": "test"}},
            "statistical_report": {"score": 1.0},
            "lock": state,
        }

    monkeypatch.setattr(FinalTestLock, "run", run)
    assert main(["--config", str(config_path)]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["checkpoint_sha256"] == {"sac": digest}
    assert artifact["lock"]["report_hash"] == "a" * 64
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
        lock.run({"sac": object()}, evaluate_fn=fail)
    assert lock.state()["status"] == "failed"
    with pytest.raises(RuntimeError, match="already consumed"):
        lock.run({"sac": object()}, evaluate_fn=fail)
    with pytest.raises(ValueError, match="identity mismatch"):
        FinalTestLock(
            lock_path,
            track=TRACK_ID,
            config_hash="config",
            checkpoint_ids={"sac": "hash-b"},
            base_seeds=[1],
        )
