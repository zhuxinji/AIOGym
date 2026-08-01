from __future__ import annotations

import json
import threading

import pytest

from aiogym.cli.benchmark import build_parser as build_benchmark_parser
from aiogym.cli.evaluate import build_parser as build_evaluate_parser
from aiogym.cli.train import build_parser as build_train_parser
from aiogym.cli.train import main as train_main
from aiogym._internal.serialization import write_json_artifact
from aiogym.rl.backends import BackendResult
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.lifecycle import TrainingLifecycleResult
from aiogym.rl.plan import resolve_training_plan
from aiogym.rl.run_claim import (
    RunClaim,
    RunClaimIdentity,
    SeedSweepClaim,
    SeedSweepIdentity,
)
from aiogym.rl.runner import RunResult, run_experiment, run_seed_sweep


TRACK_ID = "quadruple-regulation-generalist-v1"


def _config(tmp_path):
    return RLTrainingConfig(
        track_id=TRACK_ID,
        algorithm_id="sac",
        training_seed=3,
        total_transitions=2,
        n_envs=1,
        output={"directory": str(tmp_path), "name": "claimed-run"},
    )


def _backend(plan):
    plan.policy_path.write_bytes(b"policy")
    return BackendResult(
        algorithm_id="sac",
        policy_path=plan.policy_path,
        final_step=2,
        checkpoint_selection="final",
    )


def _lifecycle(plan, backend):
    plan.artifact_dir.mkdir(parents=True, exist_ok=True)
    (plan.artifact_dir / "benchmark.json").write_text(
        json.dumps({"run": plan.run_name}), encoding="utf-8"
    )
    return TrainingLifecycleResult(
        policy_sha256="f" * 64,
        validation={},
        artifact_check={"ok": True},
    )


def _sweep_result(config, seed):
    name = f"claimed-run-seed{seed}"
    return RunResult(
        config_hash=config.config_hash,
        track_id=config.track_id,
        track_hash="a" * 64,
        algorithm_id=config.algorithm_id,
        training_seed=seed,
        output_dir=str(config.output["directory"]),
        policy_path=f"{name}.zip",
        artifact_dir=f"{name}_artifacts",
        resolved_config_path=f"{name}.resolved.json",
        validation_plan_hash="b" * 64,
        policy_sha256="c" * 64,
        backend={},
    )


def test_training_does_not_overwrite_existing_policy_by_default(tmp_path):
    config = _config(tmp_path)
    run_experiment(
        config,
        backend_runner=_backend,
        lifecycle_finalizer=_lifecycle,
    )

    with pytest.raises(FileExistsError, match="run output|run claim"):
        run_experiment(
            config,
            backend_runner=_backend,
            lifecycle_finalizer=_lifecycle,
        )


def test_explicit_overwrite_replaces_only_owned_paths(tmp_path):
    config = _config(tmp_path)
    first = run_experiment(
        config,
        backend_runner=_backend,
        lifecycle_finalizer=_lifecycle,
    )
    notes = first.artifact_dir and (
        resolve_training_plan(config).artifact_dir / "notes.txt"
    )
    notes.write_text("preserve user note", encoding="utf-8")

    second = run_experiment(
        config,
        backend_runner=_backend,
        lifecycle_finalizer=_lifecycle,
        overwrite=True,
    )

    assert second.policy_path == first.policy_path
    assert notes.read_text(encoding="utf-8") == "preserve user note"
    claim = json.loads(
        (tmp_path / ".claimed-run.claim.json").read_text(encoding="utf-8")
    )
    assert claim["status"] == "complete"
    assert len(claim["overwrite_audit"]) == 1


def test_resume_and_overwrite_are_mutually_exclusive(tmp_path):
    config = _config(tmp_path)
    resumed = RLTrainingConfig.from_mapping(
        {**config.as_dict(), "resume_checkpoint": "checkpoint.zip"}
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_experiment(resumed, overwrite=True)

    parser = build_train_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--config", "config.json", "--resume", "x.zip", "--overwrite"]
        )


def test_parallel_claim_is_rejected_atomically(tmp_path):
    identity = RunClaimIdentity(
        run_name="atomic",
        config_hash="a" * 64,
        track_hash="b" * 64,
        algorithm_id="sac",
        training_seed=1,
    )
    path = tmp_path / ".atomic.claim.json"
    RunClaim.acquire(path, identity)

    with pytest.raises(FileExistsError, match="run claim"):
        RunClaim.acquire(path, identity)


def test_failed_run_retains_failure_provenance(tmp_path):
    config = _config(tmp_path)

    def fail_backend(plan):
        raise RuntimeError("injected training failure")

    with pytest.raises(RuntimeError, match="injected training failure"):
        run_experiment(config, backend_runner=fail_backend)

    claim = json.loads(
        (tmp_path / ".claimed-run.claim.json").read_text(encoding="utf-8")
    )
    assert claim["status"] == "failed"
    assert "injected training failure" in claim["failure"]


def test_benchmark_overwrite_defaults_false():
    args = build_benchmark_parser().parse_args(
        [TRACK_ID, "--controllers", "pid"]
    )

    assert args.overwrite is False


def test_evaluate_overwrite_defaults_false():
    args = build_evaluate_parser().parse_args(
        [
            "--checkpoint",
            "policy.zip",
            "--track",
            TRACK_ID,
            "--algorithm",
            "sac",
        ]
    )

    assert args.overwrite is False


def test_json_artifact_rejects_existing_target_and_overwrite_replaces(tmp_path):
    target = tmp_path / "result.json"
    target.write_bytes(b"original\n")

    with pytest.raises(FileExistsError, match="artifact already exists"):
        write_json_artifact(target, {"value": 1})
    assert target.read_bytes() == b"original\n"

    assert write_json_artifact(target, {"z": 1, "a": 2}, overwrite=True) == target
    assert target.read_text(encoding="utf-8") == (\
        '{\n  "a": 2,\n  "z": 1\n}\n'
    )


def test_json_artifact_serialization_failure_preserves_old_file(tmp_path):
    target = tmp_path / "result.json"
    target.write_bytes(b"original\n")

    with pytest.raises(TypeError):
        write_json_artifact(target, {"bad": object()}, overwrite=True)

    assert target.read_bytes() == b"original\n"
    assert list(tmp_path.iterdir()) == [target]


def test_json_artifact_replace_failure_preserves_old_file(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    target.write_bytes(b"original\n")

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr("aiogym._internal.serialization.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        write_json_artifact(target, {"value": 2}, overwrite=True)

    assert target.read_bytes() == b"original\n"
    assert list(tmp_path.iterdir()) == [target]


def test_json_artifact_concurrent_writer_is_rejected(tmp_path):
    target = tmp_path / "result.json"
    entered = threading.Event()
    release = threading.Event()
    failures = []

    class BlockingValue:
        def item(self):
            entered.set()
            assert release.wait(timeout=5)
            return 1

    def first_writer():
        try:
            write_json_artifact(target, {"value": BlockingValue()})
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=first_writer)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(FileExistsError):
            write_json_artifact(target, {"value": 2})
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert failures == []
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
    assert list(tmp_path.iterdir()) == [target]


def test_seed_sweep_existing_summary_fails_before_first_child(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    summary = tmp_path / "claimed-run.multi-seed.json"
    summary.write_text("{}\n", encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        "aiogym.rl.runner.run_experiment",
        lambda *args, **kwargs: calls.append(args),
    )

    with pytest.raises(FileExistsError, match="summary already exists"):
        run_seed_sweep(config, (1, 2))

    assert calls == []


def test_seed_sweep_claim_is_exclusive_and_identity_binds_seeds(tmp_path):
    first = SeedSweepIdentity(
        base_name="sweep",
        config_hash="a" * 64,
        track_id=TRACK_ID,
        track_hash="b" * 64,
        algorithm_id="sac",
        seeds=(1, 2),
    )
    second = SeedSweepIdentity(
        base_name="sweep",
        config_hash="a" * 64,
        track_id=TRACK_ID,
        track_hash="b" * 64,
        algorithm_id="sac",
        seeds=(1, 3),
    )
    path = tmp_path / ".sweep.multi-seed.claim.json"
    summary = tmp_path / "sweep.multi-seed.json"
    claim = SeedSweepClaim.acquire(path, first, summary_path=summary)

    with pytest.raises(FileExistsError, match="claim already exists"):
        SeedSweepClaim.acquire(path, first, summary_path=summary)

    claim.complete(summary_hash="c" * 64)
    with pytest.raises(ValueError, match="identity mismatch"):
        SeedSweepClaim.acquire(
            path,
            second,
            summary_path=summary,
            overwrite=True,
        )


def test_seed_sweep_child_failure_records_completed_seed(tmp_path, monkeypatch):
    config = _config(tmp_path)

    def child(resolved, *, overwrite=False):
        if resolved.training_seed == 2:
            raise RuntimeError("injected child failure")
        return _sweep_result(resolved, resolved.training_seed)

    monkeypatch.setattr("aiogym.rl.runner.run_experiment", child)
    with pytest.raises(RuntimeError, match="injected child failure"):
        run_seed_sweep(config, (1, 2, 3))

    claim = json.loads(
        (tmp_path / ".claimed-run.multi-seed.claim.json").read_text()
    )
    assert claim["status"] == "failed"
    assert claim["completed_seeds"] == [1]
    assert claim["child_results"][0]["seed"] == 1
    assert claim["failure"] == {
        "type": "RuntimeError",
        "message": "injected child failure",
    }


def test_seed_sweep_summary_failure_does_not_complete_claim(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path)
    monkeypatch.setattr(
        "aiogym.rl.runner.run_experiment",
        lambda resolved, *, overwrite=False: _sweep_result(
            resolved,
            resolved.training_seed,
        ),
    )
    monkeypatch.setattr(
        "aiogym.rl.runner.write_json_artifact",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("injected summary failure")
        ),
    )

    with pytest.raises(OSError, match="injected summary failure"):
        run_seed_sweep(config, (1,))

    claim = json.loads(
        (tmp_path / ".claimed-run.multi-seed.claim.json").read_text()
    )
    assert claim["status"] == "failed"
    assert claim["summary_hash"] is None
    assert not (tmp_path / "claimed-run.multi-seed.json").exists()


def test_seed_sweep_completes_only_after_durable_summary(tmp_path, monkeypatch):
    config = _config(tmp_path)
    claim_path = tmp_path / ".claimed-run.multi-seed.claim.json"
    observed = []
    real_writer = write_json_artifact
    monkeypatch.setattr(
        "aiogym.rl.runner.run_experiment",
        lambda resolved, *, overwrite=False: _sweep_result(
            resolved,
            resolved.training_seed,
        ),
    )

    def checked_writer(path, data, *, overwrite=False):
        observed.append(json.loads(claim_path.read_text())["status"])
        result = real_writer(path, data, overwrite=overwrite)
        assert result.is_file()
        return result

    monkeypatch.setattr("aiogym.rl.runner.write_json_artifact", checked_writer)
    summary = run_seed_sweep(config, (1, 2))

    claim = json.loads(claim_path.read_text())
    assert observed == ["running"]
    assert claim["status"] == "complete"
    assert claim["completed_seeds"] == [1, 2]
    assert claim["summary_hash"]
    assert summary["sweep_id"] == claim["sweep_id"]
    assert summary["track_hash"] == claim["track_hash"]


def test_seed_sweep_explicit_overwrite_is_audited(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(
        "aiogym.rl.runner.run_experiment",
        lambda resolved, *, overwrite=False: _sweep_result(
            resolved,
            resolved.training_seed,
        ),
    )
    first = run_seed_sweep(config, (1, 2))
    second = run_seed_sweep(config, (1, 2), overwrite=True)

    claim = json.loads(
        (tmp_path / ".claimed-run.multi-seed.claim.json").read_text()
    )
    assert first["sweep_id"] == second["sweep_id"]
    assert claim["status"] == "complete"
    assert len(claim["overwrite_audit"]) == 1


def test_train_cli_rejects_resume_with_seed_sweep(tmp_path):
    config = _config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="--resume cannot be combined"):
        train_main(
            [
                "--config",
                str(config_path),
                "--seeds",
                "1,2",
                "--resume",
                "checkpoint.zip",
            ]
        )


@pytest.mark.parametrize("seeds", ([], [True], [-1], [1, 1]))
def test_seed_sweep_rejects_invalid_seed_sequences(tmp_path, seeds):
    with pytest.raises((TypeError, ValueError)):
        run_seed_sweep(_config(tmp_path), seeds)
