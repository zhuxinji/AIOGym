from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aiogym.cli.main import main as cli_main
from aiogym.rl import RLTrainingConfig
from aiogym.rl.backends import BackendResult, validate_backend_result
from aiogym.rl.lifecycle import (
    TrainingLifecycleResult,
    finalize_training_lifecycle,
)
from aiogym.rl.plan import resolve_training_plan
from aiogym.rl.runner import RunResult, run_experiment


TRACK_ID = "quadruple-regulation-generalist-v1"


def _config(tmp_path, **overrides):
    data = {
        "track_id": TRACK_ID,
        "algorithm_id": "sac",
        "training_seed": 7,
        "total_transitions": 20,
        "n_envs": 1,
        "algorithm": {"batch_size": 2, "utd_ratio": 1.0},
        "replay": {"capacity": 32},
        "validation_seeds": [101, 102],
        "resume_mode": "restart_episode",
        "output": {
            "directory": str(tmp_path),
            "name": "unified-smoke",
            "save_rollout": False,
        },
    }
    data.update(overrides)
    return RLTrainingConfig.from_mapping(data)


def test_v2_config_hash_covers_lifecycle_and_output_fields(tmp_path):
    config = _config(tmp_path)
    assert config.schema_version == "aiogym.rl_training_config.v2"
    restored = RLTrainingConfig.from_mapping(config.as_dict())
    assert restored.config_hash == config.config_hash
    assert restored.validation_seeds == (101, 102)
    changed = RLTrainingConfig.from_mapping(
        {
            **config.as_dict(),
            "output": {
                "directory": str(tmp_path),
                "name": "different",
            },
        }
    )
    assert changed.config_hash != config.config_hash


def test_algorithm_and_workflow_mappings_reject_unknown_fields(tmp_path):
    with pytest.raises(ValueError, match="unknown algorithm for sac"):
        _config(
            tmp_path,
            algorithm={"batch_size": 2, "mystery": 1},
        )
    with pytest.raises(ValueError, match="unknown output"):
        _config(
            tmp_path,
            output={
                "directory": str(tmp_path),
                "name": "bad-output",
                "backend_flag": True,
            },
        )


def test_resolved_plan_binds_track_validation_and_paths(tmp_path):
    plan = resolve_training_plan(_config(tmp_path))
    assert plan.track.id == TRACK_ID
    assert plan.validation_plan.base_seeds == (101, 102)
    assert plan.policy_path == tmp_path / "unified-smoke.zip"
    assert plan.validation_plan.plan_hash
    assert plan.config.algorithm["learning_rate"] == 3e-4
    assert plan.config.algorithm["vector_backend"] == "subproc"
    assert plan.config.replay["learning_starts"] == 100
    assert plan.config.evaluation["every_transitions"] == 10_000
    assert plan.config.output["artifact_dir"] == str(plan.artifact_dir)


def test_private_sb3_backend_uses_the_exact_resolved_config(tmp_path):
    from aiogym.rl.backends.sb3 import unified_config as sb3_config

    plan = resolve_training_plan(_config(tmp_path))
    assert sb3_config(plan) is plan.config


def test_backend_result_rejects_algorithm_and_path_mismatches(tmp_path):
    plan = resolve_training_plan(_config(tmp_path))
    plan.policy_path.write_bytes(b"selected")
    wrong_algorithm = BackendResult(
        algorithm_id="td3",
        policy_path=plan.policy_path,
        final_step=20,
        checkpoint_selection="final",
    )
    with pytest.raises(ValueError, match="algorithm_id"):
        validate_backend_result(plan, wrong_algorithm)

    other_path = tmp_path / "other.zip"
    other_path.write_bytes(b"other")
    wrong_path = BackendResult(
        algorithm_id="sac",
        policy_path=other_path,
        final_step=20,
        checkpoint_selection="final",
    )
    with pytest.raises(ValueError, match="policy_path"):
        validate_backend_result(plan, wrong_path)


def test_lifecycle_reloads_checkpoint_and_replaces_stale_artifact(
    tmp_path,
    monkeypatch,
):
    plan = resolve_training_plan(_config(tmp_path))
    plan.artifact_dir.mkdir(parents=True)
    plan.policy_path.write_bytes(b"selected-policy")
    (plan.artifact_dir / "benchmark.json").write_text(
        json.dumps({"marker": "stale"}),
        encoding="utf-8",
    )
    user_file = plan.artifact_dir / "notes.txt"
    user_file.write_text("keep", encoding="utf-8")
    backend = BackendResult(
        algorithm_id="sac",
        policy_path=plan.policy_path,
        final_step=20,
        checkpoint_selection="final",
    )
    events = []
    sentinel_spec = object()
    sentinel_controller = object()

    def fake_spec(path, algorithm_id, digest, track):
        events.append("spec")
        assert path == plan.policy_path
        assert algorithm_id == "sac"
        assert track is plan.track
        assert len(digest) == 64
        return sentinel_spec

    def fake_load(spec, *, device):
        events.append("load")
        assert spec is sentinel_spec
        assert device == plan.config.device
        return sentinel_controller

    def fake_evaluate(controller, validation_plan, *, include_episodes):
        events.append("evaluate")
        assert controller is sentinel_controller
        assert validation_plan is plan.validation_plan
        assert include_episodes
        return {
            "split": "validation",
            "track_id": plan.track.id,
            "track_hash": plan.track.track_hash,
            "episode_plan_hash": plan.validation_plan.plan_hash,
            "seed_namespace": plan.track.seed_namespace("validation"),
            "base_seeds": list(plan.validation_plan.base_seeds),
            "case_count": 0,
            "results": [],
            "aggregate": {
                "metric": "regulation_cost_rate",
                "metric_direction": "minimize",
                "metric_value": 1.0,
                "official_score": 0.0,
                "ranking_eligible": True,
                "case_values": [1.0],
            },
        }

    def fake_write(directory, payload, *, replace_existing):
        events.append("write")
        assert directory == plan.artifact_dir
        assert replace_existing is True
        (directory / "benchmark.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return payload

    def fake_check(directory):
        events.append("check")
        assert directory == plan.artifact_dir
        return {
            "schema_version": "test",
            "ok": True,
            "failed": [],
        }

    monkeypatch.setattr(
        "aiogym.rl.lifecycle.learned_policy_spec_for_track",
        fake_spec,
    )
    monkeypatch.setattr(
        "aiogym.rl.lifecycle.load_policy_checkpoint",
        fake_load,
    )
    monkeypatch.setattr(
        "aiogym.rl.lifecycle.evaluate_validation_policy",
        fake_evaluate,
    )
    monkeypatch.setattr(
        "aiogym.rl.lifecycle.write_rl_artifacts",
        fake_write,
    )
    monkeypatch.setattr(
        "aiogym.rl.lifecycle.check_benchmark_artifacts",
        fake_check,
    )

    result = finalize_training_lifecycle(plan, backend)

    assert result.policy_sha256 == hashlib.sha256(
        b"selected-policy"
    ).hexdigest()
    assert events == ["spec", "load", "evaluate", "write", "check"]
    assert user_file.read_text(encoding="utf-8") == "keep"
    persisted = json.loads(
        (plan.artifact_dir / "benchmark.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["training"]["training_config_hash"] == (
        plan.config.config_hash
    )


def test_runner_owns_resolution_execution_and_result_artifacts(tmp_path):
    def fake_backend(plan):
        plan.policy_path.write_bytes(b"policy")
        return BackendResult(
            algorithm_id="sac",
            policy_path=plan.policy_path,
            final_step=20,
            checkpoint_selection="final",
            runtime={"seconds": 1.5},
        )

    def fake_lifecycle(plan, backend):
        assert backend.policy_path == plan.policy_path
        return TrainingLifecycleResult(
            policy_sha256="f" * 64,
            validation={},
            artifact_check={"ok": True},
        )

    result = run_experiment(
        _config(tmp_path),
        backend_runner=fake_backend,
        lifecycle_finalizer=fake_lifecycle,
    )
    assert result.policy_path == str(tmp_path / "unified-smoke.zip")
    assert result.schema_version == "aiogym.run_result.v3"
    assert result.policy_sha256 == "f" * 64
    assert result.backend == {
        "final_step": 20,
        "checkpoint_selection": "final",
        "runtime": {"seconds": 1.5},
        "exports": {},
    }
    assert "adapter_state" not in result.as_dict()
    resolved = json.loads(
        Path(result.resolved_config_path).read_text(encoding="utf-8")
    )
    assert resolved["validation_seeds"] == [101, 102]
    assert resolved["algorithm_id"] == "sac"
    assert (
        tmp_path / "unified-smoke.run-result.json"
    ).is_file()


def test_cli_train_is_config_first_and_has_no_backend_subcommand(
    tmp_path,
    monkeypatch,
    capsys,
):
    config = _config(tmp_path)
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(config.as_dict()),
        encoding="utf-8",
    )
    result = RunResult(
        config_hash=config.config_hash,
        track_id=TRACK_ID,
        track_hash="a" * 64,
        algorithm_id="sac",
        training_seed=7,
        output_dir=str(tmp_path),
        policy_path=str(tmp_path / "policy.zip"),
        artifact_dir=str(tmp_path / "artifacts"),
        resolved_config_path=str(tmp_path / "resolved.json"),
        validation_plan_hash="b" * 64,
        policy_sha256="c" * 64,
        backend={
            "final_step": 20,
            "checkpoint_selection": "final",
            "runtime": {},
            "exports": {},
        },
    )
    monkeypatch.setattr(
        "aiogym.cli.train.run_experiment",
        lambda resolved: result,
    )
    assert cli_main(["train", "--config", str(config_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["algorithm_id"] == "sac"
    assert "--split" not in payload["next_command"]
    assert payload["next_command"] == (
        f"aiogym evaluate --checkpoint {result.policy_path} "
        f"--track {TRACK_ID} --algorithm sac"
    )
    with pytest.raises(SystemExit):
        cli_main(["train", "sb3", "--track", TRACK_ID])


def test_evaluate_cli_parser_rejects_test_split():
    with pytest.raises(SystemExit):
        cli_main(
            [
                "evaluate",
                "--checkpoint",
                "unused.zip",
                "--track",
                TRACK_ID,
                "--algorithm",
                "sac",
                "--split",
                "test",
            ]
        )


def test_public_cli_surface_has_algorithms_and_no_backend_train(
    capsys,
):
    assert cli_main(["list", "algorithms"]) is None
    assert capsys.readouterr().out.splitlines() == [
        "bc",
        "ppo",
        "rlpd",
        "sac",
        "td3",
    ]
    with pytest.raises(SystemExit) as exc:
        cli_main(["train", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--config" in help_text
    assert "BACKEND" not in help_text
    assert "Stable-Baselines3" not in help_text


def test_benchmark_cli_rejects_direct_test_split():
    with pytest.raises(SystemExit):
        cli_main(
            [
                "benchmark",
                TRACK_ID,
                "--controllers",
                "pid",
                "--split",
                "test",
            ]
        )


def test_rlpd_backend_accepts_only_a_resolved_plan():
    import inspect
    from aiogym.rl.backends.rlpd import run_rlpd

    assert tuple(inspect.signature(run_rlpd).parameters) == ("plan",)


def test_sb3_callback_records_unique_training_episode_specs(tmp_path):
    from aiogym.rl.backends.sb3 import make_learning_curve_callback

    plan = resolve_training_plan(
        _config(
            tmp_path,
            evaluation={"every_transitions": 0},
        )
    )
    callback = make_learning_curve_callback(plan)
    callback.locals = {
        "infos": [
            {
                "episode_index": 0,
                "episode_spec_id": "episode-0",
                "episode_spec_hash": "hash-0",
                "distribution_id": "distribution",
                "distribution_hash": "distribution-hash",
            },
            {
                "episode_index": 1,
                "episode_spec_id": "episode-1",
                "episode_spec_hash": "hash-1",
                "distribution_id": "distribution",
                "distribution_hash": "distribution-hash",
            },
        ]
    }
    callback.num_timesteps = 2
    assert callback._on_step()
    assert set(callback.training_episode_specs) == {"hash-0", "hash-1"}
