from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiogym.cli.main import main as cli_main
from aiogym.rl import RLTrainingConfig
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


def test_resolved_plan_binds_track_validation_and_paths(tmp_path):
    plan = resolve_training_plan(_config(tmp_path))
    assert plan.track.id == TRACK_ID
    assert plan.validation_plan.base_seeds == (101, 102)
    assert plan.policy_path == tmp_path / "unified-smoke.zip"
    assert plan.validation_plan.plan_hash


def test_runner_owns_resolution_execution_and_result_artifacts(tmp_path):
    class FakeAdapter:
        def build(self, plan, envs=None):
            assert envs is None
            self.plan = plan
            return self

        def train_chunk(self, transitions):
            assert transitions == 20
            self.plan.policy_path.write_bytes(b"policy")

        def controller(self):
            return object()

        def state_dict(self):
            return {"trained_transitions": 20}

        def load_state_dict(self, state):
            raise AssertionError(state)

        def save_policy(self, path):
            assert Path(path).read_bytes() == b"policy"

    result = run_experiment(
        _config(tmp_path),
        adapter_factory=lambda plan: FakeAdapter(),
    )
    assert result.policy_path == str(tmp_path / "unified-smoke.zip")
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
        adapter_state={},
    )
    monkeypatch.setattr(
        "aiogym.cli.train.run_experiment",
        lambda resolved: result,
    )
    assert cli_main(["train", "--config", str(config_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["algorithm_id"] == "sac"
    assert "--split validation" in payload["next_command"]
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
