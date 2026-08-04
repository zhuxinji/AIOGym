from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiogym.cli.train import main as train_main
from aiogym.datasets.collect import main as collect_main
from aiogym.rl.runner import RunResult


def _result(config):
    directory = Path(config.output["directory"])
    name = config.output["name"]
    return RunResult(
        config_hash=config.config_hash,
        track_id=config.track_id,
        track_hash="a" * 64,
        algorithm_id=config.algorithm_id,
        training_seed=config.training_seed,
        output_dir=str(directory),
        policy_path=str(directory / f"{name}.zip"),
        artifact_dir=str(directory / f"{name}_artifacts"),
        resolved_config_path=str(directory / f"{name}.resolved.json"),
        validation_plan_hash="b" * 64,
        policy_sha256="c" * 64,
        backend={},
    )


def test_guided_train_resolves_canonical_config_and_run_reference(
    tmp_path,
    monkeypatch,
    capsys,
):
    captured = []

    def run(config, *, overwrite=False):
        captured.append((config, overwrite))
        return _result(config)

    monkeypatch.setattr("aiogym.cli.train.run_experiment", run)
    assert train_main(
        [
            "quadruple",
            "sac",
            "--profile",
            "quick",
            "--seed",
            "3",
            "--output",
            str(tmp_path),
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    config = captured[0][0]
    assert config.track_id == "quadruple-regulation-generalist-v1"
    assert config.output["name"] == "quadruple-sac-quick-seed3"
    assert payload["requested_target"] == "quadruple"
    assert payload["profile_id"] == "quick-v1"
    assert payload["next_command"] == (
        f"aiogym evaluate {tmp_path / 'quadruple-sac-quick-seed3.run-result.json'}"
    )


def test_guided_train_dry_run_creates_no_artifacts(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        "aiogym.cli.train.run_experiment",
        lambda *args, **kwargs: pytest.fail("runner must not execute"),
    )
    assert train_main(
        [
            "quadruple",
            "sac",
            "--dry-run",
            "--output",
            str(tmp_path),
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["track_id"] == "quadruple-regulation-generalist-v1"
    assert payload["reward_spec_id"] == "regulation-v1"
    assert payload["config"]["track_id"] == payload["track_id"]
    assert list(tmp_path.iterdir()) == []


def test_train_config_and_guided_fields_are_mutually_exclusive(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        train_main(["quadruple", "sac", "--config", str(config)])
    with pytest.raises(SystemExit):
        train_main(["--config", str(config), "--profile", "quick"])


def test_guided_collection_resolves_exact_counts(
    tmp_path,
    monkeypatch,
    capsys,
):
    captured = []

    def collect(config, *, resume=False):
        captured.append((config, resume))
        return {
            "dataset_id": config.dataset_id,
            "path": str(config.output),
            "config_hash": config.config_hash,
            "episodes": 1,
            "requested_transitions": config.target_transitions,
            "actual_transitions": config.target_transitions,
            "workers": config.workers,
        }

    monkeypatch.setattr("aiogym.datasets.collect.collect_dataset", collect)
    assert collect_main(
        [
            "quadruple",
            "--profile",
            "quick",
            "--transitions",
            "10k",
            "--workers",
            "2",
            "--seed",
            "5",
            "--output",
            str(tmp_path / "dataset"),
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    config, resume = captured[0]
    assert not resume
    assert config.track_id == "quadruple-regulation-generalist-v1"
    assert config.target_transitions == 10_000
    assert config.workers == 2
    assert config.dataset_id == "quadruple-quick-seed5"
    assert payload["profile_id"] == "quick-v1"


def test_guided_collection_dry_run_creates_no_directory(
    tmp_path,
    monkeypatch,
    capsys,
):
    output = tmp_path / "dataset"
    monkeypatch.setattr(
        "aiogym.datasets.collect.collect_dataset",
        lambda *args, **kwargs: pytest.fail("collector must not execute"),
    )
    assert collect_main(
        ["cascade", "--dry-run", "--output", str(output)]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["target_transitions"] == 1600
    assert payload["config"]["track_id"] == (
        "cascade-regulation-generalist-v1"
    )
    assert not output.exists()


def test_collect_config_rejects_guided_fields(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        collect_main(["quadruple", "--config", str(config)])
    with pytest.raises(SystemExit):
        collect_main(["--config", str(config), "--workers", "2"])
