from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiogym.cli import run as run_cli
from aiogym.experiments import ExperimentSpec, run_experiment_spec
from aiogym.rl.runner import RunResult


def _declaration(tmp_path: Path) -> dict:
    return {
        "schema_version": "aiogym.experiment.v1",
        "name": "quadruple-sac-demo",
        "description": "Small workflow test.",
        "output_dir": str(tmp_path / "run"),
        "seeds": [0],
        "training": {
            "schema_version": "aiogym.rl_training_config.v3",
            "track_id": "quadruple-regulation-generalist-v2",
            "algorithm_id": "sac",
            "budget": {
                "unit": "environment_transitions",
                "value": 100,
            },
            "n_envs": 1,
            "algorithm": {"vector_backend": "dummy"},
            "output": {"onnx": False},
            "validation_seeds": [5000],
        },
    }


def _result(config) -> RunResult:
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
        validation={
            "aggregate": {
                "metric": "regulation_cost_rate",
                "metric_direction": "minimize",
                "metric_value": 1.25,
                "official_score": 80.0,
                "ranking_eligible": True,
            }
        },
    )


def test_experiment_spec_owns_seeds_and_output_paths(tmp_path):
    spec = ExperimentSpec.from_mapping(_declaration(tmp_path))
    config = spec.training_config()

    assert spec.seeds == (0,)
    assert config.training_seed == 0
    assert config.output["directory"] == str(tmp_path / "run/internal/training")
    assert config.output["name"] == "model"
    assert spec.as_dict()["training"].get("training_seed") is None


def test_experiment_dry_run_creates_nothing(tmp_path):
    spec = ExperimentSpec.from_mapping(_declaration(tmp_path))

    result = run_experiment_spec(spec, dry_run=True)

    assert result["dry_run"] is True
    assert result["public_artifacts"]["run"].endswith("run/run.json")
    assert not spec.output_dir.exists()


def test_experiment_writes_small_public_surface(tmp_path, monkeypatch):
    spec = ExperimentSpec.from_mapping(_declaration(tmp_path))
    captured = []

    def fake_run(config, *, overwrite=False):
        captured.append((config, overwrite))
        result = _result(config)
        Path(result.policy_path).parent.mkdir(parents=True, exist_ok=True)
        Path(result.policy_path).write_bytes(b"policy")
        Path(result.resolved_config_path).write_text("{}", encoding="utf-8")
        Path(result.resolved_config_path).with_name(
            "model.run-result.json"
        ).write_text("{}", encoding="utf-8")
        return result

    monkeypatch.setattr("aiogym.experiments.run_experiment", fake_run)
    result = run_experiment_spec(spec)

    assert result["status"] == "complete"
    assert result["models"][0]["path"] == "internal/training/model.zip"
    assert sorted(path.name for path in spec.output_dir.iterdir()) == [
        "internal",
        "metrics.json",
        "report.md",
        "run.json",
    ]
    assert json.loads((spec.output_dir / "metrics.json").read_text())[
        "validation_summary"
    ]["mean_metric_value"] == 1.25
    assert captured[0][1] is False


def test_run_cli_loads_one_config_and_supports_dry_run(
    tmp_path, capsys
):
    config = tmp_path / "experiment.json"
    config.write_text(json.dumps(_declaration(tmp_path)), encoding="utf-8")

    assert run_cli.main([str(config), "--dry-run"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "quadruple-sac-demo"
    assert payload["track_id"] == "quadruple-regulation-generalist-v2"


def test_multi_seed_experiment_keeps_details_internal(tmp_path, monkeypatch):
    declaration = _declaration(tmp_path)
    declaration["seeds"] = [0, 1, 2]
    spec = ExperimentSpec.from_mapping(declaration)

    def fake_sweep(config, seeds, *, overwrite=False):
        runs = []
        for seed in seeds:
            run = _result(config)
            row = run.as_dict()
            row["training_seed"] = seed
            row["policy_path"] = str(spec.training_dir / f"model-seed{seed}.zip")
            runs.append(row)
        return {
            "summary_path": str(spec.training_dir / "model.multi-seed.json"),
            "runs": runs,
            "validation_summary": {"run_count": 3, "eligible_runs": 3},
            "training_seed_statistics": {"training_seeds": [0, 1, 2]},
        }

    monkeypatch.setattr("aiogym.experiments.run_seed_sweep", fake_sweep)
    result = run_experiment_spec(spec)

    assert len(result["models"]) == 3
    assert result["internal"]["training_result"] == (
        "internal/training/model.multi-seed.json"
    )
    assert json.loads((spec.output_dir / "metrics.json").read_text())[
        "training_seed_statistics"
    ]["training_seeds"] == [0, 1, 2]


def test_experiment_rejects_partial_public_output_before_training(
    tmp_path, monkeypatch
):
    spec = ExperimentSpec.from_mapping(_declaration(tmp_path))
    spec.output_dir.mkdir(parents=True)
    (spec.output_dir / "metrics.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "aiogym.experiments.run_experiment",
        lambda *args, **kwargs: pytest.fail("training must not start"),
    )

    with pytest.raises(FileExistsError, match="metrics.json"):
        run_experiment_spec(spec)


def test_experiment_rejects_nested_output_ownership(tmp_path):
    declaration = _declaration(tmp_path)
    declaration["training"]["output"]["directory"] = "elsewhere"

    with pytest.raises(ValueError, match="experiment owns output paths"):
        ExperimentSpec.from_mapping(declaration)
