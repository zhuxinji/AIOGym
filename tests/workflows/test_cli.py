from __future__ import annotations

import json

import pytest

from aiogym.cli.main import main


def test_cli_lists_current_resources(capsys):
    assert main(["list", "scenarios"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "cascade",
        "crystallization",
        "cstr",
        "extraction",
        "heater",
        "hvac",
        "quadruple",
        "three_tank",
    ]
    assert main(["list", "controllers"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "hold",
        "mpc",
        "pid",
        "random",
    ]
    assert main(["list", "algorithms"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "ddpg",
        "ppo",
        "rlpd",
        "sac",
        "td3",
    ]


def test_cli_lists_parameter_defaults_and_units(capsys):
    assert main(["list", "parameters", "--scenario", "quadruple"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["NAME", "DEFAULT", "UNIT"]
    assert any("tank_area" in line and "cm^2" in line for line in lines[1:])
    assert any("pump_gain" in line and "cm^3/(s*V)" in line for line in lines[1:])


def test_cli_lists_benchmark_summaries(capsys):
    assert main(["list", "benchmarks", "--scenario", "three_tank"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == [
        "ID",
        "HORIZON",
        "REWARD",
        "PARAMETERS",
        "RANKING_METRICS",
    ]
    assert any(
        line.split()
        == [
            "tracking",
            "600",
            "regulation",
            "scenario-defaults",
            "unsafe_rate,return",
        ]
        for line in lines[1:]
    )
    assert any(
            line.split()
            == [
                "disturbance-rejection",
                "600",
                "regulation",
            "scenario-defaults",
            "unsafe_rate,return",
        ]
        for line in lines[1:]
    )


def test_cli_parameter_listing_requires_scenario(capsys):
    with pytest.raises(SystemExit):
        main(["list", "parameters"])
    assert "--scenario is required for parameters" in capsys.readouterr().err


def test_cli_help_describes_the_small_workflow_set(capsys):
    assert main([]) == 0
    output = capsys.readouterr().out
    assert "collect an episode-oriented Dataset" in output
    assert "compare controllers and checkpoints on identical seeds" in output


def test_cli_workflow_help_explains_arguments_and_hides_invalid_benchmark(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["train", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--steps" in output
    assert "--algorithm-kwargs" in output
    assert "--dataset" in output
    assert "--resume-from MODEL_ZIP" in output
    assert "--benchmark" not in output
    assert "--boundary-probability PROBABILITY" in output
    assert "--no-randomize" in output
    assert "Training and output:" in output
    assert "Validation and progress:" in output
    assert "Examples:" in output
    assert "--steps adds steps" in output
    assert "RLPD requires a compatible Dataset" in output

    with pytest.raises(SystemExit) as exit_info:
        main(["compare", "--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "Compare controllers and checkpoints on identical seeds" in output
    assert "LABEL MODEL_ZIP" in output
    assert "--boundary-probability PROBABILITY" in output


def test_non_training_cli_does_not_discover_algorithm_plugins(monkeypatch, capsys):
    from aiogym.rl import algorithms

    def broken_discovery():
        raise RuntimeError("broken training plugin")

    monkeypatch.setattr(algorithms, "list_algorithms", broken_discovery)
    with pytest.raises(SystemExit) as exit_info:
        main(["compare", "--help"])
    assert exit_info.value.code == 0
    assert "Compare controllers and checkpoints" in capsys.readouterr().out


def test_cli_collects_dataset_with_file_parameters(tmp_path, capsys):
    parameters = tmp_path / "parameters.json"
    parameters.write_text(
        json.dumps({"pump_flow_max": 20.0 / 60000.0}), encoding="utf-8"
    )
    output = tmp_path / "dataset"
    assert (
        main(
            [
                "collect",
                "three_tank",
                "--parameters",
                str(parameters),
                "--disturbance",
                "on",
                "--randomize",
                "--boundary-probability",
                "0.3",
                "--controller",
                "hold",
                "--max-steps",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert result == {
        "episodes": 1,
        "output": str(output.resolve()),
        "schema_version": "aiogym.collect.v2",
        "transitions": 2,
    }
    assert metadata["schema_version"] == "aiogym.dataset.v3"
    assert metadata["environment"]["parameters"]["pump_flow_max"] == pytest.approx(
        20.0 / 60000.0
    )
    assert metadata["environment"]["disturbance"] is True
    assert metadata["environment"]["randomize"] is True
    assert metadata["environment"]["boundary_probability"] == pytest.approx(0.3)
    assert result["transitions"] == 2


def test_cli_evaluates_to_one_json(tmp_path, capsys):
    output = tmp_path / "evaluation.json"
    assert (
        main(
            [
                "evaluate",
                "quadruple",
                "--controller",
                "pid",
                "--seeds",
                "3",
                "4",
                "--max-steps",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert summary["seeds"] == [3, 4]
    assert summary["output"] == str(output.resolve())
    assert summary["aggregate"] == result["aggregate"]
    assert summary["policy"] == result["policy"]
    assert summary["ranking_metrics"] == result["ranking_metrics"]
    assert "episodes" not in summary
    assert result["episodes"][0]["length"] == 2


def test_cli_compares_to_default_benchmark_directory(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "compare",
                "three_tank",
                "--benchmark",
                "tracking",
                "--controllers",
                "pid",
                "mpc",
                "--seeds",
                "0",
                "--max-steps",
                "2",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    output = tmp_path / "runs" / "three-tank" / "benchmarks" / "tracking"
    result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert summary["seeds"] == [0]
    assert summary["output"] == str(output.resolve())
    assert [row["policy"] for row in summary["ranking"]] == result["ordering"]
    assert "evaluations" not in summary
    assert (output / "comparison.svg").is_file()
    assert (output / "trajectories.npz").is_file()


@pytest.mark.parametrize(
    ("randomize_args", "randomize"),
    [([], True), (["--randomize"], True), (["--no-randomize"], False)],
)
def test_cli_train_builds_env_calls_workflow_and_closes(
    tmp_path, capsys, monkeypatch, randomize_args, randomize
):
    import aiogym

    real_make_env = aiogym.make_env
    captured = {}

    def make_env(*args, **kwargs):
        env = real_make_env(*args, **kwargs)
        original_close = env.close

        def close():
            captured["closed"] = True
            original_close()

        env.close = close
        return env

    def train(**kwargs):
        from aiogym.workflows._metadata import environment_metadata

        captured.update(kwargs)
        return {
            "schema_version": "aiogym.training.v14",
            "path": str(output.resolve()),
            "algorithm": "sac",
            "seed": kwargs["seed"],
            "record_every": kwargs["record_every"],
            "environment": environment_metadata(kwargs["env"]),
            "evaluation": None,
            "initial_steps": 0,
            "added_steps": kwargs["steps"],
            "actual_steps": kwargs["steps"],
            "resume_from": None,
            "checkpoint": str((output / "model.zip").resolve()),
            "best_checkpoint": None,
            "training_curve": str((output / "training_curve.json").resolve()),
            "behavior_cloning_artifact": str(
                (output / "behavior_cloning.json").resolve()
            ),
        }

    monkeypatch.setattr(aiogym, "make_env", make_env)
    monkeypatch.setattr(aiogym, "train", train)
    output = tmp_path / "train"
    dataset = tmp_path / "dataset"
    assert (
        main(
            [
                "train",
                "quadruple",
                "sac",
                *randomize_args,
                "--record-every",
                "1",
                "--dataset",
                str(dataset),
                "--behavior-cloning-epochs",
                "3",
                "--behavior-cloning-batch-size",
                "4",
                "--behavior-cloning-learning-rate",
                "0.002",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    streams = capsys.readouterr()
    summary = json.loads(streams.out)
    assert "final; no validation selection" in streams.err
    assert "--no-evaluation" in streams.err
    assert str(output / "model.zip") in streams.err
    assert summary["schema_version"] == "aiogym.training.v14"
    assert summary["algorithm"] == "sac"
    assert summary["initial_steps"] == 0
    assert summary["added_steps"] == 500_000
    assert summary["actual_steps"] == 500_000
    assert summary["resume_from"] is None
    assert summary["output"] == str(output.resolve())
    assert summary["behavior_cloning_artifact"] == str(
        (output / "behavior_cloning.json").resolve()
    )
    assert captured["env"].unwrapped.scenario.id == "quadruple"
    assert captured["env"].unwrapped.runtime_config["randomize"] is randomize
    assert captured["algorithm"] == "sac"
    assert captured["steps"] == 500_000
    assert captured["record_every"] == 1
    assert captured["evaluation_env"].unwrapped.runtime_config["randomize"] is True
    assert captured["evaluate_every"] == 5_000
    assert captured["dataset"] == dataset
    assert captured["behavior_cloning_epochs"] == 3
    assert captured["behavior_cloning_batch_size"] == 4
    assert captured["behavior_cloning_learning_rate"] == pytest.approx(0.002)
    assert captured["resume_from"] is None
    assert captured["closed"] is True


def test_cli_compare_combines_controllers_and_checkpoints(
    tmp_path, capsys, monkeypatch
):
    import aiogym

    captured = {}
    learned_policy = object()

    def load_policy(checkpoint, *, env):
        captured["load"] = (checkpoint, env)
        return learned_policy

    def compare_policies(**kwargs):
        captured["compare"] = kwargs
        evaluations = {
            label: {"aggregate": {"return": {"mean": float(index)}}}
            for index, label in enumerate(kwargs["policies"])
        }
        return {
            "schema_version": "aiogym.comparison.v6",
            "seeds": list(kwargs["seeds"]),
            "ranking_metrics": [
                {
                    "name": "return",
                    "direction": "maximize",
                    "aggregate": "mean",
                }
            ],
            "ordering": list(kwargs["policies"]),
            "evaluations": evaluations,
        }

    monkeypatch.setattr(aiogym, "load_policy", load_policy)
    monkeypatch.setattr(aiogym, "compare_policies", compare_policies)
    checkpoint = tmp_path / "model.zip"
    output = tmp_path / "comparison"
    assert (
        main(
            [
                "compare",
                "quadruple",
                "--controllers",
                "pid",
                "mpc",
                "--checkpoint",
                "sac-best",
                str(checkpoint),
                "--seeds",
                "0",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert captured["load"][0] == checkpoint
    assert captured["load"][1].scenario.id == "quadruple"
    assert captured["compare"]["policies"] == {
        "pid": "pid",
        "mpc": "mpc",
        "sac-best": learned_policy,
    }
    assert summary["output"] == str(output.resolve())
    assert [row["policy"] for row in summary["ranking"]] == [
        "pid",
        "mpc",
        "sac-best",
    ]


@pytest.mark.parametrize("payload", ("{", "[]"))
def test_cli_rejects_invalid_parameter_files(tmp_path, capsys, payload):
    parameters = tmp_path / "parameters.json"
    parameters.write_text(payload, encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "collect",
                "quadruple",
                "--parameters",
                str(parameters),
                "--output",
                str(tmp_path / "dataset"),
            ]
        )
    assert "parameters" in capsys.readouterr().err
