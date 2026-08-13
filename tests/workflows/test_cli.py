from __future__ import annotations

import json

import pytest

from aiogym.cli.main import main


def test_cli_lists_current_resources(capsys):
    assert main(["list", "scenarios"]) == 0
    assert capsys.readouterr().out.splitlines() == ["quadruple", "three_tank"]
    assert main(["list", "controllers"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "hold",
        "mpc",
        "pid",
        "random",
    ]
    assert main(["list", "algorithms"]) == 0
    assert capsys.readouterr().out.splitlines() == ["ddpg", "ppo", "sac", "td3"]


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
        "MEASUREMENT_NOISE",
        "RANKING_METRICS",
    ]
    assert any(
        line.split()
        == [
            "tracking",
            "4200",
            "regulation",
            "scenario-defaults",
            "std=0.001,bias_std=0",
            "unsafe_rate,tracking_iae",
        ]
        for line in lines[1:]
    )
    assert any(
        line.split()
        == [
            "disturbance-rejection",
            "2400",
            "regulation",
            "scenario-defaults",
            "std=0.001,bias_std=0",
            "unsafe_rate,disturbance_iae,recovery_time",
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
    assert "compare built-in controllers on identical seeds" in output


def test_cli_collects_dataset_with_file_parameters(tmp_path, capsys):
    parameters = tmp_path / "parameters.json"
    parameters.write_text(json.dumps({"heater_power": 1800.0}), encoding="utf-8")
    output = tmp_path / "dataset"
    assert main([
        "collect",
        "three_tank",
        "--parameters",
        str(parameters),
        "--controller",
        "hold",
        "--max-steps",
        "2",
        "--output",
        str(output),
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["metadata"]["schema_version"] == "aiogym.dataset.v2"
    assert result["metadata"]["environment"]["parameters"]["heater_power"] == 1800.0
    assert result["transitions"] == 2


def test_cli_evaluates_to_one_json(tmp_path, capsys):
    output = tmp_path / "evaluation.json"
    assert main([
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
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["seeds"] == [3, 4]
    assert result["episodes"][0]["length"] == 2
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_cli_compares_to_default_benchmark_directory(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main([
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
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    output = tmp_path / "runs" / "three-tank" / "tracking"
    assert result["seeds"] == [0]
    assert json.loads((output / "comparison.json").read_text(encoding="utf-8")) == result
    assert (output / "comparison.svg").is_file()


def test_cli_train_builds_env_calls_workflow_and_closes(tmp_path, capsys, monkeypatch):
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
        captured.update(kwargs)
        return {"schema_version": "aiogym.training.v4"}

    monkeypatch.setattr(aiogym, "make_env", make_env)
    monkeypatch.setattr(aiogym, "train", train)
    output = tmp_path / "train"
    assert main([
        "train",
        "quadruple",
        "sac",
        "--steps",
        "2",
        "--record-every",
        "1",
        "--output",
        str(output),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "aiogym.training.v4"
    assert captured["env"].scenario.id == "quadruple"
    assert captured["algorithm"] == "sac"
    assert captured["steps"] == 2
    assert captured["record_every"] == 1
    assert captured["closed"] is True


@pytest.mark.parametrize("payload", ("{", "[]"))
def test_cli_rejects_invalid_parameter_files(tmp_path, capsys, payload):
    parameters = tmp_path / "parameters.json"
    parameters.write_text(payload, encoding="utf-8")
    with pytest.raises(SystemExit):
        main([
            "collect",
            "quadruple",
            "--parameters",
            str(parameters),
            "--output",
            str(tmp_path / "dataset"),
        ])
    assert "parameters" in capsys.readouterr().err
