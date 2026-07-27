from __future__ import annotations

from pathlib import Path

import pytest

import aiogym.cli.main as cli
from aiogym.catalog import list_controllers, list_scenarios, list_suites, list_tasks


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["list", "scenarios"], list_scenarios()),
        (["list", "tasks"], list_tasks()),
        (["list", "suites"], list_suites()),
        (["list", "controllers"], list_controllers()),
    ],
)
def test_list_commands_print_canonical_ids(argv, expected, capsys):
    assert cli.main(argv) is None
    assert capsys.readouterr().out.splitlines() == list(expected)


def test_list_tasks_can_filter_by_scenario(capsys):
    scenarios_with_tasks = sorted({task.split("/", 1)[0] for task in list_tasks()})
    scenario = scenarios_with_tasks[0]

    cli.main(["list", "tasks", "--scenario", scenario])

    assert capsys.readouterr().out.splitlines() == list(list_tasks(scenario))


@pytest.mark.parametrize(
    ("route", "target"),
    [
        (("benchmark", "suite"), "_benchmark"),
        (("train", "sb3"), "_train_sb3"),
        (("train", "rlpd"), "_train_rlpd"),
        (("artifacts", "report"), "_artifact_report"),
        (("artifacts", "check"), "_artifact_check"),
    ],
)
def test_unified_commands_delegate_arguments_unchanged(monkeypatch, route, target):
    received = []

    def fake_handler(argv):
        received.append(argv)
        return 17

    monkeypatch.setattr(cli, target, fake_handler)

    result = cli.main([*route, "--example", "value", "--help"])

    assert result == 17
    assert received == [["--example", "value", "--help"]]


def test_group_without_leaf_prints_group_help(capsys):
    assert cli.main(["benchmark"]) == 0
    output = capsys.readouterr().out
    assert "usage: aiogym benchmark" in output
    assert "suite" in output


def test_suite_command_requires_an_explicit_suite_id():
    from aiogym.cli.benchmark import main

    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["benchmark", "run", "quadruple", "minimum-phase"],
        ["benchmark", "--scenario", "quadruple", "--task", "minimum-phase"],
    ],
)
def test_removed_benchmark_entrypoints_are_rejected(argv):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)

    assert exc_info.value.code == 2


def test_benchmark_help_exposes_unified_target_and_task(capsys):
    from aiogym.cli.benchmark import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"], prog="aiogym benchmark suite")

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "TARGET [TASK]" in output
    assert "--setpoint-step" in output
    assert "--fail-on-degraded" in output


def test_benchmark_setpoint_parser():
    from aiogym.cli.benchmark import parse_setpoint_step

    assert parse_setpoint_step("120:13.25,11.75") == {
        "at_step": 120,
        "values": [13.25, 11.75],
    }
    with pytest.raises(ValueError, match="STEP:VALUE1"):
        parse_setpoint_step("120")


def test_console_script_metadata_exposes_only_unified_cli():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text()

    assert 'aiogym = "aiogym.cli.main:main"' in text
    for removed in (
        "aiogym-artifact-check", "aiogym-report",
        "aiogym-single-benchmark", "aiogym-suite-benchmark",
        "aiogym-train-rlpd", "aiogym-train-sb3",
    ):
        assert removed not in text
