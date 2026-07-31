from __future__ import annotations

import json
from importlib.metadata import distribution
from pathlib import Path

import pytest

import aiogym
from aiogym.benchmarks.ranking import rank_track_results
from aiogym.cli.benchmark import build_parser as build_benchmark_parser
from aiogym.cli.main import build_parser


GOLDEN = Path(__file__).with_name("golden")


def test_cli_command_and_checkpoint_surfaces_match_golden_contract():
    fixture = json.loads(
        (GOLDEN / "cli-output-contract-v1.json").read_text(encoding="utf-8")
    )
    parser = build_parser()
    commands = set(parser._subparsers._group_actions[0].choices)
    assert commands == set(fixture["workflow_commands"] + fixture["support_commands"])

    benchmark = build_benchmark_parser()
    options = {
        option
        for action in benchmark._actions
        for option in action.option_strings
    }
    assert set(fixture["benchmark_checkpoint_options"]) <= options
    assert set(fixture["retired_benchmark_options"]).isdisjoint(options)


def test_package_installs_one_console_script():
    scripts = {
        entry.name: entry.value
        for entry in distribution("aiogym").entry_points
        if entry.group == "console_scripts"
    }
    assert scripts == {"aiogym": "aiogym.cli.main:main"}


@pytest.mark.parametrize(
    "filename",
    ("quadruple-ranking-v1.json", "cascade-ranking-v1.json"),
)
def test_ranking_formula_and_anchor_identity_match_golden(filename):
    fixture = json.loads((GOLDEN / filename).read_text(encoding="utf-8"))
    track = aiogym.load_track(fixture["track_id"])
    assert track.track_hash == fixture["track_hash"]
    actual = rank_track_results(fixture["inputs"], track)
    expected = fixture["expected"]
    for field in (
        "ranking_spec_id",
        "case_score_rule",
        "track_aggregation",
        "ranking_eligible",
        "anchor_id",
        "anchor_artifact_hash",
    ):
        assert actual[field] == expected[field]
    assert actual["case_weights"] == expected["case_weights"]
    assert actual["case_scores"] == pytest.approx(expected["case_scores"])
    assert actual["official_score"] == pytest.approx(expected["official_score"])
