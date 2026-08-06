"""CLI contract tests for user-facing resource discovery."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from aiogym.benchmarks.tracks.registry import list_tracks
from aiogym.cli.main import main
from aiogym.rewards.registry import get_reward_spec


def test_list_tracks_defaults_to_selectors_and_explanatory_columns(capsys):
    assert main(["list", "tracks"]) is None
    output = capsys.readouterr().out

    assert output.splitlines()[0].split() == [
        "SELECTOR",
        "GOAL",
        "SCOPE",
        "CANONICAL",
        "ID",
    ]
    assert "quadruple" in output
    assert "cascade:economic" in output
    assert "quadruple-regulation-generalist-v2" in output
    assert "quadruple-regulation-generalist-v1" not in output


def test_list_tracks_ids_preserves_canonical_script_surface(capsys):
    main(["list", "tracks", "--ids"])

    assert capsys.readouterr().out.splitlines() == list(list_tracks())


def test_list_tracks_all_includes_compatibility_versions(capsys):
    main(["list", "tracks", "--all"])
    output = capsys.readouterr().out

    assert "quadruple-regulation-generalist-v1" in output
    assert "quadruple-regulation-generalist-v2" in output

    main(["list", "tracks", "--all", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert any(
        row["track_id"] == "quadruple-regulation-generalist-v1"
        for row in rows
    )


def test_list_tracks_structured_output_and_option_conflict(capsys):
    main(["list", "tracks", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["selector"] == "quadruple"
    assert all("track_id" in row for row in rows)

    with pytest.raises(SystemExit):
        main(["list", "tracks", "--ids", "--json"])


def test_list_rewards_and_profiles_are_filterable(capsys):
    main(["list", "rewards", "--json"])
    rewards = json.loads(capsys.readouterr().out)
    assert [row["selector"] for row in rewards] == [
        "regulation",
        "economic",
    ]

    main(
        [
            "list",
            "profiles",
            "--kind",
            "training",
            "--target",
            "quadruple",
            "--algorithm",
            "sac",
            "--json",
        ]
    )
    profiles = json.loads(capsys.readouterr().out)
    assert len(profiles) == 1
    assert profiles[0]["profile_id"] == "quick-v1"
    assert profiles[0]["track_id"] == (
        "quadruple-regulation-generalist-v2"
    )

    main(
        [
            "list",
            "profiles",
            "--kind",
            "collection",
            "--target",
            "cascade-recirculating",
            "--json",
        ]
    )
    collection_profiles = json.loads(capsys.readouterr().out)
    assert len(collection_profiles) == 1
    assert collection_profiles[0]["profile_id"] == "quick-v1"
    assert collection_profiles[0]["track_id"] == (
        "cascade-recirculating-regulation-generalist-v2"
    )


def test_describe_commands_read_canonical_registry_values(capsys):
    main(["describe", "track", "quadruple", "--json"])
    track = json.loads(capsys.readouterr().out)
    assert track["track_id"] == "quadruple-regulation-generalist-v2"
    assert track["reward_spec_id"] == "regulation-v1"

    main(["describe", "reward", "regulation", "--json"])
    reward = json.loads(capsys.readouterr().out)
    canonical = get_reward_spec("regulation-v1")
    assert reward["term_weights"] == dict(canonical.term_weights)
    assert reward["cost_weights"] == dict(canonical.cost_weights)
    assert reward["reward_spec_hash"] == canonical.spec_hash

    main(
        [
            "describe",
            "profile",
            "quick",
            "--target",
            "quadruple",
            "--algorithm",
            "sac",
            "--json",
        ]
    )
    profile = json.loads(capsys.readouterr().out)
    assert profile["source"] == (
        "package:aiogym/resources/profiles/training/quadruple/sac-quick-v1.json"
    )


def test_discovery_commands_do_not_eager_import_optional_frameworks():
    script = """
import sys
from aiogym.cli.main import main
main(['list', 'tracks'])
main(['list', 'rewards'])
main(['list', 'profiles'])
main(['describe', 'track', 'quadruple'])
loaded = sorted(
    name for name in ('stable_baselines3', 'onnxruntime', 'casadi')
    if name in sys.modules
)
print('OPTIONAL=' + repr(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines()[-1] == "OPTIONAL=[]"
