"""Public API checks after removing the pre-redesign compatibility surface."""
from __future__ import annotations

import pytest

import aiogym
from aiogym.cli.main import main as cli_main


@pytest.mark.parametrize(
    "retired_name",
    (
        "AIOGymNativeEnv",
        "BenchmarkProtocol",
        "ObjectiveSpec",
        "TASK_PROFILE_SCHEMA_VERSION",
        "list_tasks",
        "list_suites",
        "load_benchmark_track",
        "load_case_profile",
        "load_task_profile",
        "objective_spec",
        "resolve_objective",
        "reward_mode_for_objective",
        "run_benchmark",
        "validate_task_profile",
    ),
)
def test_retired_names_are_not_public(retired_name):
    assert not hasattr(aiogym, retired_name)


def test_canonical_case_and_track_api():
    case = aiogym.load_case("quadruple/minimum-phase")
    track = aiogym.load_track("quadruple-regulation-generalist-v1")

    assert case["schema_version"] == aiogym.CASE_PROFILE_SCHEMA_VERSION
    assert track.id in aiogym.list_tracks()


def test_environment_uses_case_and_rejects_task_keyword():
    env = aiogym.AIOGymEnv(
        "quadruple",
        case="minimum-phase",
        reward_spec="regulation-v1",
        episode_steps=1,
    )
    try:
        observation, _ = env.reset(seed=7)
        assert env.observation_space.contains(observation)
        assert env.case_profile["name"] == "minimum-phase"
        assert not hasattr(env, "task_profile")
    finally:
        env.close()

    with pytest.raises(TypeError, match="unexpected keyword argument 'task'"):
        aiogym.AIOGymEnv("quadruple", task="minimum-phase")


def test_task_v1_input_is_not_migrated():
    legacy = {
        "schema_version": "aiogym.task_profile.v1",
        "name": "old",
        "scenario": "quadruple",
        "status": "legacy",
        "environment": {},
    }

    with pytest.raises(ValueError, match="unsupported case profile schema"):
        aiogym.load_case(legacy)


def test_cli_discovers_only_case_and_track_resources(capsys):
    assert cli_main(["list", "cases", "--scenario", "quadruple"]) is None
    assert "quadruple/minimum-phase" in capsys.readouterr().out

    assert cli_main(["list", "tracks"]) is None
    assert "quadruple-regulation-generalist-v1" in capsys.readouterr().out
