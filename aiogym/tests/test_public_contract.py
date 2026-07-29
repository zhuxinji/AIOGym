"""End-to-end checks for the post-redesign public contract."""
from __future__ import annotations

import json

import gymnasium as gym
import pytest

import aiogym
from aiogym.cli.main import main
from aiogym.evaluation.artifact import (
    check_benchmark_artifacts,
    finalize_benchmark_artifacts,
)
from aiogym.evaluation.results import compact_result_row


def test_public_vocabulary_is_case_goal_reward_track():
    expected = {
        "AIOGymEnv",
        "GoalSpec",
        "RewardSpec",
        "TrackSpec",
        "load_case",
        "list_cases",
        "load_track",
        "list_tracks",
    }
    assert expected <= set(dir(aiogym))
    retired = {
        "AIOGymNativeEnv",
        "BenchmarkProtocol",
        "ObjectiveSpec",
        "load_task_profile",
        "list_tasks",
        "load_suite",
        "list_suites",
        "load_evaluation_artifact",
    }
    assert retired.isdisjoint(dir(aiogym))


@pytest.mark.parametrize("reward_spec", ["regulation-v1", "economic-v1"])
def test_environment_and_evaluator_follow_canonical_contract(reward_spec):
    env = aiogym.AIOGymEnv(
        "cstr",
        case="default" if "cstr/default" in aiogym.list_cases() else None,
        reward_spec=reward_spec,
        episode_steps=2,
        auto_events=False,
    )
    try:
        controller = aiogym.make_controller("pid", scenario="cstr")
        result = aiogym.evaluate_controller(controller, env, seed=4)
    finally:
        env.close()
    assert result["goal"] == aiogym.get_reward_spec(reward_spec).goal
    assert result["reward_spec_id"] == reward_spec
    assert "scorecard" in result
    assert "objective" not in result
    assert "protocol" not in result


def test_gymnasium_registration_uses_canonical_environment():
    env = gym.make(
        "AIOGym/CSTR-v0",
        reward_spec="regulation-v1",
        episode_steps=1,
        auto_events=False,
    )
    try:
        assert isinstance(env.unwrapped, aiogym.AIOGymEnv)
        observation, _ = env.reset(seed=1)
        assert observation.shape == env.observation_space.shape
    finally:
        env.close()


def test_compact_result_row_has_no_retired_fields():
    env = aiogym.AIOGymEnv(
        "cstr",
        reward_spec="regulation-v1",
        episode_steps=1,
        auto_events=False,
    )
    try:
        result = aiogym.evaluate_controller(
            aiogym.make_controller("pid", scenario="cstr"),
            env,
        )
    finally:
        env.close()
    row = compact_result_row(
        result,
        scenario="cstr",
        goal="regulation",
        run_case_id="smoke",
    )
    assert row["run_case_id"] == "smoke"
    assert row["goal"] == "regulation"
    assert {"task", "objective", "protocol", "suite_case"}.isdisjoint(row)


def test_current_artifact_round_trip_is_checkable(tmp_path):
    payload = {
        "schema_version": "aiogym.benchmark_artifact.v1",
        "benchmark": "canonical-smoke",
        "scenario": "cstr",
        "goal": "regulation",
        "rows": [],
        "results": [],
    }
    finalized = finalize_benchmark_artifacts(tmp_path, payload)
    persisted = json.loads((tmp_path / "benchmark.json").read_text())
    assert finalized["goal"] == "regulation"
    assert persisted["goal"] == "regulation"
    assert check_benchmark_artifacts(tmp_path)["ok"]


def test_cli_lists_only_canonical_resource_types(capsys):
    main(["list", "cases"])
    assert capsys.readouterr().out.strip()
    main(["list", "tracks"])
    assert capsys.readouterr().out.strip()
    with pytest.raises(SystemExit):
        main(["list", "tasks"])


def test_case_and_track_loaders_reject_unknown_ids():
    with pytest.raises(FileNotFoundError, match="unknown case ID"):
        aiogym.load_case("quadruple/not-a-case")
    with pytest.raises(FileNotFoundError, match="unknown Track ID"):
        aiogym.load_track("not-a-track")
