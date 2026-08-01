"""End-to-end checks for the post-redesign public contract."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

import aiogym
from aiogym.cli.main import main
from aiogym.evaluation.artifact import (
    check_benchmark_artifacts,
    finalize_benchmark_artifacts,
)
from aiogym.evaluation.results import compact_result_row
from aiogym.rewards import get_reward_spec


def test_public_vocabulary_is_case_goal_reward_track():
    expected = {
        "make_env",
        "list_scenarios",
        "load_case",
        "list_cases",
        "load_track",
        "list_tracks",
        "list_controllers",
        "make_controller",
        "evaluate_controller",
        "__version__",
    }
    assert set(aiogym.__all__) == expected
    assert len(aiogym.__all__) <= 12
    assert {
        name for name in dir(aiogym) if not name.startswith("_")
    } == expected - {"__version__"}
    advanced = {
        "TrackSpec",
        "DistributionSpec",
        "DatasetReader",
        "RLTrainingConfig",
        "ObservationContract",
        "ProjectionSafetyShield",
        "LagrangianSAC",
        "builtin_gym_ids",
    }
    assert advanced.isdisjoint(dir(aiogym))
    assert all(not hasattr(aiogym, name) for name in advanced)


def test_controller_discovery_includes_optional_registered_ids():
    assert aiogym.list_controllers() == ("hold", "mpc", "oracle", "pid")


def test_controller_requires_explicit_model_or_scenario():
    with pytest.raises(
        ValueError,
        match="requires an explicit scenario or model",
    ):
        aiogym.make_controller("pid")


def test_models_do_not_expose_retired_gym_id_helpers():
    import aiogym.models as models

    assert "builtin_gym_ids" not in models.__all__
    assert "gym_id_name" not in models.__all__
    assert not hasattr(models, "builtin_gym_ids")
    assert not hasattr(models, "gym_id_name")


@pytest.mark.parametrize("reward_spec", ["regulation-v1", "economic-v1"])
def test_environment_and_evaluator_follow_canonical_contract(reward_spec):
    env = aiogym.make_env(
        config={
            "scenario": "cstr",
            "case": (
                "default"
                if "cstr/default" in aiogym.list_cases()
                else None
            ),
            "reward_spec": reward_spec,
            "environment": {
                "episode_steps": 2,
                "auto_events": False,
            },
        }
    )
    try:
        controller = aiogym.make_controller("pid", scenario="cstr")
        result = aiogym.evaluate_controller(controller, env, seed=4)
    finally:
        env.close()
    assert result["goal"] == get_reward_spec(reward_spec).goal
    assert result["reward_spec_id"] == reward_spec
    assert "scorecard" in result
    assert "objective" not in result
    assert "protocol" not in result


def test_import_has_no_gymnasium_registration_side_effect():
    from gymnasium.envs.registration import registry

    assert not any(name.startswith("AIOGym/") for name in registry)


def test_environment_factory_has_one_resolved_seedless_contract():
    from aiogym._environment.spec import ResolvedEnvSpec, resolve_env_spec
    from aiogym._environment.runtime import _AIOGymEnv

    direct = resolve_env_spec(
        "quadruple",
        case="minimum-phase",
        reward_spec="regulation-v1",
    )
    configured = resolve_env_spec(
        config={
            "scenario": "quadruple",
            "case": "minimum-phase",
            "reward_spec": "regulation-v1",
            "environment": {},
        }
    )
    assert direct.spec_hash == configured.spec_hash
    detached = direct.runtime()
    detached["observation"]["integral_obs"] = not (
        detached["observation"]["integral_obs"]
    )
    assert direct.spec_hash == configured.spec_hash
    assert direct.runtime() != detached

    env = aiogym.make_env(
        config={
            "scenario": "cstr",
            "reward_spec": "regulation-v1",
            "environment": {"episode_steps": 1},
        }
    )
    try:
        assert isinstance(env.env_spec, ResolvedEnvSpec)
        assert len(env.env_spec.spec_hash) == 64
        with pytest.raises(FrozenInstanceError):
            env.env_spec.scenario = "quadruple"
        with pytest.raises(TypeError):
            env.env_spec.observation["integral_obs"] = True
    finally:
        env.close()

    with pytest.raises(TypeError, match="seed"):
        aiogym.make_env("cstr", seed=4)
    with pytest.raises(ValueError, match="mutually exclusive"):
        aiogym.make_env("cstr", config={"scenario": "cstr"})
    with pytest.raises(TypeError, match="ResolvedEnvSpec"):
        _AIOGymEnv("cstr")


def test_track_has_no_environment_factory_method():
    assert not hasattr(
        aiogym.load_track("quadruple-regulation-generalist-v1"),
        "make_env",
    )


def test_compact_result_row_has_no_retired_fields():
    env = aiogym.make_env(
        config={
            "scenario": "cstr",
            "reward_spec": "regulation-v1",
            "environment": {
                "episode_steps": 1,
                "auto_events": False,
            },
        }
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
