"""Public API checks after removing the pre-redesign compatibility surface."""
from __future__ import annotations

import importlib.util
import inspect

import pytest

import aiogym
from aiogym.cli.main import main as cli_main
from aiogym.models.cases import CASE_PROFILE_SCHEMA_VERSION


@pytest.mark.parametrize(
    "retired_name",
    (
        "AIOGymEnv",
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


def test_legacy_transition_and_compat_packages_are_absent():
    assert importlib.util.find_spec("aiogym.rl.transitions") is None
    assert importlib.util.find_spec("aiogym.compat") is None
    assert importlib.util.find_spec("aiogym.datasets.migration") is None
    assert not hasattr(aiogym, "TransitionDataset")


@pytest.mark.rl
def test_rlpd_does_not_define_a_second_replay_buffer():
    from aiogym.rl import rlpd

    assert not hasattr(rlpd, "ReplayBuffer")


def test_backend_modules_are_not_standalone_cli_entrypoints():
    assert importlib.util.find_spec("aiogym.rl.train_offline") is None
    assert importlib.util.find_spec("aiogym.rl.train_rlpd") is None
    assert importlib.util.find_spec("aiogym.rl.train_sb3") is None


def test_backend_kernels_accept_only_resolved_plans():
    import inspect
    from aiogym.rl.backends.bc import run_bc
    from aiogym.rl.backends.rlpd import run_rlpd
    from aiogym.rl.backends.sb3 import run_sb3

    for backend in (run_bc, run_rlpd, run_sb3):
        assert tuple(inspect.signature(backend).parameters) == ("plan",)


def test_checkpoint_cli_has_one_canonical_option_family():
    from aiogym.cli.benchmark import build_parser

    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    assert {"--checkpoint", "--algorithm", "--sha256"} <= options
    assert {
        "--policy-path",
        "--policy-algorithm",
        "--policy-sha256",
        "--sb3-path",
        "--sb3-algo",
        "--onnx-path",
        "--checkpoint-sha256",
    }.isdisjoint(options)


@pytest.mark.rl
def test_removed_trainer_helpers_and_physical_rlpd_api_are_absent():
    from aiogym.rl.rlpd import RLPD, RLPD_STATE_SCHEMA_VERSION

    assert RLPD_STATE_SCHEMA_VERSION == "aiogym.rlpd_state.v3"
    assert not hasattr(RLPD, "push_physical")
    assert importlib.util.find_spec("aiogym.rl.training_config") is None


def test_legacy_randomization_compiler_is_retired():
    import aiogym.generation as generation

    assert not hasattr(generation, "compile_legacy_distribution")


def test_canonical_case_and_track_api():
    case = aiogym.load_case("quadruple/minimum-phase")
    track = aiogym.load_track("quadruple-regulation-generalist-v1")

    assert case["schema_version"] == CASE_PROFILE_SCHEMA_VERSION
    assert track.id in aiogym.list_tracks()


def test_controller_factory_has_no_retired_policy_injection():
    assert tuple(inspect.signature(aiogym.make_controller).parameters) == (
        "name",
        "model",
        "scenario",
        "config",
    )


def test_environment_uses_case_and_rejects_task_keyword():
    env = aiogym.make_env(
        config={
            "scenario": "quadruple",
            "case": "minimum-phase",
            "reward_spec": "regulation-v1",
            "environment": {"episode_steps": 1},
        }
    )
    try:
        observation, _ = env.reset(seed=7)
        assert env.observation_space.contains(observation)
        assert env.case_profile["name"] == "minimum-phase"
        assert not hasattr(env, "task_profile")
    finally:
        env.close()

    with pytest.raises(TypeError, match="unexpected keyword argument 'task'"):
        aiogym.make_env("quadruple", task="minimum-phase")


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
    assert "quadruple-regulation-generalist-v2" in capsys.readouterr().out
