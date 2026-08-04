"""Checks for responsibility-based package boundaries."""

import importlib.util
import subprocess
import sys


def test_top_level_import_keeps_optional_feature_groups_lazy():
    code = """
import sys
import aiogym
from gymnasium.envs.registration import registry

unexpected = {
    "aiogym.compat",
    "aiogym.controllers",
    "aiogym.evaluation",
    "aiogym.rl",
    "casadi",
    "onnx",
    "onnxruntime",
    "optuna",
    "stable_baselines3",
    "torch",
}.intersection(sys.modules)
assert not unexpected, sorted(unexpected)
assert len(aiogym.__all__) <= 12
assert not any(name.startswith("AIOGym/") for name in registry)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_catalog_remains_the_top_level_discovery_facade():
    import aiogym
    from aiogym import catalog

    assert set(catalog.__all__) == {
        "list_cases",
        "list_controllers",
        "list_scenarios",
    }
    assert aiogym.list_cases is catalog.list_cases
    assert aiogym.list_controllers is catalog.list_controllers
    assert aiogym.list_scenarios is catalog.list_scenarios
    assert not hasattr(catalog, "build_parser")
    assert not hasattr(catalog, "main")


def test_controller_facade_exposes_only_stable_construction_contract():
    import aiogym.controllers as facade
    from aiogym.controllers import contracts, registry

    assert set(facade.__all__) == {
        "Controller",
        "make_controller",
        "register_controller",
        "unregister_controller",
    }
    assert facade.Controller is contracts.Controller
    assert facade.make_controller is registry.make_controller
    assert not hasattr(facade, "LearnedPolicySpec")
    assert not hasattr(facade, "PolicyController")


def test_evaluation_facade_exposes_only_happy_path_workflows():
    import aiogym.evaluation as public
    from aiogym.evaluation import execution

    assert len(public.__all__) == 4
    assert public.evaluate_controller is execution.evaluate_controller
    assert not hasattr(public, "rollout_controller")
    assert not hasattr(public, "GoalSpec")
    assert not hasattr(public, "ScorecardAccumulator")
    assert importlib.util.find_spec("aiogym.evaluation.core") is None


def test_model_core_uses_backend_and_integrator_implementations():
    import aiogym.models.core as core
    from aiogym.models import backends, integration

    assert core.Integrator is integration.Integrator
    assert core._NUMERIC_OPS is backends._NUMERIC_OPS
    assert core._casadi_ops is backends._casadi_ops
    assert core._maxv is backends._maxv


def test_model_metadata_has_a_single_current_module():
    from aiogym.models import metadata

    assert importlib.util.find_spec("aiogym.models.cards") is None
    assert metadata.MODEL_METADATA_SCHEMA_VERSION == "aiogym.model_metadata.v1"


def test_backend_contract_is_separate_from_dispatcher():
    from aiogym.rl import backends
    from aiogym.rl.backends import contracts

    assert backends.BackendResult is contracts.BackendResult
    assert backends.validate_backend_result is contracts.validate_backend_result


def test_generation_public_api_exposes_episode_contracts():
    import aiogym.generation as public
    from aiogym.generation import specs

    assert len(public.__all__) == 5
    assert public.DistributionSpec is specs.DistributionSpec
    assert public.EpisodeSpec is specs.EpisodeSpec
    assert not hasattr(public, "SeedTree")
    assert not hasattr(public, "FixedCaseEpisodeSampler")


def test_dataset_public_api_exposes_persistent_v2_backend():
    import aiogym.datasets as public
    from aiogym.datasets import reader

    assert len(public.__all__) == 4
    assert public.DatasetReader is reader.DatasetReader
    assert not hasattr(public, "DatasetEpisode")
    assert not hasattr(public, "DatasetWriter")


def test_rl_public_api_is_limited_to_config_and_runner():
    import aiogym.rl as public
    from aiogym.rl import config, runner

    assert len(public.__all__) == 5
    assert public.RLTrainingConfig is config.RLTrainingConfig
    assert public.RunResult is runner.RunResult
    assert public.run_experiment is runner.run_experiment
    assert not hasattr(public, "BackendResult")
    assert not hasattr(public, "CheckpointManager")
    assert public.list_algorithms is config.list_algorithms
    assert public.list_algorithms() == ("bc", "ppo", "rlpd", "sac", "td3")
    assert {
        "EpisodeCoordinator",
        "NormalizedActionWrapper",
        "UTDController",
        "DatasetReplay",
        "CompleteValidationCallback",
        "FinalTestLock",
    }.isdisjoint(dir(public))
    assert importlib.util.find_spec("aiogym.rl.adapters") is None


def test_rl_hybrid_contracts_require_explicit_modules():
    import aiogym.rl as public
    from aiogym.rl import behavior_cloning, dataset_replay, hybrid_replay

    assert not hasattr(public, "DatasetReplay")
    assert dataset_replay.DatasetReplay
    assert behavior_cloning.BehaviorCloningTrainer
    assert hybrid_replay.RLPDBatchSampler


def test_validation_contracts_require_explicit_modules():
    import aiogym.evaluation as evaluation
    import aiogym.rl as public
    from aiogym.evaluation import statistics
    from aiogym.rl import final_test, validation

    assert not hasattr(public, "CompleteValidationCallback")
    assert validation.CompleteValidationCallback
    assert final_test.FinalTestLock
    assert (
        evaluation.build_final_statistical_report
        is statistics.build_final_statistical_report
    )


def test_research_rl_api_is_explicitly_experimental():
    import aiogym
    import aiogym.evaluation as evaluation
    import aiogym.experimental.rl as experimental
    import aiogym.rl as public
    from aiogym.evaluation import statistics
    from aiogym.rl import constrained, observations, safety

    assert not hasattr(public, "ObservationContract")
    assert "experimental" not in aiogym.__all__
    assert experimental.ObservationContract is observations.ObservationContract
    assert (
        experimental.ProjectionSafetyShield
        is safety.ProjectionSafetyShield
    )
    assert experimental.LagrangianSAC is constrained.LagrangianSAC
    assert not hasattr(evaluation, "build_intervention_report")
    assert statistics.build_intervention_report


def test_environment_class_composes_focused_runtime_mixins():
    from aiogym._environment.disturbances import DisturbanceRuntimeMixin
    from aiogym._environment.observations import ObservationRuntimeMixin
    from aiogym._environment.transitions import TransitionRuntimeMixin
    from aiogym._environment.env import _AIOGymEnv

    assert _AIOGymEnv._env is DisturbanceRuntimeMixin._env
    assert _AIOGymEnv._obs is ObservationRuntimeMixin._obs
    assert _AIOGymEnv.evaluate_transition is TransitionRuntimeMixin.evaluate_transition


def test_environment_runtime_has_one_canonical_module():
    from aiogym._environment.env import _AIOGymEnv

    assert _AIOGymEnv.__module__ == "aiogym._environment.env"
    assert importlib.util.find_spec("aiogym._environment.runtime") is None


def test_generic_tools_package_is_removed():
    assert importlib.util.find_spec("aiogym.tools") is None


def test_unified_benchmark_cli_uses_track_modules():
    from aiogym.cli import benchmark as facade
    from aiogym.benchmarks import load_track

    assert importlib.util.find_spec("aiogym.cli.single_benchmark") is None
    assert importlib.util.find_spec("aiogym.cli.suite_benchmark") is None
    assert facade.load_track is load_track


def test_artifact_adapters_have_role_specific_module_names():
    from aiogym.cli import artifact_commands
    from aiogym.evaluation import artifact
    from aiogym.rl import training_artifacts

    assert importlib.util.find_spec("aiogym.cli.artifact_tools") is None
    assert importlib.util.find_spec("aiogym.rl.artifacts") is None
    assert artifact_commands.check_benchmark_artifacts is artifact.check_benchmark_artifacts
    assert (
        training_artifacts.finalize_benchmark_artifacts
        is artifact.finalize_benchmark_artifacts
    )


def test_removed_evaluation_modules_are_absent():
    for module in (
        "aiogym.evaluation.aggregation",
        "aiogym.evaluation.artifact_checks",
        "aiogym.evaluation.artifact_plotting",
        "aiogym.evaluation.artifact_tables",
        "aiogym.evaluation.artifact_writers",
        "aiogym.evaluation.artifacts",
        "aiogym.evaluation.benchmark",
        "aiogym.evaluation.evaluator",
        "aiogym.evaluation.legacy_artifacts",
        "aiogym.evaluation.metadata",
        "aiogym.evaluation.plots",
        "aiogym.evaluation.report_rendering",
        "aiogym.evaluation.reports",
        "aiogym.evaluation.rollouts",
        "aiogym.evaluation.rows",
        "aiogym.evaluation.runner",
        "aiogym.evaluation.suite_cases",
        "aiogym.evaluation.suite_loading",
        "aiogym.evaluation.suite_results",
        "aiogym.evaluation.task_profiles",
        "aiogym.evaluation.task_acceptance",
        "aiogym.evaluation.protocols",
        "aiogym.evaluation.objective_specs",
        "aiogym.evaluation.suite",
    ):
        assert importlib.util.find_spec(module) is None


def test_superseded_environment_and_rl_modules_are_absent():
    for module in (
        "aiogym.benchmarks.sampler",
        "aiogym.generation.protocols",
        "aiogym.rl.transitions",
        "aiogym.rl.environment_factory",
    ):
        assert importlib.util.find_spec(module) is None


def test_live_execution_does_not_import_offline_migration_modules():
    code = """
import sys
from aiogym.rl import RLTrainingConfig, run_experiment
from aiogym.evaluation import evaluate_controller
assert "aiogym.compat" not in sys.modules
assert "aiogym.evaluation.legacy_artifacts" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
