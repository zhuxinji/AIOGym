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
    "aiogym.controllers",
    "aiogym.evaluation",
    "aiogym.rl",
    "casadi",
    "onnx",
    "onnxruntime",
    "stable_baselines3",
    "torch",
}.intersection(sys.modules)
assert not unexpected, sorted(unexpected)
assert len(aiogym.__all__) <= 12
assert not any(name.startswith("AIOGym/") for name in registry)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_controller_public_api_exposes_current_implementations():
    import aiogym.controllers as facade
    from aiogym.controllers import adapters, configs, contracts, registry

    assert facade.ControllerContext is contracts.ControllerContext
    assert facade.Controller is contracts.Controller
    assert facade.PolicyController is adapters.PolicyController
    assert facade.SB3PolicyController is adapters.SB3PolicyController
    assert facade.as_controller is adapters.as_controller
    assert facade.load_controller_config is configs.load_controller_config
    assert facade.make_controller is registry.make_controller
    assert not hasattr(facade, "registered_controllers")


def test_evaluation_public_api_exposes_current_implementations():
    import aiogym.evaluation as public
    from aiogym.evaluation import execution, goal_specs, scorecard

    assert public.evaluate_controller is execution.evaluate_controller
    assert public.rollout_controller is execution.rollout_controller
    assert public.GoalSpec is goal_specs.GoalSpec
    assert public.ScorecardAccumulator is scorecard.ScorecardAccumulator
    assert importlib.util.find_spec("aiogym.evaluation.core") is None


def test_model_core_uses_backend_and_integrator_implementations():
    import aiogym.models as public
    import aiogym.models.core as core
    from aiogym.models import backends, integration

    assert core.Integrator is integration.Integrator
    assert public.Integrator is integration.Integrator
    assert core._NUMERIC_OPS is backends._NUMERIC_OPS
    assert core._casadi_ops is backends._casadi_ops
    assert core._maxv is backends._maxv


def test_model_metadata_has_a_single_current_module():
    from aiogym.models import metadata

    assert importlib.util.find_spec("aiogym.models.cards") is None
    assert metadata.MODEL_METADATA_SCHEMA_VERSION == "aiogym.model_metadata.v1"


def test_generation_public_api_exposes_episode_contracts():
    import aiogym.generation as public
    from aiogym.generation import adapters, samplers, seed_tree, specs

    assert public.DistributionSpec is specs.DistributionSpec
    assert public.EpisodeSpec is specs.EpisodeSpec
    assert public.SeedTree is seed_tree.SeedTree
    assert (
        public.distribution_spec_from_case
        is adapters.distribution_spec_from_case
    )
    assert (
        public.FixedCaseEpisodeSampler
        is samplers.FixedCaseEpisodeSampler
    )


def test_dataset_public_api_exposes_persistent_v2_backend():
    import aiogym.datasets as public
    from aiogym.datasets import reader, schema, writer

    assert public.DatasetEpisode is schema.DatasetEpisode
    assert public.DatasetReader is reader.DatasetReader
    assert public.DatasetWriter is writer.DatasetWriter
    assert public.DATASET_SCHEMA_VERSION == "aiogym.dataset.v2"


def test_rl_public_api_is_limited_to_config_runner_checkpoint_discovery():
    import aiogym.rl as public
    from aiogym.rl import checkpoints, config, runner

    assert public.RLTrainingConfig is config.RLTrainingConfig
    assert public.RunResult is runner.RunResult
    assert public.run_experiment is runner.run_experiment
    assert public.CheckpointManager is checkpoints.CheckpointManager
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
    import aiogym.evaluation as evaluation
    import aiogym.experimental.rl as experimental
    import aiogym.rl as public
    from aiogym.evaluation import statistics
    from aiogym.rl import constrained, observations, safety

    assert not hasattr(public, "ObservationContract")
    assert experimental.ObservationContract is observations.ObservationContract
    assert (
        experimental.ProjectionSafetyShield
        is safety.ProjectionSafetyShield
    )
    assert experimental.LagrangianSAC is constrained.LagrangianSAC
    assert (
        evaluation.build_intervention_report
        is statistics.build_intervention_report
    )


def test_environment_class_composes_focused_runtime_mixins():
    from aiogym._environment.disturbances import DisturbanceRuntimeMixin
    from aiogym._environment.observations import ObservationRuntimeMixin
    from aiogym._environment.transitions import TransitionRuntimeMixin
    from aiogym.env import _AIOGymEnv

    assert _AIOGymEnv._env is DisturbanceRuntimeMixin._env
    assert _AIOGymEnv._obs is ObservationRuntimeMixin._obs
    assert _AIOGymEnv.evaluate_transition is TransitionRuntimeMixin.evaluate_transition


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
        "aiogym.rl.transitions",
        "aiogym.rl.environment_factory",
    ):
        assert importlib.util.find_spec(module) is None
