"""Evaluation protocols, metrics, and artifact reports, loaded by responsibility."""
from __future__ import annotations

from aiogym._internal.lazy import exported_dir, resolve_export


_EXPORTS = {
    "StageRewardContext": ".objectives",
    "StageRewardResult": ".objectives",
    "stage_reward": ".objectives",
    "evaluate_task_acceptance": ".results",
    "EVALUATION_SCHEMA_VERSION": ".metric_catalog",
    "ROLLOUT_SCHEMA": ".metric_catalog",
    "METRIC_DEFINITIONS": ".metric_catalog",
    "PROTOCOL_METRICS": ".metric_catalog",
    "PRIMARY_METRICS": ".metric_catalog",
    "METRIC_DIRECTIONS": ".metric_catalog",
    "PUBLIC_BENCHMARK_SCHEMA_VERSION": ".metric_catalog",
    "primary_metric_for_objective": ".metric_catalog",
    "metric_direction": ".metric_catalog",
    "metric_definitions": ".metric_catalog",
    "BenchmarkCase": ".cases",
    "EnvironmentSpec": ".cases",
    "ObjectiveSpec": ".objective_specs",
    "metric_for_reward_mode": ".objective_specs",
    "reward_mode_for_objective": ".objective_specs",
    "objective_spec": ".objective_specs",
    "resolve_objective": ".objective_specs",
    "BenchmarkProtocol": ".protocols",
    "resolve_protocol": ".protocols",
    "result_schema": ".results",
    "build_evaluation_report": ".results",
    "evaluate_controller": ".execution",
    "rollout_controller": ".execution",
    "run_benchmark": ".execution",
    "REPORT_SCHEMA_VERSION": ".artifact",
    "ARTIFACT_CHECK_SCHEMA_VERSION": ".artifact",
    "render_benchmark_report": ".artifact",
    "check_benchmark_artifacts": ".artifact",
    "write_benchmark_artifacts": ".artifact",
    "finalize_benchmark_artifacts": ".artifact",
    "plot_results": ".artifact",
    "plot_summary": ".artifact",
    "plot_rollouts": ".artifact",
    "plot_leaderboard": ".artifact",
    "plot_constraint_timeline": ".artifact",
    "plot_learning_curve": ".artifact",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return resolve_export(globals(), __name__, _EXPORTS, name)


def __dir__():
    return exported_dir(globals(), _EXPORTS)
