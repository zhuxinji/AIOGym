"""Goal-based evaluation, scorecards, safety gates, and artifact reports."""
from __future__ import annotations

from aiogym._internal.lazy import exported_dir, resolve_export


_EXPORTS = {
    "evaluate_case_acceptance": ".results",
    "EVALUATION_SCHEMA_VERSION": ".metric_catalog",
    "ROLLOUT_SCHEMA": ".metric_catalog",
    "METRIC_DEFINITIONS": ".metric_catalog",
    "SCORECARD_GROUPS": ".metric_catalog",
    "SCORECARD_METRICS": ".metric_catalog",
    "METRIC_DIRECTIONS": ".metric_catalog",
    "PUBLIC_BENCHMARK_SCHEMA_VERSION": ".metric_catalog",
    "metric_direction": ".metric_catalog",
    "metric_definitions": ".metric_catalog",
    "scorecard_schema": ".metric_catalog",
    "GoalSpec": ".goal_specs",
    "goal_spec": ".goal_specs",
    "resolve_goal": ".goal_specs",
    "result_schema": ".results",
    "build_evaluation_report": ".results",
    "paired_robustness_report": ".results",
    "ARTIFACT_PROVENANCE_SCHEMA_VERSION": ".provenance",
    "track_provenance": ".provenance",
    "controller_access_level": ".provenance",
    "reward_spec_hash": ".provenance",
    "seed_namespace_hash": ".provenance",
    "ScorecardAccumulator": ".scorecard",
    "SafetyGateSpec": ".safety_gate",
    "apply_safety_gate": ".safety_gate",
    "evaluate_safety_gate": ".safety_gate",
    "degradation_statistics": ".metrics.robustness",
    "directional_degradation": ".metrics.robustness",
    "paired_robustness_summary": ".metrics.robustness",
    "paired_seed_metadata": ".metrics.robustness",
    "evaluate_controller": ".execution",
    "rollout_controller": ".execution",
    "REPORT_SCHEMA_VERSION": ".artifact",
    "ARTIFACT_CHECK_SCHEMA_VERSION": ".artifact",
    "render_benchmark_report": ".artifact",
    "check_benchmark_artifacts": ".artifact",
    "write_benchmark_artifacts": ".artifact",
    "compact_benchmark_artifacts": ".artifact",
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
