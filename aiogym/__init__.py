"""Native Gymnasium environments for process-control research.

The package registers built-in Gymnasium IDs on import. Public implementation
groups are otherwise loaded lazily so core environment use does not import
benchmark reporting, plotting, controller adapters, or optional integrations.
"""
from __future__ import annotations

from gymnasium.envs.registration import register, registry

from ._internal.identifiers import canonical_scenario_id
from ._internal.lazy import exported_dir, resolve_export
from .models.registry import builtin_gym_ids


_EXPORTS = {
    "AIOGymEnv": ".env",
    "make_env": ".env_factory",
    "RewardResult": ".rewards",
    "RewardScaleWrapper": ".rewards",
    "RewardSpec": ".rewards",
    "get_reward_spec": ".rewards",
    "list_reward_specs": ".rewards",
    "resolve_reward_spec": ".rewards",
    "BENCHMARK_TRACK_SCHEMA_VERSION": ".benchmarks",
    "DEFAULT_BENCHMARK_TRACK_ID": ".benchmarks",
    "CaseMixtureEnv": ".benchmarks",
    "TrackSpec": ".benchmarks",
    "derive_seed_bundle": ".benchmarks",
    "evaluate_policy_on_track": ".benchmarks",
    "list_tracks": ".benchmarks",
    "load_track": ".benchmarks",
    "ARTIFACT_CHECK_SCHEMA_VERSION": ".evaluation",
    "REPORT_SCHEMA_VERSION": ".evaluation",
    "GoalSpec": ".evaluation",
    "SafetyGateSpec": ".evaluation",
    "build_evaluation_report": ".evaluation",
    "check_benchmark_artifacts": ".evaluation",
    "evaluate_controller": ".evaluation",
    "evaluate_safety_gate": ".evaluation",
    "goal_spec": ".evaluation",
    "paired_robustness_summary": ".evaluation",
    "paired_seed_metadata": ".evaluation",
    "plot_results": ".evaluation",
    "render_benchmark_report": ".evaluation",
    "resolve_goal": ".evaluation",
    "load_controller_config": ".controllers",
    "make_controller": ".controllers",
    "register_controller": ".controllers",
    "unregister_controller": ".controllers",
    "list_controllers": ".catalog",
    "list_cases": ".catalog",
    "list_scenarios": ".catalog",
    "MODEL_METADATA_SCHEMA_VERSION": ".models",
    "PARAMETER_PROFILE_SCHEMA_VERSION": ".models",
    "CASE_PROFILE_SCHEMA_VERSION": ".models",
    "CaseSpec": ".models",
    "Integrator": ".models",
    "ProcessModelContract": ".models",
    "builtin_gym_ids": ".models",
    "collect_model_metadata": ".models",
    "define_model": ".models",
    "export_model_metadata": ".models",
    "make_model": ".models",
    "list_parameter_profiles": ".models",
    "load_parameter_profile": ".models",
    "load_case": ".models",
    "register_model": ".models",
    "unregister_model": ".models",
    "case_operation": ".models",
    "validate_case_profile": ".models",
    "validate_model_metadata": ".models",
    "validate_model_readiness": ".models",
    "validate_parameter_profile": ".models",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return resolve_export(globals(), __name__, _EXPORTS, name)


def __dir__():
    return exported_dir(globals(), _EXPORTS)


for _scenario, _gym_name in builtin_gym_ids().items():
    _env_id = f"AIOGym/{_gym_name}-v0"
    if _env_id not in registry:
        register(
            id=_env_id,
            entry_point="aiogym.env:AIOGymEnv",
            kwargs={"scenario": canonical_scenario_id(_scenario)},
        )
