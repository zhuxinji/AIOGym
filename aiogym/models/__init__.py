"""Process-model public API with implementation groups loaded lazily."""
from __future__ import annotations

from aiogym._internal.lazy import exported_dir, resolve_export


_EXPORTS = {
    "CP": ".core",
    "G": ".core",
    "RHO": ".core",
    "RHO_CP": ".core",
    "ProcessModelContract": ".core",
    "Integrator": ".integration",
    "DeclarativeProcessModel": ".declarative",
    "define_model": ".declarative",
    "CascadeModel": ".scenarios",
    "RecirculatingCascadeModel": ".scenarios",
    "QuadrupleModel": ".scenarios",
    "CSTRModel": ".scenarios",
    "HVACModel": ".scenarios",
    "ExtractionModel": ".scenarios",
    "FiredHeaterModel": ".scenarios",
    "CrystallizationModel": ".scenarios",
    "MODELS": ".registry",
    "BUILTIN_MODELS": ".registry",
    "apply_model_params": ".registry",
    "builtin_gym_ids": ".registry",
    "gym_id_name": ".registry",
    "make_model": ".registry",
    "register_model": ".registry",
    "unregister_model": ".registry",
    "validate_model_contract": ".registry",
    "MODEL_METADATA_SCHEMA_VERSION": ".metadata",
    "collect_model_metadata": ".metadata",
    "export_model_metadata": ".metadata",
    "iter_model_metadata": ".metadata",
    "validate_model_metadata": ".metadata",
    "PARAMETER_PROFILE_SCHEMA_VERSION": ".parameter_profiles",
    "list_parameter_profiles": ".parameter_profiles",
    "load_parameter_profile": ".parameter_profiles",
    "validate_parameter_profile": ".parameter_profiles",
    "validate_model_readiness": ".validation",
    "TASK_PROFILE_SCHEMA_VERSION": ".tasks",
    "configure_model_for_task": ".tasks",
    "load_task_profile": ".tasks",
    "resolve_environment_options": ".tasks",
    "task_environment": ".tasks",
    "task_identity": ".tasks",
    "task_operation": ".tasks",
    "validate_task_profile": ".tasks",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return resolve_export(globals(), __name__, _EXPORTS, name)


def __dir__():
    return exported_dir(globals(), _EXPORTS)
