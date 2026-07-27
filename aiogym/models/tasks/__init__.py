"""Model-bound task specifications and bundled task resources."""

from .registry import (
    configure_model_for_task,
    list_tasks,
    load_task_profile,
    resolve_environment_options,
    task_controller_config,
    task_environment,
    task_identity,
    task_objective_options,
    task_operation,
)
from .schema import (
    ENVIRONMENT_BOOLEAN_FIELDS,
    TASK_ENVIRONMENT_FIELDS,
    TASK_OPERATION_FIELDS,
    TASK_PROFILE_SCHEMA_VERSION,
    validate_task_profile,
)

__all__ = [
    "ENVIRONMENT_BOOLEAN_FIELDS",
    "TASK_ENVIRONMENT_FIELDS",
    "TASK_OPERATION_FIELDS",
    "TASK_PROFILE_SCHEMA_VERSION",
    "configure_model_for_task",
    "list_tasks",
    "load_task_profile",
    "resolve_environment_options",
    "task_controller_config",
    "task_environment",
    "task_identity",
    "task_objective_options",
    "task_operation",
    "validate_task_profile",
]
