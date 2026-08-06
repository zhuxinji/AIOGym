"""Scenario-oriented AIO-Gym core with no optional dependency imports."""
from .contracts import Policy, ProcessModel, ScenarioPlugin, StudyProvider
from .compat import (
    LEGACY_TASK_ALIASES,
    resolve_condition_alias,
    resolve_legacy_request,
)
from .env import ProcessControlEnv, make_env, resolve_condition, resolve_plant
from .io import (
    canonical_json_bytes,
    file_sha256,
    jsonable,
    stable_hash,
    write_json,
    write_text,
)
from .registry import (
    get_scenario,
    get_task,
    list_scenarios,
    list_plants,
    list_tasks,
    register_scenario,
    unregister_scenario,
)
from .rollout import RolloutResult, Transition, rollout
from .specs import (
    CheckResult,
    EnvironmentIdentity,
    OperatingCondition,
    PLANT_SCHEMA_VERSION,
    PlantConfig,
    ResolvedPlant,
    RunResult,
    TaskSpec,
)

__all__ = [
    "PLANT_SCHEMA_VERSION",
    "CheckResult",
    "LEGACY_TASK_ALIASES",
    "EnvironmentIdentity",
    "OperatingCondition",
    "PlantConfig",
    "Policy",
    "ProcessControlEnv",
    "ProcessModel",
    "ResolvedPlant",
    "RolloutResult",
    "RunResult",
    "ScenarioPlugin",
    "StudyProvider",
    "TaskSpec",
    "Transition",
    "canonical_json_bytes",
    "file_sha256",
    "get_scenario",
    "get_task",
    "jsonable",
    "list_scenarios",
    "list_plants",
    "list_tasks",
    "make_env",
    "register_scenario",
    "resolve_plant",
    "resolve_condition",
    "resolve_condition_alias",
    "resolve_legacy_request",
    "rollout",
    "stable_hash",
    "unregister_scenario",
    "write_json",
    "write_text",
]
