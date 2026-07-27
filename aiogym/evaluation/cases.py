"""Resolved environment and benchmark-case value objects."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from .._internal.config import resolve_auto_events
from .._internal.serialization import jsonable as _jsonable
from .metric_catalog import (
    EVALUATION_SCHEMA_VERSION,
    METRIC_DEFINITIONS,
    PROTOCOL_METRICS,
    metric_direction,
)
from .objective_specs import ObjectiveSpec

if TYPE_CHECKING:
    from .protocols import BenchmarkProtocol


@dataclass(frozen=True)
class EnvironmentSpec:
    """Task-owned environment construction data, independent of ranking."""

    scenario: str
    task: Any = None
    action_mode: str = "actuator"
    control_dt: float = 0.5
    episode_steps: int = 400
    initial_setpoint: Any = None
    setpoint_schedule: Any = None
    auto_events: bool | None = None
    randomize: bool = False
    randomize_setpoints: bool = False
    randomize_plant: bool = False
    plant_drift: bool = False
    integral_obs: bool = False
    disturbance_obs: bool = True
    previous_action_obs: bool = False
    normalize_observations: bool = False
    tracking_error_obs: bool = False
    terminate_on_runaway: bool = False
    noise: bool = False
    noise_pct: float = 0.01
    model_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        auto_events = resolve_auto_events(self.auto_events, default=False)
        object.__setattr__(self, "auto_events", auto_events)

    @classmethod
    def from_protocol(cls, protocol: BenchmarkProtocol) -> "EnvironmentSpec":
        return cls(
            scenario=protocol.scenario,
            task=protocol.task,
            action_mode=str(protocol.action_mode),
            control_dt=float(protocol.control_dt),
            episode_steps=int(protocol.episode_steps),
            initial_setpoint=protocol.initial_setpoint,
            setpoint_schedule=protocol.setpoint_schedule,
            auto_events=bool(protocol.auto_events),
            randomize=bool(protocol.randomize),
            randomize_setpoints=bool(protocol.randomize_setpoints),
            randomize_plant=bool(protocol.randomize_plant),
            plant_drift=bool(protocol.plant_drift),
            integral_obs=bool(protocol.integral_obs),
            disturbance_obs=protocol.disturbance_obs,
            previous_action_obs=protocol.previous_action_obs,
            normalize_observations=protocol.normalize_observations,
            tracking_error_obs=protocol.tracking_error_obs,
            terminate_on_runaway=bool(protocol.terminate_on_runaway),
            noise=bool(protocol.noise),
            noise_pct=float(protocol.noise_pct),
            model_params=dict(protocol.model_params),
        )

    def env_kwargs(self, objective: ObjectiveSpec) -> dict[str, Any]:
        data = asdict(self)
        data.pop("scenario")
        data["reward_mode"] = objective.reward_mode
        data.update(dict(objective.reward_options))
        return data

    def make_env(self, objective: ObjectiveSpec):
        from ..env import AIOGymNativeEnv

        return AIOGymNativeEnv(self.scenario, **self.env_kwargs(objective))

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        return _jsonable(data)


@dataclass(frozen=True)
class BenchmarkCase:
    """One resolved task/objective/controller/seed evaluation unit."""

    case_id: str
    environment: EnvironmentSpec
    objective: ObjectiveSpec
    controller: str
    seeds: tuple[int, ...]
    controller_config: Mapping[str, Any] = field(default_factory=dict)
    protocol: BenchmarkProtocol | None = field(default=None, repr=False)

    def __post_init__(self):
        seeds = tuple(int(seed) for seed in self.seeds)
        if not seeds:
            raise ValueError("benchmark case seeds must not be empty")
        if not isinstance(self.controller, str) or not self.controller.strip():
            raise ValueError("benchmark case controller must be a non-empty string")
        object.__setattr__(self, "seeds", seeds)

    @classmethod
    def from_protocol(
        cls,
        protocol: BenchmarkProtocol,
        *,
        controller: str,
        seeds: Sequence[int],
        controller_config: Mapping[str, Any] | None = None,
        case_id: str | None = None,
    ) -> "BenchmarkCase":
        from ..models.tasks import task_controller_config, task_identity

        task_name = task_identity(protocol.task)["name"]
        default_controller_config = (
            {"profile": "tracking"}
            if controller == "oracle"
            and protocol.resolved_objective().name == "tracking"
            else {}
        )
        default_controller_config = _merge_controller_config(
            default_controller_config,
            task_controller_config(protocol.task, controller),
        )
        resolved_controller_config = _merge_controller_config(
            default_controller_config,
            controller_config,
        )
        return cls(
            case_id=case_id
            or f"{protocol.objective}:{protocol.scenario}:{task_name}:{controller}",
            environment=protocol.environment_spec(),
            objective=protocol.resolved_objective(),
            controller=controller,
            seeds=tuple(seeds),
            controller_config=resolved_controller_config,
            protocol=protocol,
        )

    def metadata(self) -> dict[str, Any]:
        from ..models.tasks import task_identity

        task_meta = task_identity(self.environment.task)
        protocol_meta = self.protocol.metadata() if self.protocol is not None else None
        metrics = PROTOCOL_METRICS.get(self.objective.name, ())
        primary_metric = self.objective.primary_metric
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "case_id": self.case_id,
            "scenario": self.environment.scenario,
            "task": task_meta["name"],
            "task_status": task_meta["status"],
            "task_profile_hash": task_meta["profile_hash"],
            "objective": self.objective.name,
            "objective_source": self.objective.source,
            "objective_spec": self.objective.metadata(),
            "controller": self.controller,
            "controller_config": _jsonable(dict(self.controller_config)),
            "seed_list": list(self.seeds),
            "environment": self.environment.metadata(),
            "metrics": list(metrics),
            "primary_metric": primary_metric,
            "primary_metric_direction": metric_direction(primary_metric),
            "episode_steps": self.environment.episode_steps,
            "protocol": protocol_meta,
            "metric_definitions": {
                key: METRIC_DEFINITIONS[key]
                for key in metrics
                if key in METRIC_DEFINITIONS
            },
        }


def _merge_controller_config(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge controller configs while preserving nested parameter defaults."""

    resolved = dict(base or {})
    for key, value in dict(override or {}).items():
        if key == "parameters" and isinstance(value, Mapping):
            resolved[key] = {
                **dict(resolved.get(key) or {}),
                **dict(value),
            }
        else:
            resolved[key] = value
    return resolved
