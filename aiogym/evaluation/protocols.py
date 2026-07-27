"""Benchmark protocol declarations and metric catalog."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping

from .._internal.config import resolve_auto_events
from .._internal.identifiers import (
    canonical_scenario_id,
    internal_scenario_id,
    require_canonical_scenario_id,
)
from .cases import BenchmarkCase, EnvironmentSpec
from .metric_catalog import (
    EVALUATION_SCHEMA_VERSION,
    METRIC_DEFINITIONS,
    METRIC_DIRECTIONS,
    PRIMARY_METRICS,
    PROTOCOL_METRICS,
    PUBLIC_BENCHMARK_SCHEMA_VERSION,
    ROLLOUT_SCHEMA,
    metric_definitions,
    metric_direction,
    primary_metric_for_objective,
)
from .objective_specs import (
    DEFAULT_TRACKING_Q_Y,
    DEFAULT_TRACKING_R_MOVE,
    OBJECTIVE_REWARD_MODES,
    REWARD_MODES,
    ObjectiveSpec,
    objective_spec,
    resolve_objective,
    reward_mode_for_objective,
)


def _empty_episode_totals(ep: int, seed: int):
    return {
        "episode": int(ep),
        "seed": int(seed),
        "return": 0.0,
        "track": 0.0,
        "constraint": 0.0,
        "profit": 0.0,
        "production": 0.0,
        "energy_kwh": 0.0,
        "runtime_seconds": 0.0,
        "runtime_seconds_per_step": 0.0,
        "tracking_cost": 0.0,
        "tracking_return": 0.0,
        "tracking_error_cost": 0.0,
        "tracking_move_cost": 0.0,
        "tracking_mse": 0.0,
        "tracking_iae": 0.0,
        "tracking_ise": 0.0,
        "tracking_itae": 0.0,
        "tracking_overshoot": 0.0,
        "tracking_settling_time": 0.0,
        "tracking_raw_by_output": {},
        "constraint_violation_count": 0.0,
        "constraint_violation_duration": 0.0,
        "constraint_violation_severity": 0.0,
        "action_violation_count": 0.0,
        "action_violation_duration": 0.0,
        "action_violation_severity": 0.0,
        "runaway_count": 0.0,
        "runaway_duration": 0.0,
        "safety_margin_min": 0.0,
        "controller_solve_count": 0.0,
        "controller_solver_success_count": 0.0,
        "controller_solver_failure_count": 0.0,
        "controller_fallback_count": 0.0,
        "controller_degraded_count": 0.0,
    }


def _protocol_kwargs(defaults: Mapping[str, Any], overrides: Mapping[str, Any]):
    data = dict(defaults)
    data.update(overrides)
    return data


# Objective selection must not silently change the experiment. Task profiles or
# explicit environment overrides own these neutral protocol defaults.
_DEFAULT_ENVIRONMENT_CONDITIONS = {
    "action_mode": "actuator",
    "auto_events": False,
    "randomize": False,
    "randomize_setpoints": False,
    "randomize_plant": False,
    "plant_drift": False,
    "integral_obs": False,
    "disturbance_obs": True,
    "previous_action_obs": False,
    "normalize_observations": False,
    "tracking_error_obs": False,
    "terminate_on_runaway": False,
    "noise": False,
    "noise_pct": 0.01,
}


def _setpoint_vector(value, field_name: str):
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a numeric vector")
    values = [float(item) for item in value]
    if not values or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{field_name} must contain finite numeric values")
    return values


def _setpoint_events(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError("setpoint_schedule must be a list of event mappings")
    events = []
    seen = set()
    for event in value:
        if not isinstance(event, Mapping):
            raise TypeError("each setpoint_schedule event must be a mapping")
        at_step = event.get("at_step")
        if isinstance(at_step, bool) or not isinstance(at_step, int) or at_step < 0:
            raise ValueError("setpoint_schedule at_step must be a non-negative integer")
        if at_step in seen:
            raise ValueError("setpoint_schedule cannot contain duplicate at_step values")
        seen.add(at_step)
        events.append({
            "at_step": at_step,
            "values": _setpoint_vector(
                event.get("values"), "setpoint_schedule values"
            ),
        })
    return events


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Reproducible benchmark environment configuration.

    Use ``economic`` for supervisory/RTO benchmarks and ``tracking`` for
    setpoint-following benchmarks.  Baseline PID/MPC/oracle controllers usually
    run with ``action_mode="actuator"``; learned supervisory policies usually run
    with ``action_mode="setpoint"`` on the same protocol. ``env_reward_mode`` is
    passed to the Gym environment; benchmark metrics are chosen from ``objective``.
    """

    scenario: str = "cstr"
    objective: str = "economic"
    objective_source: str = "explicit"
    env_reward_mode: str = "economic"
    action_mode: str | None = None
    control_dt: float | None = None
    episode_steps: int | None = None
    task: Any = None
    initial_setpoint: Any = None
    setpoint_schedule: Any = None
    auto_events: bool | None = None
    randomize: bool | None = None
    randomize_setpoints: bool | None = None
    randomize_plant: bool | None = None
    plant_drift: bool | None = None
    integral_obs: bool | None = None
    disturbance_obs: bool | None = None
    previous_action_obs: bool | None = None
    normalize_observations: bool | None = None
    tracking_error_obs: bool | None = None
    terminate_on_runaway: bool | None = None
    tracking_q_y: Any = None
    tracking_r_move: float | None = None
    noise: bool | None = None
    noise_pct: float | None = None
    model_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.scenario, str) or not self.scenario:
            raise ValueError("scenario must be a non-empty string")
        require_canonical_scenario_id(self.scenario)
        canonical_scenario = canonical_scenario_id(internal_scenario_id(self.scenario))
        object.__setattr__(self, "scenario", canonical_scenario)
        if self.objective not in PRIMARY_METRICS:
            raise ValueError(
                f"objective must be one of: {', '.join(PRIMARY_METRICS)}"
            )
        from ..models import make_model

        model_supported = tuple(
            getattr(make_model(self.scenario), "supported_objectives", tuple(PRIMARY_METRICS))
        )
        if self.objective not in model_supported:
            raise ValueError(
                f"scenario {self.scenario!r} does not support objective {self.objective!r}; "
                f"supported: {', '.join(model_supported)}"
            )
        if self.env_reward_mode not in {"economic", "kpi", "tracking"}:
            raise ValueError("env_reward_mode must be one of: economic, kpi, tracking")
        expected_reward_mode = reward_mode_for_objective(self.objective)
        if self.env_reward_mode != expected_reward_mode:
            raise ValueError(
                f"objective {self.objective!r} requires env_reward_mode "
                f"{expected_reward_mode!r}, got {self.env_reward_mode!r}"
            )
        from ..models.tasks import resolve_environment_options

        auto_events = resolve_auto_events(self.auto_events)
        for name in (
            "disturbance_obs",
            "previous_action_obs",
            "normalize_observations",
            "tracking_error_obs",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean")

        task, environment_options = resolve_environment_options(
            scenario=self.scenario,
            task=self.task,
            explicit={
                "control_dt": self.control_dt,
                "episode_steps": self.episode_steps,
                "action_mode": self.action_mode,
                "auto_events": auto_events,
                "randomize": self.randomize,
                "randomize_setpoints": self.randomize_setpoints,
                "randomize_plant": self.randomize_plant,
                "plant_drift": self.plant_drift,
                "integral_obs": self.integral_obs,
                "disturbance_obs": self.disturbance_obs,
                "previous_action_obs": self.previous_action_obs,
                "normalize_observations": self.normalize_observations,
                "tracking_error_obs": self.tracking_error_obs,
                "terminate_on_runaway": self.terminate_on_runaway,
                "noise": self.noise,
                "noise_pct": self.noise_pct,
                "model_params": self.model_params,
            },
            defaults=_DEFAULT_ENVIRONMENT_CONDITIONS,
            default_control_dt=0.5,
            default_episode_steps=400,
        )
        if task is not None:
            supported = task.get("supported_objectives")
            if supported is not None and self.objective not in supported:
                raise ValueError(
                    f"task {task['name']!r} does not support objective {self.objective!r}; "
                    f"supported: {', '.join(supported)}"
                )
        from ..models.tasks import task_objective_options

        tracking_options = task_objective_options(task, "tracking")
        tracking_q_y = (
            self.tracking_q_y
            if self.tracking_q_y is not None
            else tracking_options.get("tracking_q_y", DEFAULT_TRACKING_Q_Y)
        )
        tracking_r_move = float(
            self.tracking_r_move
            if self.tracking_r_move is not None
            else tracking_options.get(
                "tracking_r_move",
                DEFAULT_TRACKING_R_MOVE,
            )
        )
        if not math.isfinite(tracking_r_move) or tracking_r_move < 0:
            raise ValueError("tracking_r_move must be finite and non-negative")
        task_setpoints = (task or {}).get("setpoints", {})
        initial_setpoint = _setpoint_vector(
            (
                task_setpoints.get("initial")
                if self.initial_setpoint is None
                else self.initial_setpoint
            ),
            "initial_setpoint",
        )
        setpoint_schedule = _setpoint_events(
            (
                task_setpoints.get("schedule", [])
                if self.setpoint_schedule is None
                else self.setpoint_schedule
            )
        )
        object.__setattr__(self, "task", task)
        for name, value in environment_options.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "initial_setpoint", initial_setpoint)
        object.__setattr__(self, "setpoint_schedule", setpoint_schedule)
        object.__setattr__(self, "tracking_q_y", tracking_q_y)
        object.__setattr__(self, "tracking_r_move", tracking_r_move)

    @classmethod
    def economic(cls, scenario: str, **kw):
        defaults = dict(objective="economic", env_reward_mode="economic")
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    @classmethod
    def tracking(cls, scenario: str, **kw):
        defaults = dict(objective="tracking", env_reward_mode="tracking")
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    @classmethod
    def kpi(cls, scenario: str, **kw):
        defaults = dict(objective="kpi", env_reward_mode="kpi")
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    @classmethod
    def robustness(cls, scenario: str, **kw):
        defaults = dict(objective="robustness", env_reward_mode="kpi")
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    @classmethod
    def safety(cls, scenario: str, **kw):
        defaults = dict(objective="safety", env_reward_mode="kpi")
        return cls(scenario=scenario, **_protocol_kwargs(defaults, kw))

    def env_kwargs(self, action_mode: str | None = None):
        data = asdict(self)
        data.pop("scenario")
        data.pop("objective")
        data.pop("objective_source")
        data["reward_mode"] = data.pop("env_reward_mode")
        if action_mode is not None:
            data["action_mode"] = action_mode
        return data

    def make_env(self, action_mode: str | None = None):
        from ..env import AIOGymNativeEnv

        return AIOGymNativeEnv(self.scenario, **self.env_kwargs(action_mode=action_mode))

    def resolved_objective(self) -> ObjectiveSpec:
        return objective_spec(
            self.objective,
            source=self.objective_source,
            reward_options={
                "tracking_q_y": self.tracking_q_y,
                "tracking_r_move": self.tracking_r_move,
            },
        )

    def environment_spec(self) -> "EnvironmentSpec":
        return EnvironmentSpec.from_protocol(self)

    def metadata(self):
        from ..models.tasks import task_identity

        primary_metric = primary_metric_for_objective(self.objective)
        data = asdict(self)
        data["resolved_reward_mode"] = self.env_reward_mode
        data["task_identity"] = task_identity(self.task)
        data["objective_spec"] = self.resolved_objective().metadata()
        data["metrics"] = list(PROTOCOL_METRICS.get(self.objective, ()))
        data["primary_metric"] = primary_metric
        data["primary_metric_direction"] = metric_direction(primary_metric)
        data["metric_definitions"] = metric_definitions(self.objective)
        return data


def resolve_protocol(
    scenario: str,
    objective=None,
    data: Mapping[str, Any] | None = None,
    *,
    case_objective: str | None = None,
    suite_objective: str | None = None,
) -> BenchmarkProtocol:
    """Resolve a protocol after loading task-owned objective defaults."""

    if isinstance(objective, BenchmarkProtocol):
        return objective
    cfg = dict(data or {})
    explicit = None
    configured = cfg.pop("objective", None)
    if isinstance(objective, Mapping):
        cfg.update(objective)
        configured = cfg.pop("objective", configured)
    elif objective is not None:
        explicit = str(objective)

    task = None
    if cfg.get("task") is not None:
        from ..models.tasks import load_task_profile

        task = load_task_profile(cfg["task"], scenario=scenario)
        cfg["task"] = task
    case_value = case_objective if case_objective is not None else configured
    resolved = resolve_objective(
        explicit=explicit,
        case_config=case_value,
        suite_config=suite_objective,
        task_profile=task,
    )
    factories = {
        "economic": BenchmarkProtocol.economic,
        "tracking": BenchmarkProtocol.tracking,
        "robustness": BenchmarkProtocol.robustness,
        "safety": BenchmarkProtocol.safety,
        "kpi": BenchmarkProtocol.kpi,
    }
    factory = factories[resolved.name]
    cfg["objective_source"] = resolved.source
    cfg["env_reward_mode"] = reward_mode_for_objective(resolved.name)
    return factory(scenario, **cfg)
