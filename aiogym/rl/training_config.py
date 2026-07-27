"""Shared configuration resolution for reinforcement-learning entrypoints.

The algorithms intentionally keep separate CLIs, while task, objective, timing,
observation, and run-identity semantics are resolved here so the two workflows
cannot drift independently.
"""
from __future__ import annotations

from typing import Any

from aiogym._internal.config import resolve_auto_events
from aiogym.evaluation.objective_specs import reward_mode_for_objective
from aiogym.evaluation.protocols import resolve_protocol


_TASK_OWNED_OPTIONS = (
    "tracking_q_y",
    "tracking_r_move",
    "disturbance_obs",
    "previous_action_obs",
    "normalize_observations",
    "tracking_error_obs",
)

_ENVIRONMENT_OPTIONS = (
    "auto_events",
    "randomize",
    "randomize_setpoints",
    "randomize_plant",
    "plant_drift",
    "integral_obs",
    "terminate_on_runaway",
    "noise",
    "noise_pct",
)


def configure_training_objective(args):
    """Resolve a public objective and its internal environment reward mode."""

    if getattr(args, "reward_mode", None) is not None:
        raise ValueError("reward_mode is not supported; use objective")
    objective = getattr(args, "objective", None)
    task_name = getattr(args, "task", None)
    if objective is None and task_name:
        from aiogym.models.tasks import load_task_profile

        task = load_task_profile(task_name, scenario=args.scenario)
        objective = task.get("default_objective")
    objective = objective or "kpi"
    args.objective = objective
    args.resolved_reward_mode = reward_mode_for_objective(objective)
    if hasattr(args, "eval_objective") and args.eval_objective is None:
        args.eval_objective = objective
    return args


def configure_training_auto_events(args):
    """Resolve the canonical automatic-event option."""

    if getattr(args, "dynamic", None) is not None:
        raise ValueError("dynamic is not supported; use auto_events")
    args.auto_events = resolve_auto_events(
        getattr(args, "auto_events", None), default=False
    )
    return args


def _set_if_present(config: dict[str, Any], args, name: str) -> None:
    value = getattr(args, name, None)
    if value is not None:
        config[name] = value


def training_protocol_config(
    args,
    *,
    action_mode: str | None = None,
    episode_steps_attr: str = "train_episode_steps",
) -> dict[str, Any]:
    """Build the shared protocol portion of a training configuration."""

    config: dict[str, Any] = {
        "action_mode": action_mode or args.action_mode,
    }
    if getattr(args, "task", None):
        config["task"] = args.task
    if getattr(args, "control_dt", None) is not None:
        config["control_dt"] = args.control_dt
    episode_steps = getattr(args, episode_steps_attr, None)
    if episode_steps is not None:
        config["episode_steps"] = episode_steps
    for name in _TASK_OWNED_OPTIONS + _ENVIRONMENT_OPTIONS:
        _set_if_present(config, args, name)
    return config


def configure_training_task(
    args,
    *,
    episode_steps_attr: str = "train_episode_steps",
    eval_episode_steps_attr: str | None = "eval_episode_steps",
):
    """Resolve task-owned settings while preserving explicit CLI overrides."""

    protocol = resolve_protocol(
        args.scenario,
        args.objective,
        training_protocol_config(
            args,
            episode_steps_attr=episode_steps_attr,
        ),
    )
    args.control_dt = protocol.control_dt
    setattr(args, episode_steps_attr, protocol.episode_steps)
    for name in _TASK_OWNED_OPTIONS:
        setattr(args, name, getattr(protocol, name))
    if (
        eval_episode_steps_attr is not None
        and getattr(args, eval_episode_steps_attr, None) is None
    ):
        default_steps = protocol.episode_steps if getattr(args, "task", None) else 80
        setattr(args, eval_episode_steps_attr, default_steps)
    return args


def training_protocol(
    args,
    *,
    action_mode: str | None = None,
    episode_steps_attr: str = "train_episode_steps",
):
    """Resolve the canonical protocol used by an RL training workflow."""

    return resolve_protocol(
        args.scenario,
        args.objective,
        training_protocol_config(
            args,
            action_mode=action_mode,
            episode_steps_attr=episode_steps_attr,
        ),
    )


def training_identity(args) -> str:
    """Return the stable scenario/task portion of generated run names."""

    task = getattr(args, "task", None)
    return f"{args.scenario}_{task}" if task else args.scenario


__all__ = [
    "configure_training_auto_events",
    "configure_training_objective",
    "configure_training_task",
    "training_identity",
    "training_protocol",
    "training_protocol_config",
]
