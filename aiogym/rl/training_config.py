"""Shared Track/Case configuration for reinforcement-learning entrypoints."""
from __future__ import annotations

import hashlib
from typing import Any

from aiogym._internal.config import resolve_auto_events
from aiogym.benchmarks import DEFAULT_BENCHMARK_TRACK_ID, load_track
from aiogym.env_factory import make_env
from aiogym.rewards import get_reward_spec


_CASE_ENVIRONMENT_OPTIONS = (
    "auto_events",
    "randomize",
    "randomize_setpoints",
    "randomize_plant",
    "plant_drift",
    "integral_obs",
    "disturbance_obs",
    "previous_action_obs",
    "normalize_observations",
    "tracking_error_obs",
    "terminate_on_runaway",
    "noise",
    "noise_pct",
)


def configure_training_track(args, *, official_default: bool = True):
    """Resolve either an official Track or an explicit specialist Case."""

    explicit_track = getattr(args, "track", None)
    custom_selectors = any(
        getattr(args, name, None) is not None
        for name in ("scenario", "case", "goal", "reward_spec")
    )
    track_reference = explicit_track
    if track_reference is None and official_default and not custom_selectors:
        track_reference = DEFAULT_BENCHMARK_TRACK_ID

    if track_reference is not None:
        track = load_track(track_reference)
        _reject_track_conflicts(args, track)
        contract = track.policy_contract
        args.track = track.id
        args.track_spec = track
        args.track_hash = track.track_hash
        args.scenario = track.scenario
        args.case = None
        args.goal = track.goal
        args.reward_spec = track.reward_spec_id
        args.resolved_reward_spec_id = track.reward_spec_id
        args.policy_scope = track.policy_scope
        args.action_mode = contract["action_mode"]
        args.control_dt = float(contract["control_dt"])
        for name in (
            "disturbance_obs",
            "previous_action_obs",
            "normalize_observations",
            "tracking_error_obs",
            "integral_obs",
        ):
            if name in contract:
                setattr(args, name, contract[name])
        if getattr(args, "steps", None) is None:
            args.steps = int(
                track.declaration["training"].get("step_budget", 10000)
            )
        _set_namespaces(
            args,
            {
                split: track.seed_namespace(split)
                for split in ("training", "validation")
            },
        )
        return args

    args.track = None
    args.track_spec = None
    args.track_hash = None
    args.scenario = getattr(args, "scenario", None) or "cstr"
    args.policy_scope = getattr(args, "policy_scope", None) or "specialist"
    if args.policy_scope != "specialist":
        raise ValueError(
            "custom single-case training requires policy_scope='specialist'"
        )
    _resolve_custom_goal_reward(args)
    if getattr(args, "steps", None) is None:
        args.steps = 10000
    _set_namespaces(
        args,
        {
            "training": "custom-training-v1",
            "validation": "custom-validation-v1",
        },
    )
    return args


def configure_training_auto_events(args):
    if getattr(args, "dynamic", None) is not None:
        raise ValueError("dynamic is not supported; use auto_events")
    args.auto_events = resolve_auto_events(
        getattr(args, "auto_events", None),
        default=False,
    )
    return args


def training_env_kwargs(
    args,
    *,
    action_mode: str | None = None,
    episode_steps_attr: str = "train_episode_steps",
) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "action_mode": action_mode or args.action_mode,
    }
    if getattr(args, "control_dt", None) is not None:
        environment["control_dt"] = args.control_dt
    episode_steps = getattr(args, episode_steps_attr, None)
    if episode_steps is not None:
        environment["episode_steps"] = episode_steps
    for name in _CASE_ENVIRONMENT_OPTIONS:
        value = getattr(args, name, None)
        if value is not None:
            environment[name] = value
    return {
        "scenario": args.scenario,
        "case": getattr(args, "case", None),
        "reward_spec": args.resolved_reward_spec_id,
        "environment": environment,
    }


def configure_training_case(
    args,
    *,
    episode_steps_attr: str = "train_episode_steps",
    eval_episode_steps_attr: str | None = "eval_episode_steps",
):
    """Resolve Case-owned defaults through the canonical environment."""

    env = make_env(
        config=training_env_kwargs(
            args,
            episode_steps_attr=episode_steps_attr,
        ),
    )
    try:
        args.control_dt = env.control_dt
        setattr(args, episode_steps_attr, env.episode_steps)
        for name in _CASE_ENVIRONMENT_OPTIONS:
            if hasattr(env, name):
                setattr(args, name, getattr(env, name))
    finally:
        env.close()
    if (
        eval_episode_steps_attr is not None
        and getattr(args, eval_episode_steps_attr, None) is None
    ):
        default_steps = (
            getattr(args, episode_steps_attr)
            if getattr(args, "case", None)
            else 80
        )
        setattr(args, eval_episode_steps_attr, default_steps)
    return args


def training_identity(args) -> str:
    track = getattr(args, "track", None)
    if track:
        return str(track)
    case = getattr(args, "case", None)
    identity = f"{args.scenario}_{case}" if case else args.scenario
    return f"{identity}_specialist"


def _resolve_custom_goal_reward(args) -> None:
    goal = getattr(args, "goal", None)
    spec_id = getattr(args, "reward_spec", None)
    if spec_id is not None:
        spec = get_reward_spec(str(spec_id))
        if goal is not None and spec.goal != goal:
            raise ValueError(
                f"reward spec {spec.id!r} conflicts with goal {goal!r}"
            )
    else:
        goal = goal or "regulation"
        spec = get_reward_spec(f"{goal}-v1")
    args.goal = spec.goal
    args.reward_spec = spec.id
    args.resolved_reward_spec_id = spec.id


def _reject_track_conflicts(args, track) -> None:
    for name in ("scenario", "case", "goal", "reward_spec"):
        value = getattr(args, name, None)
        if value is not None:
            raise ValueError(
                f"--{name.replace('_', '-')} conflicts with --track; "
                "the Track owns its scenario, cases, goal and RewardSpec"
            )
    contract = track.policy_contract
    for name in (
        "action_mode",
        "control_dt",
        "disturbance_obs",
        "previous_action_obs",
        "normalize_observations",
        "tracking_error_obs",
        "integral_obs",
    ):
        value = getattr(args, name, None)
        if value is not None and name in contract and value != contract[name]:
            raise ValueError(
                f"--{name.replace('_', '-')} conflicts with Track contract"
            )
    for name in (
        "auto_events",
        "randomize",
        "randomize_setpoints",
        "randomize_plant",
        "plant_drift",
        "terminate_on_runaway",
        "noise",
        "noise_pct",
        "train_episode_steps",
        "episode_steps",
        "eval_episode_steps",
    ):
        if getattr(args, name, None) is not None:
            raise ValueError(
                f"--{name.replace('_', '-')} conflicts with --track"
            )


def _set_namespaces(args, values: dict[str, str]) -> None:
    for split, namespace in values.items():
        setattr(args, f"{split}_seed_namespace", namespace)
        setattr(
            args,
            f"{split}_seed_namespace_hash",
            hashlib.sha256(namespace.encode()).hexdigest(),
        )


__all__ = [
    "configure_training_auto_events",
    "configure_training_case",
    "configure_training_track",
    "training_env_kwargs",
    "training_identity",
]
