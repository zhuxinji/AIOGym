"""Collector registry and environment-to-DatasetEpisode conversion."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from aiogym.generation.specs import EpisodeSpec

from .schema import DatasetEpisode


@dataclass(frozen=True)
class CollectorSpec:
    collector_id: str
    policy_id: str
    quality_tag: str
    description: str

    def __post_init__(self) -> None:
        for name in (
            "collector_id",
            "policy_id",
            "quality_tag",
            "description",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")

    def metadata(self) -> dict[str, str]:
        return {
            "collector_id": self.collector_id,
            "policy_id": self.policy_id,
            "quality_tag": self.quality_tag,
            "description": self.description,
        }


_COLLECTORS: dict[str, CollectorSpec] = {}


class SmoothExcitationPolicy:
    """Low-pass excitation proposed to the safe-excitation shield."""

    name = "safe-excitation"

    def __init__(self, *, smoothing: float = 0.85, scale: float = 0.12):
        self.smoothing = float(smoothing)
        self.scale = float(scale)
        self._rng = np.random.default_rng(0)
        self._action = None

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._action = None

    def act(self, observation, context):
        shape = context.env.action_space.shape
        if self._action is None:
            self._action = np.full(shape, 0.5, dtype=np.float64)
        target = np.clip(
            0.5 + self._rng.normal(0.0, self.scale, size=shape),
            0.05,
            0.95,
        )
        self._action = (
            self.smoothing * self._action
            + (1.0 - self.smoothing) * target
        )
        return self._action.astype(np.float32)


def register_collector(spec: CollectorSpec, *, replace: bool = False) -> None:
    if not isinstance(spec, CollectorSpec):
        raise TypeError("spec must be a CollectorSpec")
    if spec.collector_id in _COLLECTORS and not replace:
        raise ValueError(f"collector {spec.collector_id!r} is registered")
    _COLLECTORS[spec.collector_id] = spec


def unregister_collector(collector_id: str) -> None:
    if collector_id in _BUILTIN_COLLECTOR_IDS:
        raise ValueError(f"built-in collector {collector_id!r} cannot be removed")
    _COLLECTORS.pop(str(collector_id), None)


def get_collector(collector_id: str) -> CollectorSpec:
    try:
        return _COLLECTORS[str(collector_id)]
    except KeyError as exc:
        raise KeyError(
            f"unknown collector {collector_id!r}; available: "
            + ", ".join(list_collectors())
        ) from exc


def list_collectors() -> tuple[str, ...]:
    return tuple(sorted(_COLLECTORS))


def collect_episode(
    env,
    episode_spec: EpisodeSpec,
    *,
    collector_id: str,
    policy=None,
    track_id: str,
    split: str = "training",
    checkpoint_hash: str | None = None,
    episode_index: int | None = None,
    max_steps: int | None = None,
) -> DatasetEpisode:
    """Collect one resolved actuator episode into the v2 schema."""

    if not isinstance(episode_spec, EpisodeSpec):
        raise TypeError("episode_spec must be an EpisodeSpec")
    if getattr(env, "action_mode", None) != "actuator":
        raise ValueError("Dataset v2 collection currently requires actuator mode")
    collector = get_collector(collector_id)
    from aiogym.controllers.adapters import PolicyController, as_controller
    from aiogym.controllers.contracts import build_context, validate_action

    if policy is None and collector_id == "safe_excitation":
        policy = SmoothExcitationPolicy()
    controller = None
    if policy is not None:
        controller = (
            PolicyController(
                policy,
                action_mode="actuator",
                control_structure="safe_excitation",
                allow_environment_context=True,
            )
            if isinstance(policy, SmoothExcitationPolicy)
            else as_controller(policy, action_mode="actuator")
        )
    observation, reset_info = env.reset(
        options={"episode_spec": episode_spec}
    )
    recorded_initial_state = np.asarray(
        env.integ.x,
        dtype=np.float32,
    ).copy()
    requested_initial_state = np.asarray(
        episode_spec.initial_state,
        dtype=np.float64,
    ).reshape(-1)
    reset_state_delta_linf = float(
        np.max(
            np.abs(
                requested_initial_state
                - recorded_initial_state.astype(np.float64)
            )
        )
    )
    env.action_space.seed(episode_spec.component_seeds["exploration"])
    if controller is not None:
        controller.reset(seed=episode_spec.component_seeds["policy"])

    rows = {
        "observation": [],
        "true_state": [],
        "reference": [],
        "measured_disturbance": [],
        "action_policy_normalized": [],
        "action_commanded_physical": [],
        "action_applied_physical": [],
        "reward_scalar": [],
        "next_observation": [],
        "next_true_state": [],
        "terminated": [],
        "truncated": [],
        "bootstrap_mask": [],
        "step_index": [],
        "physical_time": [],
    }
    reward_rows = []
    cost_rows = []
    info = dict(reset_info or {})
    done = False
    step = 0
    while not done and (max_steps is None or step < int(max_steps)):
        state = (
            recorded_initial_state.copy()
            if step == 0
            else np.asarray(env.integ.x, dtype=np.float32).copy()
        )
        reference = np.asarray(env.y_sp, dtype=np.float32).copy()
        disturbance = (
            np.asarray(
                env.model.disturbance_vector(env._env()),
                dtype=np.float32,
            )
            if bool(getattr(env, "disturbance_obs", False))
            else np.empty((0,), dtype=np.float32)
        )
        if controller is None:
            action = env.action_space.sample()
        else:
            context = build_context(env, info)
            action = validate_action(
                controller.act(observation, context),
                env,
                controller.name,
            )
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        (
            next_observation,
            reward,
            terminated,
            truncated,
            next_info,
        ) = env.step(action)
        commanded_physical = np.asarray(
            next_info.get(
                "action_commanded_physical",
                env.model.physical_action_vector(
                    env.model.action_vector(action)
                ),
            ),
            dtype=np.float32,
        )
        applied_physical = np.asarray(
            next_info.get(
                "action_applied_physical",
                env.model.physical_action_vector(env.last_act),
            ),
            dtype=np.float32,
        )

        rows["observation"].append(np.asarray(observation, dtype=np.float32))
        rows["true_state"].append(state)
        rows["reference"].append(reference)
        rows["measured_disturbance"].append(disturbance)
        rows["action_policy_normalized"].append(
            2.0 * action.astype(np.float32) - 1.0
        )
        rows["action_commanded_physical"].append(commanded_physical)
        rows["action_applied_physical"].append(applied_physical)
        rows["reward_scalar"].append(float(reward))
        rows["next_observation"].append(
            np.asarray(next_observation, dtype=np.float32)
        )
        rows["next_true_state"].append(
            np.asarray(env.integ.x, dtype=np.float32).copy()
        )
        rows["terminated"].append(bool(terminated))
        rows["truncated"].append(bool(truncated))
        rows["bootstrap_mask"].append(0.0 if terminated else 1.0)
        rows["step_index"].append(step)
        rows["physical_time"].append((step + 1) * float(env.control_dt))
        reward_rows.append(dict(next_info.get("reward_terms") or {}))
        cost_rows.append(dict(next_info.get("costs") or {}))

        observation = next_observation
        info = dict(next_info or {})
        done = bool(terminated or truncated)
        step += 1
    if not rows["reward_scalar"]:
        raise ValueError("collector produced no transitions")

    return _finalize_collected_episode(
        env=env,
        episode_spec=episode_spec,
        collector=collector,
        collector_id=collector_id,
        controller=controller,
        checkpoint_hash=checkpoint_hash,
        episode_index=episode_index,
        split=split,
        track_id=track_id,
        rows=rows,
        reward_rows=reward_rows,
        cost_rows=cost_rows,
        info=info,
        done=done,
        recorded_initial_state=recorded_initial_state,
        requested_initial_state=requested_initial_state,
        reset_state_delta_linf=reset_state_delta_linf,
    )


def _finalize_collected_episode(
    *,
    env,
    episode_spec,
    collector,
    collector_id,
    controller,
    checkpoint_hash,
    episode_index,
    split,
    track_id,
    rows,
    reward_rows,
    cost_rows,
    info,
    done,
    recorded_initial_state,
    requested_initial_state,
    reset_state_delta_linf,
):
    termination_reason = str(
        info.get("termination_reason")
        or (
            "time_limit"
            if rows["truncated"][-1]
            else "collector_limit"
            if not done
            else "terminal"
        )
    )
    reward_terms = _channel_rows(reward_rows)
    cost_channels = _channel_rows(cost_rows)
    cost_totals = {
        name: float(np.sum(values))
        for name, values in cost_channels.items()
    }
    behavior_metadata = (
        controller.metadata()
        if controller is not None and hasattr(controller, "metadata")
        else {
            "name": "uniform-random",
            "kind": "uniform_random_action",
        }
    )
    metadata = {
        "episode_id": (
            f"{episode_spec.episode_spec_id}:collector:{collector_id}"
        ),
        "split": split,
        "track_id": track_id,
        "distribution_id": episode_spec.distribution_id,
        "distribution_hash": episode_spec.distribution_hash,
        "episode_spec_id": episode_spec.episode_spec_id,
        "resolved_hash": episode_spec.resolved_hash,
        "base_seed": episode_spec.base_seed,
        "component_seeds": episode_spec.component_seeds,
        "scenario": episode_spec.scenario_id,
        "goal": episode_spec.goal,
        "reward_spec_id": str(env.reward_spec_id),
        "reward_spec_hash": str(env.reward_spec_hash),
        "env_spec_hash": str(env.env_spec_hash),
        "env_spec_hash_schema": str(env.env_spec_hash_schema),
        "action_mode": str(env.action_mode),
        "collector_id": collector.collector_id,
        "policy_id": collector.policy_id,
        "checkpoint_hash": checkpoint_hash,
        "collector_quality_tag": collector.quality_tag,
        "collector_behavior": behavior_metadata,
        "plant_parameters": episode_spec.plant_parameters,
        "initial_state": recorded_initial_state.tolist(),
        "requested_initial_state": requested_initial_state.tolist(),
        "reset_state_delta_linf": reset_state_delta_linf,
        "reference_schedule": [
            copy.deepcopy(event)
            for event in episode_spec.reference_schedule
        ],
        "disturbance_schedule": [
            copy.deepcopy(event)
            for event in episode_spec.disturbance_schedule
        ],
        "sensor_model": episode_spec.sensor_model,
        "actuator_model": episode_spec.actuator_model,
        "difficulty_tags": list(episode_spec.difficulty_tags),
        "termination_reason": termination_reason,
        "summary": {
            "transitions": len(rows["reward_scalar"]),
            "return": float(np.sum(rows["reward_scalar"])),
            "cost_totals": cost_totals,
            "terminated": bool(rows["terminated"][-1]),
            "truncated": bool(rows["truncated"][-1]),
        },
    }
    if episode_index is not None:
        metadata["episode_index"] = int(episode_index)
    shield = getattr(env, "shield", None)
    if shield is not None and hasattr(shield, "metadata"):
        metadata["safety_shield"] = shield.metadata()
    if collector_id == "recovery":
        initial_debt = float(
            sum(
                float(values[0])
                for values in cost_channels.values()
                if len(values)
            )
        )
        later_debt = float(
            sum(
                float(np.sum(values[1:] > 0.0))
                for values in cost_channels.values()
                if len(values) > 1
            )
        )
        metadata["recovery_audit"] = {
            "initial_safety_debt": initial_debt,
            "controller_created_violation": later_debt > 0.0,
            "post_initial_violation_steps": later_debt,
        }
    return DatasetEpisode(
        metadata=metadata,
        reward_terms=reward_terms,
        cost_channels=cost_channels,
        **rows,
    )


def _channel_rows(rows):
    names = sorted({name for row in rows for name in row})
    return {
        name: np.asarray(
            [float(row.get(name, 0.0)) for row in rows],
            dtype=np.float32,
        )
        for name in names
    }


_BUILTINS = (
    CollectorSpec(
        "nominal_pid",
        "pid",
        "expert",
        "Nominal deterministic PID controller.",
    ),
    CollectorSpec(
        "parameter_randomized_pid",
        "pid-randomized",
        "medium",
        "PID parameters sampled around the nominal tuning.",
    ),
    CollectorSpec(
        "mpc",
        "mpc",
        "expert",
        "Model-predictive reference controller.",
    ),
    CollectorSpec(
        "noisy_pid",
        "pid-filtered-exploration",
        "diverse",
        "PID with temporally correlated bounded action exploration.",
    ),
    CollectorSpec(
        "checkpoint",
        "rl-checkpoint",
        "mixed",
        "A versioned partially trained RL policy checkpoint.",
    ),
    CollectorSpec(
        "safe_excitation",
        "safe-excitation",
        "diverse",
        "Smooth excitation projected through a bounded slew-rate shield.",
    ),
    CollectorSpec(
        "recovery",
        "recovery-controller",
        "stress",
        "Controller specialized for boundary recovery episodes.",
    ),
)
_BUILTIN_COLLECTOR_IDS = frozenset(
    spec.collector_id for spec in _BUILTINS
)
for _spec in _BUILTINS:
    register_collector(_spec)


__all__ = [
    "CollectorSpec",
    "SmoothExcitationPolicy",
    "collect_episode",
    "get_collector",
    "list_collectors",
    "register_collector",
    "unregister_collector",
]
