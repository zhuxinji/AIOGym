"""Behavior factories for Dataset v2 collectors."""
from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from aiogym.controllers.checkpoints import (
    learned_policy_spec_for_environment,
    load_policy_checkpoint,
)
from aiogym.controllers.registry import make_controller
from aiogym.controllers.pid import PIDAgent
from aiogym.rl.safety import ProjectionSafetyShield, SafetyShieldWrapper


@dataclass(frozen=True)
class CollectorBehavior:
    """Resolved policy, environment wrapper, and provenance for one episode."""

    env: Any
    policy: Any
    checkpoint_hash: str | None = None


CollectorAdapter = Callable[
    [Any, Any, Mapping[str, Any]],
    CollectorBehavior,
]
_ADAPTERS: dict[str, CollectorAdapter] = {}


class RandomizedPIDController:
    """PID whose complete tuning is sampled from the episode policy seed."""

    name = "parameter-randomized-pid"
    controller_api_version = "aiogym.controller.v1"
    action_mode = "actuator"
    control_structure = "episode_randomized_pid"

    def __init__(self, model, *, relative_half_width: float = 0.20) -> None:
        self.model = model
        self.relative_half_width = float(relative_half_width)
        if not 0.0 < self.relative_half_width < 1.0:
            raise ValueError("PID relative_half_width must be within (0, 1)")
        nominal = make_controller("pid", model=model)
        self._nominal_loops = copy.deepcopy(nominal.loops_config)
        self._holds = copy.deepcopy(nominal.holds)
        self._demand_u_index = nominal.demand_u_index
        self._controller = nominal
        self.sampled_tuning = copy.deepcopy(self._nominal_loops)

    def reset(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        loops = copy.deepcopy(self._nominal_loops)
        for row in loops:
            if isinstance(row, dict):
                gains = list(row["pid"])
                row["pid"] = [
                    float(
                        gain
                        * rng.uniform(
                            1.0 - self.relative_half_width,
                            1.0 + self.relative_half_width,
                        )
                    )
                    for gain in gains
                ]
            else:
                gains = list(row[2])
                row = list(row)
                row[2] = [
                    float(
                        gain
                        * rng.uniform(
                            1.0 - self.relative_half_width,
                            1.0 + self.relative_half_width,
                        )
                    )
                    for gain in gains
                ]
        self.sampled_tuning = copy.deepcopy(loops)
        self._controller = PIDAgent(
            self.model,
            loops=loops,
            holds=self._holds,
            demand_u_index=self._demand_u_index,
        )
        self._controller.reset(seed=seed)

    def act(self, observation, context):
        return np.clip(
            self._controller.act(observation, context),
            0.0,
            1.0,
        ).astype(np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "parameter_randomized_pid",
            "control_structure": self.control_structure,
            "relative_half_width": self.relative_half_width,
            "sampled_tuning": copy.deepcopy(self.sampled_tuning),
        }


class NoisyPIDController:
    """PID with seeded low-pass Gaussian exploration in action space."""

    name = "noisy-pid"
    controller_api_version = "aiogym.controller.v1"
    action_mode = "actuator"
    control_structure = "pid_low_pass_gaussian_exploration"

    def __init__(
        self,
        model,
        *,
        scale: float = 0.08,
        smoothing: float = 0.85,
    ) -> None:
        self.model = model
        self.scale = float(scale)
        self.smoothing = float(smoothing)
        if self.scale <= 0.0:
            raise ValueError("noisy PID scale must be positive")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("noisy PID smoothing must be within [0, 1)")
        self._controller = make_controller("pid", model=model)
        self._rng = np.random.default_rng(0)
        self._noise = np.zeros(model.action_dim(), dtype=np.float64)
        self.unclipped_count = 0
        self.clipped_count = 0
        self.clipped_l1 = 0.0

    def reset(self, seed: int | None = None) -> None:
        self._controller.reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._noise.fill(0.0)
        self.unclipped_count = 0
        self.clipped_count = 0
        self.clipped_l1 = 0.0

    def act(self, observation, context):
        base = np.asarray(
            self._controller.act(observation, context),
            dtype=np.float64,
        )
        innovation = self._rng.normal(
            0.0,
            self.scale,
            size=base.shape,
        )
        self._noise = (
            self.smoothing * self._noise
            + (1.0 - self.smoothing) * innovation
        )
        proposed = base + self._noise
        clipped = np.clip(proposed, 0.0, 1.0)
        self.unclipped_count += int(proposed.size)
        changed = np.abs(proposed - clipped)
        self.clipped_count += int(np.count_nonzero(changed > 1e-12))
        self.clipped_l1 += float(np.sum(changed))
        return clipped.astype(np.float32)

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "noisy_pid",
            "control_structure": self.control_structure,
            "noise": {
                "kind": "low_pass_gaussian",
                "scale": self.scale,
                "smoothing": self.smoothing,
                "space": "normalized_actuator_[0,1]",
            },
            "action_clip_statistics": {
                "values_seen": self.unclipped_count,
                "values_clipped": self.clipped_count,
                "clipped_l1": self.clipped_l1,
            },
        }


class RecoveryPIDController(RandomizedPIDController):
    """Distinct recovery tuning used only with declared recovery cases."""

    name = "recovery-pid"
    control_structure = "boundary_recovery_pid"

    def __init__(self, model) -> None:
        super().__init__(model, relative_half_width=0.05)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed=seed)
        for _, _, loop in self._controller.loops:
            loop.kp *= 1.25
            loop.ki *= 0.75
        self.sampled_tuning = [
            {
                "u_index": int(u_index),
                "y_index": int(y_index),
                "pid": [loop.kp, loop.ki, loop.kd],
                "reverse": bool(loop.reverse),
                "bias": float(loop.bias),
            }
            for u_index, y_index, loop in self._controller.loops
        ]

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "name": self.name,
                "kind": "recovery_pid",
                "control_structure": self.control_structure,
                "recovery_tuning": {
                    "proportional_multiplier": 1.25,
                    "integral_multiplier": 0.75,
                },
            }
        )
        return metadata


def register_collector_adapter(
    collector_id: str,
    factory: CollectorAdapter,
    *,
    replace: bool = False,
) -> None:
    key = str(collector_id)
    if not key:
        raise ValueError("collector_id must be non-empty")
    if not callable(factory):
        raise TypeError("collector adapter factory must be callable")
    if key in _ADAPTERS and not replace:
        raise ValueError(f"collector adapter {key!r} is registered")
    _ADAPTERS[key] = factory


def get_collector_adapter(collector_id: str) -> CollectorAdapter:
    try:
        return _ADAPTERS[str(collector_id)]
    except KeyError as exc:
        raise KeyError(
            f"collector {collector_id!r} has no behavior adapter"
        ) from exc


def make_collector_behavior(
    collector_id: str,
    env,
    episode_spec,
    *,
    options: Mapping[str, Any] | None = None,
) -> CollectorBehavior:
    return get_collector_adapter(collector_id)(
        env,
        episode_spec,
        dict(options or {}),
    )


def _nominal_pid(env, episode_spec, options):
    del episode_spec, options
    return CollectorBehavior(
        env=env,
        policy=make_controller("pid", model=env.model),
    )


def _randomized_pid(env, episode_spec, options):
    del episode_spec
    return CollectorBehavior(
        env=env,
        policy=RandomizedPIDController(
            env.model,
            relative_half_width=float(
                options.get("relative_half_width", 0.20)
            ),
        ),
    )


def _noisy_pid(env, episode_spec, options):
    del episode_spec
    return CollectorBehavior(
        env=env,
        policy=NoisyPIDController(
            env.model,
            scale=float(options.get("scale", 0.08)),
            smoothing=float(options.get("smoothing", 0.85)),
        ),
    )


def _mpc(env, episode_spec, options):
    del episode_spec, options
    return CollectorBehavior(
        env=env,
        policy=make_controller("mpc", model=env.model),
    )


def _safe_excitation(env, episode_spec, options):
    del episode_spec
    from .collector import SmoothExcitationPolicy

    shield = ProjectionSafetyShield(
        low=float(options.get("low", 0.05)),
        high=float(options.get("high", 0.95)),
        max_delta_per_step=float(options.get("max_delta_per_step", 0.08)),
        fallback_action=np.asarray(
            env.model.default_action(),
            dtype=np.float32,
        ),
        shield_id=str(
            options.get(
                "shield_id",
                "dataset-safe-excitation-projection-v1",
            )
        ),
    )
    return CollectorBehavior(
        env=SafetyShieldWrapper(env, shield),
        policy=SmoothExcitationPolicy(
            smoothing=float(options.get("smoothing", 0.85)),
            scale=float(options.get("scale", 0.12)),
        ),
    )


def _checkpoint(env, episode_spec, options):
    checkpoints = options.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(
            "checkpoint collector requires a non-empty checkpoints list"
        )
    weights = np.asarray(
        [float(row.get("weight", 1.0)) for row in checkpoints],
        dtype=np.float64,
    )
    if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("checkpoint weights must be finite and positive")
    weights /= np.sum(weights)
    rng = np.random.default_rng(episode_spec.component_seeds["policy"])
    selected = checkpoints[int(rng.choice(len(checkpoints), p=weights))]
    path = str(selected.get("path", ""))
    expected_hash = str(selected.get("sha256", ""))
    algorithm_id = str(selected.get("algorithm_id", ""))
    policy_spec = learned_policy_spec_for_environment(
        path,
        algorithm_id,
        expected_hash,
        env,
    )
    policy = load_policy_checkpoint(policy_spec)
    policy.checkpoint_metadata["mixture_weight"] = float(
        selected.get("weight", 1.0)
    )
    return CollectorBehavior(
        env=env,
        policy=policy,
        checkpoint_hash=policy_spec.sha256,
    )


def _recovery(env, episode_spec, options):
    del options
    tags = set(episode_spec.difficulty_tags)
    if not any("safety-recovery" in tag for tag in tags):
        raise ValueError(
            "recovery collector requires a safety-recovery distribution/case"
        )
    return CollectorBehavior(
        env=env,
        policy=RecoveryPIDController(env.model),
    )


for _collector_id, _factory in (
    ("nominal_pid", _nominal_pid),
    ("parameter_randomized_pid", _randomized_pid),
    ("noisy_pid", _noisy_pid),
    ("mpc", _mpc),
    ("safe_excitation", _safe_excitation),
    ("checkpoint", _checkpoint),
    ("recovery", _recovery),
):
    register_collector_adapter(_collector_id, _factory)


__all__ = [
    "CollectorAdapter",
    "CollectorBehavior",
    "NoisyPIDController",
    "RandomizedPIDController",
    "RecoveryPIDController",
    "get_collector_adapter",
    "make_collector_behavior",
    "register_collector_adapter",
]
