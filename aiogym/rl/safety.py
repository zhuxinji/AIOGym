"""Auditable safety shields for policy actions."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


SAFETY_SHIELD_SCHEMA_VERSION = "aiogym.safety_shield.v1"


@dataclass(frozen=True)
class ShieldDecision:
    proposed_action: np.ndarray
    applied_action: np.ndarray
    intervened: bool
    reasons: tuple[str, ...]
    magnitude_l1: float


class ProjectionSafetyShield:
    """Project actions through bounds, slew limits, and an optional predicate."""

    def __init__(
        self,
        *,
        low=None,
        high=None,
        max_delta_per_step=None,
        constraint: Callable[[np.ndarray, np.ndarray, gym.Env], bool]
        | None = None,
        fallback_action=None,
        shield_id: str = "projection-v1",
    ) -> None:
        if not isinstance(shield_id, str) or not shield_id:
            raise ValueError("shield_id must be a non-empty string")
        self.shield_id = shield_id
        self.low = low
        self.high = high
        self.max_delta_per_step = max_delta_per_step
        self.constraint = constraint
        self.fallback_action = fallback_action

    def project(
        self,
        proposed_action,
        *,
        observation,
        previous_action,
        env,
    ) -> ShieldDecision:
        if not isinstance(env.action_space, spaces.Box):
            raise TypeError("projection shield requires a Box action space")
        proposed = np.asarray(proposed_action, dtype=np.float64).reshape(-1)
        shape = env.action_space.shape
        if proposed.shape != shape or not np.all(np.isfinite(proposed)):
            raise ValueError(
                f"shield action must be finite with shape {shape}"
            )
        environment_low = np.asarray(
            env.action_space.low,
            dtype=np.float64,
        )
        environment_high = np.asarray(
            env.action_space.high,
            dtype=np.float64,
        )
        low = (
            environment_low
            if self.low is None
            else np.maximum(
                environment_low,
                _broadcast(self.low, shape, "shield low"),
            )
        )
        high = (
            environment_high
            if self.high is None
            else np.minimum(
                environment_high,
                _broadcast(self.high, shape, "shield high"),
            )
        )
        if not np.all(low <= high):
            raise ValueError("shield bounds must be ordered")
        applied = np.clip(proposed, low, high)
        reasons = []
        if not np.allclose(applied, proposed, rtol=0.0, atol=1e-12):
            reasons.append("action_bounds")
        if self.max_delta_per_step is not None and previous_action is not None:
            maximum_delta = _broadcast(
                self.max_delta_per_step,
                shape,
                "shield max_delta_per_step",
            )
            if np.any(maximum_delta < 0.0):
                raise ValueError(
                    "shield max_delta_per_step must be non-negative"
                )
            previous = np.asarray(previous_action, dtype=np.float64)
            limited = previous + np.clip(
                applied - previous,
                -maximum_delta,
                maximum_delta,
            )
            if not np.allclose(limited, applied, rtol=0.0, atol=1e-12):
                reasons.append("action_slew")
            applied = limited
        if self.constraint is not None and not bool(
            self.constraint(
                np.asarray(observation, dtype=np.float64).copy(),
                applied.copy(),
                env,
            )
        ):
            if self.fallback_action is not None:
                fallback = _broadcast(
                    self.fallback_action,
                    shape,
                    "shield fallback_action",
                )
            elif previous_action is not None:
                fallback = np.asarray(previous_action, dtype=np.float64)
            else:
                fallback = 0.5 * (low + high)
            applied = np.clip(fallback, low, high)
            reasons.append("safety_constraint")
        delta = applied - proposed
        return ShieldDecision(
            proposed_action=proposed.astype(np.float32),
            applied_action=applied.astype(np.float32),
            intervened=bool(np.any(np.abs(delta) > 1e-12)),
            reasons=tuple(reasons),
            magnitude_l1=float(np.sum(np.abs(delta))),
        )

    def metadata(self) -> dict:
        return {
            "schema_version": SAFETY_SHIELD_SCHEMA_VERSION,
            "shield_id": self.shield_id,
            "kind": "projection",
            "low": _parameter_metadata(self.low),
            "high": _parameter_metadata(self.high),
            "max_delta_per_step": _parameter_metadata(
                self.max_delta_per_step
            ),
            "constraint": (
                None
                if self.constraint is None
                else (
                    f"{getattr(self.constraint, '__module__', '')}."
                    f"{getattr(self.constraint, '__qualname__', '')}"
                ).strip(".")
            ),
            "fallback_action": _parameter_metadata(
                self.fallback_action
            ),
        }


class SafetyShieldWrapper(gym.Wrapper):
    """Apply a shield while preserving raw policy and intervention audit data."""

    def __init__(self, env: gym.Env, shield: ProjectionSafetyShield) -> None:
        super().__init__(env)
        if not isinstance(shield, ProjectionSafetyShield):
            raise TypeError("shield must be a ProjectionSafetyShield")
        self.shield = shield
        self._last_observation = None
        self._previous_shielded_action = None

    def __getattr__(self, name):
        wrapped = self.__dict__.get("env")
        if wrapped is None:
            raise AttributeError(name)
        return getattr(wrapped, name)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_observation = np.asarray(
            observation,
            dtype=np.float32,
        ).copy()
        self._previous_shielded_action = None
        updated = dict(info)
        updated["safety_shield"] = self.shield.metadata()
        return observation, updated

    def step(self, action):
        decision = self.shield.project(
            action,
            observation=self._last_observation,
            previous_action=self._previous_shielded_action,
            env=self.env,
        )
        observation, reward, terminated, truncated, info = self.env.step(
            decision.applied_action
        )
        updated = dict(info)
        costs = dict(updated.get("costs") or {})
        if decision.intervened:
            costs["protection_intervention"] = float(
                costs.get("protection_intervention", 0.0)
            ) + float(getattr(self, "control_dt", 1.0))
        costs["shield_intervention_magnitude"] = float(
            costs.get("shield_intervention_magnitude", 0.0)
        ) + decision.magnitude_l1
        updated.update(
            {
                "costs": costs,
                "action_policy_proposed": (
                    decision.proposed_action.tolist()
                ),
                "action_shielded": decision.applied_action.tolist(),
                "shield_intervened": decision.intervened,
                "shield_intervention_reasons": list(decision.reasons),
                "shield_intervention_magnitude": decision.magnitude_l1,
                "safety_shield": self.shield.metadata(),
            }
        )
        self._last_observation = np.asarray(
            observation,
            dtype=np.float32,
        ).copy()
        self._previous_shielded_action = (
            decision.applied_action.copy()
        )
        return observation, reward, terminated, truncated, updated


def _broadcast(value, shape, name) -> np.ndarray:
    try:
        result = np.broadcast_to(
            np.asarray(value, dtype=np.float64),
            shape,
        ).copy()
    except ValueError as exc:
        raise ValueError(f"{name} is not broadcastable to {shape}") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _parameter_metadata(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return np.asarray(value, dtype=np.float64).tolist()


__all__ = [
    "SAFETY_SHIELD_SCHEMA_VERSION",
    "ProjectionSafetyShield",
    "SafetyShieldWrapper",
    "ShieldDecision",
]
