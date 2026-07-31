"""Inference-only adapters for native Torch policy checkpoints."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from aiogym.rl.policy_spec import validate_policy_spec


class NativeBehaviorCloningPolicy:
    """Deterministic BC network returning native normalized actions."""

    algorithm_id = "bc"
    name = "BC"

    def __init__(
        self,
        policy_spec: Mapping[str, Any],
        policy_state_dict: Mapping[str, Any],
        *,
        device: str = "cpu",
    ) -> None:
        spec = validate_policy_spec(policy_spec)
        if spec["algorithm_id"] != self.algorithm_id:
            raise ValueError("BC policy requires a BC policy_spec")

        try:
            from aiogym.rl.behavior_cloning import BehaviorCloningPolicy
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "torch is required for native BC policies; install AIO-Gym "
                "with `pip install 'aiogym[rl]'`"
            ) from exc

        hidden_sizes = spec["network"]["hidden_sizes"]
        if hidden_sizes[0] != hidden_sizes[1]:
            raise ValueError("BC policy requires equal hidden layer widths")
        policy = BehaviorCloningPolicy(
            spec["observation_dim"],
            spec["action_dim"],
            hidden=hidden_sizes[0],
        )
        try:
            import torch

            self.device = torch.device(device)
            policy.model.load_state_dict(dict(policy_state_dict), strict=True)
            policy.model.to(self.device)
            policy.model.eval()
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "torch is required for native BC policies; install AIO-Gym "
                "with `pip install 'aiogym[rl]'`"
            ) from exc
        except RuntimeError as exc:
            raise ValueError(
                "BC policy_state_dict does not match policy_spec"
            ) from exc
        self.model = policy.model
        self.policy_spec = spec
        self.observation_dim = spec["observation_dim"]
        self.action_dim = spec["action_dim"]

    def predict(self, observation, deterministic: bool = True):
        del deterministic
        values, single = _validated_observation(
            observation,
            self.observation_dim,
        )
        import torch

        with torch.no_grad():
            action = self.model(
                torch.as_tensor(values, device=self.device)
            ).cpu().numpy()
        result = np.asarray(action, dtype=np.float32)
        return (result[0] if single else result), None

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "algorithm_id": self.algorithm_id,
            "scenario": self.policy_spec["scenario"],
            "action_mode": self.policy_spec["action_mode"],
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "native_action_contract": "normalized[-1,1]",
        }


class NativeRLPDPolicy:
    """Deterministic actor-only RLPD policy for evaluation."""

    algorithm_id = "rlpd"
    name = "RLPD"

    def __init__(
        self,
        policy_spec: Mapping[str, Any],
        policy_state_dict: Mapping[str, Any],
        *,
        device: str = "cpu",
    ) -> None:
        spec = validate_policy_spec(policy_spec)
        if spec["algorithm_id"] != self.algorithm_id:
            raise ValueError("RLPD policy requires an RLPD policy_spec")

        try:
            import torch

            from aiogym.rl.rlpd import Actor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "torch is required for native RLPD policies; install "
                "AIO-Gym with `pip install 'aiogym[rl]'`"
            ) from exc

        hidden_sizes = spec["network"]["hidden_sizes"]
        if hidden_sizes[0] != hidden_sizes[1]:
            raise ValueError("RLPD policy requires equal hidden layer widths")
        self.device = torch.device(device)
        actor = Actor(
            spec["observation_dim"],
            spec["action_dim"],
            hidden=hidden_sizes[0],
        )
        try:
            actor.load_state_dict(dict(policy_state_dict), strict=True)
        except RuntimeError as exc:
            raise ValueError(
                "RLPD policy_state_dict does not match policy_spec"
            ) from exc
        actor.to(self.device)
        actor.eval()
        self.actor = actor
        self.policy_spec = spec
        self.observation_dim = spec["observation_dim"]
        self.action_dim = spec["action_dim"]

    def predict(self, observation, deterministic: bool = True):
        del deterministic
        values, single = _validated_observation(
            observation,
            self.observation_dim,
        )
        import torch

        with torch.no_grad():
            mu, _ = self.actor(
                torch.as_tensor(values, device=self.device)
            )
            action = torch.tanh(mu).cpu().numpy()
        result = np.asarray(action, dtype=np.float32)
        return (result[0] if single else result), None

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "algorithm_id": self.algorithm_id,
            "scenario": self.policy_spec["scenario"],
            "action_mode": self.policy_spec["action_mode"],
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "native_action_contract": "normalized[-1,1]",
        }


def _validated_observation(
    observation,
    expected_dim: int,
) -> tuple[np.ndarray, bool]:
    values = np.asarray(observation, dtype=np.float32)
    single = values.ndim == 1
    if single:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != expected_dim:
        actual = values.shape[-1] if values.ndim else 0
        raise ValueError(
            f"native policy expects {expected_dim} observations, got {actual}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("native policy observations must be finite")
    return np.ascontiguousarray(values), single


__all__ = [
    "NativeBehaviorCloningPolicy",
    "NativeRLPDPolicy",
]
