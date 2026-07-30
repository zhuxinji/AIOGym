"""A compact reward/cost-separated Lagrangian SAC baseline."""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence

import numpy as np


LAGRANGIAN_SAC_STATE_SCHEMA_VERSION = "aiogym.lagrangian_sac.v1"


class LagrangeMultiplier:
    """Projected dual-ascent state for an expected cost budget."""

    def __init__(
        self,
        *,
        cost_limit: float,
        learning_rate: float = 1e-3,
        initial_value: float = 0.0,
        maximum: float | None = None,
    ) -> None:
        self.cost_limit = _non_negative("cost_limit", cost_limit)
        self.learning_rate = _positive("learning_rate", learning_rate)
        self.value = _non_negative("initial_value", initial_value)
        self.maximum = (
            None if maximum is None else _positive("maximum", maximum)
        )
        self.updates = 0

    def update(self, observed_cost: float) -> float:
        cost = _finite("observed_cost", observed_cost)
        value = max(
            0.0,
            self.value
            + self.learning_rate * (cost - self.cost_limit),
        )
        if self.maximum is not None:
            value = min(value, self.maximum)
        self.value = float(value)
        self.updates += 1
        return self.value

    def state_dict(self) -> dict:
        return {
            "cost_limit": self.cost_limit,
            "learning_rate": self.learning_rate,
            "value": self.value,
            "maximum": self.maximum,
            "updates": self.updates,
        }

    def load_state_dict(self, state) -> None:
        if not math.isclose(
            float(state["cost_limit"]),
            self.cost_limit,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Lagrange cost limit changed on resume")
        if (
            not math.isclose(
                float(state["learning_rate"]),
                self.learning_rate,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or state.get("maximum") != self.maximum
        ):
            raise ValueError("Lagrange update contract changed on resume")
        self.value = _non_negative("value", state["value"])
        self.updates = int(state["updates"])


class ConstrainedReplayBuffer:
    """Transition replay with a scalar cost kept separate from reward."""

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("constrained replay capacity must be positive")
        self.capacity = int(capacity)
        self.size = 0
        self.position = 0
        self._rng = np.random.default_rng(int(seed))
        self._arrays = {}

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        *,
        observation,
        action,
        reward,
        cost,
        next_observation,
        terminated,
        truncated,
        bootstrap_mask=None,
    ) -> None:
        row = {
            "observation": np.asarray(observation, dtype=np.float32),
            "action": np.asarray(action, dtype=np.float32),
            "reward": np.asarray(reward, dtype=np.float32),
            "cost": np.asarray(cost, dtype=np.float32),
            "next_observation": np.asarray(
                next_observation,
                dtype=np.float32,
            ),
            "terminated": np.asarray(terminated, dtype=np.bool_),
            "truncated": np.asarray(truncated, dtype=np.bool_),
            "bootstrap_mask": np.asarray(
                float(not bool(terminated))
                if bootstrap_mask is None
                else bootstrap_mask,
                dtype=np.float32,
            ),
        }
        if not np.all(np.isfinite(row["cost"])) or np.any(row["cost"] < 0):
            raise ValueError("constrained replay cost must be non-negative")
        if not self._arrays:
            self._arrays = {
                name: np.empty(
                    (self.capacity, *value.shape),
                    dtype=value.dtype,
                )
                for name, value in row.items()
            }
        for name, value in row.items():
            if value.shape != self._arrays[name].shape[1:]:
                raise ValueError(f"constrained replay {name} shape changed")
            self._arrays[name][self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.size == 0:
            raise ValueError("cannot sample an empty constrained replay")
        indexes = self._rng.integers(
            0,
            self.size,
            size=int(batch_size),
        )
        return {
            name: values[indexes].copy()
            for name, values in self._arrays.items()
        }

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
            "arrays": {
                name: values.copy()
                for name, values in self._arrays.items()
            },
        }

    def load_state_dict(self, state) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("constrained replay capacity changed on resume")
        self.size = int(state["size"])
        self.position = int(state["position"])
        if not 0 <= self.size <= self.capacity:
            raise ValueError("invalid constrained replay size")
        if not 0 <= self.position < self.capacity:
            raise ValueError("invalid constrained replay position")
        self._arrays = {
            str(name): np.asarray(values).copy()
            for name, values in state["arrays"].items()
        }
        self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])


class LagrangianSAC:
    """Twin reward/cost critic SAC with a projected Lagrange multiplier."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        cost_limit: float,
        hidden: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        entropy_coefficient: float = 0.1,
        learning_rate: float = 3e-4,
        multiplier_learning_rate: float = 1e-3,
        batch_size: int = 256,
        replay_capacity: int = 1_000_000,
        device: str = "cpu",
        seed: int = 0,
        cost_channels: Sequence[str] = (
            "soft_safety",
            "hard_safety",
        ),
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Lagrangian SAC requires `pip install 'aiogym[rl]'`"
            ) from exc
        from .rlpd import Actor, Critic

        self.torch = torch
        self.device = torch.device(device)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("observation_dim and action_dim must be positive")
        self.gamma = _unit_interval("gamma", gamma, include_zero=True)
        self.tau = _unit_interval("tau", tau, include_zero=False)
        self.entropy_coefficient = _non_negative(
            "entropy_coefficient",
            entropy_coefficient,
        )
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.cost_channels = tuple(str(name) for name in cost_channels)
        if not self.cost_channels or any(
            not name for name in self.cost_channels
        ):
            raise ValueError("cost_channels must contain non-empty names")
        if len(set(self.cost_channels)) != len(self.cost_channels):
            raise ValueError("cost_channels must be unique")
        self.seed = int(seed)
        torch.manual_seed(self.seed)
        self.actor = Actor(
            self.observation_dim,
            self.action_dim,
            hidden,
        ).to(self.device)
        self.reward_critics = torch.nn.ModuleList(
            [
                Critic(self.observation_dim, self.action_dim, hidden)
                for _ in range(2)
            ]
        ).to(self.device)
        self.cost_critics = torch.nn.ModuleList(
            [
                Critic(self.observation_dim, self.action_dim, hidden)
                for _ in range(2)
            ]
        ).to(self.device)
        self.reward_targets = copy.deepcopy(self.reward_critics)
        self.cost_targets = copy.deepcopy(self.cost_critics)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=learning_rate,
        )
        self.reward_optimizer = torch.optim.Adam(
            self.reward_critics.parameters(),
            lr=learning_rate,
        )
        self.cost_optimizer = torch.optim.Adam(
            self.cost_critics.parameters(),
            lr=learning_rate,
        )
        self.multiplier = LagrangeMultiplier(
            cost_limit=cost_limit,
            learning_rate=multiplier_learning_rate,
        )
        self.replay = ConstrainedReplayBuffer(
            replay_capacity,
            seed=self.seed + 1,
        )
        self.environment_transitions = 0
        self.gradient_updates = 0

    @staticmethod
    def aggregate_cost(
        info: Mapping,
        cost_channels: Sequence[str],
    ) -> float:
        costs = dict(info.get("costs") or {})
        value = float(
            sum(float(costs.get(name, 0.0)) for name in cost_channels)
        )
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "aggregated constrained cost must be finite "
                "and non-negative"
            )
        return value

    def push(
        self,
        observation,
        normalized_action,
        reward,
        cost,
        next_observation,
        terminated,
        truncated=False,
        bootstrap_mask=None,
    ) -> None:
        self.replay.add(
            observation=observation,
            action=normalized_action,
            reward=reward,
            cost=cost,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            bootstrap_mask=bootstrap_mask,
        )
        self.environment_transitions += 1

    def policy_action(self, observation, *, deterministic=False):
        torch = self.torch
        with torch.no_grad():
            value = torch.as_tensor(
                np.asarray(observation, dtype=np.float32)[None, :],
                device=self.device,
            )
            action = self.actor.act(value, deterministic=deterministic)[0]
        return action.cpu().numpy().astype(np.float32)

    def act(self, observation, deterministic=False):
        normalized = self.policy_action(
            observation,
            deterministic=deterministic,
        )
        return np.clip((normalized + 1.0) * 0.5, 0.0, 1.0).astype(
            np.float32
        )

    def update(self) -> dict[str, float]:
        torch = self.torch
        functional = torch.nn.functional
        batch = self.replay.sample(self.batch_size)
        tensor = {
            name: torch.as_tensor(values, device=self.device)
            for name, values in batch.items()
        }
        observation = tensor["observation"]
        action = tensor["action"]
        reward = tensor["reward"].reshape(-1)
        cost = tensor["cost"].reshape(-1)
        next_observation = tensor["next_observation"]
        mask = tensor["bootstrap_mask"].reshape(-1)
        with torch.no_grad():
            next_action, next_log_probability = self.actor.sample(
                next_observation
            )
            reward_next = torch.stack(
                [
                    critic(next_observation, next_action)
                    for critic in self.reward_targets
                ]
            ).min(0).values
            cost_next = torch.stack(
                [
                    critic(next_observation, next_action)
                    for critic in self.cost_targets
                ]
            ).max(0).values
            reward_target = reward + self.gamma * mask * (
                reward_next
                - self.entropy_coefficient * next_log_probability
            )
            cost_target = cost + self.gamma * mask * cost_next
        reward_loss = sum(
            functional.mse_loss(
                critic(observation, action),
                reward_target,
            )
            for critic in self.reward_critics
        )
        cost_loss = sum(
            functional.mse_loss(
                critic(observation, action),
                cost_target,
            )
            for critic in self.cost_critics
        )
        self.reward_optimizer.zero_grad(set_to_none=True)
        reward_loss.backward()
        self.reward_optimizer.step()
        self.cost_optimizer.zero_grad(set_to_none=True)
        cost_loss.backward()
        self.cost_optimizer.step()

        policy_action, log_probability = self.actor.sample(observation)
        policy_reward = torch.stack(
            [
                critic(observation, policy_action)
                for critic in self.reward_critics
            ]
        ).min(0).values
        policy_cost = torch.stack(
            [
                critic(observation, policy_action)
                for critic in self.cost_critics
            ]
        ).max(0).values
        actor_loss = (
            self.entropy_coefficient * log_probability
            - policy_reward
            + self.multiplier.value * policy_cost
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        observed_policy_cost = float(
            torch.clamp(policy_cost.detach(), min=0.0).mean()
        )
        observed_transition_cost = float(cost.detach().mean())
        multiplier = self.multiplier.update(observed_transition_cost)
        self._soft_update()
        self.gradient_updates += 1
        return {
            "reward_critic_loss": float(reward_loss.detach()),
            "cost_critic_loss": float(cost_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "policy_cost_estimate": observed_policy_cost,
            "observed_transition_cost": observed_transition_cost,
            "lagrange_multiplier": multiplier,
        }

    def _soft_update(self) -> None:
        torch = self.torch
        with torch.no_grad():
            for source_group, target_group in (
                (self.reward_critics, self.reward_targets),
                (self.cost_critics, self.cost_targets),
            ):
                for source, target in zip(
                    source_group.parameters(),
                    target_group.parameters(),
                ):
                    target.mul_(1.0 - self.tau).add_(self.tau * source)

    def state_dict(self) -> dict:
        return {
            "schema_version": LAGRANGIAN_SAC_STATE_SCHEMA_VERSION,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "cost_channels": list(self.cost_channels),
            "training_contract": {
                "gamma": self.gamma,
                "tau": self.tau,
                "entropy_coefficient": self.entropy_coefficient,
                "batch_size": self.batch_size,
                "cost_limit": self.multiplier.cost_limit,
                "multiplier_learning_rate": (
                    self.multiplier.learning_rate
                ),
            },
            "actor": self.actor.state_dict(),
            "reward_critics": self.reward_critics.state_dict(),
            "cost_critics": self.cost_critics.state_dict(),
            "reward_targets": self.reward_targets.state_dict(),
            "cost_targets": self.cost_targets.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "reward_optimizer": self.reward_optimizer.state_dict(),
            "cost_optimizer": self.cost_optimizer.state_dict(),
            "multiplier": self.multiplier.state_dict(),
            "replay": self.replay.state_dict(),
            "environment_transitions": self.environment_transitions,
            "gradient_updates": self.gradient_updates,
            "torch_rng_state": self.torch.get_rng_state(),
            "torch_cuda_rng_state_all": (
                self.torch.cuda.get_rng_state_all()
                if self.torch.cuda.is_available()
                else None
            ),
        }

    def load_state_dict(self, state) -> None:
        if state.get("schema_version") != LAGRANGIAN_SAC_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported Lagrangian SAC state schema")
        if (
            int(state["observation_dim"]) != self.observation_dim
            or int(state["action_dim"]) != self.action_dim
            or tuple(state["cost_channels"]) != self.cost_channels
        ):
            raise ValueError("Lagrangian SAC resume contract mismatch")
        expected_contract = {
            "gamma": self.gamma,
            "tau": self.tau,
            "entropy_coefficient": self.entropy_coefficient,
            "batch_size": self.batch_size,
            "cost_limit": self.multiplier.cost_limit,
            "multiplier_learning_rate": self.multiplier.learning_rate,
        }
        if state.get("training_contract") != expected_contract:
            raise ValueError(
                "Lagrangian SAC training contract changed on resume"
            )
        self.actor.load_state_dict(state["actor"])
        self.reward_critics.load_state_dict(state["reward_critics"])
        self.cost_critics.load_state_dict(state["cost_critics"])
        self.reward_targets.load_state_dict(state["reward_targets"])
        self.cost_targets.load_state_dict(state["cost_targets"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.reward_optimizer.load_state_dict(state["reward_optimizer"])
        self.cost_optimizer.load_state_dict(state["cost_optimizer"])
        self.multiplier.load_state_dict(state["multiplier"])
        self.replay.load_state_dict(state["replay"])
        self.environment_transitions = int(
            state["environment_transitions"]
        )
        self.gradient_updates = int(state["gradient_updates"])
        self.torch.set_rng_state(state["torch_rng_state"])
        cuda_state = state.get("torch_cuda_rng_state_all")
        if cuda_state is not None and self.torch.cuda.is_available():
            self.torch.cuda.set_rng_state_all(cuda_state)

    def artifact_metadata(self) -> dict:
        return {
            "algorithm_id": "lagrangian_sac",
            "state_schema_version": LAGRANGIAN_SAC_STATE_SCHEMA_VERSION,
            "cost_channels": list(self.cost_channels),
            "cost_limit": self.multiplier.cost_limit,
            "lagrange_multiplier": self.multiplier.value,
            "environment_transitions": self.environment_transitions,
            "gradient_updates": self.gradient_updates,
            "reward_cost_separation": True,
        }


def _finite(name: str, value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative(name: str, value) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(name: str, value) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _unit_interval(name: str, value, *, include_zero: bool) -> float:
    result = _finite(name, value)
    lower_ok = result >= 0.0 if include_zero else result > 0.0
    if not lower_ok or result > 1.0:
        raise ValueError(f"{name} must be in {'[0, 1]' if include_zero else '(0, 1]'}")
    return result


__all__ = [
    "LAGRANGIAN_SAC_STATE_SCHEMA_VERSION",
    "ConstrainedReplayBuffer",
    "LagrangeMultiplier",
    "LagrangianSAC",
]
