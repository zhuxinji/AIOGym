"""Behavior-cloning sanity trainer for Dataset v2 action alignment."""
from __future__ import annotations

import copy

import numpy as np

from .dataset_replay import DatasetReplay


class BehaviorCloningPolicy:
    """Small deterministic normalized-action policy."""

    def __init__(self, observation_dim: int, action_dim: int, *, hidden: int = 64):
        import torch.nn as nn

        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def normalized_action(self, observation):
        import torch

        value = torch.as_tensor(
            np.asarray(observation, dtype=np.float32).copy()
        )
        with torch.no_grad():
            action = self.model(value).cpu().numpy()
        return np.asarray(action, dtype=np.float32)

    def predict(self, observation, deterministic=True):
        normalized = self.normalized_action(observation)
        physical = np.clip(0.5 * (normalized + 1.0), 0.0, 1.0)
        return physical.astype(np.float32), None


class BehaviorCloningTrainer:
    """Fit a policy and report whether Dataset v2 actions are learnable."""

    def __init__(
        self,
        dataset: DatasetReplay,
        *,
        hidden: int = 64,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        import torch

        if not isinstance(dataset, DatasetReplay):
            raise TypeError("behavior cloning requires DatasetReplay")
        self.dataset = dataset
        self.device = torch.device(device)
        torch.manual_seed(int(seed))
        self.policy = BehaviorCloningPolicy(
            dataset.observation_dim,
            dataset.action_dim,
            hidden=hidden,
        )
        self.policy.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.model.parameters(),
            lr=float(learning_rate),
        )
        self.steps = 0

    def fit(self, *, steps: int, batch_size: int = 256) -> dict[str, float | int]:
        import torch
        import torch.nn.functional as functional

        if steps < 0 or batch_size <= 0:
            raise ValueError("steps must be non-negative and batch_size positive")
        initial_loss = self.loss(min(batch_size, max(1, len(self.dataset))))
        final_loss = initial_loss
        for _ in range(int(steps)):
            batch = self.dataset.sample(int(batch_size))
            observation = torch.as_tensor(
                batch["observation"],
                device=self.device,
            )
            target = torch.as_tensor(
                batch["action_policy_normalized"],
                device=self.device,
            )
            prediction = self.policy.model(observation)
            loss = functional.mse_loss(prediction, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            final_loss = float(loss.detach().cpu())
            self.steps += 1
        return {
            "steps": self.steps,
            "initial_mse": initial_loss,
            "final_mse": final_loss,
            **self.dataset.metadata(),
        }

    def loss(self, sample_count: int = 1024) -> float:
        import torch
        import torch.nn.functional as functional

        batch = self.dataset.sample(int(sample_count))
        with torch.no_grad():
            prediction = self.policy.model(
                torch.as_tensor(
                    batch["observation"],
                    device=self.device,
                )
            )
            target = torch.as_tensor(
                batch["action_policy_normalized"],
                device=self.device,
            )
            return float(
                functional.mse_loss(prediction, target).cpu()
            )

    def state_dict(self) -> dict:
        return {
            "policy": self.policy.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "dataset": self.dataset.state_dict(),
        }

    def load_state_dict(self, state) -> None:
        self.policy.model.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps = int(state["steps"])
        expected = self.dataset.state_dict()
        persisted = state["dataset"]
        if (
            expected["dataset_id"] != persisted["dataset_id"]
            or expected["dataset_hash"] != persisted["dataset_hash"]
        ):
            raise ValueError("behavior-cloning dataset changed")
        self.dataset._rng.bit_generator.state = copy.deepcopy(
            persisted["rng_state"]
        )
        self.dataset.sample_count = int(persisted["sample_count"])


__all__ = ["BehaviorCloningPolicy", "BehaviorCloningTrainer"]
