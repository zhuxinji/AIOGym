"""Behavior-cloning pretraining from the current Dataset format."""
from __future__ import annotations

import math
import numpy as np

from .datasets import training_dataset_metadata


BEHAVIOR_CLONING_SCHEMA_VERSION = "aiogym.behavior_cloning.v1"
BEHAVIOR_CLONING_ALGORITHMS = ("ddpg", "sac", "td3")


def load_demonstrations(reader):
    """Load policy observations and commanded actions from a validated Dataset."""

    episodes = tuple(reader.iter_episodes())
    observations = np.concatenate(
        [episode.array("observation") for episode in episodes], axis=0
    ).astype(np.float32, copy=False)
    actions = np.concatenate(
        [episode.array("commanded_action") for episode in episodes],
        axis=0,
    ).astype(np.float32, copy=False)
    metadata = training_dataset_metadata(reader)
    reader.clear_cache()
    source = {
        "dataset": metadata["path"],
        "dataset_schema": metadata["schema_version"],
        "dataset_content_sha256": metadata["content_sha256"],
        "dataset_environment": metadata["environment"],
        "policy": metadata["policy"],
        "episode_count": metadata["episode_count"],
        "transition_count": metadata["transition_count"],
        "observation_field": "observation",
        "action_field": "commanded_action",
    }
    return observations, actions, source


def behavior_clone(
    model,
    observations,
    actions,
    *,
    algorithm: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    source,
):
    """Supervise an existing SB3 off-policy actor and return its loss record."""

    if algorithm not in BEHAVIOR_CLONING_ALGORITHMS:
        raise ValueError(
            "behavior cloning supports only "
            + ", ".join(BEHAVIOR_CLONING_ALGORITHMS)
        )
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Stable-Baselines3 is required for train(); install `aiogym[rl]`"
        ) from error
    actor = model.policy.actor
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    scaled_actions = np.asarray(
        model.policy.scale_action(actions), dtype=np.float32
    )
    if scaled_actions.shape != actions.shape or not np.all(
        np.isfinite(scaled_actions)
    ):
        raise ValueError("SB3 action scaling produced invalid demonstration targets")
    rng = np.random.default_rng(seed)
    epoch_losses = []
    model.policy.set_training_mode(True)
    for epoch in range(epochs):
        order = rng.permutation(observations.shape[0])
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            observation_batch = torch.as_tensor(
                observations[indices], dtype=torch.float32, device=model.device
            )
            action_batch = torch.as_tensor(
                scaled_actions[indices], dtype=torch.float32, device=model.device
            )
            predicted = model.policy(observation_batch, deterministic=True)
            if predicted.shape != action_batch.shape:
                raise ValueError("SB3 actor output does not match demonstration actions")
            loss = torch.nn.functional.mse_loss(predicted, action_batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("behavior cloning produced a non-finite loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(indices)
        epoch_loss = total_loss / observations.shape[0]
        if not math.isfinite(epoch_loss):
            raise FloatingPointError("behavior cloning produced a non-finite epoch loss")
        epoch_losses.append({"epoch": epoch + 1, "mean_loss": epoch_loss})
    model.policy.set_training_mode(False)
    return {
        "schema_version": BEHAVIOR_CLONING_SCHEMA_VERSION,
        **dict(source),
        "algorithm": algorithm,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "loss": epoch_losses,
    }


__all__ = [
    "BEHAVIOR_CLONING_ALGORITHMS",
    "BEHAVIOR_CLONING_SCHEMA_VERSION",
    "behavior_clone",
    "load_demonstrations",
]
