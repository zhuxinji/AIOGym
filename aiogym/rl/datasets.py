"""Validated Dataset input shared by BC and offline-to-online RL."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aiogym.core.io import jsonable


def load_training_dataset(dataset: str | Path, *, env):
    """Load and validate one immutable episode snapshot for training."""

    from aiogym.workflows._metadata import (
        ENVIRONMENT_COMPATIBILITY_FIELDS,
        environment_metadata,
    )
    from aiogym.workflows.dataset import DatasetReader

    reader = DatasetReader(dataset, cache_episodes=True)
    source_environment = reader.metadata["environment"]
    target_environment = environment_metadata(env)
    for field in ENVIRONMENT_COMPATIBILITY_FIELDS:
        if field not in source_environment:
            raise ValueError(f"training dataset environment is missing {field!r}")
        if source_environment[field] != target_environment[field]:
            raise ValueError(
                f"training dataset {field} does not match training environment"
            )
    if len(reader) == 0:
        raise ValueError("training dataset must contain at least one episode")
    float32_max = np.finfo(np.float32).max
    for episode in reader.iter_episodes():
        for name, space in (
            ("observation", env.observation_space), ("next_observation", env.observation_space),
            ("action", env.action_space), ("commanded_action", env.action_space),
            ("channel_action", env.action_space), ("applied_action", env.action_space),
        ):
            array = episode.array(name)
            if array.shape != (episode.transition_count, *space.shape):
                raise ValueError(f"training dataset {name} shape does not match training space")
            if np.any(array < space.low) or np.any(array > space.high):
                raise ValueError(f"training dataset {name} must belong to training space")
            if np.any(array < -float32_max) or np.any(array > float32_max):
                raise ValueError(f"training dataset {name} exceeds float32 range")
        reward = episode.array("reward")
        if np.any(reward < -float32_max) or np.any(reward > float32_max):
            raise ValueError("training dataset reward exceeds float32 range")
    return reader


def training_dataset_metadata(reader) -> dict:
    """Return the common Dataset provenance stored by training artifacts."""

    return {
        "path": str(reader.path.resolve()),
        "schema_version": reader.metadata["schema_version"],
        "policy": dict(reader.metadata["policy"]),
        "episode_count": len(reader),
        "transition_count": reader.transition_count,
        "content_sha256": reader.content_sha256,
        "environment": jsonable(reader.metadata["environment"]),
    }


__all__ = ["load_training_dataset", "training_dataset_metadata"]
