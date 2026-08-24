"""Validated Dataset input shared by BC and offline-to-online RL."""
from __future__ import annotations

from pathlib import Path

def load_training_dataset(dataset: str | Path, *, env):
    """Open one non-empty Dataset v2 compatible with ``env``."""

    from aiogym.workflows._metadata import (
        ENVIRONMENT_COMPATIBILITY_FIELDS,
        environment_metadata,
    )
    from aiogym.workflows.dataset import DatasetReader

    reader = DatasetReader(dataset)
    source_environment = reader.metadata["environment"]
    target_environment = environment_metadata(env)
    for field in ENVIRONMENT_COMPATIBILITY_FIELDS:
        if source_environment[field] != target_environment[field]:
            raise ValueError(
                f"training dataset {field} does not match training environment"
            )
    if len(reader) == 0:
        raise ValueError("training dataset must contain at least one episode")
    return reader


def training_dataset_metadata(reader) -> dict:
    """Return the common Dataset provenance stored by training artifacts."""

    return {
        "path": str(reader.path.resolve()),
        "schema_version": reader.metadata["schema_version"],
        "policy": dict(reader.metadata["policy"]),
        "episode_count": len(reader),
        "transition_count": reader.transition_count,
    }


__all__ = ["load_training_dataset", "training_dataset_metadata"]
