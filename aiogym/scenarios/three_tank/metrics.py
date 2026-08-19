"""Task-specific episode metrics for the fixed Three-Tank interface."""
from __future__ import annotations

from aiogym.scenarios._metrics import (
    regulation_episode_metrics as _base_regulation_episode_metrics,
)

from .model import TRACKING_ERROR_SCALES


LEVEL_OUTPUT_INDICES = (0, 2, 4)


def regulation_episode_metrics(env, episode):
    return _base_regulation_episode_metrics(
        env,
        episode,
        output_scale=TRACKING_ERROR_SCALES,
        output_indices=LEVEL_OUTPUT_INDICES,
    )


def thermal_regulation_episode_metrics(env, episode):
    return _base_regulation_episode_metrics(
        env,
        episode,
        output_scale=TRACKING_ERROR_SCALES,
    )


__all__ = [
    "LEVEL_OUTPUT_INDICES",
    "regulation_episode_metrics",
    "thermal_regulation_episode_metrics",
]
