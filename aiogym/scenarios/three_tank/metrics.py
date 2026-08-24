"""Scenario-specific episode metrics for the fixed Three-Tank interface."""

from __future__ import annotations

from aiogym.scenarios._metrics import (
    regulation_episode_metrics as _base_regulation_episode_metrics,
)

from .model import TRACKING_ERROR_SCALES


def regulation_episode_metrics(env, episode):
    return _base_regulation_episode_metrics(
        env,
        episode,
        output_scale=TRACKING_ERROR_SCALES,
    )


__all__ = [
    "regulation_episode_metrics",
]
