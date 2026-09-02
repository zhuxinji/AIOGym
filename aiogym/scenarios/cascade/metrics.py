"""Episode metrics for the six-output Cascade regulation objective."""

from __future__ import annotations

from aiogym.scenarios._metrics import (
    regulation_episode_metrics as _base_regulation_episode_metrics,
)

from .model import TRACKING_ERROR_SCALES, TRACKING_SETTLING_TOLERANCES


def regulation_episode_metrics(env, episode):
    metrics = _base_regulation_episode_metrics(
        env,
        episode,
        output_scale=TRACKING_ERROR_SCALES,
        settling_tolerance=TRACKING_SETTLING_TOLERANCES,
    )
    evaluated_duration = len(episode.transitions) * float(env.unwrapped.control_dt)
    metrics["settling_rate"] = float(
        metrics["settling_time"] < evaluated_duration
    )
    return metrics


__all__ = ["regulation_episode_metrics"]
