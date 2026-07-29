"""Reward-independent service delivery measurements."""
from __future__ import annotations


def service_step_metrics(info, dt: float) -> dict[str, float]:
    costs = dict(info.get("costs", {}))
    shortfall = max(0.0, float(costs.get("service_shortfall", 0.0)))
    active = shortfall > 0.0
    return {
        "service_shortfall_count": 1.0 if active else 0.0,
        "service_shortfall_duration": float(dt) if active else 0.0,
    }


def service_availability(shortfall_duration: float, horizon_seconds: float) -> float:
    if horizon_seconds <= 0.0:
        return 1.0
    unavailable = min(
        max(float(shortfall_duration) / float(horizon_seconds), 0.0),
        1.0,
    )
    return float(1.0 - unavailable)


__all__ = ["service_availability", "service_step_metrics"]
