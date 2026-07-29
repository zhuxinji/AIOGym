"""Reward-independent accumulation of complete evaluation scorecards."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from .metric_catalog import SCORECARD_GROUPS
from .metrics.service import service_availability


_RAW_COST_NAMES = {
    "service_shortfall": "service_shortfall_cost",
    "soft_safety": "soft_safety_cost",
    "hard_safety": "hard_safety_cost",
    "protection_intervention": "protection_intervention_cost",
}


class ScorecardAccumulator:
    """Accumulate raw trajectory quantities without reading scalar reward."""

    def __init__(self) -> None:
        self._values = {
            metric: 0.0
            for metrics in SCORECARD_GROUPS.values()
            for metric in metrics
            if metric != "tracking_raw_by_output"
        }
        self._values["safety_margin_min"] = 0.0

    def accumulate(
        self,
        *,
        tracking: Mapping[str, Any],
        economic: Mapping[str, Any],
        safety: Mapping[str, Any],
        costs: Mapping[str, Any],
        model,
        action,
        previous_action,
        dt: float,
    ) -> None:
        error_cost = float(tracking["tracking_mse"])
        slew_cost, effort_cost = _action_integrals(
            model,
            action,
            previous_action,
            dt,
        )
        self._values["regulation_error_cost"] += error_cost
        self._values["regulation_slew_cost"] += slew_cost
        self._values["regulation_effort_cost"] += effort_cost
        self._values["regulation_cost"] += error_cost
        for key in (
            "tracking_iae",
            "tracking_ise",
            "tracking_itae",
        ):
            self._values[key] += float(tracking[key])
        self._values["tracking_overshoot"] = max(
            self._values["tracking_overshoot"],
            float(tracking["tracking_overshoot"]),
        )
        for key, value in economic.items():
            if key in self._values:
                self._values[key] += float(value)
        for key, value in safety.items():
            if key == "safety_margin_min":
                self._values[key] = min(self._values[key], float(value))
            elif key in self._values:
                self._values[key] += float(value)
        for raw_name, scorecard_name in _RAW_COST_NAMES.items():
            self._values[scorecard_name] += float(costs.get(raw_name, 0.0))

    def finalize(
        self,
        *,
        horizon_seconds: float,
        settling_time: float,
        tracking_raw_by_output: Mapping[str, Any],
        controller_metrics: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        values = dict(self._values)
        values["tracking_mse"] = (
            values["regulation_error_cost"] / float(horizon_seconds)
            if horizon_seconds > 0
            else 0.0
        )
        values["service_availability"] = service_availability(
            values["service_shortfall_duration"],
            horizon_seconds,
        )
        values["tracking_settling_time"] = float(settling_time)
        values["tracking_raw_by_output"] = deepcopy(
            dict(tracking_raw_by_output)
        )
        for key, value in controller_metrics.items():
            if key in values:
                values[key] = float(value)
        return {
            group: {
                metric: deepcopy(values[metric])
                for metric in metrics
            }
            for group, metrics in SCORECARD_GROUPS.items()
        }


def flatten_scorecard(scorecard: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    flattened = {}
    for group in SCORECARD_GROUPS:
        flattened.update(deepcopy(dict(scorecard.get(group, {}))))
    return flattened


def group_scorecard(values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        group: {
            metric: deepcopy(values.get(metric))
            for metric in metrics
        }
        for group, metrics in SCORECARD_GROUPS.items()
    }


def _action_integrals(model, action, previous_action, dt):
    current = _normalized_action(model, action)
    previous = _normalized_action(model, previous_action)
    if current.shape != previous.shape:
        raise ValueError("current and previous actions must have the same shape")
    if current.size == 0:
        return 0.0, 0.0
    slew = float(dt * np.mean(((current - previous) / dt) ** 2))
    effort = float(dt * np.mean(current**2))
    return slew, effort


def _normalized_action(model, action):
    values = np.asarray(model.action_vector(action), dtype=np.float64)
    rows = list(model.action_schema())
    lows = np.asarray(
        [float(row.get("low", 0.0)) for row in rows],
        dtype=np.float64,
    )
    highs = np.asarray(
        [float(row.get("high", 1.0)) for row in rows],
        dtype=np.float64,
    )
    if values.size != lows.size:
        return values
    return (values - lows) / np.maximum(highs - lows, 1e-12)


__all__ = [
    "ScorecardAccumulator",
    "flatten_scorecard",
    "group_scorecard",
]
