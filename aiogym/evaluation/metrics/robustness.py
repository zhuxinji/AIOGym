"""Paired robustness evaluation and per-case extrema."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


PAIR_ROLES = ("nominal", "shifted")


def directional_degradation(
    nominal: float,
    shifted: float,
    direction: str,
) -> float:
    """Return signed degradation where positive always means worse."""

    nominal = _finite("nominal", nominal)
    shifted = _finite("shifted", shifted)
    if direction == "minimize":
        return float(shifted - nominal)
    if direction == "maximize":
        return float(nominal - shifted)
    raise ValueError("metric direction must be one of: maximize, minimize")


def paired_seed_metadata(
    base_seeds: Sequence[int],
    *,
    pair_id: str = "robustness",
) -> tuple[dict, ...]:
    """Declare nominal/shifted members that share each base random seed."""

    seeds = tuple(int(seed) for seed in base_seeds)
    if not seeds:
        raise ValueError("paired robustness requires at least one base seed")
    return tuple(
        {
            "pair_id": str(pair_id),
            "pair_role": role,
            "base_seed": seed,
            "environment_seed": seed,
        }
        for seed in seeds
        for role in PAIR_ROLES
    )


def paired_robustness_summary(
    nominal_episodes: Sequence[Mapping],
    shifted_episodes: Sequence[Mapping],
    metric_keys: Sequence[str],
    metric_directions: Mapping[str, str],
    *,
    cvar_alpha: float = 0.9,
) -> dict:
    """Aggregate complete nominal/shifted pairs by shared episode seed."""

    alpha = float(cvar_alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("cvar_alpha must be between 0 and 1")
    nominal = _episodes_by_seed(nominal_episodes, "nominal")
    shifted = _episodes_by_seed(shifted_episodes, "shifted")
    if set(nominal) != set(shifted):
        missing_nominal = sorted(set(shifted) - set(nominal))
        missing_shifted = sorted(set(nominal) - set(shifted))
        raise ValueError(
            "paired robustness requires complete nominal/shifted pairs; "
            f"missing nominal seeds={missing_nominal}, "
            f"missing shifted seeds={missing_shifted}"
        )
    seeds = tuple(sorted(nominal))
    if not seeds:
        raise ValueError("paired robustness requires at least one complete pair")

    metrics = {}
    for metric in metric_keys:
        direction = metric_directions.get(metric)
        if direction not in {"maximize", "minimize"}:
            raise ValueError(
                f"metric {metric!r} requires direction maximize or minimize"
            )
        rows = []
        degradations = []
        for seed in seeds:
            if metric not in nominal[seed] or metric not in shifted[seed]:
                raise ValueError(
                    f"paired robustness metric {metric!r} is missing for seed {seed}"
                )
            nominal_value = _finite(
                f"nominal {metric}",
                nominal[seed][metric],
            )
            shifted_value = _finite(
                f"shifted {metric}",
                shifted[seed][metric],
            )
            degradation = directional_degradation(
                nominal_value,
                shifted_value,
                direction,
            )
            degradations.append(degradation)
            rows.append(
                {
                    "base_seed": seed,
                    "nominal": nominal_value,
                    "shifted": shifted_value,
                    "degradation": degradation,
                }
            )
        metrics[metric] = {
            "direction": direction,
            "pairs": rows,
            **degradation_statistics(degradations, cvar_alpha=alpha),
        }
    return {
        "pair_count": len(seeds),
        "base_seeds": list(seeds),
        "cvar_alpha": alpha,
        "metrics": metrics,
    }


def degradation_statistics(
    degradations: Sequence[float],
    *,
    cvar_alpha: float = 0.9,
) -> dict[str, float]:
    """Return central and upper-tail degradation summaries."""

    values = np.asarray(
        [_finite("degradation", value) for value in degradations],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError("degradation statistics require at least one value")
    alpha = float(cvar_alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError("cvar_alpha must be between 0 and 1")
    ordered = np.sort(values)
    tail_count = max(1, int(math.ceil((1.0 - alpha) * ordered.size)))
    tail = ordered[-tail_count:]
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "worst": float(np.max(values)),
        "cvar": float(np.mean(tail)),
    }


def robustness_extrema(episode_metrics, metric_keys, metric_directions=None):
    """Return best/worst aggregation for an unpaired case summary."""

    directions = dict(metric_directions or {})
    summary = {}
    for key in metric_keys:
        vals = [float(ep[key]) for ep in episode_metrics if key in ep]
        if vals:
            best, worst = (
                (min, max)
                if directions.get(key) == "minimize"
                else (max, min)
            )
            summary[f"{key}_best"] = best(vals)
            summary[f"{key}_worst"] = worst(vals)
    return summary


def _episodes_by_seed(episodes, role):
    indexed = {}
    for episode in episodes:
        if "seed" not in episode:
            raise ValueError(f"{role} episode is missing seed")
        seed = int(episode["seed"])
        if seed in indexed:
            raise ValueError(f"{role} episodes contain duplicate seed {seed}")
        indexed[seed] = dict(episode)
    return indexed


def _finite(name, value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


__all__ = [
    "PAIR_ROLES",
    "degradation_statistics",
    "directional_degradation",
    "paired_robustness_summary",
    "paired_seed_metadata",
    "robustness_extrema",
]
