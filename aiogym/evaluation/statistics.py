"""Robust multi-seed, multi-case benchmark statistics."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


STATISTICAL_REPORT_SCHEMA_VERSION = "aiogym.statistical_report.v2"
INTERVENTION_REPORT_SCHEMA_VERSION = "aiogym.intervention_report.v1"
_INTERVENTION_CHANNELS = {
    "protection": (
        "protection_intervention_count",
        "protection_intervention_duration",
        "protection_intervention_cost",
    ),
    "shield": (
        "shield_intervention_count",
        "shield_intervention_duration",
        "shield_intervention_magnitude",
    ),
    "actuator": (
        "actuator_intervention_count",
        "actuator_intervention_duration",
        "actuator_intervention_magnitude",
    ),
}


def interquartile_mean(values) -> float:
    """Return the exact mean of the central 50% empirical mass."""

    sorted_values = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if sorted_values.size == 0 or not np.all(np.isfinite(sorted_values)):
        raise ValueError("IQM values must be finite and non-empty")
    count = sorted_values.size
    weights = np.zeros(count, dtype=np.float64)
    for index in range(count):
        low = index / count
        high = (index + 1) / count
        weights[index] = max(0.0, min(high, 0.75) - max(low, 0.25))
    return float(np.sum(weights * sorted_values) / np.sum(weights))


def stratified_bootstrap_ci(
    matrix,
    *,
    repetitions: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Bootstrap runs independently inside each case stratum."""

    values = _matrix(matrix)
    if repetitions <= 0:
        raise ValueError("bootstrap repetitions must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap confidence must be in (0, 1)")
    generator = np.random.default_rng(int(seed))
    run_count, case_count = values.shape
    samples = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        resampled = np.empty_like(values)
        for case in range(case_count):
            indexes = generator.integers(0, run_count, size=run_count)
            resampled[:, case] = values[indexes, case]
        samples[repetition] = interquartile_mean(resampled)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": interquartile_mean(values),
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1.0 - alpha)),
        "confidence": float(confidence),
        "repetitions": int(repetitions),
    }


def performance_profile(
    matrices: Mapping[str, Any],
    *,
    direction: str,
    thresholds=(1.0, 1.01, 1.05, 1.1, 1.25, 1.5, 2.0),
) -> dict[str, Any]:
    """Dolan–Moré-style profile using a positive shifted loss ratio."""

    if direction not in {"minimize", "maximize"}:
        raise ValueError("performance profile direction is invalid")
    converted = {
        name: (_matrix(value) if direction == "minimize" else -_matrix(value))
        for name, value in matrices.items()
    }
    shapes = {value.shape for value in converted.values()}
    if len(shapes) != 1:
        raise ValueError("performance profile matrices must have equal shapes")
    stacked = np.stack(list(converted.values()), axis=0)
    best = np.min(stacked, axis=0)
    scale = np.maximum(np.abs(best), np.finfo(np.float64).eps)
    ratios = {
        name: 1.0 + np.maximum(0.0, loss - best) / scale
        for name, loss in converted.items()
    }
    return {
        "thresholds": [float(value) for value in thresholds],
        "profiles": {
            name: [
                float(np.mean(ratio <= float(threshold)))
                for threshold in thresholds
            ]
            for name, ratio in ratios.items()
        },
    }


def probability_of_improvement(
    candidate,
    baseline,
    *,
    direction: str,
) -> float:
    candidate_values = _matrix(candidate)
    baseline_values = _matrix(baseline)
    if candidate_values.shape != baseline_values.shape:
        raise ValueError("paired comparison matrices must have equal shapes")
    if direction == "minimize":
        better = candidate_values < baseline_values
    elif direction == "maximize":
        better = candidate_values > baseline_values
    else:
        raise ValueError("comparison direction is invalid")
    tied = candidate_values == baseline_values
    return float(np.mean(better) + 0.5 * np.mean(tied))


def build_final_statistical_report(
    evaluations: Mapping[str, Mapping[str, Any]],
    *,
    baseline: str | None = None,
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Build the final per-seed/per-case matrix and robust summaries."""

    if not evaluations:
        raise ValueError("final statistical report requires evaluations")
    matrices = {}
    case_ids = None
    seeds = None
    metric = None
    direction = None
    track_id = None
    track_hash = None
    for algorithm, evaluation in evaluations.items():
        if evaluation.get("split") != "test":
            raise ValueError("final report accepts locked test evaluations only")
        aggregate = dict(evaluation["aggregate"])
        current_metric = str(aggregate["metric"])
        current_direction = str(aggregate["metric_direction"])
        current_cases, current_seeds, matrix = _evaluation_matrix(
            evaluation,
            current_metric,
        )
        current_track_id = str(evaluation["track_id"])
        current_track_hash = str(evaluation["track_hash"])
        if case_ids is None:
            case_ids = current_cases
            seeds = current_seeds
            metric = current_metric
            direction = current_direction
            track_id = current_track_id
            track_hash = current_track_hash
        elif (
            current_cases != case_ids
            or current_seeds != seeds
            or current_metric != metric
            or current_direction != direction
            or current_track_id != track_id
            or current_track_hash != track_hash
        ):
            raise ValueError(
                "final evaluations must share Track, metric, cases, and seeds"
            )
        matrices[str(algorithm)] = matrix

    summaries = {}
    for offset, (algorithm, matrix) in enumerate(sorted(matrices.items())):
        flattened = matrix.reshape(-1)
        ordered = np.sort(flattened)
        summaries[algorithm] = {
            "official_score": float(
                evaluations[algorithm]["aggregate"].get(
                    "official_score",
                    0.0,
                )
            ),
            "ranking_eligible": bool(
                evaluations[algorithm]["aggregate"].get(
                    "ranking_eligible",
                    True,
                )
            ),
            "mean": float(np.mean(flattened)),
            "median": float(np.median(flattened)),
            "iqm": interquartile_mean(flattened),
            "iqm_bootstrap": stratified_bootstrap_ci(
                matrix,
                repetitions=bootstrap_repetitions,
                seed=bootstrap_seed + offset,
            ),
            "worst_case": (
                float(np.max(flattened))
                if direction == "minimize"
                else float(np.min(flattened))
            ),
            "worst_tail_10pct": (
                float(np.mean(ordered[-max(1, len(ordered) // 10) :]))
                if direction == "minimize"
                else float(np.mean(ordered[: max(1, len(ordered) // 10)]))
            ),
        }
    if baseline is not None:
        if baseline not in matrices:
            raise KeyError(f"unknown baseline algorithm {baseline!r}")
        for algorithm, matrix in matrices.items():
            summaries[algorithm]["probability_of_improvement_vs_baseline"] = (
                probability_of_improvement(
                    matrix,
                    matrices[baseline],
                    direction=direction,
                )
            )

    return {
        "schema_version": STATISTICAL_REPORT_SCHEMA_VERSION,
        "track_id": track_id,
        "track_hash": track_hash,
        "split": "test",
        "metric": metric,
        "metric_direction": direction,
        "seeds": list(seeds),
        "case_ids": list(case_ids),
        "per_seed_per_case_matrix": {
            algorithm: matrix.tolist()
            for algorithm, matrix in sorted(matrices.items())
        },
        "summaries": summaries,
        "performance_profile": performance_profile(
            matrices,
            direction=direction,
        ),
        "intervention_report": build_intervention_report(evaluations),
    }


def build_intervention_report(
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Report intervention matrices, rates, and upper-tail severity."""

    if not evaluations:
        raise ValueError("intervention report requires evaluations")
    split = None
    track_id = None
    track_hash = None
    seeds = None
    case_ids = None
    algorithms = {}
    for algorithm, evaluation in evaluations.items():
        current_split = str(evaluation["split"])
        current_track_id = str(evaluation["track_id"])
        current_track_hash = str(evaluation["track_hash"])
        current_seeds = tuple(
            int(seed) for seed in evaluation["base_seeds"]
        )
        current_cases = tuple(
            str(result["case_id"])
            for result in evaluation["results"]
        )
        identity = (
            current_split,
            current_track_id,
            current_track_hash,
            current_seeds,
            current_cases,
        )
        if split is None:
            (
                split,
                track_id,
                track_hash,
                seeds,
                case_ids,
            ) = identity
        elif identity != (
            split,
            track_id,
            track_hash,
            seeds,
            case_ids,
        ):
            raise ValueError(
                "intervention evaluations must share split, Track, "
                "cases, and seeds"
            )
        channel_rows = {}
        for channel, metrics in _INTERVENTION_CHANNELS.items():
            count = _safety_metric_matrix(evaluation, metrics[0])
            duration = _safety_metric_matrix(evaluation, metrics[1])
            severity = _safety_metric_matrix(evaluation, metrics[2])
            horizons = np.asarray(
                [
                    float(result["case_horizon_seconds"])
                    for result in evaluation["results"]
                ],
                dtype=np.float64,
            )[None, :]
            if np.any(horizons <= 0.0):
                raise ValueError(
                    "intervention rates require positive case horizons"
                )
            rate = duration / horizons
            flattened = severity.reshape(-1)
            channel_rows[channel] = {
                "count_matrix": count.tolist(),
                "duration_matrix": duration.tolist(),
                "rate_matrix": rate.tolist(),
                "severity_matrix": severity.tolist(),
                "summary": {
                    "episode_intervention_probability": float(
                        np.mean(count > 0.0)
                    ),
                    "mean_rate": float(np.mean(rate)),
                    "mean_severity": float(np.mean(flattened)),
                    "p90_severity": float(
                        np.quantile(flattened, 0.90)
                    ),
                    "p95_severity": float(
                        np.quantile(flattened, 0.95)
                    ),
                    "cvar95_severity": _upper_cvar(
                        flattened,
                        quantile=0.95,
                    ),
                },
            }
        algorithms[str(algorithm)] = channel_rows
    return {
        "schema_version": INTERVENTION_REPORT_SCHEMA_VERSION,
        "track_id": track_id,
        "track_hash": track_hash,
        "split": split,
        "seeds": list(seeds),
        "case_ids": list(case_ids),
        "algorithms": dict(sorted(algorithms.items())),
    }


def _evaluation_matrix(evaluation, metric):
    seeds = tuple(int(seed) for seed in evaluation["base_seeds"])
    results = tuple(evaluation["results"])
    case_ids = tuple(str(result["case_id"]) for result in results)
    rows = []
    for result in results:
        episodes = tuple(result.get("episode_metrics") or ())
        by_seed = {int(row["seed"]): row for row in episodes}
        if set(by_seed) != set(seeds):
            raise ValueError(
                f"case {result['case_id']!r} lacks per-seed episode metrics"
            )
        rows.append(
            [
                _episode_metric_value(by_seed[seed], result, metric)
                for seed in seeds
            ]
        )
    return case_ids, seeds, np.asarray(rows, dtype=np.float64).T


def _safety_metric_matrix(evaluation, metric):
    seeds = tuple(int(seed) for seed in evaluation["base_seeds"])
    rows = []
    for result in evaluation["results"]:
        episodes = tuple(result.get("episode_metrics") or ())
        by_seed = {int(row["seed"]): row for row in episodes}
        if set(by_seed) != set(seeds):
            raise ValueError(
                f"case {result['case_id']!r} lacks per-seed "
                "intervention metrics"
            )
        rows.append(
            [float(by_seed[seed].get(metric, 0.0)) for seed in seeds]
        )
    return _matrix(np.asarray(rows, dtype=np.float64).T)


def _upper_cvar(values, *, quantile):
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    threshold = float(np.quantile(array, quantile))
    tail = array[array >= threshold]
    return float(np.mean(tail))


def _episode_metric_value(episode, result, metric):
    if metric in episode:
        return float(episode[metric])
    if metric.endswith("_rate"):
        source = metric[: -len("_rate")]
        if source not in episode:
            raise KeyError(f"episode metrics do not contain {source!r}")
        horizon = float(result.get("case_horizon_seconds", 0.0))
        if horizon <= 0.0:
            raise ValueError("rate metric requires positive case horizon")
        return float(episode[source]) / horizon
    raise KeyError(f"episode metrics do not contain {metric!r}")


def _matrix(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError("statistical input must be a finite non-empty matrix")
    return array


__all__ = [
    "STATISTICAL_REPORT_SCHEMA_VERSION",
    "INTERVENTION_REPORT_SCHEMA_VERSION",
    "build_final_statistical_report",
    "build_intervention_report",
    "interquartile_mean",
    "performance_profile",
    "probability_of_improvement",
    "stratified_bootstrap_ci",
]
