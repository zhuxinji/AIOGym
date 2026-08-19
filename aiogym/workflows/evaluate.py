"""Evaluate one policy on an ordered set of seeds."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from aiogym.controllers import make_controller
from aiogym.core.contracts import policy_metadata
from aiogym.core.io import jsonable, write_json
from aiogym.core.rollout import rollout

from ._metadata import environment_metadata

EVALUATION_SCHEMA_VERSION = "aiogym.evaluation.v3"


def evaluate(
    *,
    env,
    policy,
    seeds: Sequence[int],
    max_steps: int | None = None,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate ``policy`` without taking ownership of ``env``."""

    resolved_seeds = _validate_seeds(seeds)
    rollout_limit = _max_steps(max_steps)
    resolved_policy = (
        make_controller(policy, env=env) if isinstance(policy, str) else policy
    )
    base_env = env.unwrapped
    resolved_policy_metadata = policy_metadata(resolved_policy)
    metric_function = (
        base_env.reward.episode_metric_function
        if base_env.benchmark is None
        else base_env.benchmark.metric_function
    )
    if metric_function is None:
        raise ValueError(
            f"Reward {base_env.reward.id!r} does not define episode_metric_function"
        )
    ranking_metrics = (
        ((base_env.reward.primary_metric, base_env.reward.metric_direction),)
        if base_env.benchmark is None
        else base_env.benchmark.ranking_metrics
    )
    if ranking_metrics[0][0] is None:
        raise ValueError("Reward must define a primary metric for evaluation")

    episodes = []
    metric_names = None
    for seed in resolved_seeds:
        episode = rollout(
            env,
            resolved_policy,
            seed=seed,
            max_steps=rollout_limit,
            policy_metadata=resolved_policy_metadata,
        )
        metrics = _episode_metrics(metric_function, env, episode)
        names = tuple(metrics)
        if metric_names is None:
            metric_names = names
        elif names != metric_names:
            raise ValueError(
                "episode_metric_function returned inconsistent metric names"
            )
        transitions = episode.transitions
        episodes.append(
            {
                "seed": seed,
                "return": episode.episode_return,
                "length": len(transitions),
                "terminated": bool(transitions and transitions[-1].terminated),
                "truncated": bool(transitions and transitions[-1].truncated),
                "episode_spec": jsonable(episode.reset_info["episode_spec"]),
                "episode_family": episode.reset_info["episode_family"],
                "episode_parameters": jsonable(
                    episode.reset_info["episode_parameters"]
                ),
                "runtime_variation": jsonable(episode.reset_info["runtime_variation"]),
                "metrics": metrics,
                "trajectory": _trajectory(episode),
            }
        )

    aggregate = {
        "episode_return": _summary([row["return"] for row in episodes]),
        "episode_length": _summary([row["length"] for row in episodes]),
    }
    aggregate.update(
        {
            metric: _summary([row["metrics"][metric] for row in episodes])
            for metric in metric_names
        }
    )
    missing_ranking_metrics = [
        name for name, _direction in ranking_metrics if name not in metric_names
    ]
    if missing_ranking_metrics:
        raise ValueError(
            f"ranking metrics are missing from episode metrics: {missing_ranking_metrics}"
        )
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "environment": environment_metadata(env),
        "policy": resolved_policy_metadata,
        "seeds": list(resolved_seeds),
        "max_steps": rollout_limit,
        "trajectory_schema": _trajectory_schema(base_env, episodes),
        "ranking_metrics": [
            {"name": name, "direction": direction}
            for name, direction in ranking_metrics
        ],
        "episodes": episodes,
        "return_distribution": [row["return"] for row in episodes],
        "trajectory_summary": _trajectory_summary(episodes),
        "aggregate": aggregate,
    }
    if output is not None:
        write_json(output, result)
    return result


def _episode_metrics(metric_function, env, episode) -> dict[str, float]:
    raw = metric_function(env, episode)
    if not isinstance(raw, Mapping):
        raise TypeError("episode_metric_function must return a mapping")
    metrics = {}
    for name in sorted(raw):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("episode metric names must be non-empty strings")
        value = raw[name]
        if isinstance(value, bool):
            raise TypeError(f"episode metric {name!r} must be numeric")
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError(f"episode metric {name!r} must be finite")
        metrics[name] = resolved
    return metrics


def _summary(values) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("metric values must be a non-empty finite sequence")
    median = float(np.median(array))
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": median,
        "mad": float(np.median(np.abs(array - median))),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _trajectory(episode) -> dict[str, Any]:
    transitions = episode.transitions
    return {
        "physical_time": [float(row.physical_time) for row in transitions],
        "true_state": [
            _finite_vector("true_state", row.info["true_state"]) for row in transitions
        ],
        "output": [_finite_vector("output", row.info["y"]) for row in transitions],
        "reference": [
            _finite_vector("reference", row.info["transition_reference"])
            for row in transitions
        ],
        "commanded_action": [
            _finite_vector("commanded_action", row.info["commanded_action"])
            for row in transitions
        ],
        "channel_action": [
            _finite_vector("channel_action", row.info["channel_action"])
            for row in transitions
        ],
        "applied_action": [
            _finite_vector("applied_action", row.info["applied_action"])
            for row in transitions
        ],
        "reward": [float(row.reward) for row in transitions],
        "disturbance": [
            _finite_mapping("disturbance", row.info["transition_disturbance"])
            for row in transitions
        ],
        "constraint_costs": [
            _finite_mapping("constraint_costs", row.info["constraint_costs"])
            for row in transitions
        ],
        "minimum_safety_margin": [
            float(row.info["minimum_safety_margin"]) for row in transitions
        ],
    }


def _trajectory_schema(base_env, episodes) -> dict[str, Any]:
    model = base_env.model
    disturbance_names = sorted(
        {
            name
            for episode in episodes
            for row in episode["trajectory"]["disturbance"]
            for name in row
        }
    )
    constraint_names = sorted(
        {
            name
            for episode in episodes
            for row in episode["trajectory"]["constraint_costs"]
            for name in row
        }
    )
    return {
        "state": _named_schema(model.state_schema()),
        "output": _named_schema(model.output_schema()),
        "action": _named_schema(model.action_schema()),
        "disturbance_names": disturbance_names,
        "constraint_cost_names": constraint_names,
    }


def _named_schema(rows) -> list[dict[str, Any]]:
    schema = []
    for row in rows:
        if "name" not in row:
            raise ValueError("trajectory schema rows must contain a name")
        resolved = {"name": str(row["name"])}
        if "unit" in row:
            resolved["unit"] = str(row["unit"])
        if "low" in row and "high" in row:
            low = float(row["low"])
            high = float(row["high"])
            if math.isfinite(low) and math.isfinite(high):
                resolved["low"] = low
                resolved["high"] = high
        schema.append(resolved)
    return schema


def _trajectory_summary(episodes) -> dict[str, Any]:
    trajectories = [row["trajectory"] for row in episodes]
    return {
        "physical_time": _scalar_band(trajectories, "physical_time"),
        "true_state": _vector_band(trajectories, "true_state"),
        "output": _vector_band(trajectories, "output"),
        "reference": _vector_band(trajectories, "reference"),
        "commanded_action": _vector_band(trajectories, "commanded_action"),
        "channel_action": _vector_band(trajectories, "channel_action"),
        "applied_action": _vector_band(trajectories, "applied_action"),
        "reward": _scalar_band(trajectories, "reward"),
        "disturbance": _mapping_bands(
            trajectories,
            "disturbance",
            missing_is_zero=False,
        ),
        "constraint_costs": _mapping_bands(
            trajectories,
            "constraint_costs",
            missing_is_zero=True,
        ),
        "minimum_safety_margin": _scalar_band(
            trajectories,
            "minimum_safety_margin",
        ),
    }


def _vector_band(trajectories, field) -> dict[str, Any]:
    medians = []
    minima = []
    maxima = []
    samples = []
    for step in range(_maximum_field_length(trajectories, field)):
        values = [
            np.asarray(trajectory[field][step], dtype=float)
            for trajectory in trajectories
            if step < len(trajectory[field])
        ]
        matrix = np.stack(values)
        if not np.isfinite(matrix).all():
            raise ValueError(f"trajectory field {field!r} must contain finite values")
        medians.append(np.median(matrix, axis=0).tolist())
        minima.append(np.min(matrix, axis=0).tolist())
        maxima.append(np.max(matrix, axis=0).tolist())
        samples.append(len(values))
    return {
        "median": medians,
        "min": minima,
        "max": maxima,
        "samples": samples,
    }


def _scalar_band(trajectories, field) -> dict[str, Any]:
    medians = []
    minima = []
    maxima = []
    samples = []
    for step in range(_maximum_field_length(trajectories, field)):
        values = np.asarray(
            [
                trajectory[field][step]
                for trajectory in trajectories
                if step < len(trajectory[field])
            ],
            dtype=float,
        )
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"trajectory field {field!r} must contain finite values")
        medians.append(float(np.median(values)))
        minima.append(float(np.min(values)))
        maxima.append(float(np.max(values)))
        samples.append(int(values.size))
    return {
        "median": medians,
        "min": minima,
        "max": maxima,
        "samples": samples,
    }


def _mapping_bands(trajectories, field, *, missing_is_zero) -> dict[str, Any]:
    names = sorted(
        {
            name
            for trajectory in trajectories
            for row in trajectory[field]
            for name in row
        }
    )
    bands = {}
    for name in names:
        expanded = []
        for trajectory in trajectories:
            values = []
            for row in trajectory[field]:
                if name in row:
                    values.append(float(row[name]))
                elif missing_is_zero:
                    values.append(0.0)
                else:
                    raise ValueError(
                        f"trajectory field {field!r} is missing required key {name!r}"
                    )
            expanded.append({field: values})
        bands[name] = _scalar_band(expanded, field)
    return bands


def _maximum_field_length(trajectories, field) -> int:
    lengths = [len(trajectory[field]) for trajectory in trajectories]
    if not lengths or max(lengths) <= 0:
        raise ValueError("trajectory summary requires at least one transition")
    return max(lengths)


def _finite_vector(name, values) -> list[float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"trajectory {name} must contain finite values")
    return array.tolist()


def _finite_mapping(name, values) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"trajectory {name} must be a mapping")
    resolved = {str(key): float(value) for key, value in values.items()}
    if not all(math.isfinite(value) for value in resolved.values()):
        raise ValueError(f"trajectory {name} must contain finite values")
    return resolved


def _validate_seeds(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("seeds must be a sequence of non-negative integers")
    seeds = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("seeds must be a sequence of non-negative integers")
        seed = int(value)
        if seed < 0:
            raise ValueError("seeds must be non-negative integers")
        seeds.append(seed)
    if not seeds:
        raise ValueError("seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    return tuple(seeds)


def _max_steps(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("max_steps must be a positive integer")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError("max_steps must be a positive integer")
    return resolved


__all__ = ["EVALUATION_SCHEMA_VERSION", "evaluate"]
