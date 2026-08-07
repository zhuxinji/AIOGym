"""Task-owned multi-seed policy evaluation without benchmark protocols."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import aiogym.scenarios  # noqa: F401
from aiogym.core import (
    resolve_condition_alias,
    rollout,
)

from ._policy_target import resolve_policy_target
from .artifacts import write_run_bundle


EVALUATION_SCHEMA_VERSION = "aiogym.evaluation.v1"


def evaluate(
    policy,
    *,
    task: str | None = None,
    plant=None,
    condition=None,
    preset: str | None = None,
    seeds: Sequence[int] = (0,),
    output: str | Path | None = None,
    overwrite: bool = False,
    max_steps: int | None = None,
    allow_plant_transfer: bool = False,
    allow_condition_transfer: bool = False,
) -> dict[str, Any]:
    resolved_seeds = _seeds(seeds)
    condition = resolve_condition_alias(condition, preset)
    target = resolve_policy_target(
        policy,
        task=task,
        plant=plant,
        condition=condition,
        allow_plant_transfer=allow_plant_transfer,
        allow_condition_transfer=allow_condition_transfer,
    )
    env = target.env
    resolved_policy = target.policy
    task_spec = env.task
    policy_metadata = dict(resolved_policy.metadata())
    episodes = []
    try:
        for seed in resolved_seeds:
            episode = rollout(env, resolved_policy, seed=seed, max_steps=max_steps)
            episodes.append(_episode_metrics(env, episode, task_spec.objective))
    finally:
        if target.created_env:
            env.close()
    metric_keys = sorted(
        set.intersection(*(set(row["metrics"]) for row in episodes))
    )
    aggregate = {
        key: _aggregate([float(row["metrics"][key]) for row in episodes])
        for key in metric_keys
    }
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "workflow": "evaluate",
        "task_id": task_spec.id,
        "task_hash": task_spec.task_hash,
        "plant_id": env.plant.id,
        "plant_hash": env.plant.plant_hash,
        "condition_id": env.condition.id,
        "condition_hash": env.condition.condition_hash,
        "interface_hash": env.identity.interface_hash,
        "env_hash": env.identity.env_hash,
        "objective": task_spec.objective,
        "primary_metric": task_spec.primary_metric,
        "metric_direction": task_spec.metric_direction,
        "seeds": list(resolved_seeds),
        "policy": policy_metadata,
        "policy_training_contract": target.training_contract,
        "target_environment_contract": target.target_contract,
        "contract_status": target.contract_status,
        "transfer_flags": dict(target.transfer_flags),
        "episodes": episodes,
        "aggregate": aggregate,
    }
    if output is not None:
        result["artifacts"] = write_run_bundle(
            result,
            output,
            workflow="evaluate",
            report_markdown=render_evaluation_report(result),
            overwrite=overwrite,
        )
    return result


def _episode_metrics(env, episode, objective):
    transitions = episode.transitions
    dt = float(env.control_dt)
    metrics = {
        "return": episode.episode_return,
        "constraint_violations": 0.0,
        "termination": float(bool(transitions and transitions[-1].terminated)),
        "energy": 0.0,
    }
    errors = []
    profit = 0.0
    production = 0.0
    for transition in transitions:
        info = transition.info
        constraints = info.get("constraint_costs", {})
        violated = any(float(value) > 0.0 for value in constraints.values())
        violated = violated or bool(info.get("state_violated", False))
        metrics["constraint_violations"] += float(violated)
        state = info.get("true_state")
        energy_kw = info.get("energy_kw")
        if energy_kw is None and state is not None:
            resolver = getattr(env.model, "action_energy_kw", None)
            if callable(resolver):
                try:
                    energy_kw = resolver(
                        transition.action,
                        state,
                        info.get("disturbance", {}),
                    )
                except TypeError:
                    energy_kw = resolver(transition.action, state)
        metrics["energy"] += float(energy_kw or 0.0) * dt / 3600.0
        profit += float(info.get("profit", transition.reward))
        production += float(
            info.get("production", info.get("product_flow_m3s", 0.0))
        ) * dt
        output, reference = _output_reference(env, transition)
        if output is not None and reference is not None:
            scale = _output_scale(env, len(reference))
            errors.append((output - reference) / scale)
    if objective == "economic":
        metrics.update(
            {
                "economic_objective": profit,
                "profit": profit,
                "production": production,
            }
        )
    else:
        matrix = np.asarray(errors, dtype=float)
        if matrix.size:
            absolute = np.abs(matrix)
            metrics.update(
                {
                    "tracking_iae": float(np.sum(absolute) * dt),
                    "tracking_ise": float(np.sum(matrix**2) * dt),
                    "tracking_itae": float(
                        sum(
                            (index + 1) * dt * float(np.sum(row)) * dt
                            for index, row in enumerate(absolute)
                        )
                    ),
                    "overshoot": float(np.max(np.maximum(matrix, 0.0))),
                    "final_error": float(np.max(absolute[-1])),
                    "settling_time": _settling_time(absolute, dt),
                }
            )
        else:
            metrics.update(
                {
                    "tracking_iae": 0.0,
                    "tracking_ise": 0.0,
                    "tracking_itae": 0.0,
                    "overshoot": 0.0,
                    "final_error": 0.0,
                    "settling_time": 0.0,
                }
            )
    return {
        "seed": episode.seed,
        "steps": len(transitions),
        "terminated": bool(transitions and transitions[-1].terminated),
        "truncated": bool(transitions and transitions[-1].truncated),
        "metrics": metrics,
    }


def _output_reference(env, transition):
    info = transition.info
    reference = info.get("reference", info.get("y_sp"))
    output = info.get("y")
    if output is None or reference is None:
        return None, None
    output = np.asarray(output, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    if output.shape != reference.shape:
        return None, None
    return output, reference


def _output_scale(env, size):
    resolver = getattr(env.model, "controlled_output_scales", None)
    if callable(resolver):
        values = np.asarray(resolver(), dtype=float).reshape(-1)
        if values.shape == (size,) and np.all(values > 0):
            return values
    return np.ones(size, dtype=float)


def _settling_time(absolute_errors, dt, tolerance=0.02):
    unsettled = np.any(absolute_errors > tolerance, axis=1)
    indexes = np.flatnonzero(unsettled)
    return 0.0 if indexes.size == 0 else float((indexes[-1] + 1) * dt)


def _aggregate(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _seeds(values):
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ValueError("seeds must be a non-empty sequence")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("seeds must contain non-negative integers")
        seed = int(value)
        if seed < 0:
            raise ValueError("seeds must contain non-negative integers")
        result.append(seed)
    if len(set(result)) != len(result):
        raise ValueError("seeds must not contain duplicates")
    return tuple(result)


def render_evaluation_report(result: Mapping[str, Any]) -> str:
    lines = [
        f"# Evaluation: {result['task_id']}",
        "",
        f"Policy: `{result['policy'].get('id', result['policy'].get('name', 'policy'))}`",
        "",
        f"Seeds: {', '.join(str(seed) for seed in result['seeds'])}",
        "",
        "## Aggregate metrics",
        "",
    ]
    for name, values in result["aggregate"].items():
        mean = values["mean"]
        rendered = "null" if not math.isfinite(mean) else f"{mean:.8g}"
        lines.append(f"- {name}: {rendered}")
    return "\n".join(lines)


__all__ = ["EVALUATION_SCHEMA_VERSION", "evaluate", "render_evaluation_report"]
