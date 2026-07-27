"""Episode evaluation execution."""
from __future__ import annotations

from time import perf_counter
from typing import Sequence

import numpy as np

from ..._internal.serialization import jsonable as _jsonable
from ...controllers import as_controller, build_context, validate_action
from ..results import _aggregate_metric_keys, evaluate_task_acceptance, result_schema
from .metadata import (
    _aggregate_controller_diagnostics,
    _controller_diagnostics,
    _controller_diagnostic_totals,
    _env_metadata,
    _env_objective,
    _model_metadata,
    _reproducibility_metadata,
)
from ..metrics.economic import economic_step_metrics
from ..metrics.safety import action_bound_metrics as _action_bound_metrics
from ..metrics.safety import safety_step_metrics as _safety_step_metrics
from ..metrics.tracking import tracking_step_metrics as _tracking_step_metrics
from ..metric_catalog import (
    EVALUATION_SCHEMA_VERSION,
    metric_definitions,
    metric_direction,
    primary_metric_for_objective,
)
from ..objective_specs import ObjectiveSpec, objective_spec
from ..protocols import BenchmarkProtocol, _empty_episode_totals


def evaluate_controller(agent, env, episodes: int = 20, seed: int = 0,
                        include_episodes: bool = False, protocol: BenchmarkProtocol | None = None,
                        seed_list: Sequence[int] | None = None,
                        objective_specification: ObjectiveSpec | str | None = None):
    """Evaluate any supported controller/policy on an AIOGymNativeEnv.

    Returns aggregate metrics plus the protocol/controller metadata needed to
    reproduce the benchmark.
    """

    if seed_list is None:
        if episodes <= 0:
            raise ValueError("episodes must be positive when seed_list is not provided")
        seeds = [int(seed) + ep for ep in range(episodes)]
    else:
        seeds = [int(value) for value in seed_list]
        if not seeds:
            raise ValueError("seed_list must contain at least one seed")
    controller = as_controller(agent, action_mode=getattr(env, "action_mode", "actuator"))
    if protocol is not None:
        resolved_objective = protocol.resolved_objective()
        if (
            objective_specification is not None
            and getattr(objective_specification, "name", objective_specification)
            != resolved_objective.name
        ):
            raise ValueError("objective specification does not match protocol objective")
    elif isinstance(objective_specification, ObjectiveSpec):
        resolved_objective = objective_specification
    elif objective_specification is not None:
        resolved_objective = objective_spec(
            str(objective_specification), source="explicit"
        )
    else:
        resolved_objective = objective_spec(
            _env_objective(env), source="environment-reward-mode"
        )
    objective = resolved_objective.name
    per_episode = []
    episode_schedules = []
    eval_start = perf_counter()
    for ep, ep_seed in enumerate(seeds):
        episode_start = perf_counter()
        obs, reset_info = env.reset(seed=ep_seed)
        controller.reset(seed=ep_seed)
        totals = _empty_episode_totals(ep, ep_seed)
        episode_schedules.append(_jsonable({
            "setpoints": getattr(env, "_episode_setpoint_events", {}),
            "task": getattr(env, "_task_disturbance_events", {}),
            "auto_events": getattr(env, "_dist_events", []),
        }))
        done = False
        steps = 0
        info = reset_info or {}
        last_unsettled_time = 0.0
        raw_last_unsettled_time = {}
        while not done:
            context = build_context(env, info)
            action = validate_action(controller.act(obs, context), env, controller.name)
            bound_metrics = _action_bound_metrics(action, env)
            obs, reward, term, trunc, info = env.step(action)
            time_sec = steps * float(env.control_dt)
            # ``env.y_sp`` may already contain the reference staged for the next
            # control step. Metrics for this completed transition must use the
            # reference recorded by its stage objective.
            active_setpoint = {
                "y_sp": list(info.get("y_sp", getattr(env, "y_sp", [])))
            }
            tracking = _tracking_step_metrics(info, active_setpoint, time_sec, float(env.control_dt), env)
            safety = _safety_step_metrics(info, bound_metrics, float(env.control_dt))
            totals["return"] += float(reward)
            totals["track"] += float(info.get("track", 0.0))
            totals["constraint"] += float(info.get("constraint", 0.0))
            economic = economic_step_metrics(info, float(env.control_dt))
            for key, value in economic.items():
                totals[key] += value
            for key in (
                "tracking_cost", "tracking_return", "tracking_error_cost", "tracking_move_cost",
                "tracking_mse", "tracking_iae", "tracking_ise", "tracking_itae",
            ):
                totals[key] += tracking[key]
            totals["tracking_overshoot"] = max(totals["tracking_overshoot"], tracking["tracking_overshoot"])
            if not tracking["tracking_settled"]:
                last_unsettled_time = time_sec
            _accumulate_raw_tracking_metrics(
                totals["tracking_raw_by_output"],
                tracking["tracking_raw_by_output"],
                raw_last_unsettled_time,
                time_sec,
            )
            for key, value in safety.items():
                if key == "safety_margin_min":
                    totals[key] = min(totals[key], value)
                else:
                    totals[key] += value
            done = bool(term or trunc)
            steps += 1
        rep = env.scorer.report()
        runtime_seconds = perf_counter() - episode_start
        totals["normalized_score"] = float(rep["score"])
        totals["steps"] = steps
        totals["runtime_seconds"] = float(runtime_seconds)
        totals["runtime_seconds_per_step"] = float(runtime_seconds / steps) if steps else 0.0
        horizon_seconds = steps * float(env.control_dt)
        totals["tracking_mse"] = float(totals["tracking_mse"] / horizon_seconds) if horizon_seconds > 0 else 0.0
        totals["tracking_settling_time"] = float(last_unsettled_time)
        totals["tracking_raw_by_output"] = _finalize_raw_tracking_metrics(
            totals["tracking_raw_by_output"],
            raw_last_unsettled_time,
            horizon_seconds,
        )
        controller_diag = _controller_diagnostics(controller)
        totals["controller_diagnostics"] = controller_diag
        totals.update(_controller_diagnostic_totals(controller_diag))
        per_episode.append(totals)
    runtime_total_seconds = perf_counter() - eval_start

    def mean(key):
        return float(np.mean([row[key] for row in per_episode]))

    def std(key):
        return float(np.std([row[key] for row in per_episode]))

    if protocol is not None:
        task_meta = protocol.metadata()["task_identity"]
    else:
        from ...models.tasks import task_identity

        task_meta = task_identity(getattr(env, "task_profile", None))
    primary_metric = primary_metric_for_objective(objective)
    aggregate_keys = _aggregate_metric_keys(per_episode)
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "controller_name": controller.name,
        "metric": primary_metric,
        "metric_direction": metric_direction(primary_metric),
        "objective": objective,
        "objective_source": resolved_objective.source,
        "objective_spec": resolved_objective.metadata(),
        "task": task_meta["name"],
        "task_status": task_meta["status"],
        "task_profile_hash": task_meta["profile_hash"],
        "episodes": len(seeds),
        "seed": int(seeds[0]) if seeds else int(seed),
        "seed_list": [int(s) for s in seeds],
        "normalized_score": mean("normalized_score"),
        "normalized_score_std": std("normalized_score"),
        "profit": mean("profit"),
        "profit_std": std("profit"),
        "return": mean("return"),
        "return_std": std("return"),
        "track": mean("track"),
        "track_std": std("track"),
        "constraint": mean("constraint"),
        "constraint_std": std("constraint"),
        "production": mean("production"),
        "production_std": std("production"),
        "runtime_total_seconds": float(runtime_total_seconds),
        "protocol": (
            protocol.metadata()
            if protocol is not None
            else {
                "environment": _env_metadata(env),
                "objective_spec": resolved_objective.metadata(),
            }
        ),
        "controller": controller.metadata(),
        "model": _model_metadata(env),
        "disturbance": {
            "schedule_source": "task_and_model_schema",
            "episode_schedules": episode_schedules,
        },
        "controller_diagnostics": _aggregate_controller_diagnostics(per_episode),
        "metric_definitions": metric_definitions(objective),
        "result_schema": result_schema(),
        "reproducibility": _reproducibility_metadata(
            env, seeds, protocol, resolved_objective
        ),
    }
    result["controller_status"] = "degraded" if result["controller_diagnostics"].get("degraded") else "ok"
    for key in aggregate_keys:
        result.setdefault(key, mean(key))
        result.setdefault(f"{key}_std", std(key))
    (
        result["tracking_raw_by_output"],
        result["tracking_raw_by_output_std"],
    ) = _aggregate_raw_tracking_metrics(per_episode)
    result["execution_status"] = (
        "degraded" if result["controller_status"] == "degraded" else "passed"
    )
    acceptance = evaluate_task_acceptance(
        protocol.task if protocol is not None else getattr(env, "task_profile", None),
        result,
    )
    result["objective_status"] = acceptance["status"]
    result["objective_acceptance"] = acceptance
    if include_episodes:
        result["episode_metrics"] = per_episode
    return result


def _accumulate_raw_tracking_metrics(
    totals,
    step_metrics,
    last_unsettled_time,
    time_sec,
):
    for name, row in step_metrics.items():
        target = totals.setdefault(
            name,
            {
                "unit": row.get("unit", ""),
                "mse_integral": 0.0,
                "iae": 0.0,
                "ise": 0.0,
                "itae": 0.0,
                "overshoot": 0.0,
            },
        )
        for key in ("mse_integral", "iae", "ise", "itae"):
            target[key] += float(row[key])
        target["overshoot"] = max(
            float(target["overshoot"]),
            float(row["overshoot"]),
        )
        if not row["settled"]:
            last_unsettled_time[name] = float(time_sec)


def _finalize_raw_tracking_metrics(totals, last_unsettled_time, horizon_seconds):
    finalized = {}
    for name, row in totals.items():
        finalized[name] = {
            "unit": row.get("unit", ""),
            "mse": (
                float(row["mse_integral"]) / float(horizon_seconds)
                if horizon_seconds > 0
                else 0.0
            ),
            "iae": float(row["iae"]),
            "ise": float(row["ise"]),
            "itae": float(row["itae"]),
            "overshoot": float(row["overshoot"]),
            "settling_time": float(last_unsettled_time.get(name, 0.0)),
        }
    return finalized


def _aggregate_raw_tracking_metrics(per_episode):
    names = sorted({
        name
        for episode in per_episode
        for name in episode.get("tracking_raw_by_output", {})
    })
    aggregated = {}
    deviations = {}
    metric_keys = ("mse", "iae", "ise", "itae", "overshoot", "settling_time")
    for name in names:
        rows = [
            episode["tracking_raw_by_output"][name]
            for episode in per_episode
            if name in episode.get("tracking_raw_by_output", {})
        ]
        unit = str(rows[0].get("unit", "")) if rows else ""
        aggregated[name] = {"unit": unit}
        deviations[name] = {"unit": unit}
        for key in metric_keys:
            values = [float(row[key]) for row in rows]
            aggregated[name][key] = float(np.mean(values)) if values else 0.0
            deviations[name][key] = float(np.std(values)) if values else 0.0
    return aggregated, deviations
