"""Episode evaluation execution."""
from __future__ import annotations

import copy
from time import perf_counter
from typing import Sequence

import numpy as np

from ..._internal.serialization import jsonable as _jsonable
from ...controllers.adapters import as_controller
from ...controllers.contracts import build_context, validate_action
from ..results import _aggregate_metric_keys, evaluate_case_acceptance, result_schema
from .metadata import (
    _aggregate_controller_diagnostics,
    _controller_diagnostics,
    _controller_diagnostic_totals,
    _env_metadata,
    _model_metadata,
    _reproducibility_metadata,
)
from ..metrics.economic import economic_step_metrics
from ..metrics.safety import SafetyDebtTracker
from ..metrics.safety import action_bound_metrics as _action_bound_metrics
from ..metrics.safety import safety_step_metrics as _safety_step_metrics
from ..metrics.service import service_step_metrics
from ..metrics.tracking import tracking_step_metrics as _tracking_step_metrics
from ..metric_catalog import (
    EVALUATION_SCHEMA_VERSION,
    metric_definitions,
    metric_direction,
)
from ..goal_specs import GoalSpec, goal_spec, resolve_goal
from ..scorecard import (
    ScorecardAccumulator,
    flatten_scorecard,
    group_scorecard,
)
from ..safety_gate import SafetyGateSpec, apply_safety_gate
from .rollouts import _rollout_payload, _rollout_step
from .totals import empty_episode_totals


def evaluate_controller(agent, env, episodes: int = 1, seed: int = 0,
                        include_episodes: bool = False,
                        seed_list: Sequence[int] | None = None,
                        *, goal_specification: GoalSpec | str | None = None,
                        safety_mode: str | None = None,
                        safety_gate_spec: SafetyGateSpec | None = None,
                        initial_safety_debt: bool | None = None,
                        seed_namespace: str | None = None,
                        worker_index: int = 0,
                        episode_specs: Sequence | None = None,
                        _rollout_capture: dict | None = None,
                        rollout_steps: int | None = None):
    """Evaluate any supported controller/policy on an environment from ``make_env``.

    Returns aggregate metrics plus the goal/controller metadata needed to
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
    resolved_episode_specs = (
        None if episode_specs is None else tuple(episode_specs)
    )
    if (
        resolved_episode_specs is not None
        and len(resolved_episode_specs) != len(seeds)
    ):
        raise ValueError(
            "episode_specs length must match the evaluation seed count"
        )
    controller = as_controller(agent, action_mode=getattr(env, "action_mode", "actuator"))
    if goal_specification is not None:
        resolved_goal = resolve_goal(
            explicit=goal_specification,
            source="explicit",
        )
    else:
        resolved_goal = goal_spec(
            str(getattr(env, "goal", "regulation")),
            source="environment-reward-spec",
        )
    case_profile = getattr(env, "case_profile", None) or {}
    if str(getattr(env, "goal", resolved_goal.name)) != resolved_goal.name:
        raise ValueError("goal specification does not match environment RewardSpec")
    evaluation_meta = dict(case_profile.get("evaluation", {}))
    if safety_gate_spec is not None and not isinstance(
        safety_gate_spec,
        SafetyGateSpec,
    ):
        raise TypeError("safety_gate_spec must be a SafetyGateSpec")
    declared_safety_mode = (
        safety_gate_spec.mode
        if safety_gate_spec is not None
        else safety_mode
    )
    if (
        safety_gate_spec is not None
        and safety_mode is not None
        and safety_gate_spec.mode != safety_mode
    ):
        raise ValueError(
            "safety_mode conflicts with safety_gate_spec.mode"
        )
    resolved_safety_mode = str(
        declared_safety_mode
        if declared_safety_mode is not None
        else evaluation_meta.get(
            "safety_mode",
            "recovery"
            if case_profile.get("name") == "safety-recovery"
            else "ordinary",
        )
    )
    resolved_initial_safety_debt = bool(
        initial_safety_debt
        if initial_safety_debt is not None
        else evaluation_meta.get(
            "initial_safety_debt",
            resolved_safety_mode == "recovery",
        )
    )
    resolved_safety_gate_spec = (
        safety_gate_spec
        if safety_gate_spec is not None
        else SafetyGateSpec(mode=resolved_safety_mode)
    )
    per_episode = []
    episode_schedules = []
    episode_seed_bundles = []
    eval_start = perf_counter()
    for ep, ep_seed in enumerate(seeds):
        episode_start = perf_counter()
        if resolved_episode_specs is not None:
            episode_spec = resolved_episode_specs[ep]
            if int(episode_spec.base_seed) != int(ep_seed):
                raise ValueError(
                    "evaluation EpisodeSpec base seed does not match seed_list"
                )
            obs, reset_info = env.reset(
                seed=ep_seed,
                options={"episode_spec": episode_spec},
            )
            episode_seed_bundles.append(
                {
                    "base_seed": int(ep_seed),
                    "namespace": seed_namespace,
                    **dict(episode_spec.component_seeds),
                    "episode_spec_id": episode_spec.episode_spec_id,
                    "episode_spec_hash": episode_spec.resolved_hash,
                }
            )
        elif seed_namespace is not None:
            from aiogym.generation.seed_tree import SeedTree

            seed_tree = SeedTree(
                ep_seed,
                seed_namespace,
                worker_index=worker_index,
            )
            component_seeds = seed_tree.component_seeds
            seed_bundle = {
                "base_seed": int(ep_seed),
                "namespace": seed_namespace,
                "namespace_hash": seed_tree.namespace_hash,
                "worker_index": int(worker_index),
                "episode_index": 0,
                "initial": component_seeds["initial_state"],
                "reference": component_seeds["reference"],
                "disturbance": component_seeds["disturbance"],
                "noise": component_seeds["sensor"],
                "plant": component_seeds["plant"],
            }
            obs, reset_info = env.reset(
                seed=seed_bundle["initial"],
                options={"seed_bundle": seed_bundle},
            )
            episode_seed_bundles.append(seed_bundle)
        else:
            obs, reset_info = env.reset(seed=ep_seed)
        controller.reset(seed=ep_seed)
        totals = empty_episode_totals(ep, ep_seed)
        scorecard_accumulator = ScorecardAccumulator()
        safety_debt_tracker = SafetyDebtTracker(
            initial_safety_debt=resolved_initial_safety_debt
        )
        episode_schedules.append(_jsonable({
            "setpoints": getattr(env, "_episode_setpoint_events", {}),
            "case": getattr(env, "_case_disturbance_events", {}),
            "auto_events": getattr(env, "_dist_events", []),
        }))
        done = False
        steps = 0
        info = reset_info or {}
        last_unsettled_time = 0.0
        raw_last_unsettled_time = {}
        capture_rollout = _rollout_capture is not None and ep == 0
        capture_rows = []
        capture_limit = (
            rollout_steps
            if rollout_steps is not None
            else getattr(env, "episode_steps", None)
        )
        captured_payload = None
        if capture_rollout and capture_limit is not None and capture_limit <= 0:
            captured_payload = _rollout_payload(
                controller,
                env,
                seed=ep_seed,
                rows=capture_rows,
            )
        while not done:
            context = build_context(env, info)
            action = validate_action(controller.act(obs, context), env, controller.name)
            state = list(getattr(env.integ, "x", []))
            previous_action = copy.deepcopy(getattr(env, "previous_act", action))
            disturbance = copy.deepcopy(env._env())
            bound_metrics = _action_bound_metrics(action, env)
            obs_next, reward, term, trunc, info_next = env.step(action)
            if (
                capture_rollout
                and captured_payload is None
                and (capture_limit is None or steps < capture_limit)
            ):
                capture_rows.append(_rollout_step(
                    step=steps,
                    env=env,
                    obs=obs,
                    state=state,
                    action=action,
                    context=context,
                    obs_next=obs_next,
                    reward=reward,
                    term=term,
                    trunc=trunc,
                    info_next=info_next,
                ))
                if (
                    capture_limit is not None
                    and len(capture_rows) >= capture_limit
                ):
                    captured_payload = _rollout_payload(
                        controller,
                        env,
                        seed=ep_seed,
                        rows=capture_rows,
                    )
            time_sec = steps * float(env.control_dt)
            # ``env.y_sp`` may already contain the reference staged for the next
            # control step. Metrics for this completed transition must use the
            # reference recorded for the completed transition.
            active_setpoint = {
                "y_sp": list(info_next.get("y_sp", getattr(env, "y_sp", [])))
            }
            tracking = _tracking_step_metrics(
                info_next,
                active_setpoint,
                time_sec,
                float(env.control_dt),
                env,
            )
            safety = _safety_step_metrics(
                info_next,
                bound_metrics,
                float(env.control_dt),
            )
            safety = safety_debt_tracker.observe(safety)
            totals["return"] += float(reward)
            totals["track"] += float(info_next.get("track", 0.0))
            totals["constraint"] += float(info_next.get("constraint", 0.0))
            economic = economic_step_metrics(
                info_next,
                float(env.control_dt),
                env=env,
                state=list(getattr(env.integ, "x", [])),
                action=action,
                disturbance=disturbance,
            )
            economic.update(
                service_step_metrics(info_next, float(env.control_dt))
            )
            for key, value in economic.items():
                totals[key] = totals.get(key, 0.0) + value
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
                    totals[key] = totals.get(key, 0.0) + value
            scorecard_accumulator.accumulate(
                tracking=tracking,
                economic=economic,
                safety=safety,
                costs=info_next.get("costs", {}),
                model=env.model,
                action=action,
                previous_action=previous_action,
                dt=float(env.control_dt),
            )
            obs = obs_next
            info = info_next
            done = bool(term or trunc)
            steps += 1
        if capture_rollout:
            if captured_payload is None:
                captured_payload = _rollout_payload(
                    controller,
                    env,
                    seed=ep_seed,
                    rows=capture_rows,
                )
            _rollout_capture.clear()
            _rollout_capture.update(captured_payload)
        runtime_seconds = perf_counter() - episode_start
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
        controller_totals = _controller_diagnostic_totals(controller_diag)
        totals.update(controller_totals)
        episode_scorecard = scorecard_accumulator.finalize(
            horizon_seconds=horizon_seconds,
            settling_time=last_unsettled_time,
            tracking_raw_by_output=totals["tracking_raw_by_output"],
            controller_metrics={
                **controller_totals,
                "runtime_seconds": totals["runtime_seconds"],
                "runtime_seconds_per_step": totals[
                    "runtime_seconds_per_step"
                ],
                "runtime_total_seconds": totals["runtime_seconds"],
            },
        )
        for key, value in flatten_scorecard(episode_scorecard).items():
            if isinstance(value, (int, float, np.number)):
                totals[key] = float(value)
        totals["scorecard"] = episode_scorecard
        totals["reward_spec_id"] = str(
            getattr(env, "reward_spec_id", "unknown")
        )
        per_episode.append(totals)
    runtime_total_seconds = perf_counter() - eval_start

    def mean(key):
        return float(np.mean([row[key] for row in per_episode]))

    def std(key):
        return float(np.std([row[key] for row in per_episode]))

    from ...models.cases import case_identity

    case_meta = case_identity(getattr(env, "case_profile", None))
    primary_metric = resolved_goal.primary_metric
    aggregate_keys = _aggregate_metric_keys(per_episode)
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "controller_name": controller.name,
        "metric": primary_metric,
        "metric_direction": metric_direction(primary_metric),
        "goal": resolved_goal.name,
        "goal_source": resolved_goal.source,
        "goal_spec": resolved_goal.metadata(),
        "safety_mode": resolved_safety_mode,
        "initial_safety_debt": resolved_initial_safety_debt,
        "case": case_meta["name"],
        "case_status": case_meta["status"],
        "case_profile_hash": case_meta["profile_hash"],
        "episodes": len(seeds),
        "seed": int(seeds[0]) if seeds else int(seed),
        "seed_list": [int(s) for s in seeds],
        "seed_namespace": seed_namespace,
        "seed_bundles": episode_seed_bundles,
        "profit": mean("profit"),
        "profit_std": std("profit"),
        "return": mean("return"),
        "return_std": std("return"),
        "reward_spec_id": str(getattr(env, "reward_spec_id", "unknown")),
        "return_metadata": {
            "reward_spec_id": str(
                getattr(env, "reward_spec_id", "unknown")
            ),
            "comparable_across_reward_specs": False,
            "description": "sum of the environment training reward",
        },
        "track": mean("track"),
        "track_std": std("track"),
        "constraint": mean("constraint"),
        "constraint_std": std("constraint"),
        "production": mean("production"),
        "production_std": std("production"),
        "runtime_total_seconds": float(runtime_total_seconds),
        "environment": _env_metadata(env),
        "controller": controller.metadata(),
        "model": _model_metadata(env),
        "disturbance": {
            "schedule_source": "case_and_model_schema",
            "episode_schedules": episode_schedules,
        },
        "controller_diagnostics": _aggregate_controller_diagnostics(per_episode),
        "metric_definitions": metric_definitions(),
        "result_schema": result_schema(),
        "reproducibility": _reproducibility_metadata(
            env, seeds, resolved_goal
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
    result["scorecard"] = group_scorecard(result)
    result = apply_safety_gate(result, resolved_safety_gate_spec)
    result["execution_status"] = (
        "degraded" if result["controller_status"] == "degraded" else "passed"
    )
    acceptance = evaluate_case_acceptance(
        getattr(env, "case_profile", None),
        result,
    )
    result["acceptance_status"] = acceptance["status"]
    result["acceptance"] = acceptance
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
