"""Evaluation environment, controller, and reproducibility metadata."""
from __future__ import annotations

import os
import subprocess

import numpy as np

from ..._internal.serialization import jsonable as _jsonable
from ..metric_catalog import EVALUATION_SCHEMA_VERSION


def _env_metadata(env):
    keys = (
        "scenario", "goal", "reward_spec_id", "action_mode",
        "control_dt", "episode_steps",
        "auto_events", "randomize", "randomize_setpoints", "randomize_plant",
        "plant_drift", "integral_obs", "terminate_on_runaway",
        "noise", "noise_pct", "observation_mode",
        "_active_sensor_model", "_active_actuator_model",
    )
    metadata = {
        key.lstrip("_"): getattr(env, key)
        for key in keys
        if hasattr(env, key)
    }
    observation_contract = getattr(env, "observation_contract", None)
    if observation_contract is not None:
        metadata["observation_contract"] = observation_contract.as_dict()
    shield = getattr(env, "shield", None)
    if shield is not None:
        metadata["safety_shield"] = shield.metadata()
    return _jsonable(metadata)

def _controller_diagnostics(controller):
    targets = [controller, getattr(controller, "agent", None), getattr(controller, "policy", None)]
    for target in targets:
        if target is None or not hasattr(target, "diagnostics"):
            continue
        return _jsonable(target.diagnostics())
    return {}

def _controller_diagnostic_totals(diagnostics):
    data = dict(diagnostics or {})
    degraded = bool(
        data.get("degraded")
        or float(data.get("solver_limited_count", 0.0) or 0.0) > 0.0
        or float(data.get("solver_failure_count", 0.0) or 0.0) > 0.0
        or float(data.get("fallback_count", 0.0) or 0.0) > 0.0
    )
    return {
        "controller_solve_count": float(data.get("solve_count", 0.0) or 0.0),
        "controller_solver_success_count": float(data.get("solver_success_count", 0.0) or 0.0),
        "controller_solver_limited_count": float(data.get("solver_limited_count", 0.0) or 0.0),
        "controller_solver_failure_count": float(data.get("solver_failure_count", 0.0) or 0.0),
        "controller_fallback_count": float(data.get("fallback_count", 0.0) or 0.0),
        "controller_degraded_count": 1.0 if degraded else 0.0,
    }

def _aggregate_controller_diagnostics(per_episode):
    rows = [dict(row.get("controller_diagnostics") or {}) for row in per_episode]
    totals = {
        "solve_count": 0.0,
        "solver_success_count": 0.0,
        "solver_limited_count": 0.0,
        "solver_failure_count": 0.0,
        "fallback_count": 0.0,
    }
    degraded = False
    last_error = None
    for row in rows:
        for key in totals:
            value = row.get(key, 0.0)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float, np.number)):
                totals[key] += float(value)
        row_degraded = bool(
            row.get("degraded")
            or float(row.get("solver_limited_count", 0.0) or 0.0) > 0.0
            or float(row.get("solver_failure_count", 0.0) or 0.0) > 0.0
            or float(row.get("fallback_count", 0.0) or 0.0) > 0.0
        )
        degraded = degraded or row_degraded
        if row.get("last_solver_error"):
            last_error = row.get("last_solver_error")
    out = {
        "degraded": degraded,
        "last_solver_error": last_error,
        "per_episode": rows,
    }
    out.update({key: int(value) if value.is_integer() else value for key, value in totals.items()})
    return _jsonable(out)

def _model_metadata(env):
    model = getattr(env, "model", None)
    if model is None or not hasattr(model, "metadata"):
        return {}
    return _jsonable(model.metadata())

def _env_disturbances(env):
    if hasattr(env, "_env"):
        return _jsonable(env._env())
    return {}

def _reproducibility_metadata(env, seeds, resolved_goal):
    return {
        "git_commit": _git_commit(),
        "seed_list": [int(seed) for seed in seeds],
        "model_version": getattr(getattr(env, "model", None), "scenario", None),
        "episode_length": int(getattr(env, "episode_steps", 0)),
        "disturbance_schedule": "case_and_model_schema",
        "metric_definition_version": EVALUATION_SCHEMA_VERSION,
        "environment": _env_metadata(env),
        "goal_spec": resolved_goal.metadata(),
    }

def _git_commit():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = out.stdout.strip()
    return commit or None
