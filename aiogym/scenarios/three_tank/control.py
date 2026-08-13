"""Three-Tank residual-to-physical action mapping."""
from __future__ import annotations

import math

import numpy as np

from .model import TANK3_INTERNAL_CONTROL


def resolve_residual_action(
    model,
    *,
    residual,
    state,
    reference,
    disturbances,
):
    command = np.asarray(residual, dtype=float).reshape(-1)
    if command.shape != (2,) or not np.isfinite(command).all():
        raise ValueError("residual action must contain two finite values")
    if np.any(command < -1.0) or np.any(command > 1.0):
        raise ValueError("residual action must stay within [-1, 1]")
    target = np.asarray(reference, dtype=float).reshape(-1)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise ValueError("Three-Tank reference must contain six finite values")
    action = model.tracking_steady_state_action(target, disturbances)
    if action is None:
        raise ValueError("Tank 3 residual target has no feasible steady action")
    equilibrium = {
        "action": action,
        "state": [
            target[0],
            target[3],
            target[1],
            target[4],
            target[2],
            target[5],
        ],
        "flow_m3s": _steady_flow(model, target, disturbances, action),
    }
    physical = np.asarray(action, dtype=float)
    physical[:3] = _upstream_action(model, state, equilibrium, physical[:3])
    for index, value in zip((3, 4), command):
        baseline = float(physical[index])
        physical[index] = (
            baseline + float(value) * (1.0 - baseline)
            if value >= 0
            else baseline + float(value) * baseline
        )
    return np.clip(physical, 0.0, 1.0).astype(np.float32)


def _steady_flow(model, target, disturbances, action):
    context = model.runtime_env(disturbances)
    head = max(float(target[1]) + model.parameter("gravity_drop")[1], 1e-12)
    return (
        float(action[2])
        * model.parameter("cv_valves")[1]
        * context["v23_flow_factor"]
        * math.sqrt(head)
    )


def _upstream_action(model, state, equilibrium, action):
    config = TANK3_INTERNAL_CONTROL
    values = np.asarray(state, dtype=float).reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("internal control requires a finite six-state vector")
    target = np.asarray(equilibrium["state"], dtype=float)
    tolerance = float(config["level_tolerance_m"])
    maximum = float(config["maximum_correction"])
    resolved = np.asarray(action, dtype=float).copy()
    resolved[0] += float(
        np.clip(
            float(config["pump_h1_kp"]) * (target[0] - values[0]) / tolerance,
            -maximum,
            maximum,
        )
    )
    resolved[1] += float(
        np.clip(
            float(config["v12_h2_kp"]) * (target[2] - values[2]) / tolerance,
            -maximum,
            maximum,
        )
    )
    if config["v23_nominal_flow_feedforward"]:
        head = max(values[2] + model.parameter("gravity_drop")[1], 1e-12)
        resolved[2] = equilibrium["flow_m3s"] / (
            model.parameter("cv_valves")[1] * math.sqrt(head)
        )
    return np.clip(resolved, 0.0, 1.0)


__all__ = ["resolve_residual_action"]
