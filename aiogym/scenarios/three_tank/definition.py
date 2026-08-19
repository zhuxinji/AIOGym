"""Scenario registration for the fixed BOM-backed three-tank system."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from aiogym.core.contracts import Scenario
from aiogym.core.specs import Reward
from .episodes import (
    BENCHMARKS,
    make_default_episode,
    sample_training_disturbance,
    sample_training_episode,
)
from .metrics import (
    LEVEL_OUTPUT_INDICES,
    regulation_episode_metrics,
    thermal_regulation_episode_metrics,
)
from .model import TRACKING_ERROR_SCALES, ThreeTankModel


_REGULATION_ERROR_SCALES = np.asarray(TRACKING_ERROR_SCALES, dtype=float)
_LEVEL_OUTPUT_INDICES = np.asarray(LEVEL_OUTPUT_INDICES, dtype=int)
_THERMAL_OUTPUT_INDICES = np.arange(len(_REGULATION_ERROR_SCALES), dtype=int)
_EARLY_TERMINATION_COST_RATE = 2.0


def _three_tank_tracking_reward(
    state,
    action,
    next_state,
    context,
    *,
    output_indices,
):
    del state, action
    output = np.asarray(context["model"].outputs(next_state), dtype=float)
    reference = np.asarray(context["reference"], dtype=float)
    tracking_rate = float(
        np.mean(
            (
                (output[output_indices] - reference[output_indices])
                / _REGULATION_ERROR_SCALES[output_indices]
            )
            ** 2
        )
    )
    dt = float(context["control_dt"])
    tracking_cost = dt * tracking_rate
    early_termination_cost = 0.0
    if any(float(value) > 0.0 for value in context["constraint_costs"].values()):
        remaining_steps = (
            int(context["episode"].horizon) - int(context["step_index"]) - 1
        )
        if remaining_steps < 0:
            raise ValueError("termination step exceeds the episode horizon")
        early_termination_cost = (
            _EARLY_TERMINATION_COST_RATE * remaining_steps * dt
        )
    return -tracking_cost - early_termination_cost, {
        "tracking_error": -tracking_cost,
        "early_termination": -early_termination_cost,
        "slew": 0.0,
        "effort": 0.0,
    }


def _three_tank_regulation_reward(state, action, next_state, context):
    return _three_tank_tracking_reward(
        state,
        action,
        next_state,
        context,
        output_indices=_LEVEL_OUTPUT_INDICES,
    )


def _three_tank_thermal_regulation_reward(state, action, next_state, context):
    return _three_tank_tracking_reward(
        state,
        action,
        next_state,
        context,
        output_indices=_THERMAL_OUTPUT_INDICES,
    )


def _pid_terms(
    *,
    hydraulic_kp_scale=1.0,
    hydraulic_ki_scale=0.0,
    heater_kp=None,
    heater_ki=0.0,
):
    rows = [
        ("pump_P101", "tank_1_level", 4.0),
        ("valve_V12", "tank_1_level", -3.0),
        ("valve_V12", "tank_2_level", 3.0),
        ("valve_V23", "tank_2_level", -3.0),
        ("valve_V23", "tank_3_level", 3.0),
        ("valve_V34", "tank_3_level", -3.0),
    ]
    terms = [
        {
            "actuator": actuator,
            "output": output,
            "kp": gain * hydraulic_kp_scale,
            "ki": gain * hydraulic_ki_scale,
            "kd": 0.0,
        }
        for actuator, output, gain in rows
    ]
    if heater_kp is not None:
        terms.append(
            {
                "actuator": "heater_H1",
                "output": "tank_1_temperature",
                "kp": heater_kp,
                "ki": heater_ki,
                "kd": 0.0,
            }
        )
    return terms


_LEVEL_PID = {
    "matrix_terms": _pid_terms(
        hydraulic_kp_scale=8.0,
        hydraulic_ki_scale=0.01,
    ),
    "bias": "default_action",
}

_THERMAL_PID = {
    "matrix_terms": _pid_terms(
        hydraulic_kp_scale=8.0,
        hydraulic_ki_scale=0.01,
        heater_kp=0.3,
        heater_ki=0.005,
    ),
    "bias": "default_action",
}

_LEVEL_MPC = {
    "Ts": 1.0,
    "P": 60,
    "cv_scale": list(TRACKING_ERROR_SCALES),
    "move_supp": [50.0, 50.0, 50.0, 50.0, 0.05],
    "steady_input_weight": [40.0, 40.0, 40.0, 40.0, 3.0],
    "q_y": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    "reseed_on_feedforward_change": True,
}

_THERMAL_MPC = {
    "Ts": 1.0,
    "P": 60,
    "cv_scale": list(TRACKING_ERROR_SCALES),
    "move_supp": [50.0, 50.0, 50.0, 50.0, 0.05],
    "steady_input_weight": [40.0, 40.0, 40.0, 40.0, 3.0],
    "q_y": 1.0,
    "reseed_on_feedforward_change": True,
}


def _controller_config(controller_id, reward_id):
    profiles = {
        "regulation": {"pid": _LEVEL_PID, "mpc": _LEVEL_MPC},
        "thermal_regulation": {
            "pid": _THERMAL_PID,
            "mpc": _THERMAL_MPC,
        },
    }
    try:
        return deepcopy(profiles[reward_id][controller_id])
    except KeyError as error:
        raise ValueError(
            f"three_tank has no {controller_id!r} controller for reward {reward_id!r}"
        ) from error


def _regulation_reward():
    return Reward(
        id="regulation",
        function=_three_tank_regulation_reward,
        episode_metric_function=regulation_episode_metrics,
        primary_metric="return",
        metric_direction="maximize",
        safety_violation_penalty=100.0,
    )


def _thermal_regulation_reward():
    return Reward(
        id="thermal_regulation",
        function=_three_tank_thermal_regulation_reward,
        episode_metric_function=thermal_regulation_episode_metrics,
        primary_metric="return",
        metric_direction="maximize",
        safety_violation_penalty=100.0,
    )


SCENARIO = Scenario(
    id="three_tank",
    make_model=ThreeTankModel,
    control_dt=1.0,
    make_default_episode=make_default_episode,
    sample_training_episode=sample_training_episode,
    sample_training_disturbance=sample_training_disturbance,
    benchmarks=BENCHMARKS,
    rewards={
        "regulation": _regulation_reward(),
        "thermal_regulation": _thermal_regulation_reward(),
    },
    default_reward="regulation",
    controller_config=_controller_config,
)


__all__ = ["SCENARIO"]
