"""Scenario definition for the fixed BOM-backed three-tank system."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from aiogym.core.contracts import Scenario
from aiogym.core.specs import Reward
from aiogym.scenarios._metrics import REGULATION_SUCCESS_CRITERION
from .episodes import (
    BENCHMARKS,
    make_default_episode,
    sample_training_disturbance,
    sample_training_episode,
)
from .metrics import regulation_episode_metrics
from .model import TRACKING_ERROR_SCALES, ThreeTankModel


_REGULATION_ERROR_SCALES = np.asarray(TRACKING_ERROR_SCALES, dtype=float)
_EARLY_TERMINATION_COST_RATE = 2.0
_ACTION_SLEW_COST_RATE = 1.0


def _three_tank_regulation_reward(state, action, next_state, context):
    del state
    output = np.asarray(context["model"].outputs(next_state), dtype=float)
    reference = np.asarray(context["reference"], dtype=float)
    tracking_rate = float(
        np.mean(((output - reference) / _REGULATION_ERROR_SCALES) ** 2)
    )
    dt = float(context["control_dt"])
    tracking_cost = dt * tracking_rate
    applied_action = np.asarray(action, dtype=float)
    previous_applied_action = np.asarray(
        context["previous_applied_action"], dtype=float
    )
    slew_cost = (
        dt
        * _ACTION_SLEW_COST_RATE
        * float(np.mean((applied_action - previous_applied_action) ** 2))
    )
    early_termination_cost = 0.0
    if any(float(value) > 0.0 for value in context["constraint_costs"].values()):
        remaining_steps = (
            int(context["episode"].horizon) - int(context["step_index"]) - 1
        )
        if remaining_steps < 0:
            raise ValueError("termination step exceeds the episode horizon")
        early_termination_cost = _EARLY_TERMINATION_COST_RATE * remaining_steps * dt
    return -tracking_cost - slew_cost - early_termination_cost, {
        "tracking_error": -tracking_cost,
        "early_termination": -early_termination_cost,
        "slew": -slew_cost,
        "effort": 0.0,
    }


def _pid_terms(
    *,
    hydraulic_kp_scale=1.0,
    hydraulic_ki_scale=0.0,
):
    rows = [
        ("pump_P101", "tank_1_level", 4.0),
        ("valve_V12", "tank_1_level", -3.0),
        ("valve_V12", "tank_2_level", 3.0),
        ("valve_V23", "tank_2_level", -3.0),
        ("valve_V23", "tank_3_level", 3.0),
        ("valve_V34", "tank_3_level", -3.0),
    ]
    return [
        {
            "actuator": actuator,
            "output": output,
            "kp": gain * hydraulic_kp_scale,
            "ki": gain * hydraulic_ki_scale,
            "kd": 0.0,
        }
        for actuator, output, gain in rows
    ]


_PID = {
    "matrix_terms": _pid_terms(
        hydraulic_kp_scale=8.0,
        hydraulic_ki_scale=0.02,
    ),
    "bias": "default_action",
}

_MPC = {
    "prediction_horizon": 20,
    "cv_scale": list(TRACKING_ERROR_SCALES),
    "move_supp": [5.0, 5.0, 5.0, 5.0],
    "steady_input_weight": [0.5, 0.5, 0.5, 0.5],
    "q_y": 1.0,
}


def _controller_config(controller_id, reward_id):
    if reward_id != "regulation":
        raise ValueError(f"three_tank has no controllers for reward {reward_id!r}")
    configs = {"pid": _PID, "mpc": _MPC}
    try:
        return deepcopy(configs[controller_id])
    except KeyError as error:
        raise ValueError(f"three_tank has no {controller_id!r} controller") from error


SCENARIO = Scenario(
    id="three_tank",
    make_model=ThreeTankModel,
    control_dt=1.0,
    make_default_episode=make_default_episode,
    sample_training_episode=sample_training_episode,
    sample_training_disturbance=sample_training_disturbance,
    benchmarks=BENCHMARKS,
    rewards={
        "regulation": Reward(
            id="regulation",
            success_criterion=REGULATION_SUCCESS_CRITERION,
            function=_three_tank_regulation_reward,
            episode_metric_function=regulation_episode_metrics,
            primary_metric="return",
            metric_direction="maximize",
            safety_violation_penalty=100.0,
        ),
    },
    default_reward="regulation",
    controller_config=_controller_config,
)


__all__ = ["SCENARIO"]
