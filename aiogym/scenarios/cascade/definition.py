"""Scenario definition for the heated three-tank Cascade."""

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
from .model import CascadeModel, TRACKING_ERROR_SCALES


_ERROR_SCALES = np.asarray(TRACKING_ERROR_SCALES, dtype=float)
_EARLY_TERMINATION_COST_RATE = 2.0


def _cascade_regulation_reward(state, action, next_state, context):
    del state, action
    output = np.asarray(context["model"].outputs(next_state), dtype=float)
    reference = np.asarray(context["reference"], dtype=float)
    tracking_rate = float(np.mean(((output - reference) / _ERROR_SCALES) ** 2))
    dt = float(context["control_dt"])
    tracking_cost = dt * tracking_rate
    early_termination_cost = 0.0
    if any(float(value) > 0.0 for value in context["constraint_costs"].values()):
        remaining_steps = (
            int(context["episode"].horizon) - int(context["step_index"]) - 1
        )
        if remaining_steps < 0:
            raise ValueError("termination step exceeds the episode horizon")
        early_termination_cost = _EARLY_TERMINATION_COST_RATE * remaining_steps * dt
    return -tracking_cost - early_termination_cost, {
        "tracking_error": -tracking_cost,
        "early_termination": -early_termination_cost,
        "slew": 0.0,
        "effort": 0.0,
    }


_PID = {
    "matrix_terms": [
        {
            "actuator": "valve_V12",
            "output": "tank_1_level",
            "kp": -48.0,
            "ki": -0.12,
            "kd": 0.0,
        },
        {
            "actuator": "valve_V12",
            "output": "tank_2_level",
            "kp": 48.0,
            "ki": 0.12,
            "kd": 0.0,
        },
        {
            "actuator": "valve_V23",
            "output": "tank_2_level",
            "kp": -48.0,
            "ki": -0.12,
            "kd": 0.0,
        },
        {
            "actuator": "valve_V23",
            "output": "tank_3_level",
            "kp": 48.0,
            "ki": 0.12,
            "kd": 0.0,
        },
        {
            "actuator": "valve_V34",
            "output": "tank_3_level",
            "kp": -48.0,
            "ki": -0.12,
            "kd": 0.0,
        },
        {
            "actuator": "heater_H1",
            "output": "tank_1_temperature",
            "kp": 0.75,
            "ki": 0.0075,
            "kd": 0.0,
        },
        {
            "actuator": "heater_H1",
            "output": "tank_2_temperature",
            "kp": 0.15,
            "ki": 0.0015,
            "kd": 0.0,
        },
        {
            "actuator": "heater_H1",
            "output": "tank_3_temperature",
            "kp": 0.10,
            "ki": 0.0010,
            "kd": 0.0,
        },
        {
            "actuator": "heater_H2",
            "output": "tank_2_temperature",
            "kp": 1.0,
            "ki": 0.01,
            "kd": 0.0,
        },
        {
            "actuator": "heater_H3",
            "output": "tank_3_temperature",
            "kp": 1.0,
            "ki": 0.01,
            "kd": 0.0,
        },
    ],
    "bias": "default_action",
    "tracking_steady_state_bias": True,
}

_MPC = {
    "prediction_horizon": 20,
    "solve_every": 2,
    "cv_scale": list(TRACKING_ERROR_SCALES),
    "move_supp": [25.0, 25.0, 25.0, 25.0, 2.0, 2.0, 2.0],
    "steady_input_weight": [0.5] * 7,
    "q_y": [3.0, 3.0, 3.0, 2.0, 2.5, 4.0],
}


def _controller_config(controller_id, reward_id):
    if reward_id != "regulation":
        raise ValueError(f"cascade has no controllers for reward {reward_id!r}")
    configs = {"pid": _PID, "mpc": _MPC}
    try:
        return deepcopy(configs[controller_id])
    except KeyError as error:
        raise ValueError(f"cascade has no {controller_id!r} controller") from error


SCENARIO = Scenario(
    id="cascade",
    make_model=CascadeModel,
    control_dt=2.0,
    make_default_episode=make_default_episode,
    sample_training_episode=sample_training_episode,
    sample_training_disturbance=sample_training_disturbance,
    benchmarks=BENCHMARKS,
    rewards={
        "regulation": Reward(
            id="regulation",
            success_criterion=REGULATION_SUCCESS_CRITERION,
            function=_cascade_regulation_reward,
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
