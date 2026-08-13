"""Scenario registration for the fixed BOM-backed three-tank system."""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from aiogym.core.contracts import Scenario
from aiogym.core.specs import Reward
from aiogym.scenarios._shared import economic_reward
from aiogym.scenarios._metrics import (
    TANK3_OUTPUT_INDICES,
    TANK3_TRACKING_TOLERANCES,
    TANK3_TRACKING_WEIGHTS,
    economic_episode_metrics,
    regulation_episode_metrics,
    tank3_regulation_episode_metrics,
)

from .episodes import BENCHMARKS, make_default_episode, sample_training_episode
from .model import ThreeTankModel


_TANK3_MOVE_WEIGHTS = (0.0, 0.0, 0.0, 0.01, 0.01)
_REGULATION_ERROR_SCALES = np.asarray((0.4, 0.4, 0.4, 10.0, 10.0, 10.0))
_REGULATION_FEEDFORWARD_WEIGHT = 0.1


def _three_tank_regulation_reward(state, action, next_state, context):
    del state
    output = np.asarray(context["model"].outputs(next_state), dtype=float)
    reference = np.asarray(context["reference"], dtype=float)
    tracking_rate = float(
        np.mean(((output - reference) / _REGULATION_ERROR_SCALES) ** 2)
    )
    target_action = context["model"].tracking_steady_state_action(
        reference,
        context["disturbances"],
    )
    if target_action is None:
        feedforward_rate = 0.0
    else:
        action_error = np.asarray(action, dtype=float) - np.asarray(
            target_action, dtype=float
        )
        feedforward_rate = _REGULATION_FEEDFORWARD_WEIGHT * float(
            np.sum(action_error**2)
        )
    dt = float(context["control_dt"])
    tracking_cost = dt * tracking_rate
    feedforward_cost = dt * feedforward_rate
    return -(tracking_cost + feedforward_cost), {
        "tracking_error": -tracking_cost,
        "feedforward": -feedforward_cost,
        "slew": 0.0,
        "effort": 0.0,
    }


def _tank3_regulation_reward(state, action, next_state, context):
    del state
    output = np.asarray(context["model"].outputs(next_state), dtype=float)
    reference = np.asarray(context["reference"], dtype=float)
    indices = np.asarray(TANK3_OUTPUT_INDICES, dtype=int)
    tolerances = np.asarray(TANK3_TRACKING_TOLERANCES, dtype=float)
    weights = np.asarray(TANK3_TRACKING_WEIGHTS, dtype=float)
    error = (output[indices] - reference[indices]) / tolerances
    tracking = weights * error**2
    previous = np.asarray(context["previous_applied_action"], dtype=float)
    move_weights = np.asarray(_TANK3_MOVE_WEIGHTS, dtype=float)
    move = move_weights * (np.asarray(action, dtype=float) - previous) ** 2
    dt = float(context["control_dt"])
    level_cost = dt * float(tracking[0])
    temperature_cost = dt * float(tracking[1])
    move_cost = dt * float(np.sum(move))
    total = level_cost + temperature_cost + move_cost
    return -total, {
        "tank3_level_tracking": -level_cost,
        "tank3_temperature_tracking": -temperature_cost,
        "move": -move_cost,
    }


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


_PID_CONFIGS = {
    "regulation": {
        "matrix_terms": _pid_terms(
            hydraulic_kp_scale=8.0,
            hydraulic_ki_scale=0.0005,
            heater_kp=0.15,
            heater_ki=0.001,
        ),
        "bias": "default_action",
        "feedforward": "tracking_steady_state_action",
    },
    "tank3-regulation": {
        "matrix_terms": _pid_terms(
            hydraulic_kp_scale=2.0,
            hydraulic_ki_scale=0.02,
            heater_kp=0.12,
            heater_ki=0.001,
        ),
        "bias": "default_action",
        "feedforward": "tracking_steady_state_action",
    },
}

_MPC_CONFIGS = {
    "regulation": {
        "Ts": 1.0,
        "P": 120,
        "move_supp": [10.0, 10.0, 10.0, 10.0, 0.01],
        "steady_input_weight": [0.001, 0.001, 0.001, 0.001, 0.015],
        "q_y": 1.0,
        "reseed_on_feedforward_change": True,
    },
    "tank3-regulation": {
        "Ts": 2.0,
        "P": 60,
        "move_supp": [100000.0, 100000.0, 100000.0, 0.001, 0.005],
        "steady_input_weight": [0.0, 0.0, 0.0, 0.01, 0.01],
        "cv_scale": [0.4, 0.4, 0.01, 100.0, 100.0, 0.5],
        "q_y": [0.0, 0.0, 2.0, 0.0, 0.0, 1.0],
        "reseed_on_feedforward_change": True,
    },
}


def _controller_config(controller_id, reward_id):
    config_id = "tank3-regulation" if reward_id == "tank3-regulation" else "regulation"
    if controller_id == "pid":
        return deepcopy(_PID_CONFIGS[config_id])
    if controller_id == "mpc":
        return deepcopy(_MPC_CONFIGS[config_id])
    raise ValueError(f"three_tank has no controller {controller_id!r}")


def _regulation_reward():
    return Reward(
        id="regulation",
        function=_three_tank_regulation_reward,
        episode_metric_function=regulation_episode_metrics,
        primary_metric="tracking_iae",
        metric_direction="minimize",
        safety_violation_penalty=100.0,
    )


def _tank3_reward():
    return Reward(
        id="tank3-regulation",
        function=_tank3_regulation_reward,
        episode_metric_function=tank3_regulation_episode_metrics,
        primary_metric="tank3_tracking_iae",
        metric_direction="minimize",
        safety_violation_penalty=100.0,
    )


def _economic_reward():
    return Reward(
        id="economic",
        function=economic_reward,
        episode_metric_function=economic_episode_metrics,
        primary_metric="economic_objective",
        metric_direction="maximize",
        safety_violation_penalty=100.0,
    )


SCENARIO = Scenario(
    id="three_tank",
    make_model=ThreeTankModel,
    control_dt=1.0,
    make_default_episode=make_default_episode,
    sample_training_episode=sample_training_episode,
    benchmarks=BENCHMARKS,
    rewards={
        "regulation": _regulation_reward(),
        "tank3-regulation": _tank3_reward(),
        "economic": _economic_reward(),
    },
    default_reward="regulation",
    controller_config=_controller_config,
)


__all__ = ["SCENARIO"]
