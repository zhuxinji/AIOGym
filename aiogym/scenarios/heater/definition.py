"""Scenario registration for the fired-heater model."""
from __future__ import annotations

from copy import deepcopy

from aiogym.core.contracts import Scenario
from aiogym.core.specs import Reward
from aiogym.scenarios._metrics import regulation_episode_metrics
from aiogym.scenarios._shared import regulation_reward

from .episodes import (
    BENCHMARKS,
    make_default_episode,
    sample_training_disturbance,
    sample_training_episode,
)
from .model import HeaterModel


_PID = {
    "matrix_terms": [
        {
            "actuator": "air_damper",
            "output": "flue_oxygen",
            "kp": 0.09,
            "ki": 0.001,
            "kd": 0.0,
        },
        {
            "actuator": "fuel_valve",
            "output": "outlet_temperature",
            "kp": 0.025,
            "ki": 0.0003,
            "kd": 0.0,
        },
    ],
    "bias": "default_action",
}
_MPC = {
    "Ts": 2.0,
    "P": 20,
    "move_supp": [5.0, 5.0],
    "steady_input_weight": [50.0, 50.0],
    "cv_scale": [3.2, 8.0],
    "q_y": [1.0, 1.0],
    "reseed_on_feedforward_change": True,
}


def _controller_config(controller_id, reward_id):
    if reward_id != "regulation":
        raise ValueError(f"heater has no controllers for reward {reward_id!r}")
    configs = {"pid": _PID, "mpc": _MPC}
    try:
        return deepcopy(configs[controller_id])
    except KeyError as error:
        raise ValueError(f"heater has no {controller_id!r} controller") from error


SCENARIO = Scenario(
    id="heater",
    make_model=HeaterModel,
    control_dt=1.0,
    make_default_episode=make_default_episode,
    sample_training_episode=sample_training_episode,
    sample_training_disturbance=sample_training_disturbance,
    benchmarks=BENCHMARKS,
    rewards={
        "regulation": Reward(
            id="regulation",
            function=regulation_reward,
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
