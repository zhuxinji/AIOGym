"""Scenario definition for the two-zone HVAC model."""
from __future__ import annotations

from copy import deepcopy

from aiogym.core.contracts import Scenario
from aiogym.core.specs import Reward
from aiogym.scenarios._metrics import REGULATION_SUCCESS_CRITERION
from aiogym.scenarios._metrics import regulation_episode_metrics
from aiogym.scenarios._shared import regulation_reward

from .episodes import (
    BENCHMARKS,
    make_default_episode,
    sample_training_disturbance,
    sample_training_episode,
)
from .model import HVACModel


_PID = {
    "matrix_terms": [
        {
            "actuator": "hvac_zone_0",
            "output": "zone_0_temperature",
            "kp": 0.32,
            "ki": 0.01,
            "kd": 0.0,
        },
        {
            "actuator": "hvac_zone_1",
            "output": "zone_1_temperature",
            "kp": 0.32,
            "ki": 0.01,
            "kd": 0.0,
        },
    ],
    "bias": "default_action",
}
_MPC = {
    "prediction_horizon": 1,
    "move_supp": [0.0, 0.0],
    "steady_input_weight": [0.0, 0.0],
    "cv_scale": [8.0, 8.0],
    "q_y": [1.0, 1.0],
}


def _controller_config(controller_id, reward_id):
    if reward_id != "regulation":
        raise ValueError(f"HVAC has no controllers for reward {reward_id!r}")
    configs = {"pid": _PID, "mpc": _MPC}
    try:
        return deepcopy(configs[controller_id])
    except KeyError as error:
        raise ValueError(f"HVAC has no {controller_id!r} controller") from error


SCENARIO = Scenario(
    id="hvac",
    make_model=HVACModel,
    control_dt=5.0,
    make_default_episode=make_default_episode,
    sample_training_episode=sample_training_episode,
    sample_training_disturbance=sample_training_disturbance,
    benchmarks=BENCHMARKS,
    rewards={
        "regulation": Reward(
            id="regulation",
            success_criterion=REGULATION_SUCCESS_CRITERION,
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
