"""Scenario definition for the two-input exothermic CSTR."""
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
from .model import CSTRModel


_PID = {
    "matrix_terms": [
        {
            "actuator": "feed_pump",
            "output": "reactant_concentration",
            "kp": 16.0,
            "ki": 0.4,
            "kd": 0.0,
        },
        {
            "actuator": "cooling",
            "output": "reactor_temperature",
            "kp": -0.2,
            "ki": -0.005,
            "kd": 0.0,
        },
    ],
    "bias": "default_action",
}
_MPC = {
    "prediction_horizon": 1,
    "move_supp": [0.0, 0.0],
    "cv_scale": [0.18, 45.0],
    "q_y": [1.0, 1.0],
}


def _controller_config(controller_id, reward_id):
    if reward_id != "regulation":
        raise ValueError(f"CSTR has no controllers for reward {reward_id!r}")
    configs = {"pid": _PID, "mpc": _MPC}
    try:
        return deepcopy(configs[controller_id])
    except KeyError as error:
        raise ValueError(f"CSTR has no {controller_id!r} controller") from error


SCENARIO = Scenario(
    id="cstr",
    make_model=CSTRModel,
    control_dt=1.0,
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
