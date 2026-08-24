"""Scenario registration for the five-stage extraction column."""
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
from .model import ExtractionModel


_PID = {
    "matrix_terms": [
        {
            "actuator": "liquid_feed_flow",
            "output": "stage_5_liquid_concentration",
            "kp": 2.2,
            "ki": 6.0,
            "kd": 0.0,
        },
        {
            "actuator": "gas_feed_flow",
            "output": "stage_5_liquid_concentration",
            "kp": 0.0,
            "ki": 0.0,
            "kd": 0.0,
        },
    ],
    "bias": "default_action",
}
_MPC = {
    "Ts": 0.01,
    "P": 10,
    "move_supp": [0.5, 20.0],
    "steady_input_weight": [5.0, 20.0],
    "cv_scale": [0.45],
    "q_y": [1.0],
    "reseed_on_feedforward_change": True,
}


def _controller_config(controller_id, reward_id):
    if reward_id != "regulation":
        raise ValueError(
            f"extraction has no controllers for reward {reward_id!r}"
        )
    configs = {"pid": _PID, "mpc": _MPC}
    try:
        return deepcopy(configs[controller_id])
    except KeyError as error:
        raise ValueError(
            f"extraction has no {controller_id!r} controller"
        ) from error


SCENARIO = Scenario(
    id="extraction",
    make_model=ExtractionModel,
    control_dt=0.1,
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
