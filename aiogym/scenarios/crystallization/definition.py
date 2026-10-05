"""Scenario definition for batch crystallization."""
from __future__ import annotations

from copy import deepcopy

from aiogym.core.contracts import Scenario
from aiogym.core.specs import Reward

from .episodes import (
    BENCHMARKS,
    make_default_episode,
    sample_training_disturbance,
    sample_training_episode,
)
from .model import CrystallizationModel
from .metrics import SUCCESS_CRITERION, batch_quality_metrics, batch_quality_reward


_PID = {
    "matrix_terms": [
        {
            "actuator": "cooling_temperature_fraction",
            "output": "coefficient_of_variation",
            "kp": 1.0,
            "ki": 0.001,
            "kd": 0.0,
        },
        {
            "actuator": "cooling_temperature_fraction",
            "output": "mean_crystal_size",
            "kp": -0.15,
            "ki": -0.0005,
            "kd": 0.0,
        },
    ],
    "bias": "default_action",
}
_MPC = {
    "prediction_horizon": 10,
    "move_supp": [0.05],
    "cv_scale": [0.5, 3.5],
    "q_y": [1.0, 1.0],
}


def _controller_config(controller_id, reward_id):
    if reward_id != "batch-quality":
        raise ValueError(
            f"crystallization has no controllers for reward {reward_id!r}"
        )
    configs = {"pid": _PID, "mpc": _MPC}
    try:
        return deepcopy(configs[controller_id])
    except KeyError as error:
        raise ValueError(
            f"crystallization has no {controller_id!r} controller"
        ) from error


SCENARIO = Scenario(
    id="crystallization",
    make_model=CrystallizationModel,
    control_dt=1.0,
    make_default_episode=make_default_episode,
    sample_training_episode=sample_training_episode,
    sample_training_disturbance=sample_training_disturbance,
    benchmarks=BENCHMARKS,
    rewards={
        "batch-quality": Reward(
            id="batch-quality",
            function=batch_quality_reward,
            episode_metric_function=batch_quality_metrics,
            primary_metric="return",
            metric_direction="maximize",
            safety_violation_penalty=100.0,
            success_criterion=SUCCESS_CRITERION,
        ),
    },
    default_reward="batch-quality",
    controller_config=_controller_config,
)


__all__ = ["SCENARIO"]
