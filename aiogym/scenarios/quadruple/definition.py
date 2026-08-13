from copy import deepcopy

import numpy as np

from aiogym.core.contracts import Scenario
from aiogym.core.specs import Reward
from aiogym.scenarios._metrics import regulation_episode_metrics
from aiogym.scenarios._shared import regulation_reward

from .episodes import BENCHMARKS, make_default_episode, sample_training_episode
from .model import QuadrupleModel


_SMOOTH_MOVE_WEIGHT = 0.1
_SMOOTH_REWARD_SCALE = 100.0


def _smooth_regulation_reward(state, action, next_state, context):
    reward, terms = regulation_reward(state, action, next_state, context)
    terms["tracking_error"] *= _SMOOTH_REWARD_SCALE
    previous = np.asarray(context["previous_applied_action"], dtype=float)
    delta = np.asarray(action, dtype=float) - previous
    move_cost = (
        float(context["control_dt"])
        * _SMOOTH_MOVE_WEIGHT
        * float(np.sum(delta**2))
    )
    scaled_move_cost = _SMOOTH_REWARD_SCALE * move_cost
    terms["slew"] = -scaled_move_cost
    return _SMOOTH_REWARD_SCALE * reward - scaled_move_cost, terms


_PID = {
    "loops": [
        {
            "u_index": 0,
            "y_index": 0,
            "pid": [1.0159437333249033, 0.1, 0.0],
            "bias": 0.3,
        },
        {
            "u_index": 1,
            "y_index": 1,
            "pid": [2.0, 0.09001698960588032, 0.0],
            "bias": 0.3,
        },
    ]
}
_MPC = {"Ts": 1.0, "P": 1, "move_supp": 0.0, "cv_scale": [1.0, 1.0]}


def _controller_config(controller_id, reward_id):
    del reward_id
    profiles = {"pid": _PID, "mpc": _MPC}
    try:
        return deepcopy(profiles[controller_id])
    except KeyError as error:
        raise ValueError(f"quadruple has no {controller_id!r} controller") from error


SCENARIO = Scenario(
    id="quadruple",
    make_model=QuadrupleModel,
    control_dt=1.0,
    make_default_episode=make_default_episode,
    sample_training_episode=sample_training_episode,
    benchmarks=BENCHMARKS,
    rewards={
        "regulation": Reward(
            id="regulation",
            function=regulation_reward,
            episode_metric_function=regulation_episode_metrics,
            primary_metric="tracking_iae",
            metric_direction="minimize",
            safety_violation_penalty=100.0,
        ),
        "smooth-regulation": Reward(
            id="smooth-regulation",
            function=_smooth_regulation_reward,
            episode_metric_function=regulation_episode_metrics,
            primary_metric="tracking_iae",
            metric_direction="minimize",
            safety_violation_penalty=100.0,
        ),
    },
    default_reward="regulation",
    controller_config=_controller_config,
)

__all__ = ["SCENARIO"]
