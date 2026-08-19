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
from .model import QuadrupleModel


_PID = {
    "matrix_terms": [
        {
            "actuator": "pump_1_voltage",
            "output": "lower_tank_1_level",
            "kp": 1.0159437333249033,
            "ki": 0.1,
            "kd": 0.0,
        },
        {
            "actuator": "pump_2_voltage",
            "output": "lower_tank_2_level",
            "kp": 2.0,
            "ki": 0.09001698960588032,
            "kd": 0.0,
        },
    ],
    "bias": [0.3, 0.3],
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
