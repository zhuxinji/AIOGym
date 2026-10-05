"""Terminal quality objective and batch completion criteria."""
from __future__ import annotations

import numpy as np

from aiogym.scenarios._metrics import episode_trace


QUALITY_TOLERANCES = (0.01, 0.07)  # CV; mean crystal size in um.
SUCCESS_CRITERION = (
    f"Safe full batch; endpoint |CV error| <= {QUALITY_TOLERANCES[0]:g} "
    f"and |mean size error| <= {QUALITY_TOLERANCES[1]:g} um."
)


def batch_quality_reward(state, action, next_state, context):
    del state, action
    cost = 0.0
    if context["step_index"] + 1 == context["episode"].horizon:
        model = context["model"]
        error = (np.asarray(model.outputs(next_state)) - context["reference"])
        cost = float(np.mean((error / model.output_scales()) ** 2))
    return -cost, {"terminal_quality": -cost}


def batch_quality_metrics(env, episode):
    trace = episode_trace(env, episode)
    metrics = trace["metrics"]
    error = np.asarray(trace["errors"][-1])
    physical_error = np.abs(error * trace["output_scale"])
    metrics.update(
        terminal_quality_cost=float(np.mean(error ** 2)),
        final_error=float(np.max(np.abs(error))),
        control_success=float(
            metrics["safe_completion"]
            and np.all(physical_error <= QUALITY_TOLERANCES)
        ),
    )
    metrics.update({
        f"endpoint_{row['name']}_error": float(value)
        for row, value in zip(env.unwrapped.model.output_schema(), physical_error)
    })
    return metrics
