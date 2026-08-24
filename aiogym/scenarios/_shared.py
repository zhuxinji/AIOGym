"""Reward functions shared by built-in process scenarios."""

from __future__ import annotations

import numpy as np


def regulation_reward(state, action, next_state, context):
    del state, action
    model = context["model"]
    output = np.asarray(model.outputs(next_state), dtype=float)
    reference = np.asarray(context["reference"], dtype=float)
    scale = np.asarray(model.output_scales(), dtype=float)
    rate = float(np.mean(((output - reference) / scale) ** 2))
    cost = float(context["control_dt"]) * rate
    return -cost, {"tracking_error": -cost, "slew": 0.0, "effort": 0.0}


__all__ = ["regulation_reward"]
