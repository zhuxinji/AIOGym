"""Reward functions shared by built-in process scenarios."""
from __future__ import annotations

import numpy as np


PRODUCT_VALUE_PER_M3 = 100000.0
ELECTRICITY_PRICE_PER_KWH = 0.7


def regulation_reward(state, action, next_state, context):
    del state, action
    model = context["model"]
    output = np.asarray(model.outputs(next_state), dtype=float)
    reference = np.asarray(context["reference"], dtype=float)
    scale = np.asarray(model.controlled_output_scales(), dtype=float)
    rate = float(np.mean(((output - reference) / scale) ** 2))
    cost = float(context["control_dt"]) * rate
    return -cost, {"tracking_error": -cost, "slew": 0.0, "effort": 0.0}


def economic_reward(state, action, next_state, context):
    del state
    model = context["model"]
    control_dt = float(context["control_dt"])
    power_kw = float(
        model.action_energy_kw(action, next_state, context["disturbances"])
    )
    product_flow_m3s = float(
        model.production(next_state, action, context["disturbances"])
    )
    product_value = PRODUCT_VALUE_PER_M3 * product_flow_m3s * control_dt
    energy_cost = ELECTRICITY_PRICE_PER_KWH * power_kw * control_dt / 3600.0
    reward = product_value - energy_cost
    return reward, {
        "product_value": product_value,
        "energy_cost": -energy_cost,
    }


__all__ = ["economic_reward", "regulation_reward"]
