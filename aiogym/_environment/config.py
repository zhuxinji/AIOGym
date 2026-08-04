"""Environment defaults, limits, and value validation."""
from __future__ import annotations

import numpy as np


T_HIGH, T_TRIP = 80.0, 92.0
H_HIGH_FRAC, H_LOW_FRAC, H_OVERFLOW_FRAC = 0.90, 0.15, 0.97
DIRECT_ENV_DEFAULTS = {
    "auto_events": True,
    "randomize": True,
    "randomize_setpoints": True,
    "randomize_plant": False,
    "plant_drift": False,
    "integral_obs": False,
    "disturbance_obs": True,
    "previous_action_obs": False,
    "normalize_observations": False,
    "tracking_error_obs": False,
    "observation_mode": "full_state",
    "action_mode": "actuator",
    "noise": False,
    "noise_pct": 0.01,
    "terminate_on_runaway": False,
}


def validated_range(name, value):
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{name} must contain two numeric bounds") from ex
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two bounds")
    if not all(np.isfinite(item) for item in values):
        raise ValueError(f"{name} bounds must be finite")
    if values[0] > values[1]:
        raise ValueError(f"{name} lower bound must not exceed its upper bound")
    return values
