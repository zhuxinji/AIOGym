"""Helpers for the single mapping-based core policy context."""
from __future__ import annotations

from collections.abc import Mapping


def controller_inputs(env, observation, context):
    if not isinstance(context, Mapping):
        raise TypeError("policy context must be a mapping")
    if "info" not in context:
        raise ValueError("policy context must contain step info")
    info = context["info"]
    if not isinstance(info, Mapping):
        raise ValueError("policy context must contain step info")
    if "disturbance" not in info:
        raise ValueError("policy step info must contain disturbance")
    disturbances = info["disturbance"]
    measurement = env.model.measurement_from_observation(
        observation,
        disturbances,
    )
    if "reference" not in info:
        raise ValueError("policy step info must contain reference")
    reference = info["reference"]
    return (
        measurement,
        {"y_sp": list(reference)},
        float(env.control_dt),
    )


__all__ = ["controller_inputs"]
