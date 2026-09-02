"""Helpers for the single mapping-based core policy context."""
from __future__ import annotations

from collections.abc import Mapping


def controller_inputs(env, observation, context):
    if not isinstance(context, Mapping):
        raise TypeError("policy context must be a mapping")
    measurement = env.model.measurement_from_observation(
        observation,
    )
    if "reference" not in context:
        raise ValueError("policy context must contain reference")
    reference = context["reference"]
    return (
        measurement,
        {"y_sp": list(reference)},
        float(env.control_dt),
    )


__all__ = ["controller_inputs"]
