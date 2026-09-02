"""Shared forward pre-run used to construct reachable boundary starts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from aiogym.core.model import apply_action_slew, integrate_process_state


def forward_preroll(
    model,
    *,
    command: Sequence[float],
    reached: Callable[[np.ndarray], bool],
    control_dt: float,
    maximum_steps: int,
    disturbances: Mapping[str, float] | None = None,
    initial_state: Sequence[float] | None = None,
    initial_action: Sequence[float] | None = None,
):
    """Drive a normal process state until a scenario-owned safe boundary."""

    if maximum_steps <= 0:
        raise ValueError("boundary pre-run maximum_steps must be positive")
    context = (
        model.default_disturbances()
        if disturbances is None
        else dict(disturbances)
    )
    state = np.asarray(
        model.initial_state() if initial_state is None else initial_state,
        dtype=float,
    )
    previous = np.asarray(
        model.default_action() if initial_action is None else initial_action,
        dtype=float,
    )
    requested = np.asarray(model.action_vector(command), dtype=float)
    for step in range(1, maximum_steps + 1):
        applied = apply_action_slew(model, previous, requested)
        state = integrate_process_state(
            model,
            state,
            applied,
            context,
            duration=control_dt,
        )
        costs = model.constraint_costs(state, context)
        if any(float(value) > 0.0 for value in costs.values()):
            raise ValueError(
                "boundary pre-run crossed a hard process constraint before "
                "reaching its stop condition"
            )
        if reached(state):
            return {
                "state": tuple(float(value) for value in state),
                "action": tuple(float(value) for value in applied),
                "steps": step,
            }
        previous = applied
    raise ValueError(
        "boundary pre-run did not reach its stop condition within "
        f"{maximum_steps} steps"
    )


__all__ = ["forward_preroll"]
