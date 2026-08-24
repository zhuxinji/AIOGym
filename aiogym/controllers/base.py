"""Environment-oriented controller construction for the core workflows."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def make_controller(
    controller_id: str,
    *,
    env,
    config: Mapping | None = None,
):
    import aiogym.scenarios  # noqa: F401
    from aiogym.core.registry import get_scenario

    key = str(controller_id).lower()
    base_env = env.unwrapped
    scenario = get_scenario(base_env.scenario.id)
    if key in {"pid", "mpc"}:
        if scenario.controller_config is None:
            raise ValueError(f"scenario {scenario.id!r} has no controller config")
        defaults = dict(
            scenario.controller_config(key, base_env.reward.id)
        )
    else:
        defaults = {}
    defaults.update({} if config is None else dict(config))
    defaults = _compile_controller_config(defaults, base_env)
    if key == "pid":
        from .pid import PIDPolicy

        if "kp" not in defaults:
            raise ValueError(f"scenario {scenario.id!r} has no PID controller")
        bias = defaults.pop("bias") if "bias" in defaults else None
        if isinstance(bias, str) and bias == "default_action":
            bias = base_env.model.default_action()
        return PIDPolicy(base_env, bias=bias, **defaults)
    if key == "mpc":
        from .mpc import FixedSetpointMPCPolicy

        return FixedSetpointMPCPolicy(base_env, **defaults)
    if key == "hold":
        from .policies import HoldPolicy

        action = defaults["action"] if "action" in defaults else None
        return HoldPolicy(base_env, action=action)
    if key == "random":
        from .policies import RandomPolicy

        return RandomPolicy(env)
    raise ValueError(f"unsupported core controller {controller_id!r}")


def _compile_controller_config(config, env):
    resolved = dict(config)
    if "matrix_terms" not in resolved:
        return resolved
    action_names = [row["name"] for row in env.model.action_schema()]
    output_names = [row["name"] for row in env.model.output_schema()]

    def index(values, name, kind):
        try:
            return values.index(name)
        except ValueError as error:
            raise ValueError(f"unknown PID {kind} {name!r}") from error

    terms = resolved.pop("matrix_terms")
    rows = len(action_names)
    columns = len(output_names)
    matrices = {
        name: np.zeros((rows, columns), dtype=float)
        for name in ("kp", "ki", "kd")
    }
    for term in terms:
        row = index(action_names, term["actuator"], "actuator")
        column = index(output_names, term["output"], "output")
        for name in matrices:
            matrices[name][row, column] += float(term[name])
    resolved.update({name: value.tolist() for name, value in matrices.items()})
    return resolved


__all__ = ["make_controller"]
