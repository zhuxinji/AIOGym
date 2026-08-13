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
    defaults = _compile_named_profile(defaults, base_env)
    if key == "pid":
        from .pid import MatrixPIDPolicy

        if "kp" in defaults:
            bias = defaults.pop("bias") if "bias" in defaults else None
            if isinstance(bias, str) and bias == "default_action":
                bias = base_env.model.default_action()
            return MatrixPIDPolicy(base_env, bias=bias, **defaults)
        if "loops" not in defaults or not defaults["loops"]:
            raise ValueError(f"scenario {scenario.id!r} has no PID controller")
        from .pid import FixedSetpointPIDPolicy

        return FixedSetpointPIDPolicy(base_env, **defaults)
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


def _compile_named_profile(profile, env):
    resolved = dict(profile)
    named_loops = "loops" in resolved and any(
        "actuator" in row or "output" in row for row in resolved["loops"]
    )
    named_holds = "holds" in resolved and any(
        "actuator" in row for row in resolved["holds"]
    )
    if not named_loops and not named_holds and "matrix_terms" not in resolved:
        return resolved
    action_names = [row["name"] for row in env.model.action_schema()]
    output_names = [row["name"] for row in env.model.output_schema()]

    def index(values, name, kind):
        try:
            return values.index(name)
        except ValueError as error:
            raise ValueError(f"unknown PID {kind} {name!r}") from error

    if named_loops:
        resolved["loops"] = [
            {
                **{key: value for key, value in row.items() if key not in {"actuator", "output"}},
                "u_index": index(action_names, row["actuator"], "actuator"),
                "y_index": index(output_names, row["output"], "output"),
            }
            for row in resolved["loops"]
        ]
    if named_holds:
        resolved["holds"] = [
            {
                **{key: value for key, value in row.items() if key != "actuator"},
                "u_index": index(action_names, row["actuator"], "actuator"),
            }
            for row in resolved["holds"]
        ]
    terms = resolved.pop("matrix_terms", None)
    if terms is not None:
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
