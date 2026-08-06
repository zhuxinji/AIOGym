"""Environment-oriented controller construction for the core workflows."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class _LegacyPolicyAdapter:
    def __init__(self, controller, env):
        self.controller = controller
        self.env = env

    def reset(self, seed=None):
        self.controller.reset(seed=seed)

    def act(self, observation, context):
        from .contracts import build_context

        legacy_context = build_context(self.env, context.get("info"))
        return np.asarray(
            self.controller.act(observation, legacy_context), dtype=np.float32
        )

    def metadata(self):
        metadata = dict(self.controller.metadata())
        metadata["action_contract"] = "env.action_space"
        metadata["adapter"] = "core-policy"
        metadata["interface_hash"] = self.env.identity.interface_hash
        return metadata


def make_controller(
    controller_id: str,
    *,
    env,
    profile: str | None = None,
    config: Mapping | None = None,
):
    import aiogym.scenarios  # noqa: F401
    from aiogym.core import get_scenario

    key = str(controller_id).lower()
    plugin = get_scenario(env.task.scenario)
    profiles = plugin.controller_defaults.get(key, {})
    profile_id = profile or env.condition.id
    if plugin.resolve_controller_profile is not None:
        defaults = dict(
            plugin.resolve_controller_profile(
                key,
                plant_id=env.plant.id,
                condition_id=env.condition.id,
                objective=env.task.objective,
            )
        )
    else:
        defaults = dict(profiles.get(profile_id, profiles.get(env.task.objective, {})))
    defaults.update(dict(config or {}))
    defaults = _compile_named_profile(defaults, env)
    if key == "pid":
        from .pid import MatrixPIDPolicy

        if defaults.get("kp"):
            bias = defaults.pop("bias", None)
            if isinstance(bias, str) and bias == "default_action":
                bias = env.model.default_action()
            return MatrixPIDPolicy(env, bias=bias, **defaults)
        if not defaults.get("loops"):
            raise ValueError(
                f"scenario {plugin.id!r} has no PID profile {profile_id!r}"
            )
        from .pid import PIDAgent

        model = getattr(env.model, "_model", env.model)
        return _LegacyPolicyAdapter(PIDAgent(model, **defaults), env)
    if key == "mpc":
        from .mpc import MPCAgent

        model = getattr(env.model, "_model", env.model)
        return _LegacyPolicyAdapter(MPCAgent(model, **defaults), env)
    if key == "hold":
        from .policies import HoldPolicy

        return HoldPolicy(env, action=defaults.get("action"))
    if key == "random":
        from .policies import RandomPolicy

        return RandomPolicy(env)
    if key == "sb3":
        from .policies import SB3CheckpointPolicy

        return SB3CheckpointPolicy.load(
            defaults["checkpoint"],
            algorithm=defaults["algorithm"],
            device=defaults.get("device", "auto"),
        )
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
                matrices[name][row, column] += float(term.get(name, 0.0))
        resolved.update({name: value.tolist() for name, value in matrices.items()})
    return resolved


__all__ = ["make_controller"]
