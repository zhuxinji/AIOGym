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
    profile_id = profile or env.preset.id
    defaults = dict(profiles.get(profile_id, profiles.get(env.task.objective, {})))
    defaults.update(dict(config or {}))
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
        if plugin.id == "three_tank":
            raise ValueError(
                "MPC is unsupported for three_tank until the six-action model "
                "has validated linearization defaults"
            )
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


__all__ = ["make_controller"]
