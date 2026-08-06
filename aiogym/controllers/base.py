"""Environment-oriented controller construction for the core workflows."""
from __future__ import annotations

from collections.abc import Mapping


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
    profile_id = profile or env.preset.id or env.task.objective
    defaults = dict(
        plugin.controller_defaults.get(key, {}).get(profile_id, {})
    )
    defaults.update(dict(config or {}))
    if key == "pid":
        from .pid import MatrixPIDPolicy

        if not defaults.get("kp"):
            raise ValueError(
                f"scenario {plugin.id!r} has no matrix PID profile {profile_id!r}"
            )
        bias = defaults.pop("bias", None)
        if bias == "default_action":
            bias = env.model.default_action()
        return MatrixPIDPolicy(env, bias=bias, **defaults)
    raise ValueError(f"unsupported core controller {controller_id!r}")


__all__ = ["make_controller"]
