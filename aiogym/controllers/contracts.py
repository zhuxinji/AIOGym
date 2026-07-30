"""Controller contracts and environment-facing helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np


CONTROLLER_API_VERSION = "aiogym.controller.v1"


@dataclass(frozen=True)
class ControllerContext:
    """Per-step information exposed to controllers."""

    measurement: Mapping[str, Any]
    setpoint: Mapping[str, Any]
    info: Mapping[str, Any]
    action_mode: str
    control_dt: float
    env: Any = None


class Controller(Protocol):
    name: str
    action_mode: str
    control_structure: str
    controller_api_version: str

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, obs: np.ndarray, context: ControllerContext) -> np.ndarray: ...

    def metadata(self) -> dict[str, Any]: ...


def build_context(env, info: Mapping[str, Any] | None = None) -> ControllerContext:
    return ControllerContext(
        measurement=make_meas(env),
        setpoint={"y_sp": list(getattr(env, "y_sp", []))},
        info=dict(info or {}),
        action_mode=getattr(env, "action_mode", "actuator"),
        control_dt=float(env.control_dt),
        env=env,
    )


def make_meas(env):
    """Build a measured state dictionary from the native environment."""

    state = (
        env._measured_state()
        if hasattr(env, "_measured_state")
        else env.model.state_vector(env.integ.x)
    )
    if getattr(env, "observation_mode", "full_state") == "measured_output":
        return {
            "y": list(env.model.controlled_output(state)),
            "observation_mode": "measured_output",
        }
    return env.model.measurement(state, env._env())


def validate_action(action: Any, env, controller_name: str) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).reshape(-1)
    expected = env.action_space.shape[0]
    if out.shape[0] != expected:
        raise ValueError(
            f"{controller_name} produced {out.shape[0]} actions for a "
            f"{expected}-action {getattr(env, 'action_mode', 'unknown')!r} environment"
        )
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{controller_name} produced a non-finite action")
    return out


def controller_metadata(obj) -> dict[str, Any]:
    if hasattr(obj, "metadata"):
        return dict(obj.metadata())
    return {
        "name": getattr(obj, "name", obj.__class__.__name__),
        "class": obj.__class__.__name__,
    }
