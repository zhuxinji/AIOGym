"""Explicit Three-Tank residual-control wrapper."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .control import resolve_residual_action


class Tank3ResidualWrapper(gym.Wrapper):
    """Expose a two-dimensional Tank 3 residual action over the physical env."""

    def __init__(self, env):
        super().__init__(env)
        base = env.unwrapped
        if base.scenario.id != "three_tank" or base.action_space.shape != (5,):
            raise ValueError(
                "Tank3ResidualWrapper requires a physical Three-Tank env"
            )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

    def step(self, action):
        residual = np.asarray(action, dtype=np.float32).reshape(-1)
        if residual.shape != (2,) or not self.action_space.contains(residual):
            raise ValueError("residual action must belong to wrapper action_space")
        base = self.env.unwrapped
        model = base.model
        controller_disturbances = dict(base.disturbances)
        defaults = model.default_disturbances()
        for name in (
            "pump_flow_factor",
            "v12_flow_factor",
            "v23_flow_factor",
            "v34_flow_factor",
            "heater_efficiency",
            "heat_loss_factor",
        ):
            controller_disturbances[name] = defaults[name]
        physical = resolve_residual_action(
            model,
            residual=residual,
            state=base.state,
            reference=base.y_sp,
            disturbances=controller_disturbances,
        )
        observation, reward, terminated, truncated, info = self.env.step(physical)
        info = dict(info)
        info["commanded_action"] = residual.copy()
        info["resolved_physical_action"] = physical.copy()
        return observation, reward, terminated, truncated, info


__all__ = ["Tank3ResidualWrapper"]
