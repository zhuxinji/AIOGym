"""Adapters for learned policy controller interfaces."""
from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import (
    CONTROLLER_API_VERSION,
    Controller,
    ControllerContext,
    controller_metadata,
)


class PolicyController:
    """Adapter for learned policies with predict(obs) or act(obs)."""

    controller_api_version = CONTROLLER_API_VERSION

    def __init__(
        self,
        policy,
        name: str | None = None,
        action_mode: str = "actuator",
        control_structure: str = "learned_policy",
    ):
        self.policy = policy
        self.name = name or getattr(policy, "name", policy.__class__.__name__)
        self.action_mode = action_mode
        self.control_structure = control_structure

    def reset(self, seed: int | None = None) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset(seed=seed)

    def act(self, obs: np.ndarray, context: ControllerContext) -> np.ndarray:
        if hasattr(self.policy, "predict"):
            out = self.policy.predict(obs, deterministic=True)
            return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)
        if hasattr(self.policy, "act"):
            out = self.policy.act(obs, context)
            return np.asarray(out, dtype=np.float32)
        raise TypeError(f"{self.policy!r} has neither predict(obs) nor act(obs)")

    def metadata(self) -> dict[str, Any]:
        data = controller_metadata(self.policy)
        data.setdefault("name", self.name)
        data.setdefault("class", self.policy.__class__.__name__)
        data["api"] = self.controller_api_version
        data["adapter"] = self.__class__.__name__
        data["action_mode"] = self.action_mode
        data.setdefault("control_structure", self.control_structure)
        return data


class SB3PolicyController(PolicyController):
    """Stable-Baselines3 policy adapter with optional lazy loading."""

    @classmethod
    def load(cls, path: str, algo: str = "sac", **kw):
        algo_key = algo.lower()
        try:
            if algo_key == "sac":
                from stable_baselines3 import SAC

                policy = SAC.load(path)
            elif algo_key == "ppo":
                from stable_baselines3 import PPO

                policy = PPO.load(path)
            elif algo_key == "td3":
                from stable_baselines3 import TD3

                policy = TD3.load(path)
            else:
                raise ValueError(f"unsupported SB3 algorithm: {algo}")
        except ModuleNotFoundError as ex:
            raise RuntimeError(
                "stable-baselines3 is required for SB3 policies; install "
                "AIO-Gym with `pip install 'aiogym[rl]'`"
            ) from ex
        return cls(policy, name=kw.pop("name", f"SB3-{algo_key.upper()}"), **kw)


def as_controller(
    agent,
    action_mode: str = "actuator",
    name: str | None = None,
    control_structure: str | None = None,
) -> Controller:
    if getattr(agent, "controller_api_version", None) == CONTROLLER_API_VERSION:
        return agent
    if hasattr(agent, "predict") or hasattr(agent, "act"):
        return PolicyController(
            agent,
            name=name,
            action_mode=action_mode,
            control_structure=control_structure
            or controller_metadata(agent).get("control_structure", "learned_policy"),
        )
    raise TypeError(f"{agent!r} is not a supported controller or policy")
