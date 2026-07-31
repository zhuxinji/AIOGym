"""Adapters for learned policy controller interfaces."""
from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import (
    CONTROLLER_API_VERSION,
    Controller,
    ControllerContext,
    controller_metadata,
    normalized_to_environment_action,
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
        normalized_actions: bool = False,
        allow_environment_context: bool = False,
    ):
        self.policy = policy
        self.name = name or getattr(policy, "name", policy.__class__.__name__)
        self.action_mode = action_mode
        self.control_structure = control_structure
        self.normalized_actions = bool(normalized_actions)
        self.allow_environment_context = bool(
            allow_environment_context
        )

    def reset(self, seed: int | None = None) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset(seed=seed)

    def act(self, obs: np.ndarray, context: ControllerContext) -> np.ndarray:
        if hasattr(self.policy, "predict"):
            out = self.policy.predict(obs, deterministic=True)
            action = np.asarray(
                out[0] if isinstance(out, tuple) else out,
                dtype=np.float32,
            )
            return self._environment_action(action)
        if hasattr(self.policy, "act"):
            policy_context = (
                context
                if self.allow_environment_context
                else ControllerContext(
                    measurement=_policy_measurement(context),
                    setpoint=context.setpoint,
                    info={},
                    action_mode=context.action_mode,
                    control_dt=context.control_dt,
                    env=None,
                )
            )
            out = self.policy.act(obs, policy_context)
            return self._environment_action(np.asarray(out, dtype=np.float32))
        raise TypeError(f"{self.policy!r} has neither predict(obs) nor act(obs)")

    def _environment_action(self, action: np.ndarray) -> np.ndarray:
        if not self.normalized_actions:
            return action
        return normalized_to_environment_action(action)

    def metadata(self) -> dict[str, Any]:
        data = controller_metadata(self.policy)
        data.setdefault("name", self.name)
        data.setdefault("class", self.policy.__class__.__name__)
        data["api"] = self.controller_api_version
        data["adapter"] = self.__class__.__name__
        data["action_mode"] = self.action_mode
        data["normalized_actions"] = self.normalized_actions
        data["environment_context_access"] = (
            self.allow_environment_context
        )
        data.setdefault("control_structure", self.control_structure)
        return data


def _policy_measurement(context: ControllerContext) -> dict:
    env = context.env
    if env is None:
        return dict(context.measurement)
    state = (
        env._measured_state()
        if hasattr(env, "_measured_state")
        else env.model.state_vector(env.integ.x)
    )
    if getattr(env, "observation_mode", "full_state") == "measured_output":
        measurement = {
            "y": list(env.model.controlled_output(state)),
            "observation_mode": "measured_output",
        }
    else:
        measurement = dict(env.model.outputs(state))
    if bool(getattr(env, "disturbance_obs", False)):
        names = list(env.model.dynamics_disturbance_names())
        values = list(env.model.disturbance_vector(env._env()))
        measurement["measured_disturbance"] = {
            name: value for name, value in zip(names, values)
        }
    return measurement


class SB3PolicyController(PolicyController):
    """Stable-Baselines3 policy adapter with optional lazy loading."""

    @classmethod
    def load(cls, path: str, algo: str = "sac", **kw):
        algo_key = algo.lower()
        device = kw.pop("device", "auto")
        try:
            if algo_key == "sac":
                from stable_baselines3 import SAC

                policy = SAC.load(path, device=device)
            elif algo_key == "ppo":
                from stable_baselines3 import PPO

                policy = PPO.load(path, device=device)
            elif algo_key == "td3":
                from stable_baselines3 import TD3

                policy = TD3.load(path, device=device)
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
