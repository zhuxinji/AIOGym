"""Three-Tank hydraulic PID with learned Cascade heater actions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from aiogym.controllers import make_controller
from aiogym.core.contracts import environment_interface, policy_metadata, validate_policy
from aiogym.core.env import make_env
from aiogym.core.information import EnvironmentInformation
from aiogym.scenarios.three_tank.model import ThreeTankModel


_CASCADE_OBSERVATION_DIM = 16
_CASCADE_ACTION_DIM = 7
_HYDRAULIC_ACTION_DIM = 4
_HEATER_ACTION_DIM = 3
_TEMPERATURE_ERROR_SCALE_C = 5.0


class CascadeHydraulicPIDEnv(EnvironmentInformation, gym.Wrapper):
    """Expose only heater actions while Three-Tank PID controls hydraulics."""

    def __init__(self, env: gym.Env, *, temperature_training: bool = False):
        _validate_direct_cascade_env(env)
        if not isinstance(temperature_training, bool):
            raise TypeError("temperature_training must be a boolean")
        super().__init__(env)
        self.temperature_training = temperature_training
        self._hydraulic_env, self._hydraulic_pid = _make_hydraulic_pid(
            env.unwrapped.model
        )
        self.action_space = gym.spaces.Box(
            low=np.asarray(env.action_space.low[-_HEATER_ACTION_DIM:], dtype=np.float32),
            high=np.asarray(
                env.action_space.high[-_HEATER_ACTION_DIM:], dtype=np.float32
            ),
            dtype=np.float32,
        )
        observation_low = [
            *np.asarray(env.observation_space.low, dtype=np.float32),
            *np.zeros(_HYDRAULIC_ACTION_DIM, dtype=np.float32),
        ]
        observation_high = [
            *np.asarray(env.observation_space.high, dtype=np.float32),
            *np.ones(_HYDRAULIC_ACTION_DIM, dtype=np.float32),
        ]
        if temperature_training:
            temperature_rows = env.unwrapped.model.output_schema()[3:]
            observation_low.extend(
                (float(row["low"]) - float(row["high"]))
                / _TEMPERATURE_ERROR_SCALE_C
                for row in temperature_rows
            )
            observation_high.extend(
                (float(row["high"]) - float(row["low"]))
                / _TEMPERATURE_ERROR_SCALE_C
                for row in temperature_rows
            )
        self.observation_space = gym.spaces.Box(
            low=np.asarray(observation_low, dtype=np.float32),
            high=np.asarray(observation_high, dtype=np.float32),
            dtype=np.float32,
        )
        self._pending_hydraulic_action: np.ndarray | None = None

    def policy_interface(self):
        interface = environment_interface(self.env)
        observation = list(interface["observation"])
        observation.extend(
            {**row, "name": f"pending_{row['name']}", "kind": "action"}
            for row in interface["action"][:_HYDRAULIC_ACTION_DIM]
        )
        if self.temperature_training:
            for index, row in enumerate(self.unwrapped.model.output_schema()[3:]):
                observation.append({
                    "name": f"{row['name']}_error",
                    "kind": "error",
                    "unit": "normalized",
                    "scale": _TEMPERATURE_ERROR_SCALE_C,
                    "low": float(self.observation_space.low[-3 + index]),
                    "high": float(self.observation_space.high[-3 + index]),
                })
        return {
            **interface,
            "adapter": "cascade-hydraulic-pid.v1",
            "observation": [{**row, "index": i} for i, row in enumerate(observation)],
            "action": [
                {**row, "index": i}
                for i, row in enumerate(interface["action"][-_HEATER_ACTION_DIM:])
            ],
        }

    @property
    def observations(self):
        rows = super().observations
        for name, action in self.env.actions.items():
            pending = f"pending_{name}"
            if pending in rows:
                rows[pending].update(
                    description=f"Next hydraulic PID command: {action['description']}",
                    source=name,
                )
        if self.temperature_training:
            for name, output in self.outputs.items():
                error = f"{name}_error"
                if error in rows:
                    rows[error].update(
                        description=f"Temperature tracking error: {output['description']}",
                        source=name,
                        normalization=f"(reference - measurement) / {_TEMPERATURE_ERROR_SCALE_C:g} degC",
                    )
        return rows

    def describe(self):
        report = super().describe()
        report["physical_actions"] = self.env.actions
        if self.temperature_training:
            report["environment"]["reward_adapter"] = "temperature_tracking"
        return report

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._hydraulic_pid.reset(seed=seed)
        hydraulic = _hydraulic_action(
            self._hydraulic_pid,
            self._hydraulic_env,
            observation,
        )
        self._pending_hydraulic_action = hydraulic
        updated = dict(info)
        updated["policy_action"] = None
        updated["hydraulic_action"] = hydraulic.copy()
        return (
            _hybrid_observation(
                observation,
                hydraulic,
                self.unwrapped.model,
                include_temperature_error=self.temperature_training,
            ),
            updated,
        )

    def step(self, action: Sequence[float]):
        heaters = _heater_vector(action, self.action_space)
        if self._pending_hydraulic_action is None:
            raise RuntimeError("CascadeHydraulicPIDEnv must be reset before step")
        hydraulic = self._pending_hydraulic_action.copy()
        physical_action = np.concatenate([hydraulic, heaters]).astype(np.float32)
        observation, reward, terminated, truncated, info = self.env.step(
            physical_action
        )
        next_hydraulic = _hydraulic_action(
            self._hydraulic_pid,
            self._hydraulic_env,
            observation,
        )
        self._pending_hydraulic_action = next_hydraulic
        updated = dict(info)
        updated["policy_action"] = heaters.copy()
        updated["hydraulic_action"] = hydraulic
        updated["next_hydraulic_action"] = next_hydraulic.copy()
        resolved_reward = (
            _temperature_training_reward(info, reward, self.unwrapped.control_dt)
            if self.temperature_training
            else float(reward)
        )
        if self.temperature_training:
            updated["physical_reward"] = float(reward)
            updated["physical_reward_terms"] = dict(info["reward_terms"])
            updated["reward_terms"] = {
                "temperature_tracking": resolved_reward
                - float(info["reward_terms"].get("early_termination", 0.0))
                - float(info["reward_terms"].get("safety", 0.0)),
                "early_termination": float(
                    info["reward_terms"].get("early_termination", 0.0)
                ),
                "safety": float(info["reward_terms"].get("safety", 0.0)),
            }
            updated["experimental_reward"] = "temperature_tracking"
        return (
            _hybrid_observation(
                observation,
                next_hydraulic,
                self.unwrapped.model,
                include_temperature_error=self.temperature_training,
            ),
            resolved_reward,
            terminated,
            truncated,
            updated,
        )

    def close(self) -> None:
        self._hydraulic_env.close()
        self.env.close()


class CascadeHydraulicPIDPolicyAdapter:
    """Deploy one three-heater policy through the direct Cascade interface."""

    def __init__(
        self,
        policy,
        *,
        env: gym.Env,
        temperature_error_observation: bool = False,
    ):
        _validate_direct_cascade_env(env)
        if not isinstance(temperature_error_observation, bool):
            raise TypeError("temperature_error_observation must be a boolean")
        self.env = env
        self.policy = validate_policy(policy)
        self.temperature_error_observation = temperature_error_observation
        self._hydraulic_env, self._hydraulic_pid = _make_hydraulic_pid(
            env.unwrapped.model
        )
        self._heater_action_space = gym.spaces.Box(
            low=np.asarray(env.action_space.low[-_HEATER_ACTION_DIM:], dtype=np.float32),
            high=np.asarray(
                env.action_space.high[-_HEATER_ACTION_DIM:], dtype=np.float32
            ),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None) -> None:
        self._hydraulic_pid.reset(seed=seed)
        self.policy.reset(seed=seed)

    def act(
        self,
        observation: np.ndarray,
        context: Mapping[str, Any],
    ) -> np.ndarray:
        hydraulic = _hydraulic_action(
            self._hydraulic_pid,
            self._hydraulic_env,
            observation,
        )
        hybrid_observation = _hybrid_observation(
            observation,
            hydraulic,
            self.env.unwrapped.model,
            include_temperature_error=self.temperature_error_observation,
        )
        heaters = _heater_vector(
            self.policy.act(hybrid_observation, context),
            self._heater_action_space,
        )
        return np.concatenate([hydraulic, heaters]).astype(np.float32)

    def metadata(self) -> Mapping[str, Any]:
        return {
            "id": "cascade_three_tank_pid_heater_policy",
            "kind": "hybrid_controller",
            "temperature_error_observation": self.temperature_error_observation,
            "hydraulic_controller": policy_metadata(self._hydraulic_pid),
            "heater_policy": policy_metadata(self.policy),
        }

    def close(self) -> None:
        self._hydraulic_env.close()


def three_tank_pid_heater_control(env: gym.Env) -> CascadeHydraulicPIDEnv:
    """Wrap a direct Cascade environment for three-heater policy training."""

    return CascadeHydraulicPIDEnv(env)


def three_tank_pid_temperature_control(env: gym.Env) -> CascadeHydraulicPIDEnv:
    """Expose heaters, temperature errors, and a temperature-only reward."""

    return CascadeHydraulicPIDEnv(env, temperature_training=True)


def as_hybrid_physical_policy(
    policy,
    *,
    env: gym.Env,
    temperature_error_observation: bool = False,
) -> CascadeHydraulicPIDPolicyAdapter:
    """Combine one three-heater policy with Three-Tank hydraulic PID."""

    return CascadeHydraulicPIDPolicyAdapter(
        policy,
        env=env,
        temperature_error_observation=temperature_error_observation,
    )


def _make_hydraulic_pid(cascade_model):
    parameter_names = tuple(ThreeTankModel().resolved_parameters)
    parameters = {
        name: cascade_model.parameter(name)
        for name in parameter_names
    }
    hydraulic_env = make_env("three_tank", parameters=parameters)
    return hydraulic_env, make_controller("pid", env=hydraulic_env)


def _hydraulic_action(pid, hydraulic_env, observation) -> np.ndarray:
    cascade_measurement = np.asarray(observation, dtype=np.float32).reshape(-1)
    if cascade_measurement.shape != (_CASCADE_OBSERVATION_DIM,):
        raise ValueError("Cascade hydraulic PID requires a 16-value observation")
    levels = cascade_measurement[[0, 2, 4]]
    flows = cascade_measurement[6:10]
    level_references = cascade_measurement[10:13]
    hydraulic_observation = np.concatenate([levels, flows, level_references]).astype(
        np.float32
    )
    reference = level_references * np.asarray(
        hydraulic_env.unwrapped.model.output_scales(), dtype=np.float32
    )
    action = np.asarray(
        pid.act(hydraulic_observation, {"reference": reference}),
        dtype=np.float32,
    ).reshape(-1)
    if (
        action.shape != (_HYDRAULIC_ACTION_DIM,)
        or not hydraulic_env.action_space.contains(action)
    ):
        raise ValueError("Three-Tank PID produced an invalid hydraulic action")
    return action


def _hybrid_observation(
    observation,
    hydraulic_action,
    model,
    *,
    include_temperature_error,
) -> np.ndarray:
    parts = [
        np.asarray(observation, dtype=np.float32).reshape(-1),
        np.asarray(hydraulic_action, dtype=np.float32).reshape(-1),
    ]
    if include_temperature_error:
        direct = np.asarray(observation, dtype=np.float32).reshape(-1)
        temperature_scales = np.asarray(
            model.output_scales()[3:], dtype=np.float32
        )
        measured_temperatures = direct[[1, 3, 5]] * temperature_scales
        temperature_references = direct[13:16] * temperature_scales
        temperature_error = (
            temperature_references - measured_temperatures
        ) / _TEMPERATURE_ERROR_SCALE_C
        parts.append(temperature_error)
    values = np.concatenate(parts).astype(np.float32)
    expected = _CASCADE_OBSERVATION_DIM + _HYDRAULIC_ACTION_DIM + (
        _HEATER_ACTION_DIM if include_temperature_error else 0
    )
    if values.shape != (expected,):
        raise ValueError("hybrid observation has an invalid shape")
    return values


def _temperature_training_reward(info, physical_reward, control_dt) -> float:
    del physical_reward
    reference = np.asarray(info["transition_reference"], dtype=float)[3:]
    temperatures = np.asarray(info["y"], dtype=float)[3:]
    normalized_error = (temperatures - reference) / _TEMPERATURE_ERROR_SCALE_C
    tracking = -float(control_dt) * float(np.mean(normalized_error**2))
    return (
        tracking
        + float(info["reward_terms"].get("early_termination", 0.0))
        + float(info["reward_terms"].get("safety", 0.0))
    )


def _heater_vector(action, action_space: gym.spaces.Box) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.shape != (_HEATER_ACTION_DIM,) or not action_space.contains(values):
        raise ValueError("heater action must contain three values in [0, 1]")
    return values


def _validate_direct_cascade_env(env: gym.Env) -> None:
    if env.unwrapped.scenario.id != "cascade":
        raise ValueError("hybrid control requires the cascade scenario")
    if (
        env.observation_space.shape != (_CASCADE_OBSERVATION_DIM,)
        or env.action_space.shape != (_CASCADE_ACTION_DIM,)
    ):
        raise ValueError(
            "hybrid control requires the direct 16-observation, "
            "7-action Cascade interface"
        )


__all__ = [
    "CascadeHydraulicPIDEnv",
    "CascadeHydraulicPIDPolicyAdapter",
    "as_hybrid_physical_policy",
    "three_tank_pid_heater_control",
    "three_tank_pid_temperature_control",
]
