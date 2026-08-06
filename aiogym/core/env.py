"""Concrete process-control environment and its single resolver."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from .contracts import ProcessModel
from .registry import get_scenario, get_task
from .specs import PlantConfig, PresetSpec, ResolvedPlant, TaskSpec


def _bounds(schema: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray([item.get("low", -np.inf) for item in schema], dtype=np.float32)
    high = np.asarray([item.get("high", np.inf) for item in schema], dtype=np.float32)
    if low.shape != high.shape or np.any(low > high):
        raise ValueError("schema contains invalid bounds")
    return low, high


def _load_mapping(value: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    import json

    return json.loads(Path(value).read_text(encoding="utf-8"))


def resolve_plant(
    scenario: str,
    plant: PlantConfig | str | Path | Mapping[str, Any] | None,
) -> ResolvedPlant:
    plugin = get_scenario(scenario)
    raw = plugin.default_plant() if plant is None else plant
    if isinstance(raw, PlantConfig):
        config = raw
    else:
        config = PlantConfig.from_mapping(_load_mapping(raw))
    if config.scenario != scenario:
        raise ValueError(
            f"PlantConfig scenario {config.scenario!r} does not match task scenario {scenario!r}"
        )
    return plugin.resolve_plant(config)


class ProcessControlEnv(gym.Env):
    """A small deterministic environment over one resolved ProcessModel."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        model: ProcessModel,
        task: TaskSpec,
        plant: ResolvedPlant,
        preset: PresetSpec,
    ) -> None:
        super().__init__()
        self.model = model
        self.task = task
        self.plant = plant
        self.preset = preset
        builder = getattr(model, "build_env", None)
        if callable(builder):
            self._delegate = builder(task=task, preset=preset, plant=plant)
            self.control_dt = float(self._delegate.control_dt)
            self.episode_steps = int(self._delegate.episode_steps)
            self.action_space = self._delegate.action_space
            self.observation_space = self._delegate.observation_space
            self._step_index = 0
            return
        self._delegate = None
        self.control_dt = float(preset.config.get("control_dt", task.control_dt))
        self.episode_steps = int(preset.config.get("horizon", task.horizon))
        action_low, action_high = _bounds(model.action_schema())
        state_low, state_high = _bounds(model.state_schema())
        initial_output = np.asarray(model.outputs(model.initial_state()), dtype=np.float32)
        output_schema = getattr(model, "output_schema", None)
        if callable(output_schema):
            observation_low, observation_high = _bounds(output_schema())
        elif initial_output.shape == state_low.shape:
            observation_low, observation_high = state_low, state_high
        else:
            observation_low = np.full(initial_output.shape, -np.inf, dtype=np.float32)
            observation_high = np.full(initial_output.shape, np.inf, dtype=np.float32)
        self.action_space = gym.spaces.Box(action_low, action_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            observation_low, observation_high, dtype=np.float32
        )
        self._state = np.asarray(model.initial_state(), dtype=float)
        self._step_index = 0
        self.disturbances = dict(preset.config.get("disturbances", {}))

    @property
    def state(self) -> np.ndarray:
        if self._delegate is not None:
            return np.asarray(self._delegate.integ.x, dtype=float).copy()
        return self._state.copy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self._delegate is not None:
            observation, info = self._delegate.reset(seed=seed, options=options)
            self._step_index = 0
            return observation, self._delegate_info(info)
        del options
        sampler = getattr(self.model, "sample_initial_state", None)
        if callable(sampler):
            state = sampler(self.np_random, self.preset.config)
        else:
            state = self.model.initial_state()
        self._state = np.asarray(state, dtype=float).reshape(-1)
        self._step_index = 0
        observation = self._observation()
        return observation, self._info(
            commanded_action=None,
            applied_action=None,
            reward_terms={},
            constraints={},
        )

    def step(self, action):
        if self._delegate is not None:
            result = self._delegate.step(action)
            observation, reward, terminated, truncated, info = result
            self._step_index += 1
            return (
                observation,
                reward,
                terminated,
                truncated,
                self._delegate_info(info),
            )
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != self.action_space.shape:
            raise ValueError(
                f"action must have shape {self.action_space.shape}, got {values.shape}"
            )
        if not self.action_space.contains(values):
            raise ValueError("policy action must belong to env.action_space")
        previous = self._state.copy()
        self._state = self._integrate(previous, values)
        if not np.all(np.isfinite(self._state)):
            raise FloatingPointError("model produced a non-finite state")
        self._step_index += 1
        context = {
            "reference": self._reference(),
            "step_index": self._step_index,
            "physical_time": self._step_index * self.control_dt,
            "plant": self.plant,
            "preset": self.preset,
        }
        reward_result = self.task.reward(previous, values, self._state, context)
        if isinstance(reward_result, tuple):
            reward, reward_terms = reward_result
        else:
            reward, reward_terms = reward_result, {"reward": float(reward_result)}
        constraints = self._constraints()
        terminated = bool(any(float(value) > 0 for value in constraints.values()))
        truncated = self._step_index >= self.episode_steps
        info = self._info(
            commanded_action=values,
            applied_action=values,
            reward_terms=reward_terms,
            constraints=constraints,
        )
        return self._observation(), float(reward), terminated, truncated, info

    def _integrate(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        maximum_step = float(getattr(self.model, "dt_micro", self.control_dt))
        substeps = max(1, math.ceil(self.control_dt / maximum_step - 1e-12))
        step = self.control_dt / substeps

        def derivative(values):
            output = np.asarray(
                self.model.dynamics(
                    values,
                    action,
                    disturbances=self.disturbances,
                ),
                dtype=float,
            ).reshape(-1)
            if output.shape != state.shape:
                raise ValueError("model dynamics shape does not match state shape")
            return output

        result = state.copy()
        for _ in range(substeps):
            k1 = derivative(result)
            k2 = derivative(result + 0.5 * step * k1)
            k3 = derivative(result + 0.5 * step * k2)
            k4 = derivative(result + step * k3)
            result = result + (step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            clamp = getattr(self.model, "clamp_state", None)
            if callable(clamp):
                result = np.asarray(clamp(result), dtype=float)
        return result

    def close(self) -> None:
        if self._delegate is not None:
            self._delegate.close()

    def _delegate_info(self, info: Mapping[str, Any] | None) -> dict[str, Any]:
        enriched = dict(info or {})
        enriched.update(
            {
                "task_id": self.task.id,
                "task_hash": self.task.task_hash,
                "plant_id": self.plant.id,
                "plant_hash": self.plant.plant_hash,
                "preset": self.preset.id,
            }
        )
        return enriched

    def __getattr__(self, name: str) -> Any:
        delegate = self.__dict__.get("_delegate")
        if delegate is not None:
            return getattr(delegate, name)
        raise AttributeError(name)

    def _observation(self) -> np.ndarray:
        observation = np.asarray(self.model.outputs(self._state), dtype=np.float32)
        if observation.shape != self.observation_space.shape:
            raise ValueError("model output shape changed during the episode")
        return observation

    def _reference(self) -> np.ndarray:
        if self.task.reference:
            return np.asarray(self.task.reference, dtype=np.float32)
        return np.asarray(self.model.default_setpoint_vector(), dtype=np.float32)

    def _constraints(self) -> dict[str, float]:
        function = getattr(self.model, "constraint_costs", None)
        if not callable(function):
            return {}
        return {
            str(key): float(value)
            for key, value in function(self._state, self.disturbances).items()
        }

    def _info(
        self,
        *,
        commanded_action: np.ndarray | None,
        applied_action: np.ndarray | None,
        reward_terms: Mapping[str, float],
        constraints: Mapping[str, float],
    ) -> dict[str, Any]:
        info = {
            "task_id": self.task.id,
            "task_hash": self.task.task_hash,
            "plant_id": self.plant.id,
            "plant_hash": self.plant.plant_hash,
            "preset": self.preset.id,
            "step_index": self._step_index,
            "physical_time": self._step_index * self.control_dt,
            "true_state": self.state,
            "reference": self._reference(),
            "commanded_action": None if commanded_action is None else commanded_action.copy(),
            "applied_action": None if applied_action is None else applied_action.copy(),
            "reward_terms": dict(reward_terms),
            "constraint_costs": dict(constraints),
            "disturbance": dict(getattr(self, "disturbances", {})),
        }
        step_info = getattr(self.model, "step_info", None)
        if callable(step_info):
            info.update(
                step_info(
                    self._state,
                    applied_action,
                    dict(getattr(self, "disturbances", {})),
                )
            )
        return info


def make_env(
    task: str,
    *,
    plant: PlantConfig | str | Path | Mapping[str, Any] | None = None,
    preset: str | None = None,
) -> ProcessControlEnv:
    task_spec = get_task(task)
    plugin = get_scenario(task_spec.scenario)
    resolved_plant = resolve_plant(task_spec.scenario, plant)
    preset_id = preset or task_spec.default_preset
    try:
        preset_spec = task_spec.presets[preset_id]
    except KeyError as error:
        raise KeyError(f"unknown preset {preset_id!r} for task {task!r}") from error
    model = plugin.make_model(resolved_plant)
    return ProcessControlEnv(model, task_spec, resolved_plant, preset_spec)


__all__ = ["ProcessControlEnv", "make_env", "resolve_plant"]
