"""Gymnasium environment for one process model and resolved episode."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np

from .contracts import ProcessModel, Scenario
from .registry import get_benchmark, get_reward, get_scenario
from .specs import Benchmark, EpisodeSpec, Reward
from .variations import (
    ActionChannelWrapper,
    EpisodeSamplingWrapper,
    ObservationChannelWrapper,
    resolve_delay_option,
    resolve_fault_option,
    resolve_noise_option,
)


def _bounds(schema: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray([item["low"] for item in schema], dtype=np.float32)
    high = np.asarray([item["high"] for item in schema], dtype=np.float32)
    if low.shape != high.shape or np.any(low > high):
        raise ValueError("schema contains invalid bounds")
    return low, high


def validate_episode(model: ProcessModel, episode: EpisodeSpec) -> EpisodeSpec:
    if not isinstance(episode, EpisodeSpec):
        raise TypeError("episode must be an EpisodeSpec")
    state_low, state_high = _bounds(model.state_schema())
    initial_state = np.asarray(episode.initial_state, dtype=np.float32)
    if initial_state.shape != state_low.shape:
        raise ValueError(f"episode initial_state must contain {len(state_low)} values")
    if np.any(initial_state < state_low) or np.any(initial_state > state_high):
        raise ValueError("episode initial_state must belong to the model state bounds")
    output_low, output_high = _bounds(model.output_schema())
    reference = np.asarray(episode.reference, dtype=np.float32)
    if reference.shape != output_low.shape:
        raise ValueError(f"episode reference must contain {len(output_low)} values")
    all_references = [reference]
    all_references.extend(
        np.asarray(values, dtype=np.float32)
        for values in episode.reference_schedule.values()
    )
    if any(np.any(values < output_low) or np.any(values > output_high) for values in all_references):
        raise ValueError("episode references must belong to model output bounds")
    known_disturbances = set(model.default_disturbances())
    supplied = set(episode.disturbances)
    supplied.update(
        name
        for values in episode.disturbance_schedule.values()
        for name in values
    )
    unknown = supplied - known_disturbances
    if unknown:
        raise ValueError(f"unknown disturbances: {sorted(unknown)}")
    return episode


class ProcessControlEnv(gym.Env):
    """Deterministic process simulation with physical actions."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        model: ProcessModel,
        reward: Reward,
        scenario: Scenario,
        episode: EpisodeSpec,
        *,
        benchmark: Benchmark | None,
    ) -> None:
        super().__init__()
        self.model = model
        self.reward = reward
        self.scenario = scenario
        self.benchmark = benchmark
        self.control_dt = scenario.control_dt
        self.default_episode = validate_episode(model, episode)
        self.episode = self.default_episode
        self.episode_steps = episode.horizon
        self.episode_family = (
            "default" if benchmark is None else f"benchmark:{benchmark.id}"
        )
        action_low, action_high = _bounds(model.action_schema())
        observation_low, observation_high = _bounds(model.observation_schema())
        self.action_space = gym.spaces.Box(action_low, action_high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            observation_low, observation_high, dtype=np.float32
        )
        self._state = np.asarray(episode.initial_state, dtype=float)
        self._step_index = 0
        self._disturbance_overrides: dict[str, float] = {}
        self.disturbances: dict[str, float] = {}
        self._reference_state = np.asarray(episode.reference, dtype=float)
        self._reference_events: dict[int, tuple[float, ...]] = {}
        self._disturbance_events: dict[int, Mapping[str, float]] = {}
        self._previous_applied_action = np.asarray(
            model.default_action(), dtype=np.float32
        )
        if (
            self._previous_applied_action.shape != self.action_space.shape
            or not self.action_space.contains(self._previous_applied_action)
        ):
            raise ValueError("model default action must belong to action_space")
        self.episode_parameters: dict[str, Any] = {}
        self.runtime_config: dict[str, Any] = {}
        self.runtime_variation: dict[str, Any] = {}

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def y_sp(self) -> np.ndarray:
        return self._reference_state.copy()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        supplied = {} if options is None else options
        if not isinstance(supplied, Mapping):
            raise TypeError("reset options must be a mapping")
        unknown = set(supplied) - {"episode"}
        if unknown:
            raise ValueError(f"unknown reset options: {sorted(unknown)}")
        episode = validate_episode(
            self.model,
            self.default_episode if "episode" not in supplied else supplied["episode"],
        )
        self.episode = episode
        self.episode_steps = episode.horizon
        self._state = np.asarray(episode.initial_state, dtype=float)
        self._step_index = 0
        self.disturbances = {
            **dict(self.model.default_disturbances()),
            **dict(episode.disturbances),
            **self._disturbance_overrides,
        }
        self._reference_state = np.asarray(episode.reference, dtype=float)
        self._reference_events = dict(episode.reference_schedule)
        self._disturbance_events = dict(episode.disturbance_schedule)
        self._apply_events()
        self._previous_applied_action = np.asarray(
            self.model.default_action(), dtype=np.float32
        )
        self.episode_parameters = {"reset_seed": seed}
        self.runtime_variation = {}
        constraints = self._constraints()
        margins = self._safety_margins()
        observation = self._observation()
        return observation, self._info(
            commanded_action=None,
            channel_action=None,
            applied_action=None,
            reward_terms={},
            constraints=constraints,
            safety_margins=margins,
            transition_reference=None,
            transition_disturbance=None,
            transition_step_index=None,
        )

    def step(self, action):
        commanded = np.asarray(action, dtype=np.float32).reshape(-1)
        if commanded.shape != self.action_space.shape:
            raise ValueError(
                f"action must have shape {self.action_space.shape}, got {commanded.shape}"
            )
        if not self.action_space.contains(commanded):
            raise ValueError("action must belong to env.action_space")
        transition_reference = self._reference()
        transition_disturbance = dict(self.disturbances)
        transition_step_index = self._step_index
        previous_state = self._state.copy()
        previous_applied = self._previous_applied_action.copy()
        applied = self._apply_action_slew(commanded)
        self._state = self._integrate(previous_state, applied)
        if not np.isfinite(self._state).all():
            raise FloatingPointError("model produced a non-finite state")

        constraints = self._constraints(transition_disturbance)
        safety_margins = self._safety_margins(transition_disturbance)
        terminated = bool(any(float(value) > 0 for value in constraints.values()))
        context = {
            "reference": transition_reference.copy(),
            "step_index": transition_step_index,
            "physical_time": transition_step_index * self.control_dt,
            "scenario": self.scenario,
            "benchmark": self.benchmark,
            "episode": self.episode,
            "model": self.model,
            "reward": self.reward,
            "control_dt": self.control_dt,
            "disturbances": transition_disturbance,
            "commanded_action": commanded.copy(),
            "applied_action": applied.copy(),
            "previous_applied_action": previous_applied,
            "constraint_costs": constraints,
            "safety_margins": safety_margins,
        }
        reward_result = self.reward.function(
            previous_state, applied, self._state, context
        )
        if isinstance(reward_result, tuple):
            reward, raw_terms = reward_result
            reward_terms = {str(key): float(value) for key, value in raw_terms.items()}
        else:
            reward = reward_result
            reward_terms = {"reward": float(reward_result)}
        reward = float(reward)
        if not math.isfinite(reward):
            raise FloatingPointError("reward function produced a non-finite value")
        if terminated and self.reward.safety_violation_penalty:
            penalty = self.reward.safety_violation_penalty
            reward -= penalty
            reward_terms["safety"] = -penalty

        self._step_index += 1
        truncated = self._step_index >= self.episode_steps
        if not (terminated or truncated):
            self._apply_events()
        info = self._info(
            commanded_action=commanded,
            channel_action=commanded,
            applied_action=applied,
            reward_terms=reward_terms,
            constraints=constraints,
            safety_margins=safety_margins,
            transition_reference=transition_reference,
            transition_disturbance=transition_disturbance,
            transition_step_index=transition_step_index,
        )
        self._previous_applied_action = applied.copy()
        return self._observation(), reward, terminated, truncated, info

    def set_disturbances(self, values: Mapping[str, float]) -> None:
        defaults = self.model.default_disturbances()
        unknown = set(values) - set(defaults)
        if unknown:
            raise ValueError(f"unknown disturbances: {sorted(unknown)}")
        resolved = {str(name): float(value) for name, value in values.items()}
        if not all(math.isfinite(value) for value in resolved.values()):
            raise ValueError("disturbances must be finite")
        self._disturbance_overrides = resolved
        self.disturbances.update(resolved)

    def _apply_action_slew(self, requested: np.ndarray) -> np.ndarray:
        limits = self.model.action_slew_limits()
        if limits is None:
            return requested.copy()
        maximum_step = np.asarray(limits, dtype=float).reshape(-1)
        if (
            maximum_step.shape != requested.shape
            or not np.isfinite(maximum_step).all()
            or np.any(maximum_step < 0)
        ):
            raise ValueError("model action slew limits must match actions")
        return np.clip(
            requested,
            self._previous_applied_action - maximum_step,
            self._previous_applied_action + maximum_step,
        ).astype(np.float32)

    def _integrate(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        maximum_step = float(self.model.dt_micro)
        substeps = max(1, math.ceil(self.control_dt / maximum_step - 1e-12))
        step = self.control_dt / substeps

        def derivative(values):
            output = np.asarray(
                self.model.dynamics(values, action, disturbances=self.disturbances),
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
            result = np.asarray(self.model.clamp_state(result), dtype=float)
        return result

    def _observation(self) -> np.ndarray:
        observation = np.asarray(
            self.model.observation(
                self._state,
                self._reference(),
                self._previous_applied_action,
                self.disturbances,
            ),
            dtype=float,
        )
        if observation.shape != self.observation_space.shape:
            raise ValueError("model observation shape changed during the episode")
        if not np.isfinite(observation).all():
            raise FloatingPointError("model produced a non-finite observation")
        return np.clip(
            observation,
            self.observation_space.low,
            self.observation_space.high,
        ).astype(np.float32)

    def _reference(self) -> np.ndarray:
        return self._reference_state.copy()

    def _apply_events(self) -> None:
        if self._step_index in self._reference_events:
            self._reference_state = np.asarray(
                self._reference_events[self._step_index], dtype=float
            )
        if self._step_index in self._disturbance_events:
            self.disturbances.update(self._disturbance_events[self._step_index])

    def _constraints(
        self, disturbances: Mapping[str, float] | None = None
    ) -> dict[str, float]:
        return {
            str(key): float(value)
            for key, value in self.model.constraint_costs(
                self._state,
                self.disturbances if disturbances is None else disturbances,
            ).items()
        }

    def _safety_margins(
        self, disturbances: Mapping[str, float] | None = None
    ) -> dict[str, float]:
        margins = {
            str(key): float(value)
            for key, value in self.model.safety_margins(
                self._state,
                self.disturbances if disturbances is None else disturbances,
            ).items()
        }
        if not margins or not all(math.isfinite(value) for value in margins.values()):
            raise ValueError("model safety_margins must contain finite values")
        return margins

    def _info(
        self,
        *,
        commanded_action: np.ndarray | None,
        channel_action: np.ndarray | None,
        applied_action: np.ndarray | None,
        reward_terms: Mapping[str, float],
        constraints: Mapping[str, float],
        safety_margins: Mapping[str, float],
        transition_reference: np.ndarray | None,
        transition_disturbance: Mapping[str, float] | None,
        transition_step_index: int | None,
    ) -> dict[str, Any]:
        info = {
            "scenario_id": self.scenario.id,
            "benchmark_id": None if self.benchmark is None else self.benchmark.id,
            "reward_id": self.reward.id,
            "step_index": self._step_index,
            "physical_time": self._step_index * self.control_dt,
            "true_state": self.state,
            "y": np.asarray(self.model.outputs(self._state), dtype=float),
            "reference": self._reference(),
            "commanded_action": None if commanded_action is None else commanded_action.copy(),
            "channel_action": None if channel_action is None else channel_action.copy(),
            "applied_action": None if applied_action is None else applied_action.copy(),
            "previous_applied_action": self._previous_applied_action.copy(),
            "episode_spec": self.episode.as_dict(),
            "episode_family": self.episode_family,
            "episode_parameters": dict(self.episode_parameters),
            "runtime_variation": dict(self.runtime_variation),
            "reward_terms": dict(reward_terms),
            "constraint_costs": dict(constraints),
            "safety_margins": dict(safety_margins),
            "minimum_safety_margin": min(safety_margins.values()),
            "disturbance": dict(self.disturbances),
            "transition_reference": (
                None if transition_reference is None else transition_reference.copy()
            ),
            "transition_disturbance": (
                None if transition_disturbance is None else dict(transition_disturbance)
            ),
            "transition_step_index": transition_step_index,
        }
        info.update(
            self.model.step_info(
                self._state,
                applied_action,
                self.disturbances
                if transition_disturbance is None
                else transition_disturbance,
            )
        )
        return info

    def close(self) -> None:
        return None


def make_env(
    scenario: str,
    *,
    reward: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    benchmark: str | None = None,
    randomize: bool = False,
    noise: bool | float | Mapping[str, Any] = False,
    delay: bool | Mapping[str, Any] = False,
    fault: bool | Mapping[str, Any] = False,
) -> gym.Env:
    if not isinstance(randomize, bool):
        raise TypeError("randomize must be a boolean")
    definition = get_scenario(scenario)
    if benchmark is not None and (reward is not None or parameters is not None):
        raise ValueError(
            "benchmark fixes the scenario default model parameters and reward; "
            "do not pass parameters or reward"
        )
    reward_id = definition.default_reward if reward is None else reward
    resolved_reward = get_reward(scenario, reward_id)
    resolved_noise = resolve_noise_option(noise)
    resolved_delay = resolve_delay_option(delay)
    resolved_fault = resolve_fault_option(fault)
    if benchmark is not None and (
        randomize
        or resolved_noise is not None
        or resolved_delay is not None
        or resolved_fault is not None
    ):
        raise ValueError(
            "benchmark cannot be combined with randomize, noise, delay, or fault"
        )
    model = definition.make_model(parameters)
    resolved_benchmark = (
        None if benchmark is None else get_benchmark(scenario, benchmark)
    )
    benchmark_noise = (
        None
        if resolved_benchmark is None or resolved_benchmark.measurement_noise is None
        else dict(resolved_benchmark.measurement_noise)
    )
    effective_noise = resolved_noise if resolved_benchmark is None else benchmark_noise
    episode = (
        definition.make_default_episode(model)
        if resolved_benchmark is None
        else resolved_benchmark.make_episode(model)
    )
    base = ProcessControlEnv(
        model,
        resolved_reward,
        definition,
        episode,
        benchmark=resolved_benchmark,
    )
    base.runtime_config = {
        "scenario": scenario,
        "reward": reward_id,
        "parameters": dict(model.resolved_parameters),
        "benchmark": benchmark,
        "randomize": randomize,
        "noise": effective_noise,
        "delay": resolved_delay,
        "fault": resolved_fault,
        "control_dt": definition.control_dt,
    }
    env: gym.Env = base
    if randomize:
        env = EpisodeSamplingWrapper(env, definition.sample_training_episode)
    if resolved_delay is not None or resolved_fault is not None:
        env = ActionChannelWrapper(
            env, delay=resolved_delay, fault=resolved_fault
        )
    if effective_noise is not None or resolved_delay is not None:
        env = ObservationChannelWrapper(
            env, delay=resolved_delay, noise=effective_noise
        )
    return env


__all__ = ["ProcessControlEnv", "make_env", "validate_episode"]
