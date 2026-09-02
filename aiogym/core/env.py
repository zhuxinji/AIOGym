"""Gymnasium environment for one process model and resolved episode."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from numbers import Real
from typing import Any

import gymnasium as gym
import numpy as np

from .contracts import ProcessModel, Scenario
from .model import apply_action_slew, integrate_process_state
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
    action_low, action_high = _bounds(model.action_schema())
    initial_action = np.asarray(episode.initial_action, dtype=np.float32)
    if initial_action.shape != action_low.shape:
        raise ValueError(f"episode initial_action must contain {len(action_low)} values")
    if np.any(initial_action < action_low) or np.any(initial_action > action_high):
        raise ValueError("episode initial_action must belong to the model action bounds")
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
    _validate_disturbance_values(model, episode.disturbances)
    for values in episode.disturbance_schedule.values():
        _validate_disturbance_values(model, values)
    return episode


def _validate_disturbance_values(
    model: ProcessModel, values: Mapping[str, float]
) -> dict[str, float]:
    defaults = model.default_disturbances()
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(f"unknown disturbances: {sorted(unknown)}")
    resolved = {str(name): float(value) for name, value in values.items()}
    if not all(math.isfinite(value) for value in resolved.values()):
        raise ValueError("disturbances must be finite")
    schema = {
        row["name"]: row for row in getattr(model, "input_disturbances", ())
    }
    for name, value in resolved.items():
        row = schema.get(name)
        if row is None:
            continue
        if "bounds" in row:
            lower, upper = (float(item) for item in row["bounds"])
            if value < lower or value > upper:
                raise ValueError(
                    f"disturbance {name!r} must be within [{lower}, {upper}]"
                )
        if row.get("unit") == "binary" and value not in (0.0, 1.0):
            raise ValueError(f"disturbance {name!r} must be binary")
    return resolved


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
            episode.initial_action, dtype=np.float32
        )
        if (
            self._previous_applied_action.shape != self.action_space.shape
            or not self.action_space.contains(self._previous_applied_action)
        ):
            raise ValueError("episode initial_action must belong to action_space")
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
        if self.benchmark is not None and supplied:
            raise ValueError("reset options cannot override a Benchmark episode")
        if self.benchmark is not None and seed is None:
            raise ValueError("Benchmark reset requires a case seed")
        if self.benchmark is not None:
            resolved_episode = self.benchmark.make_episode(self.model, seed)
        elif "episode" in supplied:
            resolved_episode = supplied["episode"]
        else:
            resolved_episode = self.default_episode
        episode = validate_episode(
            self.model,
            resolved_episode,
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
            episode.initial_action, dtype=np.float32
        )
        self.episode_parameters = (
            {"case_seed": int(seed)}
            if self.benchmark is not None
            else {"reset_seed": seed}
        )
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
        resolved = _validate_disturbance_values(self.model, values)
        self._disturbance_overrides = resolved
        self.disturbances.update(resolved)

    def _apply_action_slew(self, requested: np.ndarray) -> np.ndarray:
        return apply_action_slew(
            self.model,
            self._previous_applied_action,
            requested,
        ).astype(np.float32)

    def _integrate(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return integrate_process_state(
            self.model,
            state,
            action,
            self.disturbances,
            duration=self.control_dt,
        )

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
    heater: Sequence[int] | None = None,
    initial_state: Sequence[float] | None = None,
    benchmark: str | None = None,
    randomize: bool = False,
    boundary_probability: float = 0.0,
    disturbance: bool = False,
    disturbance_schedule: Mapping[int, Mapping[str, float]] | None = None,
    noise: bool | float | Mapping[str, Any] = False,
    delay: bool | Mapping[str, Any] = False,
    fault: bool | Mapping[str, Any] = False,
) -> gym.Env:
    if not isinstance(randomize, bool):
        raise TypeError("randomize must be a boolean")
    if isinstance(boundary_probability, bool) or not isinstance(
        boundary_probability, Real
    ):
        raise TypeError("boundary_probability must be a number")
    boundary_probability = float(boundary_probability)
    if not math.isfinite(boundary_probability) or not 0.0 <= boundary_probability <= 1.0:
        raise ValueError("boundary_probability must be between 0 and 1")
    if boundary_probability > 0.0 and not randomize:
        raise ValueError("boundary_probability requires randomize=True")
    if not isinstance(disturbance, bool):
        raise TypeError("disturbance must be a boolean")
    if disturbance_schedule is not None and not isinstance(
        disturbance_schedule, Mapping
    ):
        raise TypeError("disturbance_schedule must be a mapping or None")
    if disturbance_schedule is not None:
        for values in disturbance_schedule.values():
            if not isinstance(values, Mapping):
                raise TypeError("disturbance_schedule values must be mappings")
    if disturbance and disturbance_schedule is not None:
        raise ValueError(
            "disturbance_schedule cannot be combined with disturbance=True"
        )
    if initial_state is not None and isinstance(initial_state, (str, bytes)):
        raise TypeError("initial_state must be a numeric sequence or None")
    if initial_state is not None and randomize:
        raise ValueError("initial_state cannot be combined with randomize=True")
    if heater is not None and scenario != "cascade":
        raise ValueError("heater is supported only by the cascade scenario")
    definition = get_scenario(scenario)
    if benchmark is not None and (reward is not None or parameters is not None):
        raise ValueError(
            "benchmark fixes model parameters and reward; "
            "do not pass parameters or reward"
        )
    if benchmark is not None and initial_state is not None:
        raise ValueError(
            "benchmark fixes the initial state; do not pass initial_state"
        )
    resolved_benchmark = (
        None if benchmark is None else get_benchmark(scenario, benchmark)
    )
    reward_id = (
        resolved_benchmark.reward_id
        if resolved_benchmark is not None
        else definition.default_reward if reward is None else reward
    )
    resolved_reward = get_reward(scenario, reward_id)
    resolved_noise = resolve_noise_option(noise)
    resolved_delay = resolve_delay_option(delay)
    resolved_fault = resolve_fault_option(fault)
    if benchmark is not None and (
        randomize
        or boundary_probability > 0.0
        or disturbance
        or disturbance_schedule is not None
        or resolved_noise is not None
        or resolved_delay is not None
        or resolved_fault is not None
    ):
        raise ValueError(
            "benchmark cannot be combined with randomize, boundary_probability, "
            "disturbance, disturbance_schedule, noise, delay, or fault"
        )
    model = (
        definition.make_model(parameters, heater=heater)
        if scenario == "cascade"
        else definition.make_model(parameters)
    )
    episode = (
        definition.make_default_episode(model)
        if resolved_benchmark is None
        else resolved_benchmark.make_episode(model, 0)
    )
    if initial_state is not None:
        try:
            supplied_initial_state = tuple(initial_state)
        except TypeError as error:
            raise TypeError(
                "initial_state must be a numeric sequence or None"
            ) from error
        episode = replace(episode, initial_state=supplied_initial_state)
    if disturbance_schedule is not None:
        episode = replace(episode, disturbance_schedule=disturbance_schedule)
    base = ProcessControlEnv(
        model,
        resolved_reward,
        definition,
        episode,
        benchmark=resolved_benchmark,
    )
    runtime_parameters = dict(model.resolved_parameters)
    if scenario == "cascade":
        runtime_parameters["heater"] = list(model.heater)
    base.runtime_config = {
        "scenario": scenario,
        "reward": reward_id,
        "parameters": runtime_parameters,
        "initial_state": (
            None if initial_state is None else list(episode.initial_state)
        ),
        "benchmark": benchmark,
        "randomize": randomize,
        "boundary_probability": boundary_probability,
        "disturbance": True if disturbance else None,
        "disturbance_schedule": (
            None
            if disturbance_schedule is None
            else {
                str(step): dict(values)
                for step, values in episode.disturbance_schedule.items()
            }
        ),
        "noise": resolved_noise,
        "delay": resolved_delay,
        "fault": resolved_fault,
        "control_dt": definition.control_dt,
    }
    env: gym.Env = base
    if randomize or disturbance:
        env = EpisodeSamplingWrapper(
            env,
            episode_sampler=(
                definition.sample_training_episode if randomize else None
            ),
            disturbance_sampler=(
                definition.sample_training_disturbance if disturbance else None
            ),
            disturbance_schedule=(
                None if disturbance_schedule is None else episode.disturbance_schedule
            ),
            reward_id=reward_id,
            boundary_probability=boundary_probability,
        )
    if resolved_delay is not None or resolved_fault is not None:
        env = ActionChannelWrapper(
            env, delay=resolved_delay, fault=resolved_fault
        )
    if resolved_noise is not None or resolved_delay is not None:
        env = ObservationChannelWrapper(
            env, delay=resolved_delay, noise=resolved_noise
        )
    return env


__all__ = ["ProcessControlEnv", "make_env", "validate_episode"]
