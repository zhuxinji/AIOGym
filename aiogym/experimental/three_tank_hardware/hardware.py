"""Experimental hardware backend with fail-closed safety and shadow operation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from aiogym.core.env import make_env

from .calibration import validate_calibration
from .real_log import RealLogWriter, build_real_step_record


HARDWARE_BACKEND_VERSION = "aiogym.three_tank.hardware.v6"
DEFAULT_HARDWARE_MAXIMUM_ACTION_STEP = (0.05, 0.08, 0.08, 0.08, 0.05)


@dataclass(frozen=True)
class HardwareSample:
    """One engineering-unit sample returned by an injected transport."""

    measurement: tuple[float, ...]
    flow_measurement: tuple[float, ...]
    source_monotonic_time_s: float
    received_monotonic_time_s: float
    wall_time_utc: str
    applied_action: tuple[float, ...] | None = None
    boundary: Mapping[str, float] | None = None
    interlocks: Mapping[str, bool] | None = None
    raw_channels: Mapping[str, Any] | None = None

    def __post_init__(self):
        measurement = _finite_vector("measurement", self.measurement, 6)
        flow_measurement = _finite_vector(
            "flow_measurement", self.flow_measurement, 3
        )
        if any(value < 0.0 for value in flow_measurement):
            raise ValueError("flow_measurement must be non-negative")
        applied = (
            None
            if self.applied_action is None
            else _finite_vector("applied_action", self.applied_action, 5)
        )
        timestamp_values = {
            "source_monotonic_time_s": self.source_monotonic_time_s,
            "received_monotonic_time_s": self.received_monotonic_time_s,
        }
        for name, raw_value in timestamp_values.items():
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if not str(self.wall_time_utc).strip():
            raise ValueError("wall_time_utc must be non-empty")
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "flow_measurement", flow_measurement)
        object.__setattr__(self, "applied_action", applied)
        boundary = {} if self.boundary is None else dict(self.boundary)
        if "reservoir_temperature_degC" not in boundary:
            raise ValueError(
                "boundary must contain reservoir_temperature_degC"
            )
        reservoir_temperature = float(
            boundary["reservoir_temperature_degC"]
        )
        if not math.isfinite(reservoir_temperature):
            raise ValueError(
                "boundary reservoir_temperature_degC must be finite"
            )
        boundary["reservoir_temperature_degC"] = reservoir_temperature
        object.__setattr__(self, "boundary", boundary)
        interlocks = {} if self.interlocks is None else dict(self.interlocks)
        required_interlocks = {
            "emergency_stop",
            "reservoir_available",
            "watchdog_healthy",
        }
        missing_interlocks = sorted(required_interlocks - set(interlocks))
        if missing_interlocks:
            raise ValueError(
                f"interlocks are missing required fields: {missing_interlocks}"
            )
        for name in required_interlocks:
            if not isinstance(interlocks[name], bool):
                raise TypeError(f"interlock {name} must be bool")
        object.__setattr__(self, "interlocks", interlocks)
        object.__setattr__(
            self,
            "raw_channels",
            {} if self.raw_channels is None else dict(self.raw_channels),
        )


class HardwareTransport(Protocol):
    """PLC/DAQ adapters implement this boundary outside AIO-Gym."""

    def reset(self) -> HardwareSample: ...

    def exchange(self, action: Sequence[float] | None) -> HardwareSample: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class SafetyConfig:
    low_heater_level_m: float = 0.11
    high_pump_level_m: float = 0.415
    high_heater_temperature_degC: float = 60.0
    maximum_sample_age_s: float = 2.0
    maximum_action_step: tuple[float, ...] = DEFAULT_HARDWARE_MAXIMUM_ACTION_STEP
    emergency_action: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0, 0.0)

    def __post_init__(self):
        limits = {
            "low_heater_level_m": self.low_heater_level_m,
            "high_pump_level_m": self.high_pump_level_m,
            "high_heater_temperature_degC": self.high_heater_temperature_degC,
            "maximum_sample_age_s": self.maximum_sample_age_s,
        }
        for name, raw_value in limits.items():
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        maximum_step = _finite_vector(
            "maximum_action_step", self.maximum_action_step, 5
        )
        emergency = _finite_vector("emergency_action", self.emergency_action, 5)
        if any(value < 0.0 for value in maximum_step):
            raise ValueError("maximum_action_step must be non-negative")
        if any(value < 0.0 or value > 1.0 for value in emergency):
            raise ValueError("emergency_action must belong to [0, 1]")
        object.__setattr__(self, "maximum_action_step", maximum_step)
        object.__setattr__(self, "emergency_action", emergency)


@dataclass(frozen=True)
class SafetyDecision:
    requested_action: tuple[float, ...]
    applied_action: tuple[float, ...]
    reasons: tuple[str, ...]
    emergency: bool

    def as_dict(self):
        return {
            "requested_action": list(self.requested_action),
            "applied_action": list(self.applied_action),
            "reasons": list(self.reasons),
            "emergency": self.emergency,
        }


class SafetyGuardian:
    """Software guard; it complements, never replaces, hardwired interlocks."""

    def __init__(self, config: SafetyConfig | None = None):
        self.config = SafetyConfig() if config is None else config

    def apply(self, requested, *, sample: HardwareSample, previous_action):
        requested_action = np.clip(
            np.asarray(_finite_vector("requested_action", requested, 5)),
            0.0,
            1.0,
        )
        previous = np.asarray(
            _finite_vector("previous_action", previous_action, 5), dtype=float
        )
        age = sample.received_monotonic_time_s - sample.source_monotonic_time_s
        interlocks = sample.interlocks
        hard_reasons = []
        if age < 0.0 or age > self.config.maximum_sample_age_s:
            hard_reasons.append("stale_measurement")
        if bool(interlocks["emergency_stop"]):
            hard_reasons.append("emergency_stop")
        if not bool(interlocks["watchdog_healthy"]):
            hard_reasons.append("watchdog_not_healthy")
        if hard_reasons:
            return SafetyDecision(
                requested_action=tuple(requested_action.tolist()),
                applied_action=self.config.emergency_action,
                reasons=tuple(hard_reasons),
                emergency=True,
            )

        maximum_step = np.asarray(self.config.maximum_action_step, dtype=float)
        applied = np.clip(requested_action, previous - maximum_step, previous + maximum_step)
        reasons = []
        levels = np.asarray(sample.measurement[0::2], dtype=float)
        temperatures = np.asarray(sample.measurement[1::2], dtype=float)
        if np.any(levels >= self.config.high_pump_level_m):
            applied[0] = 0.0
            reasons.append("high_level_pump_inhibit")
        if (
            levels[0] <= self.config.low_heater_level_m
            or temperatures[0] >= self.config.high_heater_temperature_degC
        ):
            applied[4] = 0.0
            reasons.append("heater_inhibit")
        if not bool(interlocks["reservoir_available"]):
            applied[0] = 0.0
            reasons.append("reservoir_pump_inhibit")
        return SafetyDecision(
            requested_action=tuple(requested_action.tolist()),
            applied_action=tuple(applied.tolist()),
            reasons=tuple(reasons),
            emergency=False,
        )


class ThreeTankHardwareEnv(gym.Env):
    """Gym-compatible Three-Tank environment over an injected real transport."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        transport: HardwareTransport,
        *,
        mode: str = "shadow",
        calibration: Mapping[str, Any],
        guardian: SafetyGuardian | None = None,
        armed: bool = False,
        log_writer: RealLogWriter | None = None,
        run_id: str = "three-tank-real",
        episode_id: str = "episode-0",
    ):
        super().__init__()
        resolved_mode = str(mode).strip().lower()
        if resolved_mode not in {"shadow", "closed-loop"}:
            raise ValueError("hardware mode must be 'shadow' or 'closed-loop'")
        if resolved_mode == "closed-loop" and not armed:
            raise ValueError("closed-loop hardware mode requires armed=True")
        validated_calibration = validate_calibration(
            calibration, require_measured=resolved_mode == "closed-loop"
        )
        base_env = make_env("three_tank", benchmark="tracking")
        base = base_env.unwrapped
        self.transport = transport
        self.mode = resolved_mode
        self.calibration = validated_calibration
        self.guardian = SafetyGuardian() if guardian is None else guardian
        self.log_writer = log_writer
        self.run_id = str(run_id)
        self.episode_id = str(episode_id)
        self.model = base.model
        self.reward = base.reward
        self.scenario = base.scenario
        self.benchmark = base.benchmark
        self.episode = base.episode
        self.control_dt = base.control_dt
        self.episode_steps = base.episode_steps
        self.action_space = base.action_space
        self.observation_space = base.observation_space
        self.episode_family = "benchmark:tracking"
        self.runtime_variation = {}
        self.runtime_config = {
            "scenario": "three_tank",
            "reward": self.reward.id,
            "parameters": dict(self.model.resolved_parameters),
            "benchmark": "tracking",
            "randomize": False,
            "disturbance": None,
            "noise": None,
            "delay": None,
            "fault": None,
            "control_dt": self.control_dt,
        }
        base_env.close()
        self._step_index = 0
        self._sample: HardwareSample | None = None
        self._reference_state = np.asarray(self.episode.reference, dtype=float)
        self.disturbances = self.model.default_disturbances()
        self._previous_applied_action = np.asarray(
            self.episode.initial_action, dtype=float
        )

    @property
    def state(self):
        if self._sample is None:
            raise RuntimeError("hardware environment has not been reset")
        return _measurement_to_state(
            self._sample.measurement,
            self._sample.boundary,
        )

    @property
    def y_sp(self):
        return self._reference_state.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        self._step_index = 0
        self._reference_state = np.asarray(self.episode.reference, dtype=float)
        self._sample = _require_sample(self.transport.reset())
        self._update_disturbances(self._sample)
        if self._sample.applied_action is not None:
            self._previous_applied_action = np.asarray(
                self._sample.applied_action, dtype=float
            )
        else:
            self._previous_applied_action = np.asarray(
                self.model.default_action(), dtype=float
            )
        return self._observation(), self._info(
            commanded=None,
            recommended=None,
            applied=self._previous_applied_action,
            safety=None,
            reward_terms={},
        )

    def step(self, action):
        if self._sample is None:
            raise RuntimeError("reset must be called before step")
        commanded = np.asarray(action, dtype=np.float32).reshape(-1)
        if commanded.shape != self.action_space.shape or not self.action_space.contains(
            commanded
        ):
            raise ValueError("policy action must belong to hardware action_space")
        previous_state = self.state
        previous_applied_action = self._previous_applied_action.copy()
        transition_reference = self._reference_state.copy()
        transition_disturbance = dict(self.disturbances)
        reference_feasibility = self._reference_feasibility(
            reference=transition_reference
        )
        if reference_feasibility["accepted"]:
            recommended = commanded.astype(float)
            decision = self.guardian.apply(
                recommended,
                sample=self._sample,
                previous_action=self._previous_applied_action,
            )
        else:
            recommended = np.asarray(self.guardian.config.emergency_action, dtype=float)
            decision = SafetyDecision(
                requested_action=tuple(recommended.tolist()),
                applied_action=self.guardian.config.emergency_action,
                reasons=("infeasible_reference",),
                emergency=True,
            )
        next_sample = _require_sample(
            self.transport.exchange(
                decision.applied_action if self.mode == "closed-loop" else None
            )
        )
        self._sample = next_sample
        self._update_disturbances(next_sample)
        actual = np.asarray(
            next_sample.applied_action
            if next_sample.applied_action is not None
            else (
                decision.applied_action
                if self.mode == "closed-loop"
                else self._previous_applied_action
            ),
            dtype=float,
        )
        process_constraints = self.model.constraint_costs(
            self.state, self.disturbances
        )
        if decision.emergency:
            process_constraints["software_safety_emergency"] = 1.0
        safety_margins = self.model.safety_margins(
            self.state, self.disturbances
        )
        terminated = bool(
            any(float(value) > 0.0 for value in process_constraints.values())
        )
        context = {
            "reference": transition_reference,
            "step_index": self._step_index,
            "physical_time": self._step_index * self.control_dt,
            "scenario": self.scenario,
            "benchmark": self.benchmark,
            "episode": self.episode,
            "model": self.model,
            "reward": self.reward,
            "control_dt": self.control_dt,
            "disturbances": transition_disturbance,
            "commanded_action": commanded.copy(),
            "resolved_action": recommended.copy(),
            "applied_action": actual.copy(),
            "previous_applied_action": self._previous_applied_action.copy(),
            "constraint_costs": process_constraints,
            "safety_margins": safety_margins,
        }
        reward_result = self.reward.function(
            previous_state, actual, self.state, context
        )
        if isinstance(reward_result, tuple):
            reward, reward_terms = reward_result
        else:
            reward, reward_terms = reward_result, {"reward": float(reward_result)}
        if terminated and self.reward.safety_violation_penalty:
            reward -= self.reward.safety_violation_penalty
            reward_terms = {
                **dict(reward_terms),
                "safety": -self.reward.safety_violation_penalty,
            }
        self._step_index += 1
        if self._step_index in self.episode.reference_schedule:
            self._reference_state = np.asarray(
                self.episode.reference_schedule[self._step_index], dtype=float
            )
        truncated = self._step_index >= self.episode_steps
        self._previous_applied_action = actual.copy()
        info = self._info(
            commanded=commanded,
            recommended=recommended,
            applied=actual,
            previous_applied=previous_applied_action,
            safety=decision,
            reward_terms=reward_terms,
            transition_reference=transition_reference,
            transition_disturbance=transition_disturbance,
            transition_step_index=self._step_index - 1,
            transition_reference_feasibility=reference_feasibility,
        )
        self._write_log(
            commanded,
            actual,
            decision,
            reference=transition_reference,
            transition_disturbance=transition_disturbance,
        )
        return self._observation(), float(reward), terminated, truncated, info

    def _reference_feasibility(self, *, reference=None):
        target = np.asarray(
            self._reference_state if reference is None else reference, dtype=float
        )
        maximum_action = 0.95
        action = self.model.tracking_steady_state_action(
            target,
            self.disturbances,
        )
        equilibrium_maximum = math.inf if action is None else float(max(action))
        accepted = bool(action is not None and equilibrium_maximum <= maximum_action)
        reasons = (
            []
            if action is not None
            else ["reference has no feasible steady action"]
        )
        if action is not None and equilibrium_maximum > maximum_action:
            reasons.append("equilibrium action exceeds the hardware limit")
        return {
            "accepted": accepted,
            "maximum_equilibrium_action": equilibrium_maximum,
            "hardware_action_limit": maximum_action,
            "reasons": reasons,
        }

    def _observation(self):
        values = self.model.observation(
            self.state,
            self._reference_state,
            self._previous_applied_action,
            self.disturbances,
        )
        return np.clip(
            np.asarray(values, dtype=np.float32),
            self.observation_space.low,
            self.observation_space.high,
        )

    def _update_disturbances(self, sample):
        known = set(self.model.default_disturbances())
        self.disturbances.update(
            {
                name: float(value)
                for name, value in sample.boundary.items()
                if name in known
            }
        )

    def _info(
        self,
        *,
        commanded,
        recommended,
        applied,
        previous_applied=None,
        safety,
        reward_terms,
        transition_reference=None,
        transition_disturbance=None,
        transition_step_index=None,
        transition_reference_feasibility=None,
    ):
        constraints = self.model.constraint_costs(self.state, self.disturbances)
        if safety is not None and safety.emergency:
            constraints["software_safety_emergency"] = 1.0
        safety_margins = self.model.safety_margins(
            self.state, self.disturbances
        )
        return {
            "backend_version": HARDWARE_BACKEND_VERSION,
            "backend_kind": "real",
            "hardware_mode": self.mode,
            "scenario_id": self.scenario.id,
            "benchmark_id": self.benchmark.id,
            "reward_id": self.reward.id,
            "episode_spec": self.episode.as_dict(),
            "episode_family": self.episode_family,
            "episode_parameters": {},
            "runtime_variation": dict(self.runtime_variation),
            "calibration_id": self.calibration["calibration_id"],
            "calibration_hash": self.calibration["calibration_hash"],
            "step_index": self._step_index,
            "physical_time": self._step_index * self.control_dt,
            "true_state": None,
            "measured_state": self.state.copy(),
            "y": np.asarray(self.model.outputs(self.state), dtype=float),
            "flow_measurement_m3s": np.asarray(
                self._sample.flow_measurement, dtype=float
            ),
            "reference": self._reference_state.copy(),
            "transition_reference": (
                None
                if transition_reference is None
                else np.asarray(transition_reference).copy()
            ),
            "transition_disturbance": (
                None
                if transition_disturbance is None
                else dict(transition_disturbance)
            ),
            "transition_step_index": transition_step_index,
            "transition_reference_feasibility": (
                None
                if transition_reference_feasibility is None
                else dict(transition_reference_feasibility)
            ),
            "commanded_action": None if commanded is None else np.asarray(commanded).copy(),
            "channel_action": (
                None if recommended is None else np.asarray(recommended).copy()
            ),
            "resolved_action": None if recommended is None else np.asarray(recommended).copy(),
            "applied_action": np.asarray(applied).copy(),
            "previous_applied_action": np.asarray(
                applied if previous_applied is None else previous_applied
            ).copy(),
            "reward_terms": dict(reward_terms),
            "constraint_costs": constraints,
            "safety_margins": safety_margins,
            "minimum_safety_margin": min(safety_margins.values()),
            "disturbance": dict(self.disturbances),
            "reference_feasibility": self._reference_feasibility(),
            "safety": None if safety is None else safety.as_dict(),
            "source_monotonic_time_s": self._sample.source_monotonic_time_s,
            "received_monotonic_time_s": self._sample.received_monotonic_time_s,
        }

    def _write_log(
        self,
        commanded,
        applied,
        decision,
        *,
        reference,
        transition_disturbance,
    ):
        if self.log_writer is None:
            return
        self.log_writer.append(
            build_real_step_record(
                run_id=self.run_id,
                episode_id=self.episode_id,
                sequence=self._step_index - 1,
                source_monotonic_time_s=self._sample.source_monotonic_time_s,
                received_monotonic_time_s=self._sample.received_monotonic_time_s,
                wall_time_utc=self._sample.wall_time_utc,
                measurement=self.state,
                flow_measurement=self._sample.flow_measurement,
                reference=reference,
                transition_disturbance=transition_disturbance,
                disturbance=self.disturbances,
                commanded_action=commanded,
                applied_action=applied,
                calibration_id=self.calibration["calibration_id"],
                calibration_hash=self.calibration["calibration_hash"],
                scenario_id=self.scenario.id,
                benchmark_id=self.benchmark.id,
                reward_id=self.reward.id,
                backend_version=HARDWARE_BACKEND_VERSION,
                hardware_mode=self.mode,
                safety={
                    **decision.as_dict(),
                    "mode": self.mode,
                },
                raw_channels=self._sample.raw_channels,
            )
        )

    def close(self):
        self.transport.close()


def _measurement_to_state(measurement, boundary):
    return np.asarray(
        (
            *map(float, measurement),
            float(boundary["reservoir_temperature_degC"]),
        ),
        dtype=float,
    )


def _finite_vector(name, values, length):
    resolved = tuple(float(value) for value in values)
    if len(resolved) != length or not all(math.isfinite(value) for value in resolved):
        raise ValueError(f"{name} must contain {length} finite values")
    return resolved


def _require_sample(value):
    if not isinstance(value, HardwareSample):
        raise TypeError("hardware transport must return HardwareSample")
    return value


__all__ = [
    "HARDWARE_BACKEND_VERSION",
    "HardwareSample",
    "HardwareTransport",
    "SafetyConfig",
    "SafetyDecision",
    "SafetyGuardian",
    "ThreeTankHardwareEnv",
]
