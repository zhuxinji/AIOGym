"""Deterministic sensor and actuator realism driven by ``EpisodeSpec``."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy

import numpy as np


SENSOR_MODEL_KINDS = frozenset(
    {"identity", "additive_gaussian", "sensor_dynamics_v1"}
)
ACTUATOR_MODEL_KINDS = frozenset({"identity", "actuator_dynamics_v1"})


class SensorModelRuntime:
    """Apply a resolved sensor model without exposing latent plant state."""

    def __init__(
        self,
        specification: Mapping,
        *,
        state_schema: Sequence[Mapping],
        control_dt: float,
        rng,
    ) -> None:
        self.specification = validate_sensor_model(
            specification,
            dimension=len(state_schema),
        )
        self.control_dt = float(control_dt)
        self.rng = rng
        self.dimension = len(state_schema)
        self.scale = _schema_scale(state_schema)
        self._last_step = None
        self._last_measurement = None
        self._delay_queue = []

    def reset(self, initial_state) -> np.ndarray:
        state = _finite_vector(
            "initial sensor state",
            initial_state,
            self.dimension,
        )
        delay = int(self.specification.get("delay_steps", 0))
        self._delay_queue = [state.copy() for _ in range(delay)]
        self._last_step = None
        self._last_measurement = None
        return self.observe(state, step=0)

    def observe(self, true_state, *, step: int) -> np.ndarray:
        if self._last_step == int(step):
            return self._last_measurement.copy()
        state = _finite_vector(
            "true sensor state",
            true_state,
            self.dimension,
        )
        self._delay_queue.append(state.copy())
        delayed = self._delay_queue.pop(0)
        kind = self.specification["kind"]
        if kind == "identity":
            measured = delayed
        else:
            measured = delayed.copy()
            measured += (
                _parameter_vector(
                    self.specification.get("bias_fraction", 0.0),
                    dimension=self.dimension,
                    name="sensor bias_fraction",
                )
                * self.scale
            )
            measured += (
                _parameter_vector(
                    self.specification.get("drift_fraction_per_second", 0.0),
                    dimension=self.dimension,
                    name="sensor drift_fraction_per_second",
                )
                * self.scale
                * float(step)
                * self.control_dt
            )
            noise_fraction = (
                self.specification.get("noise_fraction")
                if kind == "sensor_dynamics_v1"
                else self.specification.get("noise_pct", 0.0)
            )
            measured += self.rng.normal(
                0.0,
                _parameter_vector(
                    0.0 if noise_fraction is None else noise_fraction,
                    dimension=self.dimension,
                    name="sensor noise_fraction",
                )
                * self.scale,
            )
            dropout_probability = float(
                self.specification.get("dropout_probability", 0.0)
            )
            if (
                self._last_measurement is not None
                and dropout_probability > 0.0
                and float(self.rng.random()) < dropout_probability
            ):
                measured = self._last_measurement.copy()
            quantization = _parameter_vector(
                self.specification.get("quantization_fraction", 0.0),
                dimension=self.dimension,
                name="sensor quantization_fraction",
            ) * self.scale
            active = quantization > 0.0
            measured[active] = (
                np.round(measured[active] / quantization[active])
                * quantization[active]
            )
        self._last_step = int(step)
        self._last_measurement = np.asarray(measured, dtype=np.float64)
        return self._last_measurement.copy()

    def metadata(self) -> dict:
        return deepcopy(self.specification)


class ActuatorModelRuntime:
    """Resolve commanded actions to delayed, rate-limited applied actions."""

    def __init__(
        self,
        specification: Mapping,
        *,
        dimension: int,
        control_dt: float,
        low,
        high,
    ) -> None:
        self.specification = validate_actuator_model(
            specification,
            dimension=dimension,
        )
        self.dimension = int(dimension)
        self.control_dt = float(control_dt)
        self.low = _finite_vector("actuator lower bounds", low, dimension)
        self.high = _finite_vector("actuator upper bounds", high, dimension)
        if not np.all(self.high > self.low):
            raise ValueError("actuator bounds must be strictly ordered")
        self._applied = None
        self._delay_queue = []

    def reset(self, initial_action) -> np.ndarray:
        initial = np.clip(
            _finite_vector(
                "initial actuator action",
                initial_action,
                self.dimension,
            ),
            self.low,
            self.high,
        )
        delay = int(self.specification.get("delay_steps", 0))
        self._delay_queue = [initial.copy() for _ in range(delay)]
        self._applied = initial
        return initial.copy()

    def apply(self, commanded_action) -> tuple[np.ndarray, dict]:
        commanded = np.clip(
            _finite_vector(
                "commanded actuator action",
                commanded_action,
                self.dimension,
            ),
            self.low,
            self.high,
        )
        if self._applied is None:
            self.reset(commanded)
        self._delay_queue.append(commanded.copy())
        delayed = self._delay_queue.pop(0)
        kind = self.specification["kind"]
        if kind == "identity":
            applied = delayed
        else:
            efficiency = _parameter_vector(
                self.specification.get("efficiency", 1.0),
                dimension=self.dimension,
                name="actuator efficiency",
            )
            bias = _parameter_vector(
                self.specification.get("bias", 0.0),
                dimension=self.dimension,
                name="actuator bias",
            )
            target = delayed * efficiency + bias
            deadband = _parameter_vector(
                self.specification.get("deadband", 0.0),
                dimension=self.dimension,
                name="actuator deadband",
            )
            target = np.where(
                np.abs(target - self._applied) <= deadband,
                self._applied,
                target,
            )
            time_constant = _parameter_vector(
                self.specification.get("time_constant_seconds", 0.0),
                dimension=self.dimension,
                name="actuator time_constant_seconds",
            )
            alpha = np.ones(self.dimension, dtype=np.float64)
            dynamic = time_constant > 0.0
            alpha[dynamic] = 1.0 - np.exp(
                -self.control_dt / time_constant[dynamic]
            )
            candidate = self._applied + alpha * (target - self._applied)
            slew_rate = _parameter_vector(
                self.specification.get("slew_rate_per_second", math.inf),
                dimension=self.dimension,
                name="actuator slew_rate_per_second",
                allow_infinity=True,
            )
            maximum_delta = slew_rate * self.control_dt
            applied = self._applied + np.clip(
                candidate - self._applied,
                -maximum_delta,
                maximum_delta,
            )
            applied = np.clip(applied, self.low, self.high)
        self._applied = np.asarray(applied, dtype=np.float64)
        delta = self._applied - commanded
        return self._applied.copy(), {
            "actuator_model_kind": kind,
            "actuator_intervened": bool(np.any(np.abs(delta) > 1e-12)),
            "actuator_command_applied_delta": delta.tolist(),
            "actuator_command_applied_l1": float(np.sum(np.abs(delta))),
        }

    def metadata(self) -> dict:
        return deepcopy(self.specification)


def validate_sensor_model(
    specification: Mapping,
    *,
    dimension: int,
) -> dict:
    if not isinstance(specification, Mapping):
        raise TypeError("sensor model must be a mapping")
    data = deepcopy(dict(specification))
    kind = data.get("kind")
    if kind not in SENSOR_MODEL_KINDS:
        raise ValueError(f"unsupported sensor model kind {kind!r}")
    allowed = {
        "identity": {"kind"},
        "additive_gaussian": {"kind", "noise_pct"},
        "sensor_dynamics_v1": {
            "kind",
            "noise_fraction",
            "bias_fraction",
            "drift_fraction_per_second",
            "delay_steps",
            "quantization_fraction",
            "dropout_probability",
        },
    }[kind]
    _reject_unknown(data, allowed, "sensor model")
    if kind == "additive_gaussian" and "noise_pct" not in data:
        raise ValueError("additive_gaussian sensor model requires noise_pct")
    for name in (
        "noise_pct",
        "noise_fraction",
        "quantization_fraction",
    ):
        if name in data:
            values = _parameter_vector(
                data[name],
                dimension=dimension,
                name=f"sensor {name}",
            )
            if np.any(values < 0.0):
                raise ValueError(f"sensor {name} must be non-negative")
    for name in ("bias_fraction", "drift_fraction_per_second"):
        if name in data:
            _parameter_vector(
                data[name],
                dimension=dimension,
                name=f"sensor {name}",
            )
    _non_negative_integer("sensor delay_steps", data.get("delay_steps", 0))
    probability = _finite_scalar(
        "sensor dropout_probability",
        data.get("dropout_probability", 0.0),
    )
    if not 0.0 <= probability <= 1.0:
        raise ValueError("sensor dropout_probability must be in [0, 1]")
    return data


def validate_actuator_model(
    specification: Mapping,
    *,
    dimension: int,
) -> dict:
    if not isinstance(specification, Mapping):
        raise TypeError("actuator model must be a mapping")
    data = deepcopy(dict(specification))
    kind = data.get("kind")
    if kind not in ACTUATOR_MODEL_KINDS:
        raise ValueError(f"unsupported actuator model kind {kind!r}")
    allowed = {
        "identity": {"kind"},
        "actuator_dynamics_v1": {
            "kind",
            "efficiency",
            "bias",
            "delay_steps",
            "deadband",
            "time_constant_seconds",
            "slew_rate_per_second",
        },
    }[kind]
    _reject_unknown(data, allowed, "actuator model")
    for name in (
        "efficiency",
        "deadband",
        "time_constant_seconds",
        "slew_rate_per_second",
    ):
        if name in data:
            values = _parameter_vector(
                data[name],
                dimension=dimension,
                name=f"actuator {name}",
                allow_infinity=name == "slew_rate_per_second",
            )
            if np.any(values < 0.0):
                raise ValueError(f"actuator {name} must be non-negative")
    if "bias" in data:
        _parameter_vector(
            data["bias"],
            dimension=dimension,
            name="actuator bias",
        )
    _non_negative_integer("actuator delay_steps", data.get("delay_steps", 0))
    return data


def _schema_scale(schema) -> np.ndarray:
    scales = []
    for row in schema:
        bounds = row.get("bounds") if isinstance(row, Mapping) else None
        if (
            isinstance(bounds, (list, tuple))
            and len(bounds) == 2
            and bounds[0] is not None
            and bounds[1] is not None
            and float(bounds[1]) > float(bounds[0])
        ):
            scales.append(float(bounds[1]) - float(bounds[0]))
        else:
            scales.append(1.0)
    return np.asarray(scales, dtype=np.float64)


def _parameter_vector(
    value,
    *,
    dimension: int,
    name: str,
    allow_infinity: bool = False,
) -> np.ndarray:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if any(isinstance(item, bool) for item in value):
            raise TypeError(f"{name} must be numeric")
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size != int(dimension):
            raise ValueError(f"{name} must contain {dimension} values")
    else:
        array = np.full(int(dimension), float(value), dtype=np.float64)
    valid = ~np.isnan(array) if allow_infinity else np.isfinite(array)
    if not np.all(valid):
        raise ValueError(f"{name} must be finite")
    return array


def _finite_vector(name, value, dimension) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != int(dimension):
        raise ValueError(f"{name} must contain {dimension} values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _finite_scalar(name, value) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative_integer(name, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _reject_unknown(data, allowed, name) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValueError(f"unknown {name} fields: " + ", ".join(unknown))


__all__ = [
    "ACTUATOR_MODEL_KINDS",
    "SENSOR_MODEL_KINDS",
    "ActuatorModelRuntime",
    "SensorModelRuntime",
    "validate_actuator_model",
    "validate_sensor_model",
]
