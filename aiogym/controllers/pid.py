"""Matrix PID baseline controller."""
from __future__ import annotations

import numpy as np

from ._context import controller_inputs


class PIDPolicy:
    """MIMO PID policy whose output belongs directly to ``env.action_space``."""

    name = "pid"

    def __init__(
        self,
        env,
        *,
        kp,
        ki=None,
        kd=None,
        bias=None,
        tracking_steady_state_bias=False,
    ):
        self.env = env
        if not isinstance(tracking_steady_state_bias, bool):
            raise TypeError("PID tracking_steady_state_bias must be a boolean")
        self.tracking_steady_state_bias = tracking_steady_state_bias
        action_dim = int(env.action_space.shape[0])
        output_dim = len(env.model.outputs(env.model.initial_state()))
        self.kp = _gain_matrix("kp", kp, action_dim, output_dim)
        self.ki = _gain_matrix(
            "ki", np.zeros_like(self.kp) if ki is None else ki, action_dim, output_dim
        )
        self.kd = _gain_matrix(
            "kd", np.zeros_like(self.kp) if kd is None else kd, action_dim, output_dim
        )
        default_bias = np.asarray(env.model.default_action(), dtype=float)
        self.bias = np.asarray(default_bias if bias is None else bias, dtype=float)
        if self.bias.shape != (action_dim,):
            raise ValueError(
                f"PID bias must have shape {(action_dim,)}, got {self.bias.shape}"
            )
        self.reset()

    def reset(self, seed=None):
        del seed
        self.integral = np.zeros(self.env.action_space.shape, dtype=float)
        self.previous_measurement = None
        self.active_bias = self.bias.copy()
        self.bias_reference = None

    def act(self, observation, context):
        measured, setpoint, _ = controller_inputs(
            self.env,
            observation,
            context,
        )
        measurement = np.asarray(measured["y"], dtype=float).reshape(-1)
        reference = np.asarray(setpoint["y_sp"], dtype=float).reshape(-1)
        if reference.shape != measurement.shape:
            raise ValueError("PID reference and observation shapes must match")
        if self.tracking_steady_state_bias and (
            self.bias_reference is None
            or not np.array_equal(reference, self.bias_reference)
        ):
            steady_action = self.env.model.tracking_steady_state_action(reference)
            if steady_action is None:
                raise ValueError(
                    "PID reference has no feasible tracking steady-state action"
                )
            candidate_bias = np.asarray(
                self.env.model.action_vector(steady_action), dtype=float
            )
            if candidate_bias.shape != self.bias.shape or not np.all(
                np.isfinite(candidate_bias)
            ):
                raise ValueError(
                    "PID tracking steady-state action must match the action interface"
                )
            if np.any(candidate_bias < self.env.action_space.low) or np.any(
                candidate_bias > self.env.action_space.high
            ):
                raise ValueError(
                    "PID tracking steady-state action must belong to the action space"
                )
            self.active_bias = candidate_bias
            self.bias_reference = reference.copy()
            self.integral.fill(0.0)
        dt = float(self.env.control_dt)
        error = reference - measurement
        derivative = (
            np.zeros_like(measurement)
            if self.previous_measurement is None
            else (measurement - self.previous_measurement) / dt
        )
        self.previous_measurement = measurement.copy()
        candidate_integral = self.integral + self.ki @ error * dt
        raw = (
            self.active_bias
            + self.kp @ error
            + candidate_integral
            - self.kd @ derivative
        )
        clipped = np.clip(raw, self.env.action_space.low, self.env.action_space.high)
        correction = self.ki @ error
        accept = np.logical_or(
            np.isclose(raw, clipped),
            np.logical_or(
                np.logical_and(raw > self.env.action_space.high, correction < 0.0),
                np.logical_and(raw < self.env.action_space.low, correction > 0.0),
            ),
        )
        self.integral = np.where(accept, candidate_integral, self.integral)
        return clipped.astype(np.float32)

    def metadata(self):
        metadata = {
            "id": "pid",
            "kind": "matrix_pid",
            "kp": self.kp.tolist(),
            "ki": self.ki.tolist(),
            "kd": self.kd.tolist(),
            "bias": self.bias.tolist(),
        }
        if self.tracking_steady_state_bias:
            metadata["feedforward"] = "tracking_steady_state_action_on_reference_change"
        return metadata


def _gain_matrix(name, value, rows, columns):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (rows, columns):
        raise ValueError(
            f"PID {name} must have shape {(rows, columns)}, got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"PID {name} must contain only finite values")
    return matrix


__all__ = ["PIDPolicy"]
