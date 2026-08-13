"""Fixed-setpoint PID baseline controller."""
from __future__ import annotations

import numpy as np

from ._context import controller_inputs


def _clip01(v):
    return 0.0 if v < 0 else 1.0 if v > 1 else v




def _loop_spec(row):
    if isinstance(row, dict):
        return (
            int(row["u_index"]),
            int(row["y_index"]),
            row["pid"],
            bool(row["reverse"] if "reverse" in row else False),
            float(row["bias"] if "bias" in row else 0.0),
        )
    if len(row) == 3:
        u_index, y_index, pid = row
        return int(u_index), int(y_index), pid, False, 0.0
    if len(row) == 4:
        u_index, y_index, pid, reverse = row
        return int(u_index), int(y_index), pid, bool(reverse), 0.0
    raise ValueError(f"PID loop must be a mapping or 3/4-field row, got {row!r}")


def _hold_spec(row):
    if isinstance(row, dict):
        return int(row["u_index"]), float(row["value"])
    u_index, value = row
    return int(u_index), float(value)


class PIDLoop:
    def __init__(self, g, reverse=False, bias=0.0):
        self.kp, self.ki, self.kd = (float(value) for value in g)
        self.reverse = reverse
        self.bias = _clip01(float(bias))
        self.reset()

    def reset(self):
        self.i = self.bias
        self.prev = None

    def update(self, sp, meas, dt):
        e = (meas - sp) if self.reverse else (sp - meas)
        dmeas = 0.0 if (self.prev is None or dt <= 0) else (meas - self.prev) / dt
        self.prev = meas
        p = self.kp * e
        d = (1.0 if self.reverse else -1.0) * self.kd * dmeas
        i_cand = self.i + self.ki * e * dt
        raw = p + i_cand + d
        out = _clip01(raw)
        if not ((raw > 1 and e > 0) or (raw < 0 and e < 0)):
            self.i = i_cand
        return out


class FixedSetpointPIDPolicy:
    """Scenario-configured fixed-setpoint PID implementing the core Policy API."""

    name = "pid"
    control_structure = "fixed_sp_pid"

    def __init__(self, env, loops=None, holds=None, demand_u_index=None):
        self.env = env
        self.model = env.model
        self.nu = self.model.action_dim()
        if loops is None:
            raise ValueError("PID loops must come from Scenario controller defaults")
        self.loops_config = list(loops)
        self.demand_u_index = demand_u_index
        self.demand_valve = 0.5
        self.holds = [] if holds is None else list(holds)
        self.hold_specs = [_hold_spec(row) for row in self.holds]
        loop_specs = [_loop_spec(row) for row in self.loops_config]
        _validate_pid_config(
            self.model, loop_specs, self.hold_specs, self.demand_u_index
        )
        self.loops = [
            (u_index, y_index, PIDLoop(pid, reverse, bias))
            for u_index, y_index, pid, reverse, bias in loop_specs
        ]

    def metadata(self):
        return {
            "id": self.name,
            "class": self.__class__.__name__,
            "kind": "fixed_setpoint_pid",
            "scenario": self.model.scenario,
            "control_structure": self.control_structure,
            "loops": self.loops_config,
            "holds": self.holds,
            "demand_u_index": self.demand_u_index,
        }

    def reset(self, seed=None):
        for *_, loop in self.loops:
            loop.reset()

    def act(self, obs, context):
        measurement, setpoint, control_dt = controller_inputs(
            self.env,
            obs,
            context,
        )
        action = self.compute(measurement, setpoint, control_dt)
        return np.asarray(self.model.action_vector(action), dtype=np.float32)

    def compute(self, meas, sp, dt):
        y_sp = list(sp["y_sp"])
        y = list(meas["y"])
        u = [0.0] * self.nu
        for u_index, value in self.hold_specs:
            u[u_index] = value
        for u_index, y_index, loop in self.loops:
            u[u_index] = loop.update(y_sp[y_index], y[y_index], dt)
        if self.demand_u_index is not None:
            u[self.demand_u_index] = self.demand_valve
        return u


class MatrixPIDPolicy:
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
        feedforward=None,
    ):
        self.env = env
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
            raise ValueError(f"PID bias must have shape {(action_dim,)}, got {self.bias.shape}")
        if feedforward not in (None, "tracking_steady_state_action"):
            raise ValueError(f"unsupported PID feedforward {feedforward!r}")
        self.feedforward = feedforward
        self.reset()

    def reset(self, seed=None):
        del seed
        self.integral = np.zeros(self.env.action_space.shape, dtype=float)
        self.previous_measurement = None

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
        dt = float(self.env.control_dt)
        error = reference - measurement
        derivative = (
            np.zeros_like(measurement)
            if self.previous_measurement is None
            else (measurement - self.previous_measurement) / dt
        )
        self.previous_measurement = measurement.copy()
        candidate_integral = self.integral + self.ki @ error * dt
        bias = self._resolved_bias(reference)
        raw = bias + self.kp @ error + candidate_integral - self.kd @ derivative
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

    def _resolved_bias(self, reference):
        if self.feedforward is None:
            return self.bias
        resolver = self.env.model.tracking_steady_state_action
        disturbances = dict(self.env.disturbances)
        candidate = resolver(reference, disturbances)
        if candidate is None:
            raise ValueError("PID feedforward target has no feasible steady action")
        values = np.asarray(candidate, dtype=float).reshape(-1)
        if values.shape != self.bias.shape or not np.all(np.isfinite(values)):
            raise ValueError("PID feedforward returned an invalid action")
        return np.clip(values, self.env.action_space.low, self.env.action_space.high)

    def metadata(self):
        return {
            "id": "pid",
            "kind": "matrix_pid",
            "kp": self.kp.tolist(),
            "ki": self.ki.tolist(),
            "kd": self.kd.tolist(),
            "bias": self.bias.tolist(),
            "feedforward": self.feedforward,
        }


def _gain_matrix(name, value, rows, columns):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (rows, columns):
        raise ValueError(
            f"PID {name} must have shape {(rows, columns)}, got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"PID {name} must contain only finite values")
    return matrix


def _validate_pid_config(model, loop_specs, hold_specs, demand_u_index):
    ny = len(model.controlled_output(model.initial_state()))
    for u_index, y_index, pid, _, bias in loop_specs:
        if not 0 <= u_index < model.action_dim():
            raise ValueError(f"PID u_index {u_index} is outside action vector length {model.action_dim()}")
        if not 0 <= y_index < ny:
            raise ValueError(f"PID y_index {y_index} is outside controlled-output length {ny}")
        if not isinstance(pid, (list, tuple)) or len(pid) != 3:
            raise ValueError(f"PID gains must be a three-item sequence, got {pid!r}")
        tuple(float(value) for value in pid)
        if not 0.0 <= float(bias) <= 1.0:
            raise ValueError(f"PID bias must be in [0, 1], got {bias!r}")
    for u_index, _ in hold_specs:
        if not 0 <= u_index < model.action_dim():
            raise ValueError(f"PID hold u_index {u_index} is outside action vector length {model.action_dim()}")
    if demand_u_index is not None and not 0 <= int(demand_u_index) < model.action_dim():
        raise ValueError(
            f"PID demand_u_index {demand_u_index} is outside action vector length {model.action_dim()}"
        )


__all__ = [
    "FixedSetpointPIDPolicy",
    "MatrixPIDPolicy",
]
