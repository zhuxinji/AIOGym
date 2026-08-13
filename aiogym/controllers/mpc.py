"""Successive-linearization fixed-setpoint MPC baseline."""
from __future__ import annotations

import math

import numpy as np

from ._context import controller_inputs


def nonnegative_float(name, value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def nonnegative_weights(name, value, size):
    if isinstance(value, (list, tuple, np.ndarray)):
        values = np.asarray(value, dtype=float).reshape(-1)
        if values.shape != (size,):
            raise ValueError(f"{name} must contain {size} values")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{name} values must be finite and non-negative")
        return values, values.tolist()
    scalar = nonnegative_float(name, value)
    return np.full(size, scalar, dtype=float), scalar


def positive_float(name, value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class SuccessiveLinearizationMPC:
    """Successive-linearization, velocity-form (M=1) constrained MPC."""

    def __init__(
        self,
        model,
        Ts=0.5,
        P=40,
        move_supp=0.8,
        cv_scale=None,
        q_y=1.0,
        steady_input_weight=0.0,
        reseed_on_feedforward_change=False,
    ):
        self.m = model
        self.nu = model.action_dim()
        self.nx = len(model.initial_state())
        self.ncv = len(model.controlled_output(model.initial_state()))
        self.Ts = positive_float("Ts", Ts)
        self.P = positive_int("P", P)
        self._move_supp, self.move_supp = nonnegative_weights(
            "move_supp", move_supp, self.nu
        )
        self._steady_input_weight, self.steady_input_weight = nonnegative_weights(
            "steady_input_weight", steady_input_weight, self.nu
        )
        self.reseed_on_feedforward_change = bool(
            reseed_on_feedforward_change
        )
        self.cv_scale = self._resolve_cv_scale(cv_scale)
        self.q_y = self._resolve_q_y(q_y)
        self.reset()

    def _resolve_cv_scale(self, cv_scale):
        if cv_scale is not None:
            values = cv_scale if isinstance(cv_scale, (list, tuple)) else [cv_scale]
            values = [float(v) for v in values]
            if len(values) == 1:
                values *= self.ncv
        else:
            values = [float(value) for value in self.m.controlled_output_scales()]
        if len(values) != self.ncv:
            raise ValueError(f"cv_scale must contain 1 or {self.ncv} values, got {len(values)}")
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("cv_scale values must be finite and positive")
        return values

    def _resolve_q_y(self, q_y):
        values = q_y if isinstance(q_y, (list, tuple)) else [q_y]
        values = [nonnegative_float("q_y", value) for value in values]
        if len(values) == 1:
            values *= self.ncv
        if len(values) != self.ncv:
            raise ValueError(
                f"q_y must contain 1 or {self.ncv} values, got {len(values)}"
            )
        return values

    def metadata(self):
        return {"class": self.__class__.__name__,
                "kind": "successive_linearization_mpc", "scenario": self.m.scenario,
                "Ts": self.Ts, "horizon": self.P,
                "move_supp": self.move_supp,
                "steady_input_weight": self.steady_input_weight,
                "initialization": "tracking_steady_state_action",
                "initialization_status": self._initialization_status,
                "feedforward_reseed": (
                    "setpoint_or_feedforward_change"
                    if self.reseed_on_feedforward_change
                    else "setpoint_change"
                ),
                "cv_scale": self.cv_scale,
                "q_y": self.q_y}

    def reset(self, seed=None):
        del seed
        initial_action = self.m.mpc_init()
        self.u = np.asarray(self.m.action_vector(initial_action), dtype=np.float64)
        self._last_target = None
        self._target_u = None
        self._clock = 1e9
        self._initialization_status = "default_action"

    def _unpack(self, u):
        return self.m.action_vector(u)

    def _toX(self, meas):
        return np.asarray(meas["x"], dtype=np.float64)

    def _cv(self, x):
        return np.asarray(self.m.controlled_output(list(x)), dtype=np.float64)

    def _wcv(self):
        return np.array([
            float(weight) / max(float(scale), 1e-12) ** 2
            for weight, scale in zip(self.q_y, self.cv_scale)
        ], dtype=np.float64)

    def compute(self, meas, sp, dt):
        self._clock += dt
        if self._clock >= self.Ts:
            self._clock = 0.0
            self._solve(meas, sp)
        return self._unpack(self.u)

    def _solve(self, meas, sp):
        m, nx, nu, P, Ts = self.m, self.nx, self.nu, self.P, self.Ts
        env = {k: v for k, v in meas.items()
               if k not in ("x", "y", "levels", "temps", "conc")}
        x0 = self._toX(meas)
        target = np.asarray(m.setpoint_vector(sp["y_sp"]), dtype=np.float64)
        target_changed = (
            self._last_target is None
            or not np.array_equal(target, self._last_target)
        )
        steady_input = m.tracking_steady_state_action(target, env)
        previous_target_u = self._target_u
        self._target_u = None
        if steady_input is not None:
            candidate = np.asarray(steady_input, dtype=np.float64).reshape(-1)
            if len(candidate) == nu and np.all(np.isfinite(candidate)):
                self._target_u = np.clip(candidate, 0.0, 1.0)
                feedforward_changed = (
                    previous_target_u is None
                    or not np.allclose(
                        self._target_u,
                        previous_target_u,
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
                if target_changed or (
                    self.reseed_on_feedforward_change and feedforward_changed
                ):
                    self.u = self._target_u.copy()
                    self._initialization_status = "tracking_steady_state_action"
        if target_changed:
            self._last_target = target.copy()
        u0 = self.u.copy()
        f = lambda x: np.asarray(m.dynamics(list(x), u0, env), dtype=np.float64)
        f0 = f(x0)
        eps = 1e-5
        Ad = np.eye(nx)
        Bd = np.zeros((nx, nu))
        for j in range(nx):
            xp = x0.copy(); xp[j] += eps
            xm = x0.copy(); xm[j] -= eps
            Ad[:, j] += (f(xp) - f(xm)) / (2 * eps) * Ts
        for j in range(nu):
            up = u0.copy(); up[j] += eps
            um = u0.copy(); um[j] -= eps
            fp = np.asarray(m.dynamics(list(x0), up, env), dtype=np.float64)
            fm = np.asarray(m.dynamics(list(x0), um, env), dtype=np.float64)
            Bd[:, j] = (fp - fm) / (2 * eps) * Ts
        cv0 = self._cv(x0)
        nCV = len(cv0)
        C = np.zeros((nCV, nx))
        for j in range(nx):
            xp = x0.copy(); xp[j] += eps
            C[:, j] = (self._cv(xp) - cv0) / eps
        Wcv = self._wcv()
        c0 = (x0 + f0 * Ts) - Ad @ x0 - Bd @ u0
        xf = x0.copy()
        S = np.zeros((nx, nu))
        H = np.zeros((nu, nu))
        g = np.zeros(nu)
        for _ in range(P):
            xf = Ad @ xf + Bd @ u0 + c0
            S = Ad @ S + Bd
            G = C @ S
            e = cv0 + C @ (xf - x0) - target
            WG = Wcv[:, None] * G
            H += G.T @ WG
            g += G.T @ (Wcv * e)
        H += np.diag(self._move_supp)
        if self._target_u is not None and np.any(self._steady_input_weight):
            H += np.diag(self._steady_input_weight)
            g += self._steady_input_weight * (u0 - self._target_u)
        try:
            du = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            # With zero move suppression, unobservable/redundant actuator
            # directions can make the positive-semidefinite Hessian singular.
            # The minimum-norm least-squares solution is the corresponding
            # well-defined MPC move and avoids inventing a metric penalty solely
            # for numerical regularization.
            du = np.linalg.lstsq(H, -g, rcond=None)[0]
        self.u = np.clip(u0 + du, 0.0, 1.0)


class FixedSetpointMPCPolicy:
    """Scenario-configured MPC implementing the core Policy API."""

    name = "mpc"

    def __init__(self, env, **config):
        self.env = env
        self.controller = SuccessiveLinearizationMPC(env.model, **config)

    def reset(self, seed=None):
        self.controller.reset(seed=seed)

    def act(self, observation, context):
        measurement, setpoint, control_dt = controller_inputs(
            self.env,
            observation,
            context,
        )
        action = self.controller.compute(measurement, setpoint, control_dt)
        return np.asarray(self.env.model.action_vector(action), dtype=np.float32)

    def metadata(self):
        metadata = dict(self.controller.metadata())
        metadata.update(
            {
                "id": self.name,
                "control_structure": "fixed_sp_mpc",
            }
        )
        return metadata


__all__ = [
    "FixedSetpointMPCPolicy",
    "SuccessiveLinearizationMPC",
]
