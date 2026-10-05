"""Successive-linearization MPC with bounded action sequences."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import lsq_linear

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
    """Optimize one action per environment step over a linearized prediction.

    Execute the first ``solve_every`` actions, then replan from the observation.
    Predicted states are unconstrained; actions obey the model's box bounds.
    """

    def __init__(
        self,
        model,
        *,
        control_dt,
        prediction_horizon=10,
        solve_every=1,
        move_supp=0.8,
        cv_scale=None,
        q_y=1.0,
        steady_input_weight=0.0,
    ):
        self.m = model
        self.nu = model.action_dim()
        action_schema = model.action_schema()
        self._action_low = np.array([row["low"] for row in action_schema], dtype=float)
        self._action_high = np.array([row["high"] for row in action_schema], dtype=float)
        self.nx = len(model.initial_state())
        self.ncv = len(model.outputs(model.initial_state()))
        self.control_dt = positive_float("control_dt", control_dt)
        self._integration_substeps = max(
            1, math.ceil(self.control_dt / model.dt_micro - 1e-12)
        )
        self._integration_dt = self.control_dt / self._integration_substeps
        self.prediction_horizon = positive_int("prediction_horizon", prediction_horizon)
        self.solve_every = positive_int("solve_every", solve_every)
        if self.solve_every > self.prediction_horizon:
            raise ValueError("solve_every must not exceed prediction_horizon")
        self._move_supp, self.move_supp = nonnegative_weights(
            "move_supp", move_supp, self.nu
        )
        self._steady_input_weight, self.steady_input_weight = nonnegative_weights(
            "steady_input_weight", steady_input_weight, self.nu
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
            values = [float(value) for value in self.m.output_scales()]
        if len(values) != self.ncv:
            raise ValueError(
                f"cv_scale must contain 1 or {self.ncv} values, got {len(values)}"
            )
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
        return {
            "class": self.__class__.__name__,
            "kind": "successive_linearization_mpc",
            "scenario": self.m.scenario,
            "control_dt": self.control_dt,
            "time_unit": self.m.time_unit,
            "prediction_horizon": self.prediction_horizon,
            "prediction_duration": self.prediction_horizon * self.control_dt,
            "solve_every": self.solve_every,
            "solve_interval": self.solve_every * self.control_dt,
            "control_horizon": self.prediction_horizon,
            "linearization": "continuous_dynamics",
            "discretization": "rk4",
            "max_integration_dt": self._integration_dt,
            "min_integration_substeps": self._integration_substeps,
            "integration_refinement": "linear_model_spectral_radius",
            "action_bound_handling": "box_constrained_lsq",
            "action_bounds": np.column_stack((self._action_low, self._action_high)).tolist(),
            "prediction_state_constraints": False,
            "move_supp": self.move_supp,
            "steady_input_weight": self.steady_input_weight,
            "initialization": "default_action",
            "steady_input_role": "linearization_reference_and_penalty",
            "cv_scale": self.cv_scale,
            "q_y": self.q_y,
        }

    def reset(self, seed=None):
        del seed
        initial_action = self.m.default_action()
        self.u = np.asarray(self.m.action_vector(initial_action), dtype=np.float64)
        self._plan = np.empty((0, self.nu))
        self._plan_index = 0

    def _cv(self, x):
        return np.asarray(self.m.outputs(list(x)), dtype=np.float64)

    def compute(self, meas, sp):
        if not len(self._plan) or self._plan_index == self.solve_every:
            self._solve(meas, sp)
            self._plan_index = 0
        self.u = np.asarray(
            self.m.action_vector(self._plan[self._plan_index]), dtype=np.float64
        )
        self._plan_index += 1
        return self.u.copy()

    def _solve(self, meas, sp):
        m, nx, nu, horizon = self.m, self.nx, self.nu, self.prediction_horizon
        env = {
            k: v
            for k, v in meas.items()
            if k not in ("x", "y", "levels", "temps", "conc")
        }
        x0 = np.asarray(meas["x"], dtype=np.float64)
        target = np.asarray(sp["y_sp"], dtype=np.float64)
        steady_input = m.tracking_steady_state_action(target, env)
        # The operating input is separate from the last issued action self.u.
        # Feedforward must not reset the reference for the first move penalty.
        u0 = self.u.copy() if steady_input is None else np.clip(
            np.asarray(steady_input, dtype=float), self._action_low, self._action_high
        )
        dynamics = lambda x, u: np.asarray(
            m.dynamics(x, u, disturbances=env), dtype=np.float64
        )
        f0 = dynamics(x0, u0)
        cv0 = self._cv(x0)
        A = np.zeros((nx, nx))
        B = np.zeros((nx, nu))
        C = np.zeros((self.ncv, nx))
        eps = np.cbrt(np.finfo(float).eps)
        state_schema = m.state_schema()
        for j in range(nx):
            step = eps * max(1.0, abs(x0[j]))
            xp, xm = x0.copy(), x0.copy()
            xp[j] = min(x0[j] + step, state_schema[j]["high"])
            xm[j] = max(x0[j] - step, state_schema[j]["low"])
            width = xp[j] - xm[j]
            if width > 0:
                A[:, j] = (dynamics(xp, u0) - dynamics(xm, u0)) / width
                C[:, j] = (self._cv(xp) - self._cv(xm)) / width
        for j in range(nu):
            step = eps * max(1.0, abs(u0[j]))
            up, um = u0.copy(), u0.copy()
            up[j] = min(u0[j] + step, self._action_high[j])
            um[j] = max(u0[j] - step, self._action_low[j])
            width = up[j] - um[j]
            if width > 0:
                B[:, j] = (dynamics(x0, up) - dynamics(x0, um)) / width

        # Linearize first: dx/dt = A (x-x0) + B (u-u0) + f0.
        # Augment with the held input and a constant to preserve the affine term.
        generator = np.zeros((nx + nu + 1, nx + nu + 1))
        generator[:nx, :nx] = A
        generator[:nx, nx:nx + nu] = B
        generator[:nx, -1] = f0
        # Boundary dynamics can be much stiffer after local linearization.
        # Keep |dt * eigenvalue| <= 1, inside RK4's stable left-half-plane region.
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
        substeps = max(self._integration_substeps, math.ceil(self.control_dt * spectral_radius))
        scaled = (self.control_dt / substeps) * generator
        substep = np.eye(len(generator))
        term = substep.copy()
        for order in range(1, 5):
            term = term @ scaled / order
            substep += term
        # RK4 for a constant linear model is a matrix polynomial. Repeated
        # squaring gives exactly the same substeps without a Python time loop.
        discrete = np.linalg.matrix_power(substep, substeps)
        Ad, Bd = discrete[:nx, :nx], discrete[:nx, nx:nx + nu]
        next_x = x0 + discrete[:nx, -1]

        # Condense x[k+1] = Ad (x[k]-x0) + next_x + Bd (u[k]-u0).
        # Each column block belongs to a separate future action, not one held move.
        size = horizon * nu
        weights = np.sqrt(self.q_y) / self.cv_scale
        xf = x0.copy()
        sensitivity = np.zeros((nx, size))
        matrices, targets = [], []
        for k in range(horizon):
            xf = Ad @ (xf - x0) + next_x
            sensitivity = Ad @ sensitivity
            sensitivity[:, k * nu:(k + 1) * nu] += Bd
            matrices.append(weights[:, None] * (C @ sensitivity))
            targets.append(weights * (target - cv0 - C @ (xf - x0)))

        if np.any(self._move_supp):
            differences = np.eye(size) - np.eye(size, k=-nu)
            move_weights = np.tile(np.sqrt(self._move_supp), horizon)
            matrices.append(move_weights[:, None] * differences)
            targets.append(move_weights * np.concatenate((self.u - u0, np.zeros(size - nu))))
        if steady_input is not None and np.any(self._steady_input_weight):
            matrices.append(np.diag(np.tile(np.sqrt(self._steady_input_weight), horizon)))
            targets.append(np.zeros(size))
        matrix, target = np.vstack(matrices), np.concatenate(targets)
        lower = np.tile(self._action_low - u0, horizon)
        upper = np.tile(self._action_high - u0, horizon)
        free = lower < upper
        moves = lower.copy()
        if np.any(free):
            result = lsq_linear(
                matrix[:, free], target - matrix[:, ~free] @ moves[~free],
                bounds=(lower[free], upper[free]), method="bvls", lsq_solver="exact",
            )
            if not result.success:
                raise RuntimeError(f"MPC action sequence solve failed: {result.message}")
            moves[free] = result.x
        self._plan = np.clip(
            moves.reshape(horizon, nu) + u0, self._action_low, self._action_high
        )


class FixedSetpointMPCPolicy:
    """Scenario-configured MPC implementing the core Policy API."""

    name = "mpc"

    def __init__(self, env, **config):
        self.env = env
        self.controller = SuccessiveLinearizationMPC(
            env.model, control_dt=env.control_dt, **config
        )

    def reset(self, seed=None):
        self.controller.reset(seed=seed)

    def act(self, observation, context):
        measurement, setpoint, _ = controller_inputs(
            self.env,
            observation,
            context,
        )
        action = self.controller.compute(measurement, setpoint)
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
