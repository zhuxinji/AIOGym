"""NMPC oracle — a true nonlinear MPC baseline (CasADi + IPOPT), in the spirit of
PC-Gym's do-mpc oracle (Bloor et al., arXiv:2410.22093). It's the "upper-bound"
controller RL/PID/APC-MPC are measured against.

Selectable multiple- or single-shooting transcription: RK4 over each control
interval, IPOPT NLP.
Plant dynamics are supplied by the model contract through
``model.dynamics(..., backend="casadi")``. The resolved optimization goal is
supplied by the benchmark execution path:
  - "regulation": Σ normalized(y-y_sp)ᵀQ normalized(y-y_sp) + ΔuᵀRΔu
  - "economic": maximize economic stage profit (value − energy − violation)

Constraints are always enforced. Safety and robustness are measured by the
evaluation gate and case shifts; they are not Oracle optimization modes.

Usage:
    orc = NMPCOracle("cstr", horizon=20, goal="economic")
    u = orc.solve(x, t_cold, t_amb, disturbances=meas, y_sp=y_sp)
"""
from __future__ import annotations
import copy
import math
import numpy as np

from .._internal.validation import nonnegative_float, positive_float, positive_int
from ..rewards.registry import get_reward_spec

try:
    import casadi as ca
    _HAVE_CASADI = True
except (ImportError, OSError):          # pragma: no cover
    _HAVE_CASADI = False

from ..models.registry import make_model


def copy_economic_config(model):
    return copy.deepcopy(getattr(model, "economic_config", {}))


class NMPCOracle:
    _GOALS = frozenset({"regulation", "economic"})

    def __init__(self, scenario="cstr", horizon=20, control_dt=0.5,
                 goal=None, reward_spec=None,
                 q_y=1.0, r_move=0.05, terminal_weight=1.0,
                 steady_input_weight=0.0,
                 ipopt_max_iter=80, ipopt_tol=1e-4,
                 ipopt_hessian_approximation="exact", warm_start=False,
                 transcription="multiple_shooting", enforce_state_bounds=True,
                 enforce_temperature_cap=True, model=None):
        if not _HAVE_CASADI:
            raise RuntimeError(
                "casadi is required for the Oracle controller; "
                "install AIO-Gym with `pip install 'aiogym[oracle]'`"
            )
        self.model = make_model(model if model is not None else scenario)
        self.scenario = self.model.scenario
        self.p = self.model.p
        self.N = positive_int("horizon", horizon)
        self.dt = positive_float("control_dt", control_dt)
        self.goal, self.reward_spec_id = self._resolve_goal_reward(
            goal=goal,
            reward_spec=reward_spec,
        )
        self.integration_max_step = positive_float(
            "model solver max_step", self.model.solver_settings()["max_step"]
        )
        self.integration_substeps = max(
            1, math.ceil(self.dt / self.integration_max_step - 1e-9)
        )
        self.ipopt_max_iter = positive_int("ipopt_max_iter", ipopt_max_iter)
        self.ipopt_tol = positive_float("ipopt_tol", ipopt_tol)
        if ipopt_hessian_approximation not in {"exact", "limited-memory"}:
            raise ValueError(
                "ipopt_hessian_approximation must be one of: "
                "exact, limited-memory"
            )
        self.ipopt_hessian_approximation = ipopt_hessian_approximation
        self.warm_start = bool(warm_start)
        if transcription not in {"multiple_shooting", "single_shooting"}:
            raise ValueError("transcription must be one of: multiple_shooting, single_shooting")
        self.transcription = transcription
        self.enforce_state_bounds = bool(enforce_state_bounds)
        self.enforce_temperature_cap = bool(enforce_temperature_cap)
        self.nu = self.model.action_dim()
        self.nx = len(self.model.initial_state())
        self.ny = len(self.model.controlled_output(self.model.initial_state()))
        self.q_y = self._resolve_q_y(q_y)
        self.r_move = nonnegative_float("r_move", r_move)
        self.terminal_weight = nonnegative_float("terminal_weight", terminal_weight)
        self.steady_input_weight = nonnegative_float(
            "steady_input_weight", steady_input_weight
        )
        self.econ = copy_economic_config(self.model)
        self.nd = len(self.model.dynamics_disturbance_names())
        self.state_bounds = self._state_bounds() if self.enforce_state_bounds else [(None, None)] * self.nx
        self.state_scale, self.state_offset = self._state_scaling()
        # hard safety cap (below the 92°C runaway trip) so economic NMPC hugs the edge
        # without driving the plant unstable; HVAC has no runaway so cap loosely.
        self.t_safe = self._temperature_soft_cap() if self.enforce_temperature_cap else None
        self.u_init = self._initial_action()
        self.u_prev = self.u_init.copy()
        self._warm = None
        self._warm_target = None
        self.last_plan = None
        self.last_error = None
        self.last_solve_status = None
        self.last_solve_success = False
        self._build()

    def _resolve_q_y(self, q_y):
        values = q_y if isinstance(q_y, (list, tuple)) else [q_y]
        values = [float(v) for v in values]
        if len(values) == 1:
            values *= self.ny
        if len(values) != self.ny:
            raise ValueError(f"q_y must contain 1 or {self.ny} values, got {len(values)}")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("q_y values must be finite and non-negative")
        return values

    @classmethod
    def _resolve_goal_reward(cls, *, goal, reward_spec):
        resolved_spec = (
            get_reward_spec(str(reward_spec))
            if reward_spec is not None
            else None
        )
        if goal is None:
            goal = resolved_spec.goal if resolved_spec is not None else "economic"
        if goal not in cls._GOALS:
            raise ValueError(
                "Oracle goal must be one of: economic, regulation"
            )
        if resolved_spec is not None and resolved_spec.goal != goal:
            raise ValueError(
                f"reward spec {resolved_spec.id!r} conflicts with goal {goal!r}"
            )
        return str(goal), (
            resolved_spec.id
            if resolved_spec is not None
            else f"{goal}-v1"
        )

    def _initial_action(self):
        initializer = getattr(self.model, "mpc_init", None)
        values = (
            initializer()
            if callable(initializer)
            else self.model.default_action()
        )
        action = np.asarray(values, dtype=float).reshape(-1)
        if len(action) != self.nu:
            raise ValueError(f"model.mpc_init() must contain {self.nu} values, got {len(action)}")
        if not np.all(np.isfinite(action)):
            raise ValueError("model.mpc_init() values must be finite")
        return np.clip(action, 0.0, 1.0)

    def _state_bounds(self):
        bounds = []
        for row in self.model.state_schema():
            value = row.get("bounds")
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                bounds.append((None, None))
                continue
            lo, hi = value
            bounds.append((None if lo is None else float(lo), None if hi is None else float(hi)))
        return bounds

    def _state_scaling(self):
        initial = np.asarray(self.model.initial_state(), dtype=float)
        scales = []
        offsets = []
        for value, (lo, hi) in zip(initial, self.state_bounds):
            if lo is not None and hi is not None:
                scales.append(max(hi - lo, 1e-6))
                offsets.append(lo)
            else:
                scales.append(max(abs(float(value)), 1.0))
                offsets.append(0.0)
        return np.asarray(scales, dtype=float), np.asarray(offsets, dtype=float)

    def _temperature_soft_cap(self):
        if getattr(self.model, "oracle_temperature_cap", True) is False:
            return None
        caps = []
        for row in self.model.constraint_schema():
            name = row.get("name", "")
            bounds = row.get("bounds")
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or bounds[1] is None:
                continue
            if "temperature" in name:
                caps.append(float(bounds[1]))
        if caps:
            return min(caps) - 2.0
        return 40.0

    # Match the environment Integrator exactly: RK4 with enough fixed substeps
    # that no substep exceeds the model contract's solver max_step.
    def _rk4(self, x, u, d):
        f = lambda xx: self.model.dynamics(xx, u, d, backend="casadi", ca=ca)
        nsub = self.integration_substeps
        h = self.dt / nsub
        for _ in range(nsub):
            k1 = f(x); k2 = f(x + 0.5 * h * k1); k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def _stage_cost(self, x, u, sp, d, ideal_energy):
        if self.goal == "economic":
            return -self._econ_profit(x, u, d)            # minimize -profit
        return self._tracking_state_cost(x, sp)

    def _tracking_state_cost(self, x, sp):
        y = self.model.controlled_output(x, backend="casadi", ca=ca)
        scales = self.model.controlled_output_scales()
        c = 0
        for i, yi in enumerate(y):
            weight = self.q_y[i] if i < len(self.q_y) else 1.0
            scale = float(scales[i]) if i < len(scales) else 1.0
            c += weight * ((yi - sp["y_sp"][i]) / max(scale, 1e-12)) ** 2
        return c

    @staticmethod
    def _smooth_abs(value, epsilon=1e-2):
        return ca.sqrt(value * value + epsilon * epsilon) - epsilon

    @classmethod
    def _smooth_positive(cls, value, epsilon=1e-2):
        return 0.5 * (value + cls._smooth_abs(value, epsilon) + epsilon)

    def _normalized_absolute_tracking_error(self, x, sp):
        y = self.model.controlled_output(x, backend="casadi", ca=ca)
        scales = self.model.controlled_output_scales()
        total = 0
        for i, yi in enumerate(y):
            scale = float(scales[i]) if i < len(scales) else 1.0
            error = (yi - sp["y_sp"][i]) / max(scale, 1e-12)
            total += self._smooth_abs(error)
        return total / max(len(y), 1)

    def _safety_state_cost(self, x):
        """Continuous surrogate for the evaluator's per-step safety indicator."""

        display = self.model.display_outputs(x, backend="casadi", ca=ca)
        levels = list(display.get("levels", []))
        temps = list(display.get("temps", []))
        heights = list(getattr(self.model, "height_max", []))
        cost = 0
        for i, level in enumerate(levels):
            height = float(heights[i]) if i < len(heights) else 1.0
            scale = max(0.1 * height, 1e-9)
            cost += self._smooth_positive((0.15 * height - level) / scale)
            cost += self._smooth_positive((level - 0.90 * height) / scale)
        for temp in temps:
            cost += self._smooth_positive((temp - 80.0) / 10.0)
            cost += self._smooth_positive((temp - 92.0) / 10.0)
        return cost

    def _kpi_stage_cost(self, x, u, sp, ideal_energy):
        mean_error = self._normalized_absolute_tracking_error(x, sp)
        energy_cost = 0
        if bool(getattr(self.model, "energy_scored", True)):
            energy = self.model.energy_kw(u, backend="casadi", ca=ca)
            energy_cost = 0.5 * self._smooth_positive(energy - ideal_energy)
        safety_violation = self._safety_state_cost(x)
        safety_cost = 60.0 * safety_violation / (1.0 + safety_violation)
        return 20.0 * mean_error + energy_cost + safety_cost

    def _econ_profit(self, x, u, d):
        cfg = self.econ
        display = self.model.display_outputs(x, backend="casadi", ca=ca)
        temps = display.get("temps", [])
        env = self.model.dynamics_disturbance_map(d)
        value = self.model.economic_value(x, u, env, backend="casadi", ca=ca)
        energy = self.model.energy_kw(u, backend="casadi", ca=ca)
        shortfall_resolver = getattr(self.model, "product_flow_shortfall", None)
        shortfall = (
            shortfall_resolver(value, backend="casadi", ca=ca)
            if callable(shortfall_resolver)
            else 0.0
        )
        viol = 0
        for i, (lo, hi) in enumerate(cfg["temp_band"]):
            if lo is not None:
                viol += ca.fmax(0, lo - temps[i]) / 10.0
            if hi is not None:
                viol += ca.fmax(0, temps[i] - hi) / 10.0
        level_scale = cfg.get("level_scale", 0.1)
        y = self.model.controlled_output(x, backend="casadi", ca=ca)
        for j, (lo, hi) in enumerate(cfg["level_band"]):
            cv = y[j] if j < len(y) else 0
            if lo is not None:
                viol += ca.fmax(0, lo - cv) / level_scale
            if hi is not None:
                viol += ca.fmax(0, cv - hi) / level_scale
        return (
            cfg["w_value"] * value
            - cfg["w_energy"] * energy
            - cfg["w_viol"] * viol
            - float(cfg.get("w_product_shortfall", 0.0)) * shortfall
        )

    def _build(self):
        if self.transcription == "single_shooting":
            self._build_single_shooting()
        else:
            self._build_multiple_shooting()

    def _solver_options(self):
        return {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0,
                "ipopt.max_iter": self.ipopt_max_iter,
                "ipopt.acceptable_tol": self.ipopt_tol,
                "ipopt.tol": self.ipopt_tol,
                "ipopt.hessian_approximation": self.ipopt_hessian_approximation}

    def _constrain_state(self, opti, x, slack=None):
        slack_value = 0 if slack is None else slack
        for i, (lo, hi) in enumerate(self.state_bounds):
            if lo is not None and hi is not None:
                opti.subject_to(x[i] >= lo - slack_value)
                opti.subject_to(x[i] <= hi + slack_value)
            elif lo is not None:
                opti.subject_to(x[i] >= lo - slack_value)
            elif hi is not None:
                opti.subject_to(x[i] <= hi + slack_value)

    def _build_multiple_shooting(self):
        N, nx, nu = self.N, self.nx, self.nu
        opti = ca.Opti()
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        opti.set_linear_scale(
            X,
            ca.repmat(ca.DM(self.state_scale), 1, N + 1),
            ca.repmat(ca.DM(self.state_offset), 1, N + 1),
        )
        x0 = opti.parameter(nx)
        d = opti.parameter(self.nd)
        u_prev = opti.parameter(nu)
        u_target = opti.parameter(nu)
        ysp = opti.parameter(self.ny, N)
        ideal_energy = opti.parameter()
        J = 0
        opti.subject_to(X[:, 0] == x0)
        slack = opti.variable(1, N)                                    # shared state/cap feasibility slack
        opti.subject_to(slack >= 0)
        for i, (lo, hi) in enumerate(self.state_bounds):
            # X[:, 0] is a measured parameter-equivalent state. Do not make the
            # NLP infeasible merely because the real plant has already crossed
            # a nominal bound; softly drive predicted future states back instead.
            for k in range(1, N + 1):
                state_slack = slack[0, k - 1]
                if lo is not None and hi is not None:
                    opti.subject_to(X[i, k] >= lo - state_slack)
                    opti.subject_to(X[i, k] <= hi + state_slack)
                elif lo is not None:
                    opti.subject_to(X[i, k] >= lo - state_slack)
                elif hi is not None:
                    opti.subject_to(X[i, k] <= hi + state_slack)
        for k in range(N):
            sp = {"y_sp": [ysp[i, k] for i in range(self.ny)]}
            opti.subject_to(X[:, k + 1] == self._rk4(X[:, k], U[:, k], d))
            opti.subject_to(opti.bounded(0.0, U[:, k], 1.0))            # actuators in [0,1]
            up = u_prev if k == 0 else U[:, k - 1]
            temps = self.model.display_outputs(X[:, k + 1], backend="casadi", ca=ca).get("temps", [])
            if self.t_safe is not None:
                for temp in temps:                                     # soft safety cap on temperature outputs
                    opti.subject_to(temp <= self.t_safe + slack[0, k])
            J += (
                self._stage_cost(X[:, k + 1], U[:, k], sp, d, ideal_energy)
                + self.r_move * ca.sumsqr(U[:, k] - up)
                + self.steady_input_weight * ca.sumsqr(U[:, k] - u_target)
            )
        if self.goal == "regulation":
            terminal_sp = {"y_sp": [ysp[i, N - 1] for i in range(self.ny)]}
            J += self.terminal_weight * self._tracking_state_cost(X[:, N], terminal_sp)
        J += 1e4 * ca.sumsqr(slack)                                    # heavily discourage cap violation
        opti.minimize(J)
        opti.solver("ipopt", self._solver_options())
        self.opti, self.X, self.U, self.slack = opti, X, U, slack
        self.par = {
            "x0": x0, "d": d, "u_prev": u_prev, "u_target": u_target, "ysp": ysp,
            "ideal_energy": ideal_energy,
        }

    def _build_single_shooting(self):
        N, nx, nu = self.N, self.nx, self.nu
        opti = ca.Opti()
        U = opti.variable(nu, N)
        x0 = opti.parameter(nx)
        d = opti.parameter(self.nd)
        u_prev = opti.parameter(nu)
        u_target = opti.parameter(nu)
        ysp = opti.parameter(self.ny, N)
        ideal_energy = opti.parameter()
        slack = opti.variable(1, N)
        opti.subject_to(slack >= 0)
        x = x0
        J = 0
        for k in range(N):
            sp = {"y_sp": [ysp[i, k] for i in range(self.ny)]}
            up = u_prev if k == 0 else U[:, k - 1]
            opti.subject_to(opti.bounded(0.0, U[:, k], 1.0))
            x_next = self._rk4(x, U[:, k], d)
            J += (
                self._stage_cost(x_next, U[:, k], sp, d, ideal_energy)
                + self.r_move * ca.sumsqr(U[:, k] - up)
                + self.steady_input_weight * ca.sumsqr(U[:, k] - u_target)
            )
            x = x_next
            self._constrain_state(opti, x, slack=slack[0, k])
            temps = self.model.display_outputs(x, backend="casadi", ca=ca).get("temps", [])
            if self.t_safe is not None:
                for temp in temps:
                    opti.subject_to(temp <= self.t_safe + slack[0, k])
        if self.goal == "regulation":
            terminal_sp = {"y_sp": [ysp[i, N - 1] for i in range(self.ny)]}
            J += self.terminal_weight * self._tracking_state_cost(x, terminal_sp)
        J += 1e4 * ca.sumsqr(slack)
        opti.minimize(J)
        opti.solver("ipopt", self._solver_options())
        self.opti, self.X, self.U, self.slack = opti, None, U, slack
        self.par = {
            "x0": x0, "d": d, "u_prev": u_prev, "u_target": u_target, "ysp": ysp,
            "ideal_energy": ideal_energy,
        }

    def _numeric_rk4(self, x, u, disturbances):
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        nsub = self.integration_substeps
        h = self.dt / nsub
        for _ in range(nsub):
            k1 = np.asarray(self.model.dynamics(x, u, disturbances), dtype=float)
            k2 = np.asarray(self.model.dynamics(x + 0.5 * h * k1, u, disturbances), dtype=float)
            k3 = np.asarray(self.model.dynamics(x + 0.5 * h * k2, u, disturbances), dtype=float)
            k4 = np.asarray(self.model.dynamics(x + h * k3, u, disturbances), dtype=float)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if callable(getattr(self.model, "clamp_state", None)):
                x = np.asarray(self.model.clamp_state(list(x)), dtype=float)
        return x

    def _initial_state_guess(self, x, u, d):
        disturbances = self.model.dynamics_disturbance_map(d)
        cols = [np.asarray(x, dtype=float)]
        cur = np.asarray(x, dtype=float)
        for _ in range(self.N):
            cur = self._numeric_rk4(cur, u, disturbances)
            cols.append(cur)
        return np.column_stack(cols)

    def reset(self):
        self.u_prev = self.u_init.copy()
        self._warm = None
        self._warm_target = None
        self.last_plan = None
        self.last_error = None
        self.last_solve_status = None
        self.last_solve_success = False

    def _fallback_action(self):
        """Move safely toward the model's nominal equilibrium action."""

        return self.u_init.copy()

    def _steady_action_target(self, target):
        resolver = getattr(self.model, "tracking_steady_state_action", None)
        uses_setpoint = self.goal == "regulation"
        values = resolver(target) if uses_setpoint and callable(resolver) else None
        if values is None:
            return self.u_init.copy()
        action = np.asarray(values, dtype=float).reshape(-1)
        if len(action) != self.nu or not np.all(np.isfinite(action)):
            return self.u_init.copy()
        return np.clip(action, 0.0, 1.0)

    def _disturbance_vector(self, t_cold, t_amb, disturbances=None):
        values = dict(disturbances or {})
        values.update({"t_cold": t_cold, "t_amb": t_amb})
        return self.model.disturbance_vector(values)

    def _setpoint_trajectory(self, y_sp, y_sp_preview=None):
        current = self.model.setpoint_vector(y_sp)
        if y_sp_preview is None:
            targets = [current] * self.N
        else:
            targets = [self.model.setpoint_vector(values) for values in y_sp_preview]
            if not targets:
                targets = [current]
            targets = targets[:self.N]
            targets.extend([targets[-1]] * (self.N - len(targets)))
        return current, np.asarray(targets, dtype=float).T

    def solve(self, x, t_cold, t_amb, disturbances=None, y_sp=None,
              y_sp_preview=None, advance_steps=1):
        o = self.opti
        o.set_value(self.par["x0"], np.asarray(x, float))
        if self.nd:
            o.set_value(self.par["d"], self._disturbance_vector(t_cold, t_amb, disturbances))
        o.set_value(self.par["u_prev"], self.u_prev)
        target, target_trajectory = self._setpoint_trajectory(y_sp, y_sp_preview)
        o.set_value(self.par["ysp"], target_trajectory)
        initial_action = self._steady_action_target(target)
        o.set_value(self.par["u_target"], initial_action)
        dvec = self._disturbance_vector(t_cold, t_amb, disturbances)
        disturbance_map = self.model.dynamics_disturbance_map(dvec)
        try:
            ideal_energy = self.model.ideal_energy_kw(
                np.asarray(x, float),
                target,
                disturbance_map,
                initial_action,
            )
        except (TypeError, ValueError, KeyError):
            ideal_energy = 0.0
        if not np.isfinite(ideal_energy):
            ideal_energy = 0.0
        o.set_value(self.par["ideal_energy"], float(ideal_energy))
        target_signature = tuple(float(value) for value in target)
        try:
            use_warm_start = (
                self.warm_start
                and self._warm is not None
                and self._warm_target == target_signature
            )
            if not use_warm_start:
                if self.X is not None:
                    o.set_initial(self.X, self._initial_state_guess(x, initial_action, dvec))
                o.set_initial(self.U, np.tile(initial_action.reshape(-1, 1), (1, self.N)))
                o.set_initial(self.slack, np.zeros((1, self.N)))
            else:
                Xw, Uw, Sw = self._warm
                shift = min(max(int(advance_steps), 1), self.N)
                if self.X is not None:
                    Xw = np.column_stack([
                        np.asarray(x, float),
                        Xw[:, shift + 1:],
                        np.tile(Xw[:, -1:], (1, shift)),
                    ])
                    o.set_initial(self.X, Xw)
                Uw = np.column_stack([
                    Uw[:, shift:],
                    np.tile(Uw[:, -1:], (1, shift)),
                ])
                Sw = np.asarray(Sw, float).reshape(1, -1)
                Sw = np.column_stack([
                    Sw[:, shift:],
                    np.tile(Sw[:, -1:], (1, shift)),
                ])
                o.set_initial(self.U, Uw)
                o.set_initial(self.slack, Sw)
            sol = o.solve_limited()
            stats = sol.stats()
            self.last_solve_status = str(stats.get("return_status", "unknown"))
            self.last_solve_success = bool(stats.get("success", False))
            Xv = None if self.X is None else np.asarray(sol.value(self.X), float).reshape(self.nx, self.N + 1)
            Uv = np.asarray(sol.value(self.U), float).reshape(self.nu, self.N)
            Sv = np.asarray(sol.value(self.slack), float).reshape(1, self.N)
            if self.warm_start:
                self._warm = (Xv, Uv, Sv)
                self._warm_target = target_signature
            self.last_plan = Uv.copy()
            u = np.clip(Uv[:, 0], 0.0, 1.0)
            self.last_error = None
        except Exception as e:
            self.last_error = e
            self.last_solve_status = None
            self.last_solve_success = False
            self.last_plan = None
            u = self._fallback_action()
        self.u_prev = np.asarray(u, float).reshape(-1)
        return list(self.u_prev)


class OracleAgent:
    """Controller-v1 wrapper around the nonlinear MPC oracle."""
    name = "NMPC-oracle"
    controller_api_version = "aiogym.controller.v1"
    action_mode = "actuator"
    control_structure = "nmpc_oracle"

    def __init__(self, scenario, solve_every=1, preview_setpoints=False, model=None, **kw):
        self.orc = NMPCOracle(scenario, model=model, **kw)
        self.scenario = self.orc.scenario
        self.model = self.orc.model
        self.solve_every = positive_int("solve_every", solve_every)
        self.preview_setpoints = bool(preview_setpoints)
        self._ticks = 0
        self._last = None
        self._plan_step = 0
        self._last_plan_signature = None
        self.solve_count = 0
        self.solver_success_count = 0
        self.solver_limited_count = 0
        self.solver_failure_count = 0
        self.fallback_count = 0
        self.last_solver_error = None

    def metadata(self):
        return {"name": self.name, "class": self.__class__.__name__,
                "kind": "nonlinear_mpc_oracle", "scenario": self.scenario,
                "api": self.controller_api_version,
                "action_mode": self.action_mode, "control_structure": self.control_structure,
                "solve_every": self.solve_every,
                "preview_setpoints": self.preview_setpoints,
                "horizon": self.orc.N, "control_dt": self.orc.dt,
                "goal": self.orc.goal,
                "reward_spec_id": self.orc.reward_spec_id,
                "q_y": list(self.orc.q_y),
                "r_move": self.orc.r_move,
                "terminal_weight": self.orc.terminal_weight,
                "steady_input_weight": self.orc.steady_input_weight,
                "initialization": "tracking_steady_state_action",
                "integration_max_step": self.orc.integration_max_step,
                "integration_substeps": self.orc.integration_substeps,
                "ipopt_max_iter": self.orc.ipopt_max_iter,
                "ipopt_tol": self.orc.ipopt_tol,
                "ipopt_hessian_approximation": (
                    self.orc.ipopt_hessian_approximation
                ),
                "last_solve_status": self.orc.last_solve_status,
                "warm_start": self.orc.warm_start,
                "transcription": self.orc.transcription,
                "enforce_state_bounds": self.orc.enforce_state_bounds,
                "enforce_temperature_cap": self.orc.enforce_temperature_cap,
                "diagnostics": self.diagnostics()}

    def diagnostics(self):
        return {
            "solve_count": int(self.solve_count),
            "solver_success_count": int(self.solver_success_count),
            "solver_limited_count": int(self.solver_limited_count),
            "solver_failure_count": int(self.solver_failure_count),
            "fallback_count": int(self.fallback_count),
            "degraded": bool(
                self.solver_limited_count
                or self.solver_failure_count
                or self.fallback_count
            ),
            "last_solve_status": self.orc.last_solve_status,
            "last_solver_error": self.last_solver_error,
        }

    def reset(self, seed=None):
        self.orc.reset()
        self._ticks = 0
        self._last = None
        self._plan_step = 0
        self._last_plan_signature = None
        self.solve_count = 0
        self.solver_success_count = 0
        self.solver_limited_count = 0
        self.solver_failure_count = 0
        self.fallback_count = 0
        self.last_solver_error = None

    def act(self, obs, context):
        preview = self._preview_targets(context) if self.preview_setpoints else None
        action = self.compute(
            context.measurement,
            context.setpoint,
            context.control_dt,
            y_sp_preview=preview,
        )
        return np.asarray(self.model.action_vector(action), dtype=np.float32)

    def _preview_targets(self, context):
        """Return the active reference for each future stage in the NLP horizon."""

        current = list(self.model.setpoint_vector(context.setpoint.get("y_sp")))
        env = getattr(context, "env", None)
        if env is None:
            return [current] * self.orc.N
        step = int(getattr(env, "_k", 0))
        events = dict(getattr(env, "_episode_setpoint_events", {}) or {})
        targets = []
        active = current
        for offset in range(self.orc.N):
            event = events.get(step + offset)
            if event is not None:
                active = list(self.model.setpoint_vector(event))
            targets.append(list(active))
        return targets

    def _x_from_meas(self, meas):
        if "x" in meas:
            return list(meas["x"])
        raise ValueError("oracle controller requires measurement['x']")

    def _plan_signature(self, meas, sp, y_sp_preview=None):
        """Return exogenous values that should trigger an immediate replan."""

        target = tuple(float(value) for value in self.model.setpoint_vector(sp.get("y_sp")))
        disturbances = tuple(float(value) for value in self.model.disturbance_vector(meas))
        preview = None
        if y_sp_preview is not None:
            preview = tuple(tuple(float(value) for value in row) for row in y_sp_preview)
        return target, disturbances, preview

    def compute(self, meas, sp, dt, y_sp_preview=None):
        plan_signature = self._plan_signature(meas, sp, y_sp_preview)
        should_solve = (
            self._last is None
            or self._plan_step >= self.solve_every
            or plan_signature != self._last_plan_signature
        )
        if should_solve:
            solve_kwargs = {
                "disturbances": meas,
                "y_sp": sp.get("y_sp"),
                "advance_steps": max(self._plan_step, 1),
            }
            if y_sp_preview is not None:
                solve_kwargs["y_sp_preview"] = y_sp_preview
            self._last = self.orc.solve(
                self._x_from_meas(meas),
                meas.get("t_cold", 15.0),
                meas.get("t_amb", 20.0),
                **solve_kwargs,
            )
            self._plan_step = 0
            self._last_plan_signature = plan_signature
            self.solve_count += 1
            if self.orc.last_error is None:
                if self.orc.last_solve_success:
                    self.solver_success_count += 1
                else:
                    self.solver_limited_count += 1
            else:
                self.solver_failure_count += 1
                self.fallback_count += 1
                self.last_solver_error = f"{self.orc.last_error.__class__.__name__}: {self.orc.last_error}"
        plan = self.orc.last_plan
        if plan is not None and self._plan_step < plan.shape[1]:
            self._last = np.clip(plan[:, self._plan_step], 0.0, 1.0)
            self.orc.u_prev = np.asarray(self._last, dtype=float).reshape(-1)
        self._plan_step += 1
        self._ticks += 1
        return self._last
