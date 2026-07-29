"""Johansson's nonlinear quadruple-tank laboratory process."""
from __future__ import annotations

import math

from ..core import ProcessModelContract, _maxv


class QuadrupleModel(ProcessModelContract):
    """Four liquid levels driven by two split pump flows.

    Internal units follow Johansson (2000): level in cm, area in cm², flow in
    cm³/s, voltage in V, and time in s. AIO-Gym actions remain normalized to
    ``[0, 1]`` and are mapped to physical pump voltage by ``max_voltage``.
    """

    scenario = "quadruple"
    benchmark_goals = ("regulation",)
    display_name = "Johansson quadruple-tank process"
    summary = "Four-state, two-input nonlinear level process with adjustable minimum/nonminimum-phase behavior."
    n = 4
    dt_micro = 0.1
    state_names = ("h1", "h2", "h3", "h4")
    state_units = {name: "cm" for name in state_names}
    state_bounds = {name: (0.0, 20.0) for name in state_names}
    action_names = ("pump_1_voltage", "pump_2_voltage")
    action_units = {name: "normalized_voltage" for name in action_names}
    action_bounds = {name: (0.0, 1.0) for name in action_names}
    action_kinds = {name: "pump" for name in action_names}
    output_names = ("lower_tank_1_level", "lower_tank_2_level")
    output_units = {name: "cm" for name in output_names}
    output_bounds = {name: (0.0, 20.0) for name in output_names}
    setpoint_names = ("h1_sp", "h2_sp")
    setpoint_units = {name: "cm" for name in setpoint_names}
    setpoint_bounds = {name: (0.0, 20.0) for name in setpoint_names}
    # Model-consistent equilibrium for v1=v2=3 V at the P- parameters.
    default_y_sp = (12.2629675195507, 12.783158403008972)
    plant_regime = {"outlet_area": (0.9, 1.1), "pump_gain": (0.9, 1.1)}
    economic_config = {
        "temp_band": (),
        "level_band": ((0.0, 20.0), (0.0, 20.0)),
        "value": "none",
        "w_value": 0.0,
        "w_energy": 0.0,
        "w_viol": 30.0,
    }
    supervisory_layout = (("y_sp", 0, 5.0, 18.0), ("y_sp", 1, 5.0, 18.0))
    param_units = {
        "tank_area": "cm2",
        "outlet_area": "cm2",
        "pump_gain": "cm3/(V*s)",
        "gamma": "fraction",
        "gravity": "cm/s2",
        "max_voltage": "V",
        "max_level": "cm",
        "nominal_voltage": "V",
    }
    param_bounds = {
        "tank_area": (1.0, 1000.0),
        "outlet_area": (1e-4, 10.0),
        "pump_gain": (1e-4, 100.0),
        "gamma": (0.01, 0.99),
        "gravity": (900.0, 1100.0),
        "max_voltage": (0.1, 20.0),
        "max_level": (1.0, 100.0),
        "nominal_voltage": (0.0, 20.0),
    }
    input_disturbances = (
        {
            "name": "pump_flow_factor",
            "event": "pump_capacity_shift",
            "unit": "fraction",
            "bounds": (0.7, 1.3),
            "default": 1.0,
            "description": "common multiplicative pump-flow change",
        },
        {
            "name": "outlet_area_factor",
            "event": "outlet_restriction_shift",
            "unit": "fraction",
            "bounds": (0.7, 1.3),
            "default": 1.0,
            "description": "common multiplicative outlet-area change",
        },
    )
    event_disturbances = ProcessModelContract.event_disturbances
    safety_constraints = (
        {"name": "tank_level_bounds", "states": state_names, "bounds": (0.0, 20.0)},
    )
    randomize_common_temperatures = False
    energy_scored = False
    not_applicable_physics = ("energy_balance",)

    def __init__(self):
        self.p = {
            "tank_area": [28.0, 32.0, 28.0, 32.0],
            "outlet_area": [0.071, 0.057, 0.071, 0.057],
            "pump_gain": [3.33, 3.35],
            "gamma": [0.70, 0.60],
            "gravity": 981.0,
            "max_voltage": 10.0,
            "max_level": 20.0,
            "nominal_voltage": [3.0, 3.0],
        }

    @property
    def height_max(self):
        return [float(self.p["max_level"])] * 4

    @property
    def phase_configuration(self):
        total = float(sum(self.p["gamma"]))
        if total > 1.0:
            return "minimum-phase"
        if total < 1.0:
            return "nonminimum-phase"
        return "zero-at-origin"

    def default_action(self):
        maximum = float(self.p["max_voltage"])
        return [float(value) / maximum for value in self.p["nominal_voltage"]]

    def physical_action_vector(self, act):
        """Map canonical pump commands in [0, 1] to physical volts."""

        maximum = float(self.p["max_voltage"])
        return [value * maximum for value in self.action_vector(act)]

    def mpc_init(self):
        return self.default_action()

    def equilibrium_state(self, voltage=None):
        """Return the exact nonlinear equilibrium for a physical voltage pair."""

        a = self.p["outlet_area"]
        k = self.p["pump_gain"]
        gamma = self.p["gamma"]
        g = float(self.p["gravity"])
        values = self.p["nominal_voltage"] if voltage is None else voltage
        v1, v2 = (float(value) for value in values)
        q3 = (1.0 - gamma[1]) * k[1] * v2
        q4 = (1.0 - gamma[0]) * k[0] * v1
        h3 = (q3 / a[2]) ** 2 / (2.0 * g)
        h4 = (q4 / a[3]) ** 2 / (2.0 * g)
        q1 = gamma[0] * k[0] * v1 + q3
        q2 = gamma[1] * k[1] * v2 + q4
        h1 = (q1 / a[0]) ** 2 / (2.0 * g)
        h2 = (q2 / a[1]) ** 2 / (2.0 * g)
        return [h1, h2, h3, h4]

    def tracking_steady_state_action(self, y_sp):
        """Return the nominal steady input associated with lower-tank targets.

        The inverse uses the two steady lower-tank flow balances.  It is
        especially useful in the nonminimum-phase configuration, where a short
        prediction horizon otherwise rewards the inverse response and can pick
        the wrong pump allocation before the eventual benefit is visible.
        Returns ``None`` when the steady-state map is singular and no unique
        target-specific input exists.
        """

        target = [max(float(value), 0.0) for value in y_sp]
        if len(target) != 2:
            raise ValueError(f"quadruple y_sp must contain 2 values, got {len(target)}")
        a = self.p["outlet_area"]
        k1, k2 = (float(value) for value in self.p["pump_gain"])
        gamma1, gamma2 = (float(value) for value in self.p["gamma"])
        g = float(self.p["gravity"])
        required = [
            float(a[i]) * math.sqrt(2.0 * g * target[i])
            for i in range(2)
        ]
        matrix = (
            (gamma1 * k1, (1.0 - gamma2) * k2),
            ((1.0 - gamma1) * k1, gamma2 * k2),
        )
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        if abs(determinant) < 1e-12:
            # At the zero boundary the lower-tank steady-state map is singular,
            # so there is no unique target-specific steady action to score.
            return None
        v1 = (required[0] * matrix[1][1] - matrix[0][1] * required[1]) / determinant
        v2 = (matrix[0][0] * required[1] - required[0] * matrix[1][0]) / determinant
        vmax = float(self.p["max_voltage"])
        return [max(0.0, min(1.0, v1 / vmax)), max(0.0, min(1.0, v2 / vmax))]

    def is_setpoint_reachable(self, y_sp):
        if not super().is_setpoint_reachable(y_sp):
            return False
        action = self.tracking_steady_state_action(y_sp)
        if action is None:
            return False
        equilibrium = self.equilibrium_state(self.physical_action_vector(action))
        output = self.controlled_output(equilibrium)
        if any(abs(float(actual) - float(target)) > 1e-7
               for actual, target in zip(output, y_sp)):
            return False
        for value, row in zip(equilibrium, self.state_schema()):
            bounds = row.get("bounds")
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                continue
            lo, hi = bounds
            if lo is not None and float(value) < float(lo):
                return False
            if hi is not None and float(value) > float(hi):
                return False
        return True

    def sample_env_setpoints(self, y_sp, rng, options=None):
        options = dict(options or {})
        reference = [float(value) for value in y_sp]
        if not options.get("randomize_setpoints", False):
            return reference
        sampled = super().sample_env_setpoints(reference, rng, options)
        if sampled != reference or self.is_setpoint_reachable(sampled):
            return sampled

        # At gamma1 + gamma2 = 1 the lower-tank steady-state map is singular.
        # Sampling a safe steady actuator pair and mapping it forward still
        # produces a reproducible target on the physically reachable manifold.
        anchor = self.tracking_steady_state_action(reference)
        if anchor is None:
            anchor = self.default_action()
        for _ in range(64):
            action = [
                max(0.0, min(1.0, float(value) + float(rng.uniform(-0.05, 0.05))))
                for value in anchor
            ]
            equilibrium = self.equilibrium_state(self.physical_action_vector(action))
            safe = True
            for value, row in zip(equilibrium, self.state_schema()):
                bounds = row.get("bounds")
                if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                    continue
                lo, hi = bounds
                safe = (
                    safe
                    and (lo is None or float(value) >= float(lo))
                    and (hi is None or float(value) <= float(hi))
                )
            if safe:
                return list(self.controlled_output(equilibrium))
        return reference

    def initial_state(self):
        return self.equilibrium_state()

    def default_setpoint_vector(self):
        return list(self.controlled_output(self.initial_state()))

    def linearized_time_constants(self, state=None):
        """Return Johansson's four local time constants at ``state``."""

        x = self.initial_state() if state is None else self.state_vector(state)
        A = self.p["tank_area"]
        a = self.p["outlet_area"]
        g = float(self.p["gravity"])
        return [
            float(A[i]) / float(a[i]) * math.sqrt(2.0 * max(float(x[i]), 0.0) / g)
            for i in range(4)
        ]

    def transmission_zeros(self, state=None):
        """Return the two finite zeros of the local 2x2 transfer matrix."""

        gamma1, gamma2 = (float(value) for value in self.p["gamma"])
        T3, T4 = self.linearized_time_constants(state)[2:]
        eta = (1.0 - gamma1) * (1.0 - gamma2) / (gamma1 * gamma2)
        b = T3 + T4
        discriminant = max(0.0, b * b - 4.0 * T3 * T4 * (1.0 - eta))
        root = math.sqrt(discriminant)
        denominator = 2.0 * T3 * T4
        return [float((-b - root) / denominator), float((-b + root) / denominator)]

    def _dynamics(self, x, u, env, ops):
        A = self.p["tank_area"]
        a = self.p["outlet_area"]
        k = self.p["pump_gain"]
        gamma = self.p["gamma"]
        g = self.p["gravity"]
        vmax = self.p["max_voltage"]
        pump_factor = env.get("pump_flow_factor", 1.0)
        outlet_factor = env.get("outlet_area_factor", 1.0)
        voltage = [u[i] * vmax for i in range(2)]
        outlet = [
            # The numeric plant keeps the exact max(h, 0) law. CasADi uses the
            # smooth counterpart so the square-root derivative stays finite as
            # an NMPC prediction approaches an empty tank.
            outlet_factor * a[i] * ops.sqrt(2.0 * g * ops.smooth_max(x[i], 0.0, 1e-6))
            for i in range(4)
        ]
        pump = [pump_factor * k[i] * voltage[i] for i in range(2)]
        return ops.vector([
            (-outlet[0] + outlet[2] + gamma[0] * pump[0]) / A[0],
            (-outlet[1] + outlet[3] + gamma[1] * pump[1]) / A[1],
            (-outlet[2] + (1.0 - gamma[1]) * pump[1]) / A[2],
            (-outlet[3] + (1.0 - gamma[0]) * pump[0]) / A[3],
        ])

    def display_outputs(self, x, backend="numeric", ca=None):
        if backend == "casadi":
            return {"levels": [x[i] for i in range(4)], "temps": []}
        return {"levels": [_maxv(float(x[i]), 0.0) for i in range(4)], "temps": []}

    def controlled_output(self, x, backend="numeric", ca=None):
        return [x[0], x[1]] if backend == "casadi" else [_maxv(float(x[0]), 0.0), _maxv(float(x[1]), 0.0)]

    def clamp_state(self, x):
        return [_maxv(float(value), 0.0) for value in x]

    def sample_disturbance(self, event, current, rng):
        if event in {"pump_capacity_shift", "outlet_restriction_shift"}:
            return float(max(0.7, min(1.3, float(current) + rng.uniform(-0.2, 0.2))))
        return super().sample_disturbance(event, current, rng)

    def process_info(self, x, levels, temps, env):
        return {
            "quadruple_phase_configuration": self.phase_configuration,
            "pump_flow_factor": env.get("pump_flow_factor", 1.0),
            "outlet_area_factor": env.get("outlet_area_factor", 1.0),
        }

    def common_constraint_info(self, levels, temps):
        maximum = float(self.p["max_level"])
        return {
            "level_high": max((max(0.0, float(level) - maximum) for level in levels), default=0.0),
            "level_low": max((max(0.0, -float(level)) for level in levels), default=0.0),
        }

    def runaway_state(self, levels, temps):
        return any(float(level) > float(self.p["max_level"]) for level in levels)

    def mass_balance_residual(self, x, action, env=None):
        """Return total volumetric balance residual in cm³/s."""

        values = self.action_vector(action)
        context = self.runtime_env(self.disturbance_defaults())
        context.update(dict(env or {}))
        dx = self.dynamics(x, values, context)
        A = self.p["tank_area"]
        a = self.p["outlet_area"]
        g = self.p["gravity"]
        outlet_factor = context.get("outlet_area_factor", 1.0)
        external_out = sum(
            outlet_factor * a[i] * math.sqrt(2.0 * g * max(float(x[i]), 0.0))
            for i in (0, 1)
        )
        pump_factor = context.get("pump_flow_factor", 1.0)
        external_in = pump_factor * sum(
            self.p["pump_gain"][i] * values[i] * self.p["max_voltage"]
            for i in range(2)
        )
        accumulation = sum(A[i] * dx[i] for i in range(4))
        return float(accumulation - (external_in - external_out))

    def physical_validation_checks(self):
        """Return scenario-specific physical acceptance checks."""

        shared = {
            "tank_area": [28.0, 32.0, 28.0, 32.0],
            "outlet_area": [0.071, 0.057, 0.071, 0.057],
            "gravity": 981.0,
        }
        references = {
            "P-": {"pump_gain": [3.33, 3.35], "gamma": [0.70, 0.60], "nominal_voltage": [3.0, 3.0]},
            "P+": {"pump_gain": [3.14, 3.29], "gamma": [0.43, 0.34], "nominal_voltage": [3.15, 3.15]},
        }
        reference_name = next((
            name for name, values in references.items()
            if all(self.p[key] == value for key, value in {**shared, **values}.items())
        ), None)
        x0 = self.initial_state()
        dx0 = self.dynamics(x0, self.default_action(), self.disturbance_defaults())
        zeros = self.transmission_zeros(x0)
        phase_ok = (
            (self.phase_configuration == "minimum-phase" and max(zeros) < 0.0)
            or (self.phase_configuration == "nonminimum-phase" and max(zeros) > 0.0)
            or (self.phase_configuration == "zero-at-origin" and min(abs(value) for value in zeros) < 1e-12)
        )
        return [
            {
                "name": "reference_parameters",
                "passed": reference_name is not None,
                "detail": f"Johansson 2000 {reference_name or 'custom'} parameter set",
            },
            {
                "name": "nonlinear_equilibrium",
                "passed": max(abs(float(value)) for value in dx0) < 1e-12,
                "detail": f"max_abs_derivative={max(abs(float(value)) for value in dx0):.3e}",
            },
            {
                "name": "mass_balance",
                "passed": abs(self.mass_balance_residual(x0, self.default_action())) < 1e-12,
                "detail": f"residual_cm3_per_s={self.mass_balance_residual(x0, self.default_action()):.3e}",
            },
            {
                "name": "phase_configuration",
                "passed": phase_ok,
                "detail": (
                    f"configuration={self.phase_configuration}, gamma_sum={sum(self.p['gamma']):.3f}, "
                    f"zeros={[round(value, 8) for value in zeros]}"
                ),
            },
        ]
