"""Johansson's nonlinear quadruple-tank laboratory process."""
from __future__ import annotations

import math

from aiogym.core.backends import _maxv
from aiogym.core.model import PhysicsModelBase


class _QuadruplePhysicsKernel(PhysicsModelBase):
    """Four liquid levels driven by two split pump flows.

    Internal units follow Johansson (2000): level in cm, area in cm², flow in
    cm³/s, voltage in V, and time in s. AIO-Gym actions remain normalized to
    ``[0, 1]`` and are mapped to physical pump voltage by ``max_voltage``.
    """

    scenario = "quadruple"
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
    # Model-consistent equilibrium for v1=v2=3 V at the P- parameters.
    default_y_sp = (12.2629675195507, 12.783158403008972)
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
    event_disturbances = PhysicsModelBase.event_disturbances
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

    def initial_state(self):
        return self.equilibrium_state()

    def default_setpoint_vector(self):
        return list(self.controlled_output(self.initial_state()))

    def _dynamics(self, x, u, env, ops):
        A = self.p["tank_area"]
        a = self.p["outlet_area"]
        k = self.p["pump_gain"]
        gamma = self.p["gamma"]
        g = self.p["gravity"]
        vmax = self.p["max_voltage"]
        pump_factor = env["pump_flow_factor"]
        outlet_factor = env["outlet_area_factor"]
        voltage = [u[i] * vmax for i in range(2)]
        outlet = [
            # The numeric model keeps the exact max(h, 0) law. CasADi uses the
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

    def process_info(self, x, levels, temps, env):
        return {
            "quadruple_phase_configuration": self.phase_configuration,
            "pump_flow_factor": env["pump_flow_factor"],
            "outlet_area_factor": env["outlet_area_factor"],
        }



__all__ = []
