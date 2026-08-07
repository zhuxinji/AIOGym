"""Topology-neutral mass and energy balance kernel for three-tank plants."""
from __future__ import annotations

import math

from aiogym.core.backends import _maxv
from aiogym.core.model import RHO_CP, ProcessModelContract


class ThreeTankPhysicsKernel(ProcessModelContract):
    """Topology-neutral state, I/O, validation, and balance assembly.

    A topology supplies flow connections, mixing terms, and heat inputs. This
    kernel owns the common six-state ordering and converts those terms into
    the three mass and energy balance pairs.
    """

    n = 3
    dt_micro = 0.02
    energy_scored = True

    @staticmethod
    def _levels_temperatures(x):
        return [x[0], x[2], x[4]], [x[1], x[3], x[5]]

    def _tank_parameter(self, name, index):
        value = self.p[name]
        if isinstance(value, (tuple, list)):
            return value[index]
        return value

    @property
    def state_bounds(self):
        temperature_max = float(self.p["temperature_hard_limit"])
        bounds = {}
        for index, name in enumerate(self.state_names[0::2]):
            bounds[name] = (0.0, self.height_max[index])
        for name in self.state_names[1::2]:
            bounds[name] = (0.0, temperature_max)
        return bounds

    @property
    def output_bounds(self):
        temperature_max = float(self.p["temperature_hard_limit"])
        bounds = {}
        for index, name in enumerate(self.output_names[:3]):
            bounds[name] = (0.0, self.height_max[index])
        for name in self.output_names[3:]:
            bounds[name] = (0.0, temperature_max)
        return bounds

    def controlled_output(self, x, backend="numeric", ca=None):
        del backend, ca
        levels, temperatures = self._levels_temperatures(x)
        return [*levels, *temperatures]

    def display_outputs(self, x, backend="numeric", ca=None):
        del ca
        levels, temperatures = self._levels_temperatures(x)
        if backend != "casadi":
            levels = [_maxv(value, 0.0) for value in levels]
        return {"levels": levels, "temps": temperatures}

    def integral_observation_limits(self):
        return [8.0, 8.0, 8.0, 300.0, 300.0, 300.0]

    def mpc_init(self):
        return self.default_action()

    def _resolved_env(self, env=None, ops=None):
        values = dict(env or {})
        resolved = {}
        for row in self.input_disturbances:
            name = row["name"]
            default = row.get("default", self.p.get(name, 0.0))
            resolved[name] = values.get(name, default)
        if bool(getattr(ops, "symbolic", False)):
            return resolved

        clean = {}
        for name, value in resolved.items():
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"three_tank disturbance {name!r} must be finite"
                ) from exc
            if not math.isfinite(number):
                raise ValueError(
                    f"three_tank disturbance {name!r} must be finite"
                )
            lower, upper = self._environment_bounds[name]
            if lower is not None and number < float(lower):
                raise ValueError(
                    f"three_tank disturbance {name!r} must be within "
                    f"[{lower}, {upper}], got {number}"
                )
            if upper is not None and number > float(upper):
                raise ValueError(
                    f"three_tank disturbance {name!r} must be within "
                    f"[{lower}, {upper}], got {number}"
                )
            clean[name] = number
        return clean

    def runtime_env(self, disturbance_values):
        return self._resolved_env(super().runtime_env(disturbance_values))

    def disturbance_vector(self, values=None):
        return super().disturbance_vector(self.runtime_env(values or {}))

    def _effective_action(self, u, ops):
        if not bool(getattr(ops, "symbolic", False)):
            for value in u:
                if not math.isfinite(float(value)):
                    raise ValueError("three_tank action values must be finite")
        return [
            ops.min(ops.max(u[index], 0.0), 1.0)
            for index in range(self.action_dim())
        ]

    def _assemble_dynamics(
        self,
        levels,
        temperatures,
        flows_in,
        flows_out,
        mixing_terms,
        heat_inputs,
        env,
        ops,
    ):
        derivatives = []
        for index in range(self.n):
            area = self._tank_parameter("area", index)
            volume = area * ops.max(levels[index], self.p["h_floor"])
            heat_loss = (
                self._tank_parameter("ua_loss", index)
                * env["heat_loss_factor"]
                * (temperatures[index] - env["t_amb"])
            )
            derivatives.extend(
                (
                    (flows_in[index] - flows_out[index]) / area,
                    mixing_terms[index] / volume
                    + (heat_inputs[index] - heat_loss) / (RHO_CP * volume),
                )
            )
        return ops.vector(derivatives)


__all__ = ["ThreeTankPhysicsKernel"]
