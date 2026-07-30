"""Observation, setpoint, and supervisory-action runtime helpers."""
from __future__ import annotations

import numpy as np

from .config import I_TEMP_MAX


class ObservationRuntimeMixin:
    def _split(self, action):
        a = np.clip(np.asarray(action, np.float64), 0.0, 1.0)
        return self.model.action_vector(a)

    def _obs(self):
        measured_state = self._measured_state()
        if self.observation_mode == "full_state":
            state = list(measured_state)
            state_schema = self.model.state_schema()
        else:
            state = list(self.model.controlled_output(measured_state))
            state_schema = self.model.setpoint_schema()
        setpoint = list(self.y_sp)
        observed_output = list(
            self.model.controlled_output(measured_state)
        )
        tracking_error = [
            float(target) - float(value)
            for target, value in zip(setpoint, observed_output)
        ]
        disturbance = list(self.model.disturbance_vector(self._env()))
        previous_action = self.model.action_vector(self.previous_act)
        if self.normalize_observations:
            state = self._normalize_observation_values(
                state, state_schema
            )
            if self.tracking_error_obs:
                tracking_error = self._normalize_observation_deltas(
                    tracking_error, self.model.setpoint_schema()
                )
            else:
                setpoint = self._normalize_observation_values(
                    setpoint, self.model.setpoint_schema()
                )
            if self.disturbance_obs:
                names = list(self.model.dynamics_disturbance_names())
                by_name = {
                    row.get("name"): row for row in self.model.disturbance_schema()
                }
                disturbance = self._normalize_observation_values(
                    disturbance, [by_name.get(name, {}) for name in names]
                )
            if self.previous_action_obs:
                previous_action = self._normalize_observation_values(
                    previous_action, self.model.action_schema()
                )
        reference = tracking_error if self.tracking_error_obs else setpoint
        o = state + reference
        if self.disturbance_obs:
            o += disturbance
        if self.previous_action_obs:
            o += previous_action
        if self.integral_obs:
            o = o + [iy / I_TEMP_MAX for iy in self._iy]
        return np.asarray(o, dtype=np.float32)

    @staticmethod
    def _normalize_observation_values(values, schema):
        normalized = []
        for value, row in zip(values, schema):
            bounds = row.get("bounds") if isinstance(row, dict) else None
            if (
                isinstance(bounds, (tuple, list))
                and len(bounds) == 2
                and bounds[0] is not None
                and bounds[1] is not None
                and float(bounds[1]) > float(bounds[0])
            ):
                lo, hi = float(bounds[0]), float(bounds[1])
                normalized.append((float(value) - lo) / (hi - lo))
            else:
                normalized.append(float(value))
        return normalized

    @staticmethod
    def _normalize_observation_deltas(values, schema):
        normalized = []
        for value, row in zip(values, schema):
            bounds = row.get("bounds") if isinstance(row, dict) else None
            if (
                isinstance(bounds, (tuple, list))
                and len(bounds) == 2
                and bounds[0] is not None
                and bounds[1] is not None
                and float(bounds[1]) > float(bounds[0])
            ):
                scale = float(bounds[1]) - float(bounds[0])
                normalized.append(float(value) / scale)
            else:
                normalized.append(float(value))
        return normalized

    def _accumulate_integral(self):
        if not self.model.supports_integral_observation:
            return
        out = self.model.outputs(self.integ.x)
        y = list(out["y"])
        errors = [self.y_sp[i] - y[i] if i < len(y) else 0.0 for i in range(len(self.y_sp))]
        dt = self.control_dt
        self._iy = [float(np.clip(self._iy[i] + errors[i] * dt, -I_TEMP_MAX, I_TEMP_MAX)) for i in range(len(errors))]
    def _setpoint_bounds(self):
        bounds = []
        for row in self.model.setpoint_schema():
            raw = row.get("bounds")
            if isinstance(raw, (tuple, list)) and len(raw) == 2 and raw[0] is not None and raw[1] is not None:
                bounds.append((float(raw[0]), float(raw[1])))
            else:
                bounds.append((None, None))
        return bounds

    def _randomize_setpoints(self, rng):
        self.y_sp = self.model.sample_env_setpoints(
            self.y_sp,
            rng,
            {
                **self._model_env_options,
                "randomize_setpoints": True,
            },
        )
    def default_sp_action(self):
        """Normalized supervisory action that reproduces the default setpoints (= the
        fixed-SP PID baseline), the offline prior to learn from."""
        if self.layout is None:
            return None
        a = []
        for spec in self.layout:
            lo, hi = spec[-2], spec[-1]
            if spec[0] == "y_sp":
                v = self._ysp0[spec[1]]
            else:
                v = lo + 0.7 * (hi - lo)
            a.append(float(np.clip((v - lo) / (hi - lo), 0.0, 1.0)))
        return np.array(a, np.float32)

    def _meas(self):
        """Build the measured-state mapping consumed by the inner PID."""

        return self.model.measurement(self._measured_state(), self._env())

    def _measured_state(self):
        true_state = self.model.state_vector(self.integ.x)
        runtime = getattr(self, "_sensor_runtime", None)
        if runtime is None:
            return true_state
        return runtime.observe(true_state, step=int(self._k)).tolist()

    def _supervise(self, action):
        """Supervisory action = normalized setpoints -> set SPs, inner PID regulates
        to them; unregulated economic MVs ('mv') are applied directly."""
        a = np.clip(np.asarray(action, np.float64), 0.0, 1.0)
        mv = {}
        for i, spec in enumerate(self.layout):
            lo, hi = spec[-2], spec[-1]
            val = lo + float(a[i]) * (hi - lo)
            if spec[0] == "y_sp":
                self.y_sp[spec[1]] = val
            else:                                   # ("mv", u_index, lo, hi)
                mv[spec[1]] = val
        act = self.pid.compute(self._meas(), {"y_sp": self.y_sp}, self.control_dt)
        for u_index, value in mv.items():
            act[u_index] = value
        return act
