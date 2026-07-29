"""Disturbance state, event scheduling, and plant-regime variation."""
from __future__ import annotations

import copy

import numpy as np


class DisturbanceRuntimeMixin:
    def _copy_disturbance_value(self, value):
        return copy.deepcopy(value)

    def _reset_disturbance_values(self):
        self._disturbance_values = {
            name: self._copy_disturbance_value(value)
            for name, value in self._disturbance_defaults.items()
        }
        for name, attr in self._disturbance_attrs.items():
            if name in self._disturbance_values:
                setattr(self, attr, self._copy_disturbance_value(self._disturbance_values[name]))

    def _set_disturbance_value(self, name, value):
        self._disturbance_values[name] = self._copy_disturbance_value(value)
        attr = self._disturbance_attrs.get(name)
        if attr:
            setattr(self, attr, self._copy_disturbance_value(value))

    def _validate_case_disturbance(self, name, value):
        if name not in self._disturbance_defaults:
            available = ", ".join(sorted(self._disturbance_defaults)) or "none"
            raise ValueError(f"unknown case disturbance {name!r}; available: {available}")
        row = self._disturbance_schema_by_name.get(name, {})
        bounds = row.get("bounds")
        values = value if isinstance(value, list) else [value]
        if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            lo, hi = bounds
            for item in values:
                number = float(item)
                if lo is not None and number < float(lo):
                    raise ValueError(f"case disturbance {name!r} is below its lower bound {lo}")
                if hi is not None and number > float(hi):
                    raise ValueError(f"case disturbance {name!r} is above its upper bound {hi}")

    def _sync_known_disturbances(self):
        for name in self._disturbance_defaults:
            attr = self._disturbance_attrs.get(name)
            if attr and hasattr(self, attr):
                self._disturbance_values[name] = self._copy_disturbance_value(getattr(self, attr))

    def _env(self):
        self._sync_known_disturbances()
        return self.model.runtime_env(self._disturbance_values)
    # ---- operating-regime variation ----
    def _restore_nominal(self):
        for k, v in self._p_nominal.items():
            self.model.p[k] = (list(v) if isinstance(v, list) else v)

    def _apply_regime(self):
        """Scale plant params by per-episode multipliers (fouling / ageing / gain drift)."""
        self._regime_mult = self._sample_regime_mult()
        self._apply_mult(self._regime_mult)

    def _sample_regime_mult(self):
        rng = getattr(self, "_plant_rng", self.np_random)
        return {
            k: float(rng.uniform(lo, hi))
            for k, (lo, hi) in self._regime.items()
            if k in self._p_nominal
        }

    def _init_regime_state(self):
        self._regime_mult = {k: 1.0 for k in self._regime if k in self._p_nominal}
        self._regime_target = dict(self._regime_mult)

    def _apply_plant_drift(self):
        if not self.plant_drift or not self._regime_mult:
            return
        if not self._regime_target:
            self._regime_target = self._sample_regime_mult()
        alpha = min(0.03, max(0.002, 4.0 / max(1, self.episode_steps)))
        next_mult = {}
        settled = True
        for k, current in self._regime_mult.items():
            target = self._regime_target.get(k, current)
            value = current + alpha * (target - current)
            lo, hi = self._regime[k]
            value = float(np.clip(value, lo, hi))
            next_mult[k] = value
            settled = settled and abs(value - target) <= 0.01 * max(1.0, abs(target))
        self._regime_mult = next_mult
        if settled:
            self._regime_target = self._sample_regime_mult()
        self._apply_mult(self._regime_mult)

    def _apply_mult(self, mult):
        for k, m in mult.items():
            nom = self._p_nominal[k]
            self.model.p[k] = [x * m for x in nom] if isinstance(nom, list) else nom * m

    # ---- disturbance scheduler (the "adaptation" dimension) ----
    def _disturbance_names(self):
        return [
            row["event"]
            for row in self.model.disturbance_schema()
            if row.get("dynamic", False) and row.get("event")
        ]

    def _schedule_disturbances(self):
        self._dist_events = []
        if not self.auto_events:
            return
        rng = getattr(self, "_disturbance_rng", self.np_random)
        names = self._disturbance_names()
        if not names:
            return
        for _ in range(int(rng.integers(1, 4))):
            t = int(rng.integers(int(0.15 * self.episode_steps), max(2, self.episode_steps)))
            self._dist_events.append((t, names[int(rng.integers(0, len(names)))]))

    def _apply_disturbance(self, event):
        rng = getattr(self, "_disturbance_rng", self.np_random)
        if event == "setpoint_move":
            self._randomize_setpoints(
                getattr(self, "_reference_rng", self.np_random)
            )
        else:
            row = self._disturbance_by_event.get(event)
            if row and row.get("kind") != "setpoint":
                name = row.get("name")
                default = self._disturbance_defaults.get(name, row.get("default", 0.0))
                self._set_disturbance_value(name, self.model.sample_disturbance(event, default, rng))
