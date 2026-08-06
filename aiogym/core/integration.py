"""Fixed-step integration for process models."""
from __future__ import annotations

import math


class Integrator:
    """Fixed-step RK4 integrator for process models."""

    def __init__(self, model, *, max_step=None):
        self.model = model
        settings = model.solver_settings()
        self.method = settings["method"]
        self.dt_micro = float(settings["max_step"] if max_step is None else max_step)
        if not math.isfinite(self.dt_micro) or self.dt_micro <= 0:
            raise ValueError("integrator max_step must be finite and positive")
        self.reset()

    def reset(self, state=None):
        self.x = list(state) if state is not None else list(self.model.initial_state())
        self.t = 0.0

    def step(self, dt, act, env):
        m = self.model
        u = m.action_vector(act)
        f = lambda x: m.dynamics(x, u, env)
        nsub = max(1, math.ceil(dt / self.dt_micro - 1e-9))
        h = dt / nsub
        for _ in range(nsub):
            x = self.x
            k1 = f(x)
            k2 = f([v + 0.5 * h * k1[i] for i, v in enumerate(x)])
            k3 = f([v + 0.5 * h * k2[i] for i, v in enumerate(x)])
            k4 = f([v + h * k3[i] for i, v in enumerate(x)])
            self.x = [
                v + (h / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
                for i, v in enumerate(x)
            ]
            if hasattr(m, "clamp_state"):
                self.x = m.clamp_state(self.x)
            self.t += h
        return self.x
