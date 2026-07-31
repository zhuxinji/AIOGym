"""Fixed-setpoint PID baseline controller."""
from __future__ import annotations

import numpy as np


def _clip01(v):
    return 0.0 if v < 0 else 1.0 if v > 1 else v




def _loop_spec(row):
    if isinstance(row, dict):
        return (
            int(row["u_index"]),
            int(row["y_index"]),
            row["pid"],
            bool(row.get("reverse", False)),
            float(row.get("bias", 0.0)),
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


class PIDAgent:
    name = "PID"
    controller_api_version = "aiogym.controller.v1"
    action_mode = "actuator"
    control_structure = "fixed_sp_pid"

    def __init__(self, model, loops=None, holds=None, demand_u_index=None):
        self.model = model
        self.nu = model.action_dim()
        cfg = _default_pid_config(model) if loops is None else {}
        self.loops_config = list(loops if loops is not None else cfg["loops"])
        self.demand_u_index = cfg.get("demand_u_index") if demand_u_index is None else demand_u_index
        self.demand_valve = 0.5
        self.holds = list(holds if holds is not None else cfg.get("holds", []))
        self.hold_specs = [_hold_spec(row) for row in self.holds]
        loop_specs = [_loop_spec(row) for row in self.loops_config]
        _validate_pid_config(model, loop_specs, self.hold_specs, self.demand_u_index)
        self.loops = [
            (u_index, y_index, PIDLoop(pid, reverse, bias))
            for u_index, y_index, pid, reverse, bias in loop_specs
        ]

    def metadata(self):
        return {"name": self.name, "class": self.__class__.__name__,
                "kind": "fixed_setpoint_pid", "scenario": self.model.scenario,
                "api": self.controller_api_version,
                "action_mode": self.action_mode, "control_structure": self.control_structure,
                "loops": self.loops_config,
                "holds": self.holds,
                "demand_u_index": self.demand_u_index}

    def reset(self, seed=None):
        for *_, loop in self.loops:
            loop.reset()

    def act(self, obs, context):
        action = self.compute(context.measurement, context.setpoint, context.control_dt)
        return np.asarray(self.model.action_vector(action), dtype=np.float32)

    def compute(self, meas, sp, dt):
        y_sp = list(sp.get("y_sp") or self.model.default_setpoint_vector())
        y = list(meas.get("y") or self.model.controlled_output(meas["x"]))
        u = [0.0] * self.nu
        for u_index, value in self.hold_specs:
            if 0 <= u_index < len(u):
                u[u_index] = value
        for u_index, y_index, loop in self.loops:
            u[u_index] = loop.update(y_sp[y_index], y[y_index], dt)
        if self.demand_u_index is not None and 0 <= self.demand_u_index < len(u):
            u[self.demand_u_index] = self.demand_valve
        return u


def _default_pid_config(model):
    from .configs import load_controller_config

    params = dict(load_controller_config("pid", model.scenario).get("parameters", {}))
    if not params.get("loops"):
        raise ValueError(
            f"no PID loop config for {model.scenario!r}; pass loops explicitly or add it to pid.json"
        )
    return params


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
