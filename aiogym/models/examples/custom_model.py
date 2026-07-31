#!/usr/bin/env python3
"""Minimal custom process model example for the AIO-Gym backend."""
from __future__ import annotations

import numpy as np

if __package__ is None or __package__ == "":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from aiogym import make_env
from aiogym.models import ProcessModelContract, register_model, unregister_model


class MiniHeatedTank(ProcessModelContract):
    scenario = "mini_heated_tank"
    display_name = "Mini heated tank"
    summary = "One liquid level and one temperature state with generic action-vector inputs."
    n = 1
    dt_micro = 0.02

    state_names = ("h0", "T0")
    state_units = {"h0": "m", "T0": "degC"}
    state_bounds = {"h0": (0.0, 1.0), "T0": (0.0, 120.0)}
    action_names = ("feed_flow", "heater_power")
    action_units = {"feed_flow": "fraction", "heater_power": "fraction"}
    action_bounds = {"feed_flow": (0.0, 1.0), "heater_power": (0.0, 1.0)}
    action_kinds = {"feed_flow": "flow", "heater_power": "heat"}
    output_names = ("tank_temperature",)
    output_units = {"tank_temperature": "degC"}
    output_bounds = {"tank_temperature": (20.0, 80.0)}
    default_y_sp = (50.0,)
    param_units = {
        "area": "m2",
        "pump_flow_max": "m3/s",
        "heater_gain": "degC/s",
        "heat_loss": "1/s",
        "t_cold": "degC",
        "t_amb": "degC",
    }
    param_bounds = {
        "area": (0.01, 2.0),
        "pump_flow_max": (0.0, 0.02),
        "heater_gain": (0.0, 20.0),
        "heat_loss": (0.0, 1.0),
        "t_cold": (0.0, 40.0),
        "t_amb": (0.0, 45.0),
    }
    plant_regime = {"heater_gain": (0.8, 1.2), "heat_loss": (0.7, 1.4)}
    economic_config = {
        "temp_band": ((35.0, 70.0),),
        "level_band": (),
        "value": "none",
        "w_value": 0.0,
        "w_energy": 0.1,
        "w_viol": 10.0,
    }
    input_disturbances = ProcessModelContract.input_disturbances + (
        {
            "name": "feed_bias",
            "event": "feed_bias_step",
            "unit": "m3/s",
            "bounds": (-0.0001, 0.0001),
            "default": 0.0,
            "description": "small unmeasured feed-flow offset",
        },
    )
    safety_constraints = (
        {"name": "level_bounds", "states": ("h0",), "bounds": (0.0, 1.0)},
        {"name": "temperature_trip", "states": ("T0",), "bounds": (None, 92.0)},
    )

    def __init__(self):
        self.p = {
            "area": 0.2,
            "pump_flow_max": 0.001,
            "heater_gain": 4.0,
            "heat_loss": 0.03,
            "t_cold": 15.0,
            "t_amb": 20.0,
        }

    def initial_state(self):
        return [0.25, 20.0]

    def dynamics(self, x, u, env=None, backend="numeric", ca=None):
        env = env or {}
        pump = float(u[0])
        heater = float(u[1])
        feed = pump * self.p["pump_flow_max"] + float(env.get("feed_bias", 0.0))
        dh = feed / self.p["area"]
        dT = (
            0.02 * (float(env.get("t_cold", self.p["t_cold"])) - x[1])
            + heater * self.p["heater_gain"]
            - self.p["heat_loss"] * (x[1] - float(env.get("t_amb", self.p["t_amb"])))
        )
        return [dh, dT]

    def controlled_output(self, x, backend="numeric", ca=None):
        return [x[1]]

    def clamp_state(self, x):
        return [float(np.clip(x[0], 0.0, 1.0)), float(x[1])]


def main():
    register_model("mini_heated_tank", MiniHeatedTank, replace=True)
    try:
        env = make_env(
            config={
                "scenario": "mini_heated_tank",
                "environment": {
                    "auto_events": True,
                    "randomize": False,
                    "randomize_setpoints": False,
                    "episode_steps": 5,
                },
            }
        )
        obs, _ = env.reset(seed=0)
        total_reward = 0.0
        for _ in range(env.episode_steps):
            obs, reward, terminated, truncated, info = env.step(np.array([0.5, 0.6], dtype=np.float32))
            total_reward += reward
            if terminated or truncated:
                break
        print({
            "scenario": env.scenario,
            "obs_shape": tuple(obs.shape),
            "temperature": round(info["y"][0], 3),
            "total_reward": round(total_reward, 3),
        })
    finally:
        unregister_model("mini_heated_tank")


if __name__ == "__main__":
    main()
