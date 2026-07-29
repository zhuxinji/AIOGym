#!/usr/bin/env python3
"""Formula-only custom process model example for AIO-Gym."""
from __future__ import annotations

import numpy as np

if __package__ is None or __package__ == "":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from aiogym.env import AIOGymEnv
from aiogym.models import define_model, register_model, unregister_model


model = define_model({
    "scenario": "simple_process",
    "display_name": "Simple process",
    "summary": "One-state formula-only model with oracle-compatible dynamics.",
    "states": {
        "x0": {"initial": 0.2, "bounds": (0.0, 1.0)},
    },
    "actions": {
        "u0": {"bounds": (0.0, 1.0), "kind": "input"},
    },
    "params": {
        "gain": {"value": 0.4, "bounds": (0.0, 2.0)},
    },
    "dynamics": {
        "x0": "-0.2 * (x0 - 0.5) + gain * u0",
    },
    "outputs": {
        "y0": {"expr": "x0", "bounds": (0.0, 1.0), "setpoint": 0.6},
    },
})


def main():
    register_model("simple_process", model, replace=True)
    try:
        env = AIOGymEnv("simple_process", auto_events=False, randomize=False,
                              randomize_setpoints=False, episode_steps=5)
        obs, _ = env.reset(seed=0)
        total_reward = 0.0
        for _ in range(env.episode_steps):
            obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
            total_reward += reward
            if terminated or truncated:
                break
        print({
            "scenario": env.scenario,
            "obs_shape": tuple(obs.shape),
            "y": round(info["y"][0], 3),
            "y_sp": round(info["y_sp"][0], 3),
            "total_reward": round(total_reward, 3),
        })
    finally:
        unregister_model("simple_process")


if __name__ == "__main__":
    main()
