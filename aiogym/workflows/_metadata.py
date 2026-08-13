"""Shared environment metadata for workflow artifacts."""
from __future__ import annotations

from aiogym.core.io import jsonable


def environment_metadata(env):
    base = env.unwrapped
    config = base.runtime_config
    return {
        "scenario": config["scenario"],
        "reward": config["reward"],
        "parameters": jsonable(config["parameters"]),
        "benchmark": config["benchmark"],
        "randomize": config["randomize"],
        "noise": jsonable(config["noise"]),
        "delay": jsonable(config["delay"]),
        "fault": jsonable(config["fault"]),
        "control_dt": config["control_dt"],
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
    }


__all__ = ["environment_metadata"]
