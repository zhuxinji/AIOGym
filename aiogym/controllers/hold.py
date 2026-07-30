"""Frozen default-action controller used only for benchmark anchors."""
from __future__ import annotations

import numpy as np

from .contracts import CONTROLLER_API_VERSION


class HoldController:
    """Keep the model's versioned default feasible actuator command."""

    name = "Hold"
    action_mode = "actuator"
    control_structure = "fixed_default_action"
    controller_api_version = CONTROLLER_API_VERSION

    def __init__(self, model) -> None:
        self.model = model
        self._action = np.asarray(
            model.action_vector(model.default_action()),
            dtype=np.float32,
        )

    def reset(self, seed: int | None = None) -> None:
        del seed

    def act(self, obs, context) -> np.ndarray:
        del obs, context
        return self._action.copy()

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "kind": "frozen_default_action",
            "scenario": self.model.scenario,
            "api": self.controller_api_version,
            "action_mode": self.action_mode,
            "control_structure": self.control_structure,
        }


__all__ = ["HoldController"]
