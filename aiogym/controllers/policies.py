"""Small baseline policies and a callback adapter for external algorithms."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np


class FunctionPolicy:
    """Adapt ``act(observation, context)`` and optional ``reset(seed=...)``.

    Observation-only functions can be passed directly to evaluate and collect;
    use this adapter for reset callbacks, context, or metadata.

    The callback returns an action in the environment's action space; the
    workflow validates it without scaling or clipping. Metadata is copied and
    must be JSON-compatible. Its optional ``environment`` entry is a snapshot
    from ``aiogym.environment_metadata(training_env)``, checked by evaluation
    and collection before resetting or stepping the environment.
    """

    env = None

    def __init__(self, act, *, reset=None, metadata=None):
        if not callable(act):
            raise TypeError("act must be callable")
        if reset is not None and not callable(reset):
            raise TypeError("reset must be callable or None")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")
        self._act = act
        self._reset = reset
        self._metadata = deepcopy({
            "id": "external",
            "kind": "external_policy",
            **({} if metadata is None else metadata),
        })

    def reset(self, seed=None):
        if self._reset is not None:
            self._reset(seed=seed)

    def act(self, observation, context):
        return np.asarray(self._act(observation, context), dtype=np.float32)

    def metadata(self):
        return deepcopy(self._metadata)


class HoldPolicy:
    def __init__(self, env, action=None):
        self.env = env
        source = env.model.default_action() if action is None else action
        self.action = np.asarray(source, dtype=np.float32)
        if self.action.shape != env.action_space.shape or not env.action_space.contains(
            self.action
        ):
            raise ValueError("hold action must belong to env.action_space")

    def reset(self, seed=None):
        del seed

    def act(self, observation, context):
        del observation, context
        return self.action.copy()

    def metadata(self):
        return {
            "id": "hold",
            "kind": "fixed_action",
            "action": self.action.tolist(),
        }


class RandomPolicy:
    def __init__(self, env):
        self.env = env
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, observation, context):
        del observation, context
        return self.rng.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
        ).astype(np.float32)

    def metadata(self):
        return {
            "id": "random",
            "kind": "uniform_random",
        }


__all__ = ["FunctionPolicy", "HoldPolicy", "RandomPolicy"]
