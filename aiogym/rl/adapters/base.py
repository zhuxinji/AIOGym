"""Stable training-adapter protocol."""
from __future__ import annotations

from typing import Any, Protocol


class TrainingAdapter(Protocol):
    def build(self, plan, envs=None) -> "TrainingAdapter": ...

    def train_chunk(self, transitions: int) -> None: ...

    def controller(self) -> Any: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state) -> None: ...

    def save_policy(self, path) -> None: ...


__all__ = ["TrainingAdapter"]
