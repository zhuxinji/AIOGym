"""Temporary execution bridge while backend kernels are decomposed."""
from __future__ import annotations

from pathlib import Path


class DelegatingTrainingAdapter:
    backend = ""

    def __init__(self) -> None:
        self.plan = None
        self._trained = False
        self._state = {}

    def build(self, plan, envs=None):
        if envs is not None:
            raise ValueError(
                "delegating adapters build environments from the resolved plan"
            )
        self.plan = plan
        return self

    def train_chunk(self, transitions: int) -> None:
        if self.plan is None:
            raise RuntimeError("adapter must be built before training")
        if self._trained:
            raise RuntimeError(
                "delegating adapter consumes the transition budget once"
            )
        if int(transitions) != self.plan.config.total_transitions:
            raise ValueError(
                "delegating adapter requires the complete transition budget"
            )
        self._run_backend()
        self._trained = True
        self._state = {
            "backend": self.backend,
            "trained_transitions": int(transitions),
            "policy_path": str(self.plan.policy_path),
        }

    def controller(self):
        return None

    def state_dict(self) -> dict:
        return dict(self._state)

    def load_state_dict(self, state) -> None:
        self._state = dict(state)

    def save_policy(self, path) -> None:
        target = Path(path)
        if target != self.plan.policy_path:
            raise ValueError(
                "backend already saved the policy at the resolved path"
            )
        if not target.exists():
            raise FileNotFoundError(target)

    def _run_backend(self) -> None:
        raise NotImplementedError


def add_option(argv, flag, value) -> None:
    if value is not None:
        argv.extend((flag, str(value)))


def add_boolean_option(argv, flag, value) -> None:
    if value is True:
        argv.append(flag)
    elif value is False:
        argv.append(f"--no-{flag.removeprefix('--')}")


__all__ = [
    "DelegatingTrainingAdapter",
    "add_boolean_option",
    "add_option",
]
