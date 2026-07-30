"""Concise public API for AIO-Gym.

Advanced generation, dataset, evaluation, model, and reinforcement-learning
contracts live in their responsibility-based subpackages. Importing this
module has no Gymnasium registration side effects and does not load optional
training dependencies.
"""
from __future__ import annotations

from ._internal.lazy import exported_dir, resolve_export


__version__ = "0.1.0"

_EXPORTS = {
    "make_env": ".env_factory",
    "list_scenarios": ".catalog",
    "list_cases": ".catalog",
    "list_tracks": ".benchmarks",
    "list_controllers": ".catalog",
    "load_case": ".models",
    "load_track": ".benchmarks",
    "make_controller": ".controllers",
    "evaluate_controller": ".evaluation",
}
__all__ = sorted((*_EXPORTS, "__version__"))


def __getattr__(name):
    return resolve_export(globals(), __name__, _EXPORTS, name)


def __dir__():
    return exported_dir(globals(), _EXPORTS | {"__version__": ""})
