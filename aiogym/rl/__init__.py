"""Stable reinforcement-learning configuration and runner API.

Algorithm implementations, replay primitives, wrappers, and research
components remain importable from their explicit modules but are not part of
the stable facade.
"""
from __future__ import annotations

from aiogym._internal.lazy import exported_dir, resolve_export


_EXPORTS = {
    "RL_TRAINING_CONFIG_SCHEMA_VERSION": ".config",
    "RLTrainingConfig": ".config",
    "list_algorithms": ".config",
    "TRAINING_CHECKPOINT_SCHEMA_VERSION": ".checkpoints",
    "CheckpointManager": ".checkpoints",
    "TrainingCheckpoint": ".checkpoints",
    "RUN_RESULT_SCHEMA_VERSION": ".runner",
    "RunResult": ".runner",
    "run_experiment": ".runner",
    "run_seed_sweep": ".runner",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return resolve_export(globals(), __name__, _EXPORTS, name)


def __dir__():
    return exported_dir(globals(), _EXPORTS)
