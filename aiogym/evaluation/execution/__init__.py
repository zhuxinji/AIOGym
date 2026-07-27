"""Benchmark execution pipeline."""
from __future__ import annotations

from aiogym._internal.lazy import exported_dir, resolve_export


_EXPORTS = {
    "run_benchmark": ".benchmark",
    "evaluate_controller": ".evaluator",
    "rollout_controller": ".rollouts",
    "execute_benchmark_case": ".runner",
    "run_evaluation_case": ".runner",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return resolve_export(globals(), __name__, _EXPORTS, name)


def __dir__():
    return exported_dir(globals(), _EXPORTS)
