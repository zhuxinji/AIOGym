"""Stable evaluation and report-building facade."""
from __future__ import annotations

from aiogym._internal.lazy import exported_dir, resolve_export


_EXPORTS = {
    "build_evaluation_report": ".results",
    "evaluate_controller": ".execution",
    "render_benchmark_report": ".artifact",
    "build_final_statistical_report": ".statistics",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return resolve_export(globals(), __name__, _EXPORTS, name)


def __dir__():
    return exported_dir(globals(), _EXPORTS)
