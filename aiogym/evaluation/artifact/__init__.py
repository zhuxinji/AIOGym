"""Benchmark artifact creation, rendering, plotting, and validation."""
from __future__ import annotations

from aiogym._internal.lazy import exported_dir, resolve_export


_EXPORTS = {
    "ARTIFACT_CHECK_SCHEMA_VERSION": ".checks",
    "check_benchmark_artifacts": ".checks",
    "plot_results": ".plotting",
    "REPORT_SCHEMA_VERSION": ".report",
    "render_benchmark_report": ".report",
    "plot_constraint_timeline": ".svg",
    "plot_leaderboard": ".svg",
    "plot_learning_curve": ".svg",
    "plot_rollouts": ".svg",
    "plot_summary": ".svg",
    "finalize_benchmark_artifacts": ".writers",
    "write_benchmark_artifacts": ".writers",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    return resolve_export(globals(), __name__, _EXPORTS, name)


def __dir__():
    return exported_dir(globals(), _EXPORTS)
