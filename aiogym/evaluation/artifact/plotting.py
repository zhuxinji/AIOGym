"""Plot generation from resolved benchmark artifact data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..._internal.serialization import write_json as _write_json
from .paths import resolve_artifact_path
from .tables import (
    _benchmark_case_key,
    _leaderboard,
    _plot_row,
    _rows_by_benchmark_case,
    _rows_by_objective,
    _slug,
    _tracking_comparison_rows,
    _tracking_rollout_groups,
)
from .svg import (
    plot_constraint_timeline,
    plot_grouped_leaderboard,
    plot_learning_curve,
    plot_leaderboard,
    plot_rollouts,
    plot_summary,
    plot_tracking_control,
    plot_tracking_comparison_table,
)


def plot_results(run_dir: str | Path) -> dict[str, str]:
    """Generate summary and rollout SVGs from a benchmark artifact directory."""

    run_path = Path(run_dir)
    benchmark_path = run_path / "benchmark.json"
    with benchmark_path.open() as f:
        payload = json.load(f)
    figures_dir = run_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    _clear_comparison_figures(figures_dir)
    title = payload.get("scenario", payload.get("suite", "benchmark"))
    rows = [
        _plot_row(row)
        for row in payload.get("rows", [])
        if row.get("execution_status") in {"passed", "degraded"}
    ]
    figures = {}
    artifact_figures = {}
    if rows:
        tracking_comparison_rows = _tracking_comparison_rows(rows)
        if tracking_comparison_rows:
            tracking_comparison_path = figures_dir / "tracking_comparison.svg"
            plot_tracking_comparison_table(tracking_comparison_rows, str(tracking_comparison_path), title)
            figures["tracking_comparison"] = str(tracking_comparison_path)
            artifact_figures["tracking_comparison_figure"] = str(tracking_comparison_path)
        groups = _rows_by_objective(rows)
        benchmark_case_count = len({_benchmark_case_key(row) for row in rows})
        if len(groups) == 1 and benchmark_case_count <= 1:
            objective, objective_rows = next(iter(groups.items()))
            summary_path = figures_dir / "summary.svg"
            plot_summary(objective_rows, str(summary_path), f"{title} {objective}")
            figures["summary"] = str(summary_path)
            artifact_figures["summary_figure"] = str(summary_path)
            leaderboard_path = figures_dir / "leaderboard.svg"
            plot_leaderboard(_leaderboard(objective_rows), str(leaderboard_path), objective)
            figures["leaderboard"] = str(leaderboard_path)
            artifact_figures["leaderboard_figure"] = str(leaderboard_path)
        else:
            summary_figures = {}
            leaderboard_sections = []
            for objective, objective_rows in groups.items():
                summary_figures[objective] = {}
                for benchmark_case, scenario_rows in _rows_by_benchmark_case(objective_rows).items():
                    slug = f"{_slug(objective)}_{_slug(benchmark_case)}"
                    summary_path = figures_dir / f"summary_{slug}.svg"
                    plot_summary(scenario_rows, str(summary_path), f"{title} {objective} {benchmark_case}")
                    summary_figures[objective][benchmark_case] = str(summary_path)
                    case_title = benchmark_case
                    repeated_scenario = f"{title} / "
                    if case_title.startswith(repeated_scenario):
                        case_title = case_title[len(repeated_scenario):]
                    leaderboard_sections.append({
                        "title": f"{case_title} / {objective}",
                        "metric": scenario_rows[0].get("metric") if scenario_rows else None,
                        "board": _leaderboard(scenario_rows),
                    })
            leaderboard_path = figures_dir / "leaderboard.svg"
            plot_grouped_leaderboard(leaderboard_sections, str(leaderboard_path), title)
            figures["summary_by_scenario"] = summary_figures
            figures["leaderboard_by_scenario"] = str(leaderboard_path)
            artifact_figures["summary_figures"] = summary_figures
            artifact_figures["leaderboard_figure"] = str(leaderboard_path)
    rollouts = _resolved_rollouts(payload, run_path)
    if "rollouts" in payload:
        payload["rollouts"] = rollouts
    tracking_rollout_groups = _tracking_rollout_groups(rollouts)
    if tracking_rollout_groups:
        control_figures = {}
        for (scenario, task), case_rollouts in tracking_rollout_groups.items():
            label = scenario if task == "default" else f"{scenario} / {task}"
            control_path = figures_dir / f"tracking_control_{_slug(label)}.svg"
            plot_tracking_control(case_rollouts, str(control_path), scenario, task)
            control_figures[label] = str(control_path)
        figures["tracking_control_by_scenario"] = control_figures
        artifact_figures["tracking_control_figures"] = control_figures
    elif rollouts:
        rollout_path = figures_dir / f"{title}_rollout.svg"
        plot_rollouts(rollouts, str(rollout_path), title)
        figures["rollout"] = str(rollout_path)
        artifact_figures["rollout_figure"] = str(rollout_path)
        constraint_path = figures_dir / "constraint_timeline.svg"
        plot_constraint_timeline(rollouts, str(constraint_path), title)
        figures["constraint_timeline"] = str(constraint_path)
        artifact_figures["constraint_timeline_figure"] = str(constraint_path)
    learning_curve = payload.get("learning_curve") or []
    if learning_curve:
        curve_path = figures_dir / "learning_curve.svg"
        plot_learning_curve(learning_curve, str(curve_path), title)
        figures["learning_curve"] = str(curve_path)
        artifact_figures["learning_curve_figure"] = str(curve_path)
    payload.setdefault("artifacts", {})
    for key in (
        "summary_figure", "leaderboard_figure", "summary_figures", "leaderboard_figures",
        "tracking_control_figures", "rollout_figure", "constraint_timeline_figure",
    ):
        payload["artifacts"].pop(key, None)
    payload["artifacts"].update(artifact_figures)
    _write_json(benchmark_path, payload)
    return figures

def _clear_comparison_figures(figures_dir: Path) -> None:
    for pattern in (
        "summary*.svg", "leaderboard*.svg", "tracking_comparison.svg",
        "tracking_control_*.svg", "*_rollout.svg", "constraint_timeline.svg",
    ):
        for path in figures_dir.glob(pattern):
            path.unlink()


def _resolved_rollouts(
    payload: Mapping[str, Any], root: Path | None = None
) -> list[dict[str, Any]]:
    """Backfill rollout identity fields missing from older benchmark artifacts."""

    raw_rollouts = payload.get("rollouts")
    if raw_rollouts is None and root is not None:
        artifacts = dict(payload.get("artifacts") or {})
        raw_path = artifacts.get("rollouts")
        rollout_path = resolve_artifact_path(
            root, raw_path, "rollouts/rollouts.json"
        )
        if rollout_path.exists():
            with rollout_path.open() as stream:
                raw_rollouts = json.load(stream)
    resolved = []
    for raw_rollout in raw_rollouts or []:
        rollout = dict(raw_rollout)
        protocol = rollout.get("protocol") or {}
        protocol_task = protocol.get("task")
        if isinstance(protocol_task, Mapping):
            protocol_task = protocol_task.get("name")
        identity = {
            "scenario": protocol.get("scenario") or payload.get("scenario"),
            "task": protocol_task or payload.get("task"),
            "objective": protocol.get("objective") or payload.get("objective"),
        }
        for key, value in identity.items():
            if not rollout.get(key) and value:
                rollout[key] = value
        resolved.append(rollout)
    return resolved
