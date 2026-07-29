"""Plot generation from resolved benchmark artifact data."""
from __future__ import annotations

import gzip
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
    _rows_by_goal,
    _slug,
    _tracking_comparison_rows,
    _tracking_rollout_groups,
)
from .svg import (
    plot_constraint_timeline,
    plot_grouped_leaderboard,
    plot_learning_curve,
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
    title = payload.get("scenario", payload.get("track", "benchmark"))
    rows = _result_backed_plot_rows(payload)
    figures = {}
    artifact_figures = {}
    if rows:
        tracking_comparison_rows = _tracking_comparison_rows(rows)
        groups = _rows_by_goal(rows)
        benchmark_case_count = len({_benchmark_case_key(row) for row in rows})
        comparison_path = figures_dir / "comparison.svg"
        if (
            tracking_comparison_rows
            and len(groups) == 1
            and "regulation" in groups
        ):
            plot_tracking_comparison_table(
                tracking_comparison_rows,
                str(comparison_path),
                title,
            )
            figures["comparison"] = str(comparison_path)
            artifact_figures["comparison_figure"] = str(comparison_path)
        elif len(groups) == 1 and benchmark_case_count <= 1:
            goal, goal_rows = next(iter(groups.items()))
            plot_summary(
                goal_rows,
                str(comparison_path),
                f"{title} {goal}",
            )
            figures["comparison"] = str(comparison_path)
            artifact_figures["comparison_figure"] = str(comparison_path)
        else:
            summary_figures = {}
            leaderboard_sections = []
            for goal, goal_rows in groups.items():
                summary_figures[goal] = {}
                for benchmark_case, scenario_rows in _rows_by_benchmark_case(goal_rows).items():
                    slug = f"{_slug(goal)}_{_slug(benchmark_case)}"
                    summary_path = figures_dir / f"summary_{slug}.svg"
                    plot_summary(scenario_rows, str(summary_path), f"{title} {goal} {benchmark_case}")
                    summary_figures[goal][benchmark_case] = str(summary_path)
                    case_title = benchmark_case
                    repeated_scenario = f"{title} / "
                    if case_title.startswith(repeated_scenario):
                        case_title = case_title[len(repeated_scenario):]
                    leaderboard_sections.append({
                        "title": f"{case_title} / {goal}",
                        "metric": scenario_rows[0].get("metric") if scenario_rows else None,
                        "board": _leaderboard(scenario_rows),
                    })
            plot_grouped_leaderboard(
                leaderboard_sections,
                str(comparison_path),
                title,
            )
            figures["summary_by_scenario"] = summary_figures
            figures["comparison"] = str(comparison_path)
            artifact_figures["summary_figures"] = summary_figures
            artifact_figures["comparison_figure"] = str(comparison_path)
    rollouts = _resolved_rollouts(payload, run_path)
    if "rollouts" in payload:
        payload["rollouts"] = rollouts
    tracking_rollout_groups = _tracking_rollout_groups(rollouts)
    if tracking_rollout_groups:
        control_figures = {}
        for (scenario, case), case_rollouts in tracking_rollout_groups.items():
            label = scenario if case == "default" else f"{scenario} / {case}"
            control_path = (
                figures_dir / "tracking.svg"
                if len(tracking_rollout_groups) == 1
                else figures_dir / f"tracking_{_slug(label)}.svg"
            )
            plot_tracking_control(case_rollouts, str(control_path), scenario, case)
            control_figures[label] = str(control_path)
        figures["tracking_control_by_scenario"] = control_figures
        artifact_figures["tracking_control_figures"] = control_figures
        if len(control_figures) == 1:
            artifact_figures["tracking_figure"] = next(
                iter(control_figures.values())
            )
    elif rollouts:
        rollout_path = figures_dir / "rollout.svg"
        plot_rollouts(rollouts, str(rollout_path), title)
        figures["rollout"] = str(rollout_path)
        artifact_figures["rollout_figure"] = str(rollout_path)
        constraint_path = figures_dir / "constraints.svg"
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
        "comparison_figure", "tracking_figure",
        "summary_figure", "leaderboard_figure", "summary_figures", "leaderboard_figures",
        "tracking_comparison_figure",
        "tracking_control_figures", "rollout_figure", "constraint_timeline_figure",
    ):
        payload["artifacts"].pop(key, None)
    payload["artifacts"].update(artifact_figures)
    _write_json(benchmark_path, payload)
    return figures


def _result_backed_plot_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Restore detailed metrics omitted by compact or older summary rows."""

    results = {
        (
            str(result.get("controller_name") or ""),
            str(result.get("goal") or ""),
            str(result.get("case") or "default"),
        ): result
        for result in payload.get("results", [])
    }
    rows = []
    for row in payload.get("rows", []):
        if row.get("execution_status") not in {"passed", "degraded"}:
            continue
        key = (
            str(row.get("controller") or ""),
            str(row.get("goal") or ""),
            str(row.get("case") or "default"),
        )
        enriched = dict(results.get(key) or {})
        enriched.update(row)
        rows.append(_plot_row(enriched))
    return rows


def _clear_comparison_figures(figures_dir: Path) -> None:
    for pattern in (
        "comparison.svg", "tracking.svg", "tracking_*.svg",
        "summary*.svg", "leaderboard*.svg", "tracking_comparison.svg",
        "tracking_control_*.svg", "*_rollout.svg", "rollout.svg",
        "constraint_timeline.svg", "constraints.svg",
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
            opener = gzip.open if rollout_path.suffix == ".gz" else open
            with opener(rollout_path, "rt", encoding="utf-8") as stream:
                raw_rollouts = json.load(stream)
    resolved = []
    for raw_rollout in raw_rollouts or []:
        rollout = dict(raw_rollout)
        evaluation = rollout.get("evaluation") or {}
        evaluation_case = evaluation.get("case")
        if isinstance(evaluation_case, Mapping):
            evaluation_case = evaluation_case.get("name")
        identity = {
            "scenario": evaluation.get("scenario") or payload.get("scenario"),
            "case": evaluation_case or payload.get("case"),
            "goal": evaluation.get("goal") or payload.get("goal"),
        }
        for key, value in identity.items():
            if not rollout.get(key) and value:
                rollout[key] = value
        resolved.append(rollout)
    return resolved
