"""Markdown rendering for benchmark artifact directories."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paths import resolve_artifact_path


REPORT_SCHEMA_VERSION = "aiogym.offline_report.v1"


def render_benchmark_report(artifact_dir: str | Path, out_path: str | Path | None = None) -> str:
    """Render a Markdown report from a standard benchmark artifact directory."""

    root = Path(artifact_dir)
    benchmark = _read_json(root / "benchmark.json")
    artifacts = dict(benchmark.get("artifacts") or {})
    leaderboard = _read_json(_artifact_path(root, artifacts, "leaderboard", "summary/leaderboard.json"), default=[])
    objective_report = _read_json(_artifact_path(root, artifacts, "report", "results/report.json"), default={})
    summary_path = _artifact_path(root, artifacts, "all_summary_csv", "summary/all_summary.csv")
    if not summary_path.exists():
        summary_path = _artifact_path(root, artifacts, "summary_csv", "summary/summary.csv")
    summary_rows = _read_csv(summary_path)
    tracking_comparison = _read_csv(_artifact_path(root, artifacts, "tracking_comparison", "summary/tracking_comparison.csv"))
    model_manifest = _read_model_metadata_manifest(root, artifacts)

    title = benchmark.get("suite") or benchmark.get("scenario") or benchmark.get("benchmark", "benchmark")
    lines = [
        f"# AIO-Gym Benchmark Report: {title}",
        "",
        f"Report schema: `{REPORT_SCHEMA_VERSION}`",
        f"Artifact directory: `{root}`",
        "",
        "## Run Summary",
        "",
    ]
    lines.extend(_markdown_table(["Field", "Value"], _summary_rows(benchmark, summary_rows, model_manifest)))
    lines.extend(["", "## Scenario Coverage", ""])
    lines.extend(_scenario_section(benchmark, model_manifest))
    if tracking_comparison:
        lines.extend(["", "## Tracking Comparison", ""])
        lines.extend(_tracking_comparison_section(tracking_comparison))
    lines.extend(["", "## Leaderboard", ""])
    lines.extend(_leaderboard_sections(leaderboard))
    lines.extend(["", "## Objective Report", ""])
    lines.extend(_objective_section(objective_report))
    lines.extend(["", "## Stable Inputs", ""])
    lines.extend(_artifact_section(root, benchmark, artifacts))
    lines.append("")
    text = "\n".join(lines)
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text


def _summary_rows(benchmark: Mapping[str, Any], summary_rows: Sequence[Mapping[str, str]],
                  model_manifest: Mapping[str, Any] | None) -> list[list[str]]:
    suite_config = benchmark.get("suite_config") or {}
    counts = benchmark.get("counts") or {}
    scenarios = _scenario_names(benchmark, model_manifest)
    objectives = suite_config.get("objectives") or sorted({row.get("objective", "") for row in summary_rows if row.get("objective")})
    controllers = suite_config.get("controllers") or sorted({row.get("controller", "") for row in summary_rows if row.get("controller")})
    rows = [
        ["Benchmark", benchmark.get("benchmark", "")],
        ["Created At", benchmark.get("created_at", "")],
        ["Suite", benchmark.get("suite", benchmark.get("scenario", ""))],
        ["Scenarios", ", ".join(str(item) for item in scenarios)],
        ["Objectives", ", ".join(str(item) for item in objectives)],
        ["Controllers", ", ".join(str(item) for item in controllers)],
        ["Rows", str(len(benchmark.get("rows") or []) or len(summary_rows))],
    ]
    if benchmark.get("learning_curve"):
        rows.append(["Learning Curve Rows", str(len(benchmark.get("learning_curve") or []))])
    if counts:
        rows.append(["Counts", ", ".join(f"{key}={value}" for key, value in counts.items())])
    defaults = benchmark.get("defaults") or {}
    if defaults:
        rows.append(["Seeds", _fmt(defaults.get("seed_list", defaults.get("seed", "")))])
        rows.append(["Episode Steps", _fmt(defaults.get("episode_steps", ""))])
    return rows


def _scenario_section(benchmark: Mapping[str, Any], model_manifest: Mapping[str, Any] | None) -> list[str]:
    scenarios = _scenario_names(benchmark, model_manifest)
    if not scenarios:
        return ["No scenario metadata was found."]
    rows = []
    models = dict((model_manifest or {}).get("models") or {})
    for scenario in scenarios:
        rows.append([
            scenario,
            _rel(models.get(scenario, ""))
            if scenario in models
            else "metadata/model_metadata.json",
        ])
    return _markdown_table(["Scenario", "Model Metadata"], rows)


def _leaderboard_sections(leaderboard) -> list[str]:
    if isinstance(leaderboard, Mapping):
        lines = []
        for objective, rows in leaderboard.items():
            lines.extend([f"### {objective}", ""])
            lines.extend(_leaderboard_section(rows if isinstance(rows, list) else []))
            lines.append("")
        return lines[:-1] if lines else ["No leaderboard rows were found."]
    return _leaderboard_section(leaderboard)


def _tracking_comparison_section(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = ["scenario", "task", "best_controller", "best_tracking_cost", "best_runtime_total_seconds", "oracle_gap_vs_best"]
    controller_columns = [
        key for key in rows[0].keys()
        if key.endswith("_tracking_cost") or key.endswith("_runtime_total_seconds")
    ] if rows else []
    columns = preferred + [key for key in controller_columns if key not in preferred]
    table_rows = []
    for row in rows:
        table_rows.append([
            _fmt(row.get(column)) if not column.endswith(("_tracking_cost", "_runtime_total_seconds")) and column != "oracle_gap_vs_best"
            else _fmt_number(row.get(column))
            for column in columns
        ])
    return _markdown_table(columns, table_rows)


def _tracking_benchmark_case_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count distinct scenario/task cases represented by successful tracking rows."""
    return len({
        (
            str(row.get("scenario") or "benchmark"),
            str(row.get("task") or "default"),
        )
        for row in rows
        if row.get("objective") == "tracking" and row.get("execution_status") != "failed"
    })


def _leaderboard_section(leaderboard: Sequence[Mapping[str, Any]]) -> list[str]:
    if not leaderboard:
        return ["No leaderboard rows were found."]
    rows = []
    for row in leaderboard:
        rows.append([
            _fmt(row.get("rank")),
            _fmt(row.get("scenario")),
            _fmt(row.get("task", "default")),
            _fmt(row.get("objective")),
            _fmt(row.get("controller")),
            _fmt(row.get("execution_status")),
            _fmt(row.get("objective_status", "not-defined")),
            _fmt(row.get("metric")),
            _fmt_number(row.get("metric_value")),
            _fmt_number(row.get("normalized_score")),
            _fmt_number(row.get("profit")),
            _fmt_number(row.get("constraint_violation_count")),
        ])
    return _markdown_table(
        ["Rank", "Scenario", "Task", "Objective", "Controller", "Execution", "Objective Status", "Metric", "Value", "KPI", "Profit", "Violations"],
        rows,
    )


def _objective_section(report: Mapping[str, Any]) -> list[str]:
    if not report:
        return ["No objective-grouped report was found."]
    rows = []
    for objective, entries in report.items():
        if not isinstance(entries, list):
            continue
        metrics = _objective_metrics(entries)
        rows.append([str(objective), str(len(entries)), ", ".join(metrics)])
    return _markdown_table(["Objective", "Entries", "Primary Metrics"], rows) if rows else ["No objective entries were found."]


def _objective_metrics(entries: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "tracking_cost",
        "tracking_iae",
        "profit",
        "constraint_violation_count",
        "return",
        "normalized_score",
        "energy_kwh",
    ]
    keys = set()
    for entry in entries:
        if entry.get("metric"):
            keys.add(str(entry["metric"]))
        for key in preferred:
            if key in entry:
                keys.add(key)
    return [key for key in preferred if key in keys] + sorted(keys.difference(preferred))


def _artifact_section(root: Path, benchmark: Mapping[str, Any], artifacts: Mapping[str, str]) -> list[str]:
    keys = [
        ("benchmark", "benchmark.json"),
        ("summary_csv", "summary/summary.csv"),
        ("tracking_comparison", "summary/tracking_comparison.csv"),
        ("tracking_comparison_figure", "figures/tracking_comparison.svg"),
        ("rollouts", "rollouts/rollouts.json"),
        ("leaderboard", "summary/leaderboard.json"),
        ("all_summary_csv", "summary/all_summary.csv"),
        ("all_leaderboard", "summary/all_leaderboard.json"),
        ("report", "results/report.json"),
        ("model_metadata_manifest", "metadata/models/manifest.json"),
        ("model_metadata", "metadata/model_metadata.json"),
        ("training", "training/training.json"),
        ("learning_curve", "training/learning_curve.json"),
        ("learning_curve_csv", "training/learning_curve.csv"),
        ("summary_figure", "figures/summary.svg"),
        ("leaderboard_figure", "figures/leaderboard.svg"),
        ("learning_curve_figure", "figures/learning_curve.svg"),
    ]
    rows = []
    for key, default in keys:
        path = _artifact_path(root, artifacts, key, default)
        if path.exists():
            rows.append([key, _rel(path)])
    rows.extend(_artifact_mapping_rows(root, artifacts, "summary_csvs"))
    rows.extend(_artifact_mapping_rows(root, artifacts, "leaderboards"))
    rows.extend(_artifact_mapping_rows(root, artifacts, "summary_figures"))
    rows.extend(_artifact_mapping_rows(root, artifacts, "tracking_control_figures"))
    rows.extend(_artifact_mapping_rows(root, artifacts, "leaderboard_figures"))
    if not rows and benchmark.get("artifacts"):
        rows = [[key, _rel(value)] for key, value in sorted(artifacts.items())]
    return _markdown_table(["Input", "Path"], rows)


def _artifact_mapping_rows(root: Path, artifacts: Mapping[str, Any], key: str) -> list[list[str]]:
    value = artifacts.get(key)
    if not isinstance(value, Mapping):
        return []
    rows = []
    for name, raw in _flatten_artifact_mapping(value):
        path = resolve_artifact_path(root, raw, "")
        if path.exists():
            rows.append([f"{key}:{name}", _rel(path)])
    return rows


def _flatten_artifact_mapping(value: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    rows = []
    for name, raw in sorted(value.items()):
        label = f"{prefix}/{name}" if prefix else str(name)
        if isinstance(raw, Mapping):
            rows.extend(_flatten_artifact_mapping(raw, label))
        else:
            rows.append((label, raw))
    return rows


def _scenario_names(benchmark: Mapping[str, Any], model_manifest: Mapping[str, Any] | None) -> list[str]:
    if model_manifest and model_manifest.get("scenarios"):
        return [str(item) for item in model_manifest["scenarios"]]
    if benchmark.get("scenario"):
        return [str(benchmark["scenario"])]
    suite_config = benchmark.get("suite_config") or {}
    if suite_config.get("scenarios"):
        return [str(item) for item in suite_config["scenarios"]]
    return list(dict.fromkeys(str(row["scenario"]) for row in benchmark.get("rows", []) if row.get("scenario")))


def _read_model_metadata_manifest(
    root: Path, artifacts: Mapping[str, str]
) -> Mapping[str, Any] | None:
    path = _artifact_path(
        root,
        artifacts,
        "model_metadata_manifest",
        "metadata/models/manifest.json",
    )
    if path.exists():
        return _read_json(path)
    return None


def _artifact_path(root: Path, artifacts: Mapping[str, str], key: str, default: str) -> Path:
    raw = artifacts.get(key)
    if raw:
        return resolve_artifact_path(root, raw, default)
    return root / default


def _read_json(path: Path, default=None):
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with path.open() as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_escape_cell(cell) for cell in row) + " |")
    return out


def _escape_cell(value) -> str:
    return _fmt(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _fmt_number(value) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.4g}"


def _rel(path) -> str:
    if not path:
        return ""
    raw = str(path)
    marker = "/metadata/"
    if marker in raw:
        return "metadata/" + raw.split(marker, 1)[1]
    marker = "/summary/"
    if marker in raw:
        return "summary/" + raw.split(marker, 1)[1]
    marker = "/results/"
    if marker in raw:
        return "results/" + raw.split(marker, 1)[1]
    marker = "/training/"
    if marker in raw:
        return "training/" + raw.split(marker, 1)[1]
    marker = "/figures/"
    if marker in raw:
        return "figures/" + raw.split(marker, 1)[1]
    return raw
