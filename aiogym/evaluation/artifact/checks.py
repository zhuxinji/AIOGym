"""Structural and content checks for benchmark artifact directories."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paths import resolve_artifact_path
from .report import _read_json, _tracking_benchmark_case_count


ARTIFACT_CHECK_SCHEMA_VERSION = "aiogym.artifact_check.v1"


def check_benchmark_artifacts(artifact_dir: str | Path) -> dict[str, Any]:
    """Check a standard benchmark artifact directory and return structured status."""

    root = Path(artifact_dir)
    checks = []
    benchmark_path = root / "benchmark.json"
    _add_exists(checks, "benchmark", benchmark_path, required=True)
    if not benchmark_path.exists():
        return _check_result(root, checks)

    try:
        benchmark = _read_json(benchmark_path)
        checks.append(_check("benchmark_json", True, str(benchmark_path), "benchmark.json is valid JSON"))
    except Exception as ex:
        checks.append(_check("benchmark_json", False, str(benchmark_path), str(ex)))
        return _check_result(root, checks)

    artifacts = dict(benchmark.get("artifacts") or {})
    rows = list(benchmark.get("rows") or [])
    expected_scenarios = _artifact_scenario_names(benchmark)
    expected_rows = len(rows)
    paths = {
        "rows": _check_artifact_path(root, artifacts, "rows", "summary/rows.json"),
        "summary_csv": _check_artifact_path(root, artifacts, "summary_csv", "summary/summary.csv"),
        "tracking_comparison": _check_artifact_path(root, artifacts, "tracking_comparison", "summary/tracking_comparison.csv"),
        "tracking_comparison_figure": _check_artifact_path(root, artifacts, "tracking_comparison_figure", "figures/tracking_comparison.svg"),
        "rollouts": _check_artifact_path(root, artifacts, "rollouts", "rollouts/rollouts.json"),
        "leaderboard": _check_artifact_path(root, artifacts, "leaderboard", "summary/leaderboard.json"),
        "all_summary_csv": _check_artifact_path(root, artifacts, "all_summary_csv", "summary/all_summary.csv"),
        "all_leaderboard": _check_artifact_path(root, artifacts, "all_leaderboard", "summary/all_leaderboard.json"),
        "results": _check_artifact_path(root, artifacts, "results", "results/results.json"),
        "report": _check_artifact_path(root, artifacts, "report", "results/report.json"),
        "input_config": _check_artifact_path(root, artifacts, "input_config", "config/config.json"),
        "benchmark_config": _check_artifact_path(root, artifacts, "benchmark_config", "config/benchmark_config.json"),
        "summary_figure": _check_artifact_path(root, artifacts, "summary_figure", "figures/summary.svg"),
        "leaderboard_figure": _check_artifact_path(root, artifacts, "leaderboard_figure", "figures/leaderboard.svg"),
        "training": _check_artifact_path(root, artifacts, "training", "training/training.json"),
        "learning_curve": _check_artifact_path(root, artifacts, "learning_curve", "training/learning_curve.json"),
        "learning_curve_csv": _check_artifact_path(root, artifacts, "learning_curve_csv", "training/learning_curve.csv"),
        "learning_curve_figure": _check_artifact_path(root, artifacts, "learning_curve_figure", "figures/learning_curve.svg"),
    }
    for key in ("rows", "summary_csv", "leaderboard", "results", "report", "input_config", "benchmark_config"):
        _add_exists(checks, key, paths[key], required=True)
    tracking_rows_present = any(row.get("objective") == "tracking" for row in rows)
    if tracking_rows_present:
        _add_exists(checks, "tracking_comparison", paths["tracking_comparison"], required=True)
        _add_exists(checks, "tracking_comparison_figure", paths["tracking_comparison_figure"], required=True)
    rollouts = list(benchmark.get("rollouts") or [])
    rollout_artifact_present = bool(artifacts.get("rollouts")) or bool(rollouts)
    if not rollouts and rollout_artifact_present and paths["rollouts"].exists():
        try:
            rollouts = list(_read_json(paths["rollouts"]) or [])
        except Exception:
            pass
    tracking_control_figures = (
        artifacts.get("tracking_control_figures")
        if isinstance(artifacts.get("tracking_control_figures"), Mapping)
        else {}
    )
    if rollout_artifact_present:
        _add_exists(checks, "rollouts", paths["rollouts"], required=True)
        for benchmark_case, raw in sorted(tracking_control_figures.items()):
            path = resolve_artifact_path(root, raw, "missing")
            _add_exists(checks, f"tracking_control_figure:{benchmark_case}", path, required=True)
        expected_control_cases = len({
            (
                str(rollout.get("scenario")),
                str(rollout.get("task") or "default"),
            )
            for rollout in rollouts
            if rollout.get("objective") == "tracking" and rollout.get("scenario")
        })
        checks.append(_check(
            "tracking_control_figure_count",
            len(tracking_control_figures) == expected_control_cases,
            str(root / "figures"),
            f"expected {expected_control_cases}, found {len(tracking_control_figures)}",
        ))
    summary_csvs = artifacts.get("summary_csvs") if isinstance(artifacts.get("summary_csvs"), Mapping) else {}
    leaderboards = artifacts.get("leaderboards") if isinstance(artifacts.get("leaderboards"), Mapping) else {}
    summary_figures = artifacts.get("summary_figures") if isinstance(artifacts.get("summary_figures"), Mapping) else {}
    leaderboard_figures = artifacts.get("leaderboard_figures") if isinstance(artifacts.get("leaderboard_figures"), Mapping) else {}
    has_objective_outputs = bool(summary_csvs or leaderboards)
    if has_objective_outputs:
        _add_exists(checks, "all_summary_csv", paths["all_summary_csv"], required=True)
        _add_exists(checks, "all_leaderboard", paths["all_leaderboard"], required=True)
    for key in ("summary_figure", "leaderboard_figure"):
        _add_exists(checks, key, paths[key], required=bool(rows) and not summary_figures and not leaderboard_figures)
    if benchmark.get("training"):
        _add_exists(checks, "training", paths["training"], required=True)
    learning_curve = list(benchmark.get("learning_curve") or [])
    if learning_curve:
        for key in ("learning_curve", "learning_curve_csv", "learning_curve_figure"):
            _add_exists(checks, key, paths[key], required=True)

    row_data = _safe_json_list(checks, "rows_json", paths["rows"])
    leaderboard = _safe_json(checks, "leaderboard_json", paths["leaderboard"])
    summary_rows = _safe_csv_rows(checks, "summary_csv_rows", paths["summary_csv"])
    tracking_comparison_rows = _safe_csv_rows(checks, "tracking_comparison_rows", paths["tracking_comparison"]) if tracking_rows_present else None
    all_summary_rows = _safe_csv_rows(checks, "all_summary_csv_rows", paths["all_summary_csv"]) if has_objective_outputs else None
    all_leaderboard = _safe_json_list(checks, "all_leaderboard_json", paths["all_leaderboard"]) if has_objective_outputs else None
    curve_rows = _safe_json_list(checks, "learning_curve_json", paths["learning_curve"]) if learning_curve else None
    curve_csv_rows = _safe_csv_rows(checks, "learning_curve_csv_rows", paths["learning_curve_csv"]) if learning_curve else None
    rollout_rows = (
        _safe_json_list(checks, "rollouts_json", paths["rollouts"])
        if rollout_artifact_present
        else None
    )
    if row_data is not None:
        _add_count_check(checks, "rows_json_count", len(row_data), expected_rows, paths["rows"])
    if summary_rows is not None:
        expected_summary_rows = len(_rows_by_objective(rows)) if has_objective_outputs else expected_rows
        _add_count_check(checks, "summary_csv_count", len(summary_rows), expected_summary_rows, paths["summary_csv"])
    if tracking_comparison_rows is not None:
        expected_tracking_cases = _tracking_benchmark_case_count(rows)
        _add_count_check(checks, "tracking_comparison_count", len(tracking_comparison_rows), expected_tracking_cases, paths["tracking_comparison"])
    if all_summary_rows is not None:
        _add_count_check(checks, "all_summary_csv_count", len(all_summary_rows), expected_rows, paths["all_summary_csv"])
    if leaderboard is not None:
        active_rows = sum(1 for row in rows if row.get("execution_status") != "failed")
        leaderboard_count = _leaderboard_count(leaderboard)
        _add_count_check(checks, "leaderboard_count", leaderboard_count, active_rows, paths["leaderboard"])
    if all_leaderboard is not None:
        active_rows = sum(1 for row in rows if row.get("execution_status") != "failed")
        _add_count_check(checks, "all_leaderboard_count", len(all_leaderboard), active_rows, paths["all_leaderboard"])
    _check_objective_artifacts(root, checks, rows, summary_csvs, "summary_csv")
    _check_objective_artifacts(root, checks, rows, leaderboards, "leaderboard")
    _check_objective_artifacts(root, checks, rows, summary_figures, "summary_figure", count_rows=False)
    _check_objective_artifacts(root, checks, rows, leaderboard_figures, "leaderboard_figure", count_rows=False)
    if curve_rows is not None:
        _add_count_check(checks, "learning_curve_json_count", len(curve_rows), len(learning_curve), paths["learning_curve"])
    if curve_csv_rows is not None:
        _add_count_check(checks, "learning_curve_csv_count", len(curve_csv_rows), len(learning_curve), paths["learning_curve_csv"])
    if rollout_rows is not None:
        _add_count_check(checks, "rollouts_json_count", len(rollout_rows), len(rollouts), paths["rollouts"])

    _check_model_metadata(root, artifacts, expected_scenarios, checks)
    return _check_result(root, checks)


def _check_model_metadata(
    root: Path,
    artifacts: Mapping[str, str],
    expected_scenarios: list[str],
    checks: list[dict[str, Any]],
) -> None:
    manifest_path = _check_artifact_path(
        root,
        artifacts,
        "model_metadata_manifest",
        "metadata/models/manifest.json",
    )
    single_path = _check_artifact_path(
        root, artifacts, "model_metadata", "metadata/model_metadata.json"
    )
    models_dir = root / "metadata" / "models"
    if len(expected_scenarios) <= 1:
        _add_exists(checks, "model_metadata", single_path, required=True)
        if models_dir.exists():
            stale = sorted(path.name for path in models_dir.glob("*.json"))
            checks.append(_check(
                "stale_model_metadata",
                not stale,
                str(models_dir),
                _stale_message(stale),
            ))
        return

    _add_exists(checks, "model_metadata_manifest", manifest_path, required=True)
    if not manifest_path.exists():
        return
    try:
        manifest = _read_json(manifest_path)
    except Exception as ex:
        checks.append(_check(
            "model_metadata_manifest_json", False, str(manifest_path), str(ex)
        ))
        return
    manifest_scenarios = [str(item) for item in manifest.get("scenarios", [])]
    checks.append(_check(
        "model_metadata_manifest_scenarios",
        manifest_scenarios == expected_scenarios,
        str(manifest_path),
        f"expected {expected_scenarios}, found {manifest_scenarios}",
    ))
    models = dict(manifest.get("models") or {})
    for scenario in expected_scenarios:
        path = resolve_artifact_path(
            root, models.get(scenario), f"metadata/models/{scenario}.json"
        )
        _add_exists(checks, f"model_metadata:{scenario}", path, required=True)
    if models_dir.exists():
        expected_files = {f"{scenario}.json" for scenario in expected_scenarios} | {"manifest.json"}
        stale = sorted(
            path.name
            for path in models_dir.glob("*.json")
            if path.name not in expected_files
        )
        checks.append(_check(
            "stale_model_metadata",
            not stale,
            str(models_dir),
            _stale_message(stale),
        ))


def _safe_json_list(checks: list[dict[str, Any]], name: str, path: Path) -> list[Any] | None:
    if not path.exists():
        return None
    try:
        data = _read_json(path)
    except Exception as ex:
        checks.append(_check(name, False, str(path), str(ex)))
        return None
    ok = isinstance(data, list)
    checks.append(_check(name, ok, str(path), "valid JSON list" if ok else "expected JSON list"))
    return data if ok else None


def _safe_json(checks: list[dict[str, Any]], name: str, path: Path):
    if not path.exists():
        return None
    try:
        data = _read_json(path)
    except Exception as ex:
        checks.append(_check(name, False, str(path), str(ex)))
        return None
    ok = isinstance(data, (list, dict))
    checks.append(_check(name, ok, str(path), "valid JSON" if ok else "expected JSON list or object"))
    return data if ok else None


def _safe_csv_rows(checks: list[dict[str, Any]], name: str, path: Path) -> list[dict[str, str]] | None:
    if not path.exists():
        return None
    try:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as ex:
        checks.append(_check(name, False, str(path), str(ex)))
        return None
    checks.append(_check(name, True, str(path), "valid CSV"))
    return rows


def _add_count_check(checks: list[dict[str, Any]], name: str, actual: int, expected: int, path: Path) -> None:
    checks.append(_check(name, actual == expected, str(path), f"expected {expected}, found {actual}"))


def _add_exists(checks: list[dict[str, Any]], name: str, path: Path, required: bool) -> None:
    exists = path.exists()
    checks.append(_check(name, exists or not required, str(path), "exists" if exists else "missing"))


def _rows_by_objective(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("objective") or "benchmark"), []).append(row)
    return groups


def _leaderboard_count(data) -> int:
    if isinstance(data, list):
        return len(data)
    if isinstance(data, Mapping):
        return sum(len(rows) for rows in data.values() if isinstance(rows, list))
    return 0


def _check_objective_artifacts(root: Path, checks: list[dict[str, Any]], rows: Sequence[Mapping[str, Any]],
                               paths_by_objective, name: str, count_rows: bool = True) -> None:
    if not isinstance(paths_by_objective, Mapping) or not paths_by_objective:
        return
    groups = _rows_by_objective(rows)
    for objective, objective_rows in groups.items():
        raw = paths_by_objective.get(objective)
        if isinstance(raw, Mapping):
            for scenario, scenario_raw in sorted(raw.items()):
                path = resolve_artifact_path(root, scenario_raw, "missing")
                _add_exists(checks, f"{name}:{objective}:{scenario}", path, required=True)
            continue
        path = resolve_artifact_path(root, raw, "missing")
        check_name = f"{name}:{objective}"
        _add_exists(checks, check_name, path, required=True)
        if not count_rows or not path.exists():
            continue
        if name == "summary_csv":
            csv_rows = _safe_csv_rows(checks, f"{check_name}_rows", path)
            if csv_rows is not None:
                _add_count_check(checks, f"{check_name}_count", len(csv_rows), len(objective_rows), path)
        elif name == "leaderboard":
            data = _safe_json_list(checks, f"{check_name}_json", path)
            if data is not None:
                active_rows = sum(1 for row in objective_rows if row.get("execution_status") != "failed")
                _add_count_check(checks, f"{check_name}_count", len(data), active_rows, path)


def _check(name: str, ok: bool, path: str, message: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "path": path, "message": message}


def _check_result(root: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [row for row in checks if not row["ok"]]
    return {
        "schema_version": ARTIFACT_CHECK_SCHEMA_VERSION,
        "artifact_dir": str(root),
        "ok": not failed,
        "checks": checks,
        "failed": failed,
    }


def _artifact_scenario_names(benchmark: Mapping[str, Any]) -> list[str]:
    if benchmark.get("scenario"):
        return [str(benchmark["scenario"])]
    suite_config = benchmark.get("suite_config") or {}
    if suite_config.get("scenarios"):
        return [str(item) for item in suite_config["scenarios"]]
    return list(dict.fromkeys(str(row["scenario"]) for row in benchmark.get("rows", []) if row.get("scenario")))


def _check_artifact_path(root: Path, artifacts: Mapping[str, str], key: str, default: str) -> Path:
    raw = artifacts.get(key)
    return resolve_artifact_path(root, raw, default)


def _stale_message(stale: list[str]) -> str:
    if not stale:
        return "no stale model metadata JSON files"
    return "stale files: " + ", ".join(stale)
