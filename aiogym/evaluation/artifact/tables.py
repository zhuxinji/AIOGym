"""Artifact row grouping, leaderboard, and CSV preparation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

def _artifact_scenarios(payload: Mapping[str, Any]) -> list[str]:
    if payload.get("scenario"):
        return [str(payload["scenario"])]
    track_config = payload.get("track_config") or {}
    if track_config.get("scenarios"):
        return list(dict.fromkeys(str(scenario) for scenario in track_config["scenarios"]))
    rows = payload.get("rows") or []
    return list(dict.fromkeys(str(row["scenario"]) for row in rows if row.get("scenario")))


def _leaderboard(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        if row.get("execution_status") == "failed":
            continue
        metric = row.get("metric")
        value = row.get(metric) if metric else None
        item = {
            "rank": None,
            "controller": row.get("controller"),
            "scenario": row.get("scenario"),
            "case": row.get("case", "default"),
            "case_status": row.get("case_status", "implicit-default"),
            "case_profile_hash": row.get("case_profile_hash"),
            "goal": row.get("goal"),
            "goal_source": row.get("goal_source"),
            "acceptance_status": row.get("acceptance_status", "not-defined"),
            "execution_status": row.get("execution_status"),
            "metric": metric,
            "metric_value": value,
            "official_score": row.get("official_score", value),
            "ranking_eligible": bool(
                row.get("ranking_eligible", True)
            ),
            "safety_gate": row.get("safety_gate"),
            "profit": row.get("profit"),
            "tracking_cost": row.get("tracking_cost"),
            "tracking_return": row.get("tracking_return"),
            "tracking_error_cost": row.get("tracking_error_cost"),
            "tracking_move_cost": row.get("tracking_move_cost"),
            "tracking_mse": row.get("tracking_mse"),
            "tracking_iae": row.get("tracking_iae"),
            "constraint_violation_count": row.get("constraint_violation_count"),
            "constraint_violation_severity": row.get("constraint_violation_severity"),
        }
        key = (
            str(item.get("scenario") or "benchmark"),
            str(item.get("case") or "default"),
            str(item.get("goal") or "benchmark"),
        )
        groups.setdefault(key, []).append(item)
    out = []
    for group in groups.values():
        group.sort(key=lambda item: (
            item["execution_status"] not in {"passed", "degraded"},
            not item["ranking_eligible"],
            _sort_value(item["metric"], item["metric_value"]),
        ))
        rank = 0
        for item in group:
            if item["ranking_eligible"]:
                rank += 1
                item["rank"] = rank
            out.append(item)
    return out

def _sort_value(metric: str | None, value):
    if value is None:
        return float("inf")
    if metric in {
        "tracking_cost", "tracking_error_cost", "tracking_move_cost",
        "tracking_mse", "tracking_iae", "tracking_ise", "tracking_itae", "tracking_overshoot",
        "tracking_settling_time", "constraint_violation_count",
        "constraint_violation_severity", "action_violation_count",
        "action_violation_severity", "runaway_count", "runaway_duration",
    }:
        return float(value)
    return -float(value)


def _rows_by_goal(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        goal = str(row.get("goal") or "benchmark")
        groups.setdefault(goal, []).append(row)
    return groups


def _benchmark_case_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario") or "benchmark"),
        str(row.get("case") or "default"),
    )


def _benchmark_case_label(row: Mapping[str, Any]) -> str:
    scenario, case = _benchmark_case_key(row)
    return scenario if case == "default" else f"{scenario} / {case}"


def _rows_by_benchmark_case(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        label = _benchmark_case_label(row)
        groups.setdefault(label, []).append(row)
    return groups


def _tracking_rollout_groups(rollouts: Sequence[Mapping[str, Any]]):
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for rollout in rollouts:
        if rollout.get("goal") != "regulation" or not rollout.get("scenario"):
            continue
        key = (
            str(rollout["scenario"]),
            str(rollout.get("case") or "default"),
        )
        groups.setdefault(key, []).append(rollout)
    return groups


TRACKING_COMPARISON_METRICS = (
    ("tracking_cost", "Tracking goal"),
    ("tracking_error_cost", "Error cost"),
    ("tracking_move_cost", "Move cost"),
    ("tracking_iae", "IAE"),
    ("tracking_ise", "ISE"),
    ("tracking_itae", "ITAE"),
    ("tracking_overshoot", "Overshoot"),
    ("energy_kwh", "Energy (kWh)"),
    ("constraint_violation_count", "Constraint violations"),
    ("runtime_seconds", "Runtime (s)"),
)


def _tracking_comparison_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tracking_rows = [
        row for row in rows
        if row.get("goal") == "regulation" and row.get("execution_status") in {"passed", "degraded"}
    ]
    if not tracking_rows:
        return []
    benchmark_cases = list(dict.fromkeys(_benchmark_case_key(row) for row in tracking_rows))
    controllers = list(dict.fromkeys(str(row.get("controller") or "controller") for row in tracking_rows))
    out = []
    for scenario, case in benchmark_cases:
        scenario_rows = [row for row in tracking_rows if _benchmark_case_key(row) == (scenario, case)]
        rows_by_controller = {
            str(row.get("controller") or "controller"): row
            for row in scenario_rows
        }
        values = {
            controller: _float_or_none(source.get("tracking_cost"))
            for controller, source in rows_by_controller.items()
        }
        runtime_seconds = {
            controller: _float_or_none(source.get("runtime_seconds"))
            for controller, source in rows_by_controller.items()
        }
        ranked = [(controller, value) for controller, value in values.items() if value is not None]
        if not ranked:
            continue
        best_controller, best_value = min(ranked, key=lambda item: item[1])
        row = {
            "scenario": scenario,
            "case": case,
            "best_controller": best_controller,
            "best_tracking_cost": best_value,
            "best_runtime_seconds": runtime_seconds.get(best_controller),
        }
        oracle_value = values.get("NMPC-oracle")
        row["oracle_gap_vs_best"] = None if oracle_value is None else oracle_value - best_value
        for controller in controllers:
            source = rows_by_controller.get(controller, {})
            for metric, _ in TRACKING_COMPARISON_METRICS:
                value = runtime_seconds.get(controller) if metric == "runtime_seconds" else source.get(metric)
                row[f"{controller}_{metric}"] = _float_or_none(value)
        out.append(row)
    return out


def _write_tracking_comparison_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    base = [
        "scenario",
        "case",
        "best_controller",
        "best_tracking_cost",
        "best_runtime_seconds",
        "oracle_gap_vs_best",
    ]
    controllers = []
    for row in rows:
        for key in row:
            if key.endswith("_tracking_cost") and key not in {"best_tracking_cost"}:
                controllers.append(key[: -len("_tracking_cost")])
    controllers = list(dict.fromkeys(controllers))
    columns = base + [
        f"{controller}_{metric}"
        for controller in controllers
        for metric, _ in TRACKING_COMPARISON_METRICS
    ]
    _write_summary_csv(path, rows, columns)


def _float_or_none(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "goal"


def _write_summary_index_csv(path: Path, groups: Mapping[str, Sequence[Mapping[str, Any]]],
                             summary_csvs: Mapping[str, str], leaderboards: Mapping[str, str]) -> None:
    columns = ["goal", "rows", "metrics", "summary_csv", "leaderboard"]
    with path.open("w") as f:
        f.write(",".join(columns) + "\n")
        for goal, rows in groups.items():
            metrics = list(dict.fromkeys(str(row.get("metric", "")) for row in rows if row.get("metric")))
            row = {
                "goal": goal,
                "rows": len(rows),
                "metrics": metrics,
                "summary_csv": summary_csvs[goal],
                "leaderboard": leaderboards[goal],
            }
            f.write(",".join(_csv_cell(row.get(column)) for column in columns) + "\n")


FULL_SUMMARY_COLUMNS = [
    "run_case_id", "scenario", "case", "case_status", "case_profile_hash",
    "goal", "goal_source",
    "acceptance_status", "reward_spec_id", "action_mode", "controller",
    "control_structure", "execution_status", "ranking_eligible",
    "official_score", "safety_mode", "case_id", "pair_id", "pair_role",
    "base_case_id", "metric", "profit",
    "production",
    "return", "track", "tracking_cost", "tracking_return", "tracking_error_cost",
    "tracking_move_cost", "tracking_mse", "tracking_iae", "tracking_ise",
    "tracking_itae", "tracking_overshoot", "tracking_settling_time",
    "energy_kwh", "service_shortfall_count", "service_shortfall_duration",
    "service_availability", "constraint", "state_violation_count",
    "state_violation_duration", "state_violation_severity",
    "command_violation_count", "command_violation_duration",
    "command_violation_severity", "protection_intervention_count",
    "protection_intervention_duration", "hard_termination_count",
    "initial_safety_debt_count", "initial_safety_debt_duration",
    "controller_created_state_violation_count",
    "controller_created_state_violation_duration",
    "constraint_violation_count", "constraint_violation_severity",
    "safety_margin_min",
    "runtime_seconds_per_step", "runtime_total_seconds", "episodes", "seed_list",
]


GOAL_SUMMARY_COLUMNS = {
    "regulation": [
        "run_case_id", "scenario", "case", "case_status", "case_profile_hash",
        "goal_source", "acceptance_status", "controller", "control_structure",
        "execution_status", "ranking_eligible", "official_score",
        "metric", "tracking_cost", "tracking_return", "tracking_error_cost",
        "tracking_move_cost", "tracking_mse", "tracking_iae", "tracking_ise",
        "tracking_itae", "tracking_overshoot", "tracking_settling_time",
        "track", "energy_kwh",
        "constraint_violation_count", "constraint_violation_severity",
        "runtime_seconds_per_step", "runtime_total_seconds", "episodes", "seed_list",
    ],
    "economic": [
        "run_case_id", "scenario", "case", "case_status", "case_profile_hash",
        "goal_source", "acceptance_status", "controller", "control_structure",
        "execution_status", "ranking_eligible", "official_score",
        "metric", "profit", "production", "energy_kwh",
        "service_shortfall_count", "service_shortfall_duration",
        "service_availability",
        "constraint", "constraint_violation_count", "constraint_violation_severity",
        "safety_margin_min", "runtime_seconds_per_step", "episodes", "seed_list",
    ],
    "safety": [
        "run_case_id", "scenario", "case", "case_status", "case_profile_hash",
        "goal_source", "acceptance_status", "controller", "control_structure",
        "execution_status", "ranking_eligible", "official_score", "safety_mode",
        "metric", "state_violation_count", "state_violation_duration",
        "state_violation_severity", "command_violation_count",
        "command_violation_duration", "command_violation_severity",
        "protection_intervention_count", "protection_intervention_duration",
        "hard_termination_count", "initial_safety_debt_count",
        "controller_created_state_violation_count",
        "constraint_violation_count", "constraint_violation_duration",
        "constraint_violation_severity", "runaway_count", "safety_margin_min",
        "runtime_seconds_per_step", "episodes", "seed_list",
    ],
}


def _summary_columns_for_goal(goal: str) -> list[str]:
    return list(GOAL_SUMMARY_COLUMNS.get(goal, FULL_SUMMARY_COLUMNS))


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]],
                       columns: Sequence[str] = FULL_SUMMARY_COLUMNS) -> None:
    with path.open("w") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join(_csv_cell(row.get(column)) for column in columns) + "\n")


def _write_learning_curve_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    preferred = [
        "step", "timesteps", "phase", "metric", "metric_value", "official_score", "profit",
        "return", "track", "tracking_cost", "tracking_return", "tracking_error_cost",
        "tracking_move_cost", "tracking_mse", "tracking_iae", "constraint_violation_count",
        "constraint_violation_severity", "runtime_total_seconds",
    ]
    keys = list(dict.fromkeys(
        [key for key in preferred if any(key in row for row in rows)]
        + sorted({key for row in rows for key in row}.difference(preferred))
    ))
    with path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(_csv_cell(row.get(column)) for column in keys) + "\n")


def _csv_cell(value) -> str:
    if value is None:
        return ""
    text = json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict, tuple)) else str(value)
    if any(ch in text for ch in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def _plot_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in ("profit", "track", "constraint"):
        out.setdefault(key, 0.0)
    return out
