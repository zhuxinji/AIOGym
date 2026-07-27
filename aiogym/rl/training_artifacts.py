"""RL training payload adapters for the standard benchmark artifact system."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from aiogym._internal.serialization import jsonable as _jsonable
from aiogym.evaluation import build_evaluation_report
from aiogym.evaluation.artifact import finalize_benchmark_artifacts
from aiogym.evaluation.results import compact_result_row


RL_ARTIFACT_SCHEMA_VERSION = "aiogym.rl_training_artifact.v1"


def utc_run_id(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def result_row(result: Mapping[str, Any], scenario: str, action_mode: str,
               controller: str | None = None, suite_case: str | None = None) -> dict[str, Any]:
    """Return a summary row following the benchmark-suite artifact schema."""

    return compact_result_row(
        result,
        scenario=scenario,
        objective=result.get("objective"),
        action_mode=action_mode,
        suite_case=suite_case or f"{result.get('objective', '')}:{scenario}:{controller or result.get('name', '')}",
        controller=controller,
    )


def learning_curve_point(step: int, result: Mapping[str, Any], phase: str = "eval") -> dict[str, Any]:
    """Condense an evaluation result into one training-history row."""

    metric = str(result.get("metric") or "")
    row = {
        "step": int(step),
        "phase": phase,
        "metric": metric,
        "metric_value": result.get(metric) if metric else None,
        "metric_direction": result.get("metric_direction"),
        "episodes": result.get("episodes"),
        "runtime_total_seconds": result.get("runtime_total_seconds"),
    }
    for key in (
        "normalized_score",
        "profit",
        "return",
        "track",
        "tracking_cost",
        "tracking_return",
        "tracking_error_cost",
        "tracking_move_cost",
        "tracking_mse",
        "tracking_iae",
        "constraint_violation_count",
        "constraint_violation_severity",
        "safety_margin_min",
    ):
        if key in result:
            row[key] = result[key]
    return _jsonable(row)


def rl_payload(kind: str, scenario: str, objective: str, action_mode: str,
               training: Mapping[str, Any], protocol: Mapping[str, Any],
               results: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
               learning_curve: Sequence[Mapping[str, Any]] | None = None,
               rollouts: Sequence[Mapping[str, Any]] | None = None,
               extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a standard benchmark payload for an RL training run."""

    payload = {
        "schema_version": RL_ARTIFACT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "rl_training",
        "kind": kind,
        "scenario": scenario,
        "objective": objective,
        "action_mode": action_mode,
        "config": dict(training),
        "benchmark_config": dict(protocol),
        "training": dict(training),
        "evaluation_protocol": dict(protocol),
        "rows": list(rows),
        "results": list(results),
        "report": build_evaluation_report(results),
        "learning_curve": list(learning_curve or []),
        "rollouts": list(rollouts or []),
    }
    if extra:
        payload.update(dict(extra))
    return _jsonable(payload)


def write_rl_artifacts(artifact_dir: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write benchmark.json, standard children, figures, and Markdown report."""

    payload = dict(_jsonable(payload))
    payload["artifact_dir"] = str(artifact_dir)
    return finalize_benchmark_artifacts(
        artifact_dir,
        payload,
        create_plots=True,
        markdown_report=True,
    )
