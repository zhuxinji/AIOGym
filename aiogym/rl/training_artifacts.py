"""RL training payload adapters for the standard benchmark artifact system."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from aiogym._internal.serialization import jsonable as _jsonable
from aiogym.evaluation import build_evaluation_report
from aiogym.evaluation.artifact import finalize_benchmark_artifacts
from aiogym.evaluation.results import compact_result_row
from aiogym.evaluation.provenance import track_provenance


RL_ARTIFACT_SCHEMA_VERSION = "aiogym.rl_training_artifact.v3"


def utc_run_id(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def result_row(result: Mapping[str, Any], scenario: str, action_mode: str,
               controller: str | None = None, run_case_id: str | None = None) -> dict[str, Any]:
    """Return a canonical Case/Goal summary row."""

    return compact_result_row(
        result,
        scenario=scenario,
        goal=result.get("goal"),
        action_mode=action_mode,
        run_case_id=run_case_id or (
            f"{result.get('goal', '')}:{scenario}:"
            f"{controller or result.get('name', '')}"
        ),
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
        "ranking_eligible": result.get("ranking_eligible", True),
        "track_id": result.get("track_id"),
        "track_split": result.get("track_split"),
        "case_id": result.get("case_id"),
        "seed_namespace": result.get("seed_namespace"),
    }
    for key in (
        "official_score",
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


def rl_payload(kind: str, scenario: str, goal: str, action_mode: str,
               training: Mapping[str, Any], evaluation: Mapping[str, Any],
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
        "goal": goal,
        "action_mode": action_mode,
        "config": dict(training),
        "benchmark_config": dict(evaluation),
        "training": dict(training),
        "evaluation": dict(evaluation),
        "rows": list(rows),
        "results": list(results),
        "report": build_evaluation_report(results),
        "learning_curve": list(learning_curve or []),
        "rollouts": list(rollouts or []),
    }
    for name in (
        "track_id",
        "track_hash",
        "goal",
        "reward_spec_id",
        "policy_scope",
        "training_seed_namespace",
        "validation_seed_namespace",
        "test_seed_namespace",
        "training_seed_namespace_hash",
        "validation_seed_namespace_hash",
        "test_seed_namespace_hash",
    ):
        if training.get(name) is not None:
            payload[name] = training[name]
    track_id = training.get("track_id")
    if track_id:
        from aiogym.benchmarks import load_track

        track = load_track(
            str(track_id),
            validate_policy_contract=False,
        )
        eligibility_reasons = [
            str(reason)
            for result in results
            for reason in dict(
                result.get("safety_gate") or {}
            ).get("reasons", ())
        ]
        provenance = track_provenance(
            track,
            controller=kind,
            training_seed=training.get("seed"),
            custom_overrides={
                "environment": training.get("env_kwargs", {}),
                "controller": training.get("controller_config", {}),
            },
            eligible=all(
                bool(result.get("ranking_eligible", True))
                for result in results
            ),
            eligibility_reasons=eligibility_reasons,
        )
        payload["provenance"] = provenance
        for name in (
            "track_id",
            "track_hash",
            "goal",
            "reward_spec_id",
            "reward_spec_hash",
            "scorecard_spec_id",
            "ranking_spec_id",
            "policy_scope",
            "training_seed",
            "training_seed_namespace",
            "validation_seed_namespace",
            "test_seed_namespace",
            "training_seed_namespace_hash",
            "validation_seed_namespace_hash",
            "test_seed_namespace_hash",
            "controller_access_level",
            "model_access_level",
            "code_commit",
            "package_version",
            "custom_override_hash",
            "eligibility_status",
            "eligibility_reasons",
        ):
            payload[name] = provenance[name]
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
