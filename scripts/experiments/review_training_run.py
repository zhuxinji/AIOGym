"""Review validation-only training artifacts without running policies."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from aiogym._internal.serialization import file_sha256, write_json_artifact
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.plan import resolve_training_plan


REVIEW_SCHEMA_VERSION = "aiogym.training_run_review.v1"


def review_training_run(source: str | Path) -> dict:
    source_path = Path(source)
    payload = _read_json(source_path)
    _reject_test_content(payload)
    runs = payload.get("runs") if "runs" in payload else [payload]
    if not isinstance(runs, list) or not runs:
        raise ValueError("training review requires at least one run")
    rows = [_review_run(dict(run), source_path.parent) for run in runs]
    expected_seeds = list(payload.get("seeds") or ())
    actual_seeds = [row["training_seed"] for row in rows]
    top_errors = []
    if expected_seeds and actual_seeds != expected_seeds:
        top_errors.append("multi-seed summary has incomplete or reordered seeds")
    statistics = dict(payload.get("training_seed_statistics") or {})
    if expected_seeds and statistics.get("training_seeds") != expected_seeds:
        top_errors.append("training_seed_statistics does not cover sweep seeds")
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source": str(source_path),
        "source_sha256": file_sha256(source_path),
        "passed": not top_errors and all(row["passed"] for row in rows),
        "errors": top_errors,
        "run_count": len(rows),
        "training_seeds": actual_seeds,
        "runs": rows,
        "training_seed_statistics": statistics,
        "test_split_accessed": False,
    }


def _review_run(run: dict, base_dir: Path) -> dict:
    _require_finite_json(run, "run-result")
    errors = []
    warnings = []
    config_path = _reference_path(run.get("resolved_config_path"), base_dir)
    config = None
    plan = None
    if config_path is None or not config_path.is_file():
        errors.append("resolved config is missing")
    else:
        try:
            config = RLTrainingConfig.load(config_path)
            plan = resolve_training_plan(config)
        except Exception as exc:
            errors.append(f"resolved config is invalid: {exc}")
    if plan is not None:
        if plan.config.config_hash != run.get("config_hash"):
            errors.append("config hash mismatch")
        if plan.track.id != run.get("track_id"):
            errors.append("Track ID mismatch")
        if plan.track.track_hash != run.get("track_hash"):
            errors.append("Track hash mismatch")
        if plan.validation_plan.plan_hash != run.get("validation_plan_hash"):
            errors.append("validation plan hash mismatch")

    backend = dict(run.get("backend") or {})
    final_step = backend.get("final_step")
    selection = backend.get("checkpoint_selection")
    selected_step = backend.get("selected_checkpoint_step")
    if selection not in {"best-validation", "final", "final-fixed-dataset"}:
        errors.append("checkpoint selection type is invalid")
    if selected_step is None or int(selected_step) < 0:
        errors.append("selected checkpoint step is missing or invalid")
    if plan is not None and final_step != plan.config.budget_value:
        errors.append("executed transitions do not match declared budget")

    policy_path = _reference_path(run.get("policy_path"), base_dir)
    policy_integrity = {"path": None, "exists": False, "sha256": None}
    if policy_path is None or not policy_path.is_file():
        errors.append("selected policy file is missing")
    else:
        digest = file_sha256(policy_path)
        policy_integrity = {
            "path": str(policy_path),
            "exists": True,
            "sha256": digest,
        }
        if digest != run.get("policy_sha256"):
            errors.append("selected policy SHA-256 mismatch")

    validation = dict(run.get("validation") or {})
    _reject_test_content(validation)
    aggregate = dict(validation.get("aggregate") or {})
    if not aggregate:
        errors.append("validation aggregate is missing")
    eligible = bool(aggregate.get("ranking_eligible", False))
    if aggregate and not eligible:
        warnings.append("validation ranking is ineligible")
    case_values = list(aggregate.get("case_values") or ())
    case_rows = list(validation.get("results") or ())
    per_case = []
    for index, value in enumerate(case_values):
        identity = case_rows[index] if index < len(case_rows) else {}
        per_case.append(
            {
                "case_id": identity.get("case_id", f"case-{index}"),
                "resolved_case_hash": identity.get("resolved_case_hash"),
                "metric_value": float(value),
                "ranking_eligible": identity.get("ranking_eligible"),
            }
        )

    artifact_dir = _reference_path(run.get("artifact_dir"), base_dir)
    benchmark_path = (
        None if artifact_dir is None else artifact_dir / "benchmark.json"
    )
    benchmark = None
    if benchmark_path is None or not benchmark_path.is_file():
        errors.append("benchmark.json is missing")
    else:
        benchmark = _read_json(benchmark_path)
        _reject_test_content(benchmark)
        _require_finite_json(benchmark, "benchmark.json")
    curve = list((benchmark or {}).get("learning_curve") or ())
    curve_steps = [
        int(row.get("timesteps", row.get("step", -1))) for row in curve
    ]
    curve_monotonic = all(
        left < right for left, right in zip(curve_steps, curve_steps[1:])
    )
    if curve_steps and (not curve_monotonic or len(set(curve_steps)) != len(curve_steps)):
        errors.append("learning-curve steps are not strictly increasing")
    final_participated = bool(final_step in curve_steps) if curve_steps else False
    if curve_steps and not final_participated:
        errors.append("final checkpoint is absent from selector learning curve")
    training = dict((benchmark or {}).get("training") or {})
    runtime = dict(training.get("runtime") or backend.get("runtime") or {})
    provenance_present = all(
        training.get(name) is not None
        for name in ("training_config_hash", "track_hash", "device")
    )
    if benchmark is not None and not provenance_present:
        errors.append("training dependency/device provenance is incomplete")

    episode_specs = list(training.get("training_episode_specs") or ())
    unique_specs = training.get("unique_training_episode_specs")
    if unique_specs is None and episode_specs:
        unique_specs = len(
            {row.get("episode_spec_hash") for row in episode_specs}
        )
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "training_seed": int(run.get("training_seed", -1)),
        "track_id": run.get("track_id"),
        "track_hash": run.get("track_hash"),
        "config_hash": run.get("config_hash"),
        "validation_plan_hash": run.get("validation_plan_hash"),
        "declared_budget": None if plan is None else plan.config.budget_value,
        "executed_transitions": final_step,
        "checkpoint_selection": selection,
        "selected_checkpoint_step": selected_step,
        "policy_integrity": policy_integrity,
        "learning_curve": {
            "point_count": len(curve_steps),
            "steps": curve_steps,
            "strictly_increasing": curve_monotonic,
            "final_checkpoint_participated": final_participated,
        },
        "ranking_eligible": eligible,
        "per_case_validation": per_case,
        "unique_training_episode_specs": unique_specs,
        "runtime": runtime,
        "provenance_present": provenance_present,
    }


def _reference_path(value, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return base_dir / path


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must be a mapping: {path}")
    return value


def _reject_test_content(value) -> None:
    if isinstance(value, dict):
        execution_fields = {
            "aggregate",
            "base_seeds",
            "episode_plan_hash",
            "results",
            "runs",
            "summaries",
        }
        if value.get("split") == "test" and execution_fields & set(value):
            raise ValueError("training review refuses test-split artifacts")
        if value.get("track_split") == "test":
            raise ValueError("training review refuses test-split artifacts")
        for item in value.values():
            _reject_test_content(item)
    elif isinstance(value, list):
        for item in value:
            _reject_test_content(item)


def _require_finite_json(value, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} contains NaN or Inf")
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_json(item, context)
    elif isinstance(value, list):
        for item in value:
            _require_finite_json(item, context)


def render_markdown(report: dict) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    lines = [f"# Training Run Review — {status}", ""]
    lines.append(f"Source: `{report['source']}`")
    lines.append("")
    for row in report["runs"]:
        lines.extend(
            [
                f"## Seed {row['training_seed']}",
                "",
                f"- Status: {'PASS' if row['passed'] else 'FAIL'}",
                f"- Selected checkpoint: {row['checkpoint_selection']} at step {row['selected_checkpoint_step']}",
                f"- Validation eligible: {row['ranking_eligible']}",
                f"- Unique training EpisodeSpecs: {row['unique_training_episode_specs']}",
            ]
        )
        for error in row["errors"]:
            lines.append(f"- ERROR: {error}")
        for warning in row["warnings"]:
            lines.append(f"- WARNING: {warning}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)
    if Path(args.output_md).exists():
        raise FileExistsError(
            f"training review already exists: {args.output_md}"
        )
    report = review_training_run(args.source)
    write_json_artifact(args.output_json, report)
    Path(args.output_md).write_text(render_markdown(report), encoding="utf-8")
    print(args.output_json)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
