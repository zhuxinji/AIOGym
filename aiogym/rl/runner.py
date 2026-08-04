"""Shared public lifecycle for all stable training algorithms."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aiogym._internal.serialization import (
    file_sha256,
    stable_json_hash,
    write_json_artifact,
)
from aiogym._internal.validation import seed_sequence
from aiogym.evaluation.statistics import (
    interquartile_mean,
    stratified_bootstrap_ci,
)

from .backends import run_backend, validate_backend_result
from .config import RLTrainingConfig
from .lifecycle import finalize_training_lifecycle
from .plan import ResolvedTrainingPlan, resolve_training_plan
from .run_claim import (
    RunClaim,
    RunClaimIdentity,
    SeedSweepClaim,
    SeedSweepIdentity,
)


RUN_RESULT_SCHEMA_VERSION = "aiogym.run_result.v3"


@dataclass(frozen=True)
class RunResult:
    config_hash: str
    track_id: str
    track_hash: str
    algorithm_id: str
    training_seed: int
    output_dir: str
    policy_path: str
    artifact_dir: str
    resolved_config_path: str
    validation_plan_hash: str
    policy_sha256: str
    backend: dict[str, Any]
    validation: dict[str, Any] = field(default_factory=dict)
    artifact_check: dict[str, Any] = field(default_factory=dict)
    distribution_id: str | None = None
    schema_version: str = RUN_RESULT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "track_id": self.track_id,
            "track_hash": self.track_hash,
            "algorithm_id": self.algorithm_id,
            "training_seed": self.training_seed,
            "output_dir": self.output_dir,
            "policy_path": self.policy_path,
            "artifact_dir": self.artifact_dir,
            "resolved_config_path": self.resolved_config_path,
            "validation_plan_hash": self.validation_plan_hash,
            "policy_sha256": self.policy_sha256,
            "backend": dict(self.backend),
            "validation": _compact_validation(self.validation),
            "artifact_check": _compact_artifact_check(
                self.artifact_check
            ),
            "distribution_id": self.distribution_id,
        }


def run_experiment(
    config: RLTrainingConfig,
    *,
    validation_plan=None,
    backend_runner=run_backend,
    lifecycle_finalizer=finalize_training_lifecycle,
    overwrite: bool = False,
) -> RunResult:
    """Resolve, execute, and report one complete training lifecycle."""

    plan = resolve_training_plan(config)
    resume = bool(plan.config.resume_checkpoint)
    if resume and plan.config.algorithm_id == "bc":
        raise ValueError("BC resume is not implemented")
    if overwrite and resume:
        raise ValueError("overwrite and resume are mutually exclusive")
    if validation_plan is not None:
        _validate_external_plan(plan, validation_plan)
        plan = replace(plan, validation_plan=validation_plan)
    plan = replace(plan, replace_existing=bool(overwrite or resume))
    claim_path = plan.output_dir / f".{plan.run_name}.claim.json"
    run_result_path = plan.output_dir / f"{plan.run_name}.run-result.json"
    artifact_path = plan.artifact_dir / "benchmark.json"
    claim = RunClaim.acquire(
        claim_path,
        RunClaimIdentity(
            run_name=plan.run_name,
            config_hash=plan.config.config_hash,
            track_hash=plan.track.track_hash,
            algorithm_id=plan.config.algorithm_id,
            training_seed=plan.config.training_seed,
        ),
        occupied_paths=_owned_paths(plan, run_result_path),
        overwrite=overwrite,
        resume=resume,
        previous_artifact=artifact_path,
    )
    try:
        if overwrite:
            _clear_file_outputs(plan, run_result_path)
        plan.output_dir.mkdir(parents=True, exist_ok=True)
        plan.artifact_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(plan.resolved_config_path, plan.config.as_dict())
        backend_result = backend_runner(plan)
        validate_backend_result(plan, backend_result)
        lifecycle = lifecycle_finalizer(plan, backend_result)
        result = RunResult(
            config_hash=plan.config.config_hash,
            track_id=plan.track.id,
            track_hash=plan.track.track_hash,
            algorithm_id=plan.config.algorithm_id,
            training_seed=plan.config.training_seed,
            output_dir=str(plan.output_dir),
            policy_path=str(plan.policy_path),
            artifact_dir=str(plan.artifact_dir),
            resolved_config_path=str(plan.resolved_config_path),
            validation_plan_hash=plan.validation_plan.plan_hash,
            policy_sha256=lifecycle.policy_sha256,
            backend=backend_result.compact_summary(),
            validation=lifecycle.validation,
            artifact_check=lifecycle.artifact_check,
            distribution_id=plan.track.train_distribution_id,
        )
        _atomic_json(run_result_path, result.as_dict())
        claim.complete(
            run_result_hash=file_sha256(run_result_path),
            artifact_hash=(
                file_sha256(artifact_path) if artifact_path.is_file() else None
            ),
        )
    except BaseException as exc:
        claim.fail(exc)
        raise
    return result


def run_seed_sweep(
    config: RLTrainingConfig,
    seeds: Sequence[int],
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run independent local seeds and persist one compact comparison."""

    resolved_seeds = seed_sequence("seeds", seeds)
    if config.resume_checkpoint:
        raise ValueError(
            "--resume cannot be combined with --seeds; multi-seed sweeps "
            "are independent fresh runs"
        )
    base_plan = resolve_training_plan(config)
    base_output = dict(config.output)
    base_name = str(
        base_output.get("name")
        or f"{config.algorithm_id}-{config.track_id}"
    )
    directory = Path(str(base_output.get("directory", "runs")))
    summary_path = directory / f"{base_name}.multi-seed.json"
    identity = SeedSweepIdentity(
        base_name=base_name,
        config_hash=base_plan.config.config_hash,
        track_id=base_plan.track.id,
        track_hash=base_plan.track.track_hash,
        algorithm_id=base_plan.config.algorithm_id,
        seeds=resolved_seeds,
    )
    claim = SeedSweepClaim.acquire(
        directory / f".{base_name}.multi-seed.claim.json",
        identity,
        summary_path=summary_path,
        overwrite=overwrite,
    )
    results = []
    try:
        for seed in resolved_seeds:
            output = {
                **base_output,
                "name": f"{base_name}-seed{seed}",
            }
            result = run_experiment(
                replace(config, training_seed=seed, output=output),
                overwrite=overwrite,
            )
            results.append(result)
            claim.record_child(
                seed=seed,
                result_hash=_mapping_sha256(result.as_dict()),
            )
        summary = {
            "schema_version": "aiogym.multi_seed_run.v1",
            "sweep_id": identity.sweep_id,
            "config_hash": identity.config_hash,
            "track_id": identity.track_id,
            "track_hash": identity.track_hash,
            "algorithm_id": identity.algorithm_id,
            "summary_path": str(summary_path),
            "seeds": list(resolved_seeds),
            "runs": [result.as_dict() for result in results],
            "validation_summary": _summarize_validations(results),
            "training_seed_statistics": _training_seed_statistics(results),
        }
        write_json_artifact(summary_path, summary, overwrite=overwrite)
        claim.complete(summary_hash=file_sha256(summary_path))
    except BaseException as exc:
        claim.fail(exc)
        raise
    return summary


def _validate_external_plan(plan: ResolvedTrainingPlan, validation_plan) -> None:
    if validation_plan.track.track_hash != plan.track.track_hash:
        raise ValueError("validation plan Track does not match config")
    if tuple(validation_plan.base_seeds) != tuple(
        plan.config.validation_seeds
    ):
        raise ValueError("validation plan seeds do not match config")


def _summarize_validations(results) -> dict[str, Any]:
    evaluations = [
        result.validation
        for result in results
        if result.validation.get("aggregate")
    ]
    if not evaluations:
        return {}
    aggregates = [evaluation["aggregate"] for evaluation in evaluations]
    return {
        "metric": aggregates[0]["metric"],
        "metric_direction": aggregates[0]["metric_direction"],
        "mean_metric_value": sum(
            float(row["metric_value"]) for row in aggregates
        )
        / len(aggregates),
        "mean_official_score": sum(
            float(row.get("official_score", 0.0))
            for row in aggregates
        )
        / len(aggregates),
        "eligible_runs": sum(
            bool(row.get("ranking_eligible", True))
            for row in aggregates
        ),
        "run_count": len(aggregates),
    }


def _training_seed_statistics(results) -> dict[str, Any]:
    """Summarize validation with independent training seeds as rows."""

    evaluations = [result.validation for result in results]
    populated = [
        evaluation
        for evaluation in evaluations
        if evaluation.get("aggregate")
    ]
    if not populated:
        return {}
    if len(populated) != len(results):
        raise ValueError(
            "every training-seed run must contain a validation aggregate"
        )

    identity = None
    case_ids = None
    case_hashes = None
    matrix = []
    run_values = []
    official_scores = []
    eligibility = []
    selected_steps = []
    training_seeds = []
    for result, evaluation in zip(results, populated):
        aggregate = dict(evaluation["aggregate"])
        rows = list(evaluation.get("results") or ())
        current_case_ids = tuple(str(row["case_id"]) for row in rows)
        current_case_hashes = tuple(
            str(row.get("resolved_case_hash", "")) for row in rows
        )
        case_values = [float(value) for value in aggregate["case_values"]]
        if not current_case_ids or len(current_case_ids) != len(case_values):
            raise ValueError(
                "validation results must preserve ordered Case rows matching "
                "aggregate.case_values"
            )
        current_identity = (
            str(result.track_id),
            str(result.track_hash),
            str(evaluation.get("track_id")),
            str(evaluation.get("track_hash")),
            str(result.validation_plan_hash),
            str(evaluation.get("episode_plan_hash")),
            str(aggregate["metric"]),
            str(aggregate["metric_direction"]),
        )
        if identity is None:
            identity = current_identity
            case_ids = current_case_ids
            case_hashes = current_case_hashes
        elif (
            current_identity != identity
            or current_case_ids != case_ids
            or current_case_hashes != case_hashes
        ):
            raise ValueError(
                "training-seed validations must share Track/hash, validation "
                "plan/hash, metric, and ordered Cases"
            )
        if (
            current_identity[0] != current_identity[2]
            or current_identity[1] != current_identity[3]
            or current_identity[4] != current_identity[5]
        ):
            raise ValueError(
                "run identity does not match its validation identity"
            )
        matrix.append(case_values)
        run_values.append(float(aggregate["metric_value"]))
        official_scores.append(float(aggregate.get("official_score", 0.0)))
        eligibility.append(bool(aggregate.get("ranking_eligible", True)))
        selected_step = result.backend.get("selected_checkpoint_step")
        if selected_step is None:
            selected_step = result.backend.get("final_step")
        selected_steps.append(
            None if selected_step is None else int(selected_step)
        )
        training_seeds.append(int(result.training_seed))

    case_matrix = np.asarray(matrix, dtype=np.float64)
    run_array = np.asarray(run_values, dtype=np.float64)
    score_array = np.asarray(official_scores, dtype=np.float64)
    if not (
        np.all(np.isfinite(case_matrix))
        and np.all(np.isfinite(run_array))
        and np.all(np.isfinite(score_array))
    ):
        raise ValueError("training-seed statistics require finite values")
    direction = identity[-1]
    if direction not in {"minimize", "maximize"}:
        raise ValueError("validation metric direction is invalid")
    worst_index = int(
        np.argmax(run_array) if direction == "minimize" else np.argmin(run_array)
    )
    return {
        "unit": "independent_training_seed",
        "training_seeds": training_seeds,
        "metric": identity[-2],
        "metric_direction": direction,
        "case_ids": list(case_ids),
        "case_matrix": case_matrix.tolist(),
        "run_metric_values": run_array.tolist(),
        "mean": float(np.mean(run_array)),
        "std": float(np.std(run_array, ddof=1)) if len(run_array) > 1 else 0.0,
        "median": float(np.median(run_array)),
        "iqm": interquartile_mean(case_matrix),
        "iqm_bootstrap": stratified_bootstrap_ci(case_matrix),
        "official_score_mean": float(np.mean(score_array)),
        "official_score_std": (
            float(np.std(score_array, ddof=1)) if len(score_array) > 1 else 0.0
        ),
        "ranking_eligibility_rate": float(np.mean(eligibility)),
        "worst_training_seed": {
            "training_seed": training_seeds[worst_index],
            "metric_value": float(run_array[worst_index]),
        },
        "selected_checkpoint_steps": selected_steps,
    }


def _compact_validation(evaluation) -> dict[str, Any]:
    if not evaluation or not evaluation.get("aggregate"):
        return {}
    aggregate = dict(evaluation["aggregate"])
    return {
        "split": evaluation.get("split"),
        "seed_namespace": evaluation.get("seed_namespace"),
        "base_seeds": list(evaluation.get("base_seeds") or ()),
        "episode_plan_hash": evaluation.get("episode_plan_hash"),
        "case_count": evaluation.get("case_count"),
        "results": [
            {
                name: row.get(name)
                for name in (
                    "case_id",
                    "resolved_case_hash",
                    "ranking_utility",
                    "official_score",
                    "ranking_eligible",
                )
            }
            for row in evaluation.get("results", ())
        ],
        "aggregate": aggregate,
    }


def _compact_artifact_check(check) -> dict[str, Any]:
    if not check:
        return {}
    return {
        "schema_version": check.get("schema_version"),
        "ok": bool(check.get("ok")),
        "failed": list(check.get("failed") or ()),
    }


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _owned_paths(plan, run_result_path: Path) -> tuple[Path, ...]:
    final_suffix = ".pt" if plan.config.algorithm_id in {"bc", "rlpd"} else ".zip"
    return (
        plan.policy_path,
        Path(f"{plan.policy_path}.training.ckpt"),
        plan.output_dir / f"{plan.run_name}.final{final_suffix}",
        plan.output_dir / f"{plan.run_name}.onnx",
        plan.resolved_config_path,
        run_result_path,
        plan.artifact_dir / "benchmark.json",
        plan.artifact_dir / "report.md",
    )


def _clear_file_outputs(plan, run_result_path: Path) -> None:
    for path in _owned_paths(plan, run_result_path):
        if path.is_file() or path.is_symlink():
            path.unlink()


def _mapping_sha256(value: dict[str, Any]) -> str:
    return stable_json_hash(value)


__all__ = [
    "RUN_RESULT_SCHEMA_VERSION",
    "RunResult",
    "run_experiment",
    "run_seed_sweep",
]
