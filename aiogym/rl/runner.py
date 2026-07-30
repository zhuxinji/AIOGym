"""Shared public lifecycle for all stable training algorithms."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from .adapters import build_training_adapter
from .config import RLTrainingConfig
from .plan import ResolvedTrainingPlan, resolve_training_plan


RUN_RESULT_SCHEMA_VERSION = "aiogym.run_result.v1"


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
    adapter_state: dict[str, Any]
    validation: dict[str, Any] = field(default_factory=dict)
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
            "adapter_state": dict(self.adapter_state),
            "validation": _compact_validation(self.validation),
            "distribution_id": self.distribution_id,
        }


def run_experiment(
    config: RLTrainingConfig,
    *,
    validation_plan=None,
    adapter_factory=build_training_adapter,
) -> RunResult:
    """Resolve, execute, and report one complete training lifecycle."""

    plan = resolve_training_plan(config)
    if validation_plan is not None:
        _validate_external_plan(plan, validation_plan)
        plan = replace(plan, validation_plan=validation_plan)
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    plan.artifact_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(plan.resolved_config_path, plan.config.as_dict())
    adapter = adapter_factory(plan)
    adapter.build(plan)
    adapter.train_chunk(plan.config.total_transitions)
    adapter.save_policy(plan.policy_path)
    validation = _load_validation_artifact(plan.artifact_dir)
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
        adapter_state=adapter.state_dict(),
        validation=validation,
        distribution_id=plan.track.train_distribution_id,
    )
    _atomic_json(
        plan.output_dir / f"{plan.run_name}.run-result.json",
        result.as_dict(),
    )
    return result


def run_seed_sweep(
    config: RLTrainingConfig,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Run independent local seeds and persist one compact comparison."""

    resolved_seeds = tuple(int(seed) for seed in seeds)
    if (
        not resolved_seeds
        or min(resolved_seeds) < 0
        or len(set(resolved_seeds)) != len(resolved_seeds)
    ):
        raise ValueError("seeds must be unique non-negative integers")
    base_output = dict(config.output)
    base_name = str(
        base_output.get("name")
        or f"{config.algorithm_id}-{config.track_id}"
    )
    results = []
    for seed in resolved_seeds:
        output = {
            **base_output,
            "name": f"{base_name}-seed{seed}",
        }
        results.append(
            run_experiment(
                replace(
                    config,
                    training_seed=seed,
                    output=output,
                )
            )
        )
    summary = {
        "schema_version": "aiogym.multi_seed_run.v1",
        "track_id": config.track_id,
        "algorithm_id": config.algorithm_id,
        "seeds": list(resolved_seeds),
        "runs": [result.as_dict() for result in results],
        "validation_summary": _summarize_validations(results),
    }
    directory = Path(str(base_output.get("directory", "runs")))
    _atomic_json(directory / f"{base_name}.multi-seed.json", summary)
    return summary


def _validate_external_plan(plan: ResolvedTrainingPlan, validation_plan) -> None:
    if validation_plan.track.track_hash != plan.track.track_hash:
        raise ValueError("validation plan Track does not match config")
    if tuple(validation_plan.base_seeds) != tuple(
        plan.config.validation_seeds
    ):
        raise ValueError("validation plan seeds do not match config")


def _load_validation_artifact(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "benchmark.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    evaluation = payload.get("track_evaluation")
    return dict(evaluation) if isinstance(evaluation, dict) else {}


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
        "aggregate": aggregate,
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


__all__ = [
    "RUN_RESULT_SCHEMA_VERSION",
    "RunResult",
    "run_experiment",
    "run_seed_sweep",
]
