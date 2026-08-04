"""Shared post-training validation and artifact lifecycle."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiogym._environment.builder import build_track_case_environment
from aiogym.controllers.checkpoints import (
    checkpoint_sha256,
    learned_policy_spec_for_track,
    load_policy_checkpoint,
)
from aiogym.controllers.export import ExportResult, export_policy_checkpoint
from aiogym.evaluation.artifact import check_benchmark_artifacts
from aiogym.evaluation.execution.rollouts import rollout_controller
from aiogym.evaluation.provenance import reward_spec_hash, seed_namespace_hash

from .backends.contracts import BackendResult, validate_backend_result
from .plan import ResolvedTrainingPlan
from .training_artifacts import rl_payload, result_row, write_rl_artifacts
from .validation import evaluate_validation_policy


@dataclass(frozen=True)
class TrainingLifecycleResult:
    policy_sha256: str
    validation: dict[str, Any]
    artifact_check: dict[str, Any]


def finalize_training_lifecycle(
    plan: ResolvedTrainingPlan,
    backend_result: BackendResult,
) -> TrainingLifecycleResult:
    """Load, validate, and publish the backend's selected checkpoint."""

    validate_backend_result(plan, backend_result)
    digest = checkpoint_sha256(backend_result.policy_path)
    spec = learned_policy_spec_for_track(
        backend_result.policy_path,
        backend_result.algorithm_id,
        digest,
        plan.track,
    )
    controller = load_policy_checkpoint(spec, device=plan.config.device)
    evaluation = evaluate_validation_policy(
        controller,
        plan.validation_plan,
        include_episodes=True,
    )
    rollouts = _validation_rollouts(plan, controller)
    export_result = _optional_export(plan, spec)
    _write_runner_artifacts(
        plan,
        backend_result,
        digest,
        evaluation,
        rollouts,
        export_result,
    )
    payload = _read_json(plan.artifact_dir / "benchmark.json")
    validation = payload.get("track_evaluation")
    if not isinstance(validation, dict):
        raise ValueError(
            "training artifact is missing track_evaluation"
        )
    _validate_training_artifact(plan, payload, validation)
    artifact_check = check_benchmark_artifacts(plan.artifact_dir)
    if not artifact_check["ok"]:
        failed = ", ".join(
            str(row.get("name"))
            for row in artifact_check.get("failed", ())
        )
        raise ValueError(
            "training artifacts failed structural validation: "
            + (failed or "unknown check")
        )
    if (
        export_result.status == "failed"
        and bool(plan.config.output.get("strict_export", False))
    ):
        raise RuntimeError(
            "strict ONNX export failed after preserving native checkpoint "
            f"and artifacts: {export_result.error}"
        )
    return TrainingLifecycleResult(
        policy_sha256=digest,
        validation=dict(validation),
        artifact_check=dict(artifact_check),
    )


def _write_runner_artifacts(
    plan,
    backend_result,
    digest,
    evaluation,
    rollouts,
    export_result,
) -> None:
    track = plan.track
    config = plan.config
    action_mode = str(track.policy_contract["action_mode"])
    training = {
        "algo": config.algorithm_id,
        "algorithm_id": config.algorithm_id,
        "scenario": track.scenario,
        "action_mode": action_mode,
        "goal": track.goal,
        "reward_spec_id": track.reward_spec_id,
        "reward_spec_hash": reward_spec_hash(track.reward_spec_id),
        "track_id": track.id,
        "track_hash": track.track_hash,
        "policy_scope": track.policy_scope,
        "seed": config.training_seed,
        "training_seed": config.training_seed,
        "n_envs": config.n_envs,
        "device": config.device,
        "budget_unit": config.budget_unit,
        "budget_value": config.budget_value,
        "environment_transitions": (
            0
            if config.algorithm_id == "bc"
            else int(backend_result.final_step)
        ),
        "optimizer_updates": (
            int(backend_result.final_step)
            if config.algorithm_id == "bc"
            else None
        ),
        "offline_samples_available": None,
        "offline_samples_drawn": None,
        "training_config": config.as_dict(),
        "training_config_hash": config.config_hash,
        "validation_plan_hash": plan.validation_plan.plan_hash,
        "training_seed_namespace": track.seed_namespace("training"),
        "validation_seed_namespace": track.seed_namespace("validation"),
        "training_seed_namespace_hash": seed_namespace_hash(
            track.seed_namespace("training")
        ),
        "validation_seed_namespace_hash": seed_namespace_hash(
            track.seed_namespace("validation")
        ),
        "checkpoint_path": str(plan.policy_path),
        "checkpoint_sha256": digest,
        "checkpoint_selection": backend_result.checkpoint_selection,
        "dataset_id": plan.dataset_id,
        "dataset_hash": plan.dataset_hash,
        "dataset_path": (
            None if plan.dataset_path is None else str(plan.dataset_path)
        ),
        "final_step": backend_result.final_step,
    }
    _merge_backend_metadata(training, backend_result.training_metadata)
    training["runtime"] = dict(backend_result.runtime)
    training["exports"] = {
        "onnx": export_result.as_dict(),
    }
    results = list(evaluation["results"])
    payload = rl_payload(
        kind=f"{config.algorithm_id}_train_eval",
        scenario=track.scenario,
        goal=track.goal,
        action_mode=action_mode,
        training=training,
        evaluation=track.metadata(),
        results=results,
        rows=[
            result_row(
                result,
                track.scenario,
                action_mode,
                controller=config.algorithm_id.upper(),
                run_case_id=(
                    f"{track.id}:{result.get('case_id')}:"
                    f"{config.algorithm_id}"
                ),
            )
            for result in results
        ],
        learning_curve=backend_result.learning_curve,
        rollouts=rollouts,
        extra={"track_evaluation": evaluation},
    )
    write_rl_artifacts(
        plan.artifact_dir,
        payload,
        replace_existing=plan.replace_existing,
    )


def _merge_backend_metadata(training, backend_metadata) -> None:
    protected = {
        "track_id",
        "track_hash",
        "training_config_hash",
        "validation_plan_hash",
        "checkpoint_path",
        "checkpoint_sha256",
    }
    collisions = protected.intersection(backend_metadata)
    if collisions:
        raise ValueError(
            "backend metadata cannot override canonical fields: "
            + ", ".join(sorted(collisions))
        )
    for name, value in backend_metadata.items():
        if name in training and training[name] is None:
            training[name] = value
            continue
        if name in training and training[name] != value:
            raise ValueError(
                f"backend metadata conflicts with canonical field: {name}"
            )
        training[name] = value


def _optional_export(plan, policy_spec) -> ExportResult:
    output = dict(plan.config.output)
    if not bool(output.get("onnx", False)):
        return ExportResult(format="onnx", status="skipped")
    return export_policy_checkpoint(
        policy_spec,
        format="onnx",
        output=plan.output_dir / f"{plan.run_name}.onnx",
        device=plan.config.device,
    )


def _validation_rollouts(plan, controller) -> list[dict[str, Any]]:
    output = dict(plan.config.output)
    if not bool(output.get("save_rollout", False)):
        return []
    seed = int(plan.validation_plan.base_seeds[0])
    max_steps = output.get("rollout_steps")
    rows = []
    for case in plan.track.resolved_cases("validation"):
        env = build_track_case_environment(plan.track, case)
        try:
            rollout = rollout_controller(
                controller,
                env,
                seed=seed,
                max_steps=max_steps,
            )
        finally:
            env.close()
        rollout.update(
            {
                "track_id": plan.track.id,
                "track_split": "validation",
                "case_id": case.case_id,
                "resolved_case_hash": case.resolved_case_hash,
            }
        )
        rows.append(rollout)
    return rows


def _validate_training_artifact(plan, payload, validation) -> None:
    expected = {
        "split": "validation",
        "track_id": plan.track.id,
        "track_hash": plan.track.track_hash,
        "episode_plan_hash": plan.validation_plan.plan_hash,
    }
    for name, value in expected.items():
        if validation.get(name) != value:
            raise ValueError(
                f"training artifact {name} does not match resolved plan"
            )
    if tuple(validation.get("base_seeds") or ()) != tuple(
        plan.validation_plan.base_seeds
    ):
        raise ValueError(
            "training artifact validation seeds do not match resolved plan"
        )
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("training artifact is missing training metadata")
    if training.get("training_config_hash") != plan.config.config_hash:
        raise ValueError(
            "training artifact config hash does not match resolved config"
        )
    artifact_plan_hash = training.get("validation_plan_hash")
    if (
        artifact_plan_hash is not None
        and artifact_plan_hash != plan.validation_plan.plan_hash
    ):
        raise ValueError(
            "training metadata validation plan hash does not match"
        )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


__all__ = [
    "TrainingLifecycleResult",
    "finalize_training_lifecycle",
]
