#!/usr/bin/env python3
"""RLPD training, replay accounting, and checkpoint selection backend."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from ..checkpoints import (
    selected_checkpoint_manifest,
    validate_resume_config,
    validate_selected_checkpoint,
)
from ..config import RLTrainingConfig
from ..coordinator import EpisodeCoordinator
from ..episode_env import (
    make_track_episode_sampler,
    make_track_training_base_env,
    make_track_training_env,
)
from ..online_collection import VectorOnlineCollector
from ..training_artifacts import learning_curve_point
from ..validation import (
    CompleteValidationCallback,
    evaluate_validation_policy,
)
from . import BackendResult


def make_training_env(
    plan,
    *,
    worker_index: int = 0,
):
    """Build one normalized RLPD training worker from a resolved plan."""

    config = plan.config
    coordinator = EpisodeCoordinator(
        base_seed=config.training_seed,
        namespace=plan.track.seed_namespace("training"),
        next_episode_index=worker_index,
        stride=config.n_envs,
    )
    return make_track_training_env(
        plan.track,
        base_seed=config.training_seed,
        worker_index=worker_index,
        coordinator=coordinator,
    )


def unified_config(plan, dataset_id: str) -> RLTrainingConfig:
    """Return and cross-check the already resolved canonical config."""

    config = plan.config
    if config.algorithm_id != "rlpd":
        raise ValueError("resolved config algorithm_id must be 'rlpd'")
    if config.track_id != plan.track.id:
        raise ValueError("resolved config track_id does not match Track")
    if config.dataset_id != dataset_id:
        raise ValueError(
            "resolved config dataset_id does not match the dataset"
        )
    return config


def checkpoint_state(
    agent,
    config,
    *,
    collector=None,
    plan=None,
    validation_state=None,
    selected_checkpoint=None,
):
    """Build a self-contained policy/replay checkpoint and resume contract."""

    state = agent.state_dict()
    state["training_config"] = config.as_dict()
    state["training_config_hash"] = config.config_hash
    state["validation_state"] = validation_state
    state["selected_checkpoint"] = selected_checkpoint
    if collector is not None:
        if plan is None:
            raise TypeError("plan is required when collector is provided")
        resume_contract = collector.resume_state()
        resume_contract.update(
            {
                "track_hash": plan.track.track_hash,
                "reward_spec_id": plan.track.reward_spec_id,
                "distribution_hash": (
                    plan.track.training_distribution().distribution_hash
                ),
                "policy_contract_hash": _stable_mapping_hash(
                    plan.track.policy_contract
                ),
                "algorithm_id": "rlpd",
                "replay_schema": "aiogym.rlpd_replay.v1",
            }
        )
        state["resume_contract"] = resume_contract
    return state


def _aggregate_evaluation_result(evaluation):
    aggregate = evaluation["aggregate"]
    metric = str(aggregate["metric"])
    return {
        "metric": metric,
        "metric_direction": aggregate["metric_direction"],
        metric: float(aggregate["metric_value"]),
        f"{metric}_std": float(
            np.std(aggregate.get("case_values", [0.0]))
        ),
        "episodes": len(evaluation.get("base_seeds", [])),
        "runtime_total_seconds": sum(
            float(result.get("runtime_total_seconds", 0.0))
            for result in evaluation["results"]
        ),
        "ranking_eligible": aggregate["ranking_eligible"],
        "track_id": evaluation["track_id"],
        "track_split": evaluation["split"],
        "seed_namespace": evaluation["seed_namespace"],
        "case_count": evaluation["case_count"],
    }


def _training_evaluation(agent, plan):
    evaluation = evaluate_validation_policy(
        agent,
        plan.validation_plan,
        include_episodes=False,
    )
    return _aggregate_evaluation_result(evaluation)


def run_rlpd(plan) -> BackendResult:
    """Train RLPD and return a selected native checkpoint only."""

    config = plan.config
    if config.resume_mode != "restart_episode":
        raise ValueError(
            "RLPD supports only resume_mode='restart_episode'"
        )
    algorithm = dict(config.algorithm)
    replay = dict(config.replay)
    if plan.dataset_path is None:
        raise ValueError("RLPD requires dataset_path")
    if config.n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if config.total_transitions <= 0:
        raise ValueError("total_transitions must be positive")
    batch_size = int(algorithm["batch_size"])
    if batch_size <= 0 or batch_size % 2:
        raise ValueError(
            "algorithm.batch_size must be a positive even integer"
        )
    validation_seeds = tuple(config.validation_seeds)

    import torch

    from ..rlpd import RLPD

    np.random.seed(config.training_seed)
    torch.manual_seed(config.training_seed)
    started_at = time.monotonic()

    dimension_env = make_training_env(plan)
    try:
        observation_dim = int(dimension_env.observation_space.shape[0])
        action_dim = int(dimension_env.action_space.shape[0])
    finally:
        dimension_env.close()

    offline_fraction = float(algorithm["offline_fraction"])
    utd_ratio = config.utd_ratio
    canonical = offline_fraction == 0.5
    agent = RLPD(
        observation_dim,
        action_dim,
        n_critics=int(algorithm["n_critics"]),
        utd=int(utd_ratio),
        batch=batch_size,
        device=config.device,
        online_capacity=int(replay["capacity"]),
        offline_fraction=offline_fraction,
        canonical=canonical,
        seed=config.training_seed,
        scenario=plan.track.scenario,
        action_mode=str(plan.track.policy_contract["action_mode"]),
    )
    offline_replay = agent.load_dataset(
        str(plan.dataset_path),
        stratify=True,
        verify_checksums=True,
    )
    if (
        plan.dataset_id is not None
        and offline_replay.dataset_id != plan.dataset_id
    ):
        raise ValueError(
            f"dataset_id {plan.dataset_id!r} does not match "
            f"manifest {offline_replay.dataset_id!r}"
        )
    training_config = unified_config(plan, offline_replay.dataset_id)

    resume_contract = None
    resume_validation_state = None
    resume_selected_checkpoint = None
    if config.resume_checkpoint:
        resume_state = torch.load(
            config.resume_checkpoint,
            map_location=config.device,
            weights_only=False,
        )
        try:
            previous_config = RLTrainingConfig.from_mapping(
                resume_state["training_config"]
            )
            validate_resume_config(previous_config, training_config)
            _validate_rlpd_resume_contract(
                plan,
                resume_state.get("resume_contract"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc
        resume_contract = resume_state["resume_contract"]
        resume_validation_state = resume_state.get("validation_state")
        resume_selected_checkpoint = resume_state.get(
            "selected_checkpoint"
        )
        selection_sidecar = _load_selection_sidecar(
            config.resume_checkpoint
        )
        if selection_sidecar is not None:
            if (
                resume_selected_checkpoint is not None
                and resume_selected_checkpoint != selection_sidecar
            ):
                raise ValueError(
                    "RLPD selection sidecar does not match checkpoint"
                )
            resume_selected_checkpoint = selection_sidecar
        if resume_validation_state is None:
            raise ValueError(
                "RLPD checkpoint is missing validation selector state"
            )
        if resume_selected_checkpoint is not None:
            validate_selected_checkpoint(resume_selected_checkpoint)
        agent.load_state_dict(resume_state)
        accounting = agent.accounting()
        if accounting["dataset_id"] != offline_replay.dataset_id:
            raise ValueError(
                "resume checkpoint dataset_id does not match dataset"
            )
        if accounting["dataset_hash"] != offline_replay.dataset_hash:
            raise ValueError(
                "resume checkpoint dataset hash does not match dataset"
            )

    validation_callback = CompleteValidationCallback(
        plan.track,
        base_seeds=validation_seeds,
    )
    next_validation_boundary = max(
        1, int(config.evaluation["every_transitions"])
    )
    if resume_validation_state is not None:
        next_validation_boundary = _load_rlpd_validation_state(
            validation_callback,
            resume_validation_state,
        )
    history = []
    resume_result = None
    bc_steps = int(algorithm["bc_steps"])
    if bc_steps > 0 and not config.resume_checkpoint:
        agent.bc_warmstart(bc_steps)
        history.append(
            learning_curve_point(
                0,
                _training_evaluation(agent, plan),
                phase="bc",
            )
        )

    pretrain_updates = (
        0
        if config.resume_checkpoint
        else int(algorithm["pretrain_updates"])
    )
    pretrain_actor = bc_steps == 0
    for _ in range(pretrain_updates):
        agent.update(actor=pretrain_actor)
    if config.resume_checkpoint:
        resume_step = int(agent.environment_transitions)
        resume_evaluation = validation_callback.evaluate(
            agent,
            checkpoint_id=f"resume-step-{resume_step}",
            step=resume_step,
        )
        resume_result = _aggregate_evaluation_result(resume_evaluation)
        history.append(
            learning_curve_point(
                resume_step,
                resume_result,
                phase="resume",
            )
        )
    else:
        history.append(
            learning_curve_point(
                0,
                _training_evaluation(agent, plan),
                phase="pretrain",
            )
        )
    training_sampler = make_track_episode_sampler(plan.track)
    collector_env_fns = [
        lambda: make_track_training_base_env(
            plan.track,
            sampler=training_sampler,
        )
        for _ in range(config.n_envs)
    ]
    collector = VectorOnlineCollector(
        collector_env_fns,
        agent.online,
        base_seed=config.training_seed,
        namespace=plan.track.seed_namespace("training"),
        sampler=training_sampler,
        track=plan.track,
        coordinator_state=resume_contract,
    )
    collector.online_transitions = agent.environment_transitions
    starting_step = int(agent.environment_transitions)
    eval_every = int(config.evaluation["every_transitions"])
    historical_best = validation_callback.selector.best
    best_step = historical_best.step if historical_best is not None else None
    best_metric_value = (
        historical_best.metric_value
        if historical_best is not None
        else None
    )
    plan.policy_path.parent.mkdir(parents=True, exist_ok=True)
    final_checkpoint_path = (
        plan.output_dir / f"{plan.run_name}.final.pt"
    )
    try:
        selected_checkpoint = None
        if resume_selected_checkpoint is not None:
            selected_checkpoint = _materialize_rlpd_selected_checkpoint(
                resume_selected_checkpoint,
                plan.policy_path,
            )
        resume_improved = (
            resume_result is not None
            and validation_callback.selector.best is not None
            and validation_callback.selector.best.checkpoint_id
            == f"resume-step-{starting_step}"
        )
        if resume_improved:
            best_step = starting_step
            metric = str(resume_result["metric"])
            best_metric_value = float(resume_result[metric])
            _save_rlpd_selected_checkpoint(
                plan.policy_path,
                checkpoint_state(
                    agent,
                    training_config,
                    collector=collector,
                    plan=plan,
                    validation_state=_rlpd_validation_state(
                        validation_callback,
                        next_validation_boundary,
                    ),
                ),
                validation_callback.selector.best.__dict__,
            )
            selected_checkpoint = _load_selection_sidecar(
                plan.policy_path
            )
        online_started_at = time.monotonic()
        while agent.environment_transitions < config.total_transitions:
            remaining = (
                config.total_transitions
                - agent.environment_transitions
            )
            collected = min(config.n_envs, remaining)
            collector.collect(agent, collected)
            agent.environment_transitions += collected
            for _ in range(collected):
                agent.update()
            step = int(agent.environment_transitions)
            crossed_eval = eval_every > 0 and (
                step >= next_validation_boundary
                or step == config.total_transitions
            )
            if not crossed_eval:
                continue
            checkpoint_id = f"step-{step}"
            online_evaluation = validation_callback.evaluate(
                agent,
                checkpoint_id=checkpoint_id,
                step=step,
            )
            online_result = _aggregate_evaluation_result(
                online_evaluation
            )
            history.append(
                learning_curve_point(
                    step,
                    online_result,
                    phase="online",
                )
            )
            improved = (
                validation_callback.selector.best is not None
                and (
                    validation_callback.selector.best.checkpoint_id
                    == checkpoint_id
                )
            )
            if improved:
                best_step = step
                metric = str(online_result["metric"])
                best_metric_value = float(online_result[metric])
                while next_validation_boundary <= step:
                    next_validation_boundary += max(1, eval_every)
                _save_rlpd_selected_checkpoint(
                    plan.policy_path,
                    checkpoint_state(
                        agent,
                        training_config,
                        collector=collector,
                        plan=plan,
                        validation_state=_rlpd_validation_state(
                            validation_callback,
                            next_validation_boundary,
                        ),
                    ),
                    validation_callback.selector.best.__dict__,
                )
                selected_checkpoint = _load_selection_sidecar(
                    plan.policy_path
                )
            elif eval_every > 0:
                while next_validation_boundary <= step:
                    next_validation_boundary += max(1, eval_every)

        online_seconds = time.monotonic() - online_started_at
        final_step = int(agent.environment_transitions)
        final_curve = _training_evaluation(agent, plan)
        history.append(
            learning_curve_point(
                final_step,
                final_curve,
                phase="final",
            )
        )
        collector_accounting = collector.accounting()
        replay_accounting = agent.accounting()
        final_state = checkpoint_state(
            agent,
            training_config,
            collector=collector,
            plan=plan,
            validation_state=_rlpd_validation_state(
                validation_callback,
                next_validation_boundary,
            ),
            selected_checkpoint=selected_checkpoint,
        )
        torch.save(final_state, final_checkpoint_path)

        selected_by_validation = best_step is not None
        if selected_by_validation:
            selected_state = torch.load(
                plan.policy_path,
                map_location=config.device,
                weights_only=False,
            )
            agent.load_state_dict(selected_state)
            checkpoint_selection = "best-validation"
        else:
            _save_rlpd_selected_checkpoint(
                plan.policy_path,
                final_state,
                {
                    "checkpoint_id": f"final-step-{final_step}",
                    "step": final_step,
                },
            )
            checkpoint_selection = "final"

        total_seconds = time.monotonic() - started_at
        training_metadata = {
            "dataset_id": replay_accounting["dataset_id"],
            "dataset_hash": replay_accounting["dataset_hash"],
            "offline_transitions": replay_accounting[
                "offline_transitions"
            ],
            "online_transitions": collector_accounting[
                "online_transitions"
            ],
            "online_replay_transitions": replay_accounting[
                "online_replay_transitions"
            ],
            "offline_samples": replay_accounting["offline_samples"],
            "environment_transitions": final_step,
            "optimizer_updates": int(agent.gradient_updates),
            "offline_samples_available": replay_accounting[
                "offline_transitions"
            ],
            "offline_samples_drawn": replay_accounting[
                "offline_samples"
            ],
            "online_samples": replay_accounting["online_samples"],
            "sampled_offline_fraction": replay_accounting[
                "sampled_offline_fraction"
            ],
            "source_stratified": True,
            "offline_fraction": offline_fraction,
            "rlpd_variant": (
                "canonical-50-50"
                if canonical
                else "offline-ratio-ablation"
            ),
            "bc_steps": bc_steps,
            "pretrain_updates": pretrain_updates,
            "n_critics": int(algorithm["n_critics"]),
            "utd_ratio": utd_ratio,
            "batch_size": batch_size,
            "starting_step": starting_step,
            "best_step": best_step,
            "best_metric_value": best_metric_value,
            "best_checkpoint_path": (
                str(plan.policy_path)
                if selected_by_validation
                else None
            ),
            "final_checkpoint_path": str(final_checkpoint_path),
            "collector_accounting": collector_accounting,
            "replay_accounting": replay_accounting,
        }
        return BackendResult(
            algorithm_id="rlpd",
            policy_path=plan.policy_path,
            final_step=final_step,
            checkpoint_selection=checkpoint_selection,
            learning_curve=tuple(history),
            training_metadata=training_metadata,
            runtime={
                "seconds": total_seconds,
                "online_seconds": online_seconds,
                "collected_transitions": final_step - starting_step,
            },
        )
    finally:
        collector.close()


def _validate_rlpd_resume_contract(plan, state) -> None:
    if not isinstance(state, dict):
        raise ValueError(
            "RLPD checkpoint is missing restart-episode coordinator state"
        )
    if state.get("resume_mode") != "restart_episode":
        raise ValueError("RLPD supports only restart_episode resume")
    expected = {
        "track_hash": plan.track.track_hash,
        "reward_spec_id": plan.track.reward_spec_id,
        "distribution_hash": (
            plan.track.training_distribution().distribution_hash
        ),
        "policy_contract_hash": _stable_mapping_hash(
            plan.track.policy_contract
        ),
        "algorithm_id": "rlpd",
        "replay_schema": "aiogym.rlpd_replay.v1",
    }
    for name, value in expected.items():
        if state.get(name) != value:
            raise ValueError(
                f"resume checkpoint {name} does not match training contract"
            )


def _stable_mapping_hash(value) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _rlpd_validation_state(callback, next_boundary: int) -> dict:
    return {
        "schema_version": "aiogym.rlpd_validation_callback.v1",
        "validation_callback": callback.state_dict(),
        "next_validation_boundary": int(next_boundary),
    }


def _load_rlpd_validation_state(callback, state) -> int:
    if not isinstance(state, dict):
        raise TypeError("RLPD validation state must be a mapping")
    payload = dict(state)
    if payload.pop("schema_version", None) != (
        "aiogym.rlpd_validation_callback.v1"
    ):
        raise ValueError("unsupported RLPD validation state")
    callback.load_state_dict(payload.pop("validation_callback", None))
    boundary = payload.pop("next_validation_boundary", None)
    if (
        isinstance(boundary, bool)
        or not isinstance(boundary, int)
        or boundary <= 0
    ):
        raise ValueError("next validation boundary is invalid")
    if payload:
        raise ValueError(
            "RLPD validation state contains unknown fields: "
            + ", ".join(sorted(payload))
        )
    return boundary


def _selection_sidecar_path(path) -> Path:
    return Path(f"{path}.selection.json")


def _write_selection_sidecar(path, manifest) -> None:
    target = _selection_sidecar_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_selection_sidecar(path):
    target = _selection_sidecar_path(path)
    if not target.is_file():
        return None
    with target.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("RLPD selection sidecar must be a mapping")
    return payload


def _save_rlpd_selected_checkpoint(path, state, record) -> None:
    import torch

    target = Path(path)
    torch.save(state, target)
    manifest = selected_checkpoint_manifest(target, record)
    _write_selection_sidecar(target, manifest)


def _materialize_rlpd_selected_checkpoint(manifest, destination) -> dict:
    source = validate_selected_checkpoint(manifest)
    destination = Path(destination).resolve()
    if source.resolve() != destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    updated = selected_checkpoint_manifest(
        destination,
        manifest["selection_record"],
    )
    _write_selection_sidecar(destination, updated)
    return updated


__all__ = [
    "checkpoint_state",
    "make_training_env",
    "run_rlpd",
    "unified_config",
]
