#!/usr/bin/env python3
"""Stable-Baselines3 training and checkpoint selection backend."""
from __future__ import annotations

import math
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from aiogym.controllers.adapters import PolicyController
from aiogym._internal.serialization import stable_json_hash

from ..checkpoints import (
    CheckpointManager,
    TrainingCheckpoint,
    capture_rng_state,
    restore_rng_state,
    selected_checkpoint_manifest,
    validate_selected_checkpoint,
)
from ..coordinator import EpisodeCoordinator
from ..episode_env import make_track_training_env
from ..statistics import NormalizedActionWrapper
from ..utd import sb3_update_schedule
from ..validation import CompleteValidationCallback, evaluate_validation_policy
from .contracts import BackendResult
from .sb3_algorithms import (
    get_algorithm_adapter,
    load_algorithm_checkpoint,
)


def make_training_env(
    plan,
    rank: int = 0,
    *,
    coordinator: EpisodeCoordinator | None = None,
    resume_coordinator_state=None,
):
    """Build one deterministic Track-distribution worker."""

    def _init():
        track = plan.track
        if coordinator is not None:
            resolved_coordinator = coordinator
        else:
            start = (
                int(resume_coordinator_state["next_episode_index"])
                if resume_coordinator_state is not None
                else 0
            )
            resolved_coordinator = EpisodeCoordinator(
                base_seed=plan.config.training_seed,
                namespace=track.seed_namespace("training"),
                next_episode_index=start + rank,
                stride=max(1, int(plan.config.n_envs)),
            )
        env = make_track_training_env(
            track,
            base_seed=plan.config.training_seed,
            worker_index=rank,
            coordinator=resolved_coordinator,
            info_level="minimal",
        )
        return NormalizedActionWrapper(env)

    return _init


def default_n_envs() -> int:
    return min(16, max(1, (os.cpu_count() or 4) - 2))


def best_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def unified_config(plan):
    """Return and cross-check the already resolved canonical config."""

    config = plan.config
    if config.algorithm_id not in {"ppo", "sac", "td3"}:
        raise ValueError("SB3 backend requires ppo, sac, or td3")
    if config.track_id != plan.track.id:
        raise ValueError("resolved config track_id does not match Track")
    return config


def build_algo(
    plan,
    env,
    *,
    resume_training_checkpoint=None,
):
    config = unified_config(plan)
    resume_path = config.resume_checkpoint
    if resume_path:
        model = load_algorithm_checkpoint(
            config.algorithm_id,
            resume_path,
            env=env,
            device=config.device,
        )
        replay_path = f"{resume_path}.replay.pkl"
        if (
            get_algorithm_adapter(config.algorithm_id).off_policy
            and os.path.exists(replay_path)
        ):
            model.load_replay_buffer(replay_path)
        state_path = f"{resume_path}.training.ckpt"
        if os.path.exists(state_path):
            checkpoint = (
                resume_training_checkpoint
                or CheckpointManager.load(
                    state_path,
                    expected_config=config,
                    allow_runtime_changes=True,
                )
            )
            restore_rng_state(checkpoint.rng_state)
        return model
    return get_algorithm_adapter(config.algorithm_id).build(
        config,
        env,
        verbose=int(config.algorithm["verbose"]),
    )


def evaluate_training_policy(
    plan,
    model,
    step: int,
    phase: str = "eval",
    *,
    episode_plan=None,
    validation_callback=None,
    checkpoint_id=None,
):
    """Evaluate a live model only for learning curves/checkpoint selection."""

    controller = PolicyController(
        model,
        name=f"SB3-{plan.config.algorithm_id.upper()}",
        action_mode=str(plan.track.policy_contract["action_mode"]),
        control_structure="sb3_policy",
        normalized_actions=True,
    )
    resolved_plan = episode_plan or plan.validation_plan
    evaluation = (
        validation_callback.evaluate(
            controller,
            checkpoint_id=(
                str(checkpoint_id)
                if checkpoint_id is not None
                else f"step-{step}"
            ),
            step=step,
        )
        if validation_callback is not None
        else evaluate_validation_policy(
            controller,
            resolved_plan,
            include_episodes=False,
        )
    )
    aggregate = evaluation["aggregate"]
    return {
        "step": int(step),
        "phase": phase,
        **aggregate,
        "track_id": plan.track.id,
        "track_split": "validation",
        "seed_namespace": evaluation["seed_namespace"],
        "case_values": list(aggregate["case_values"]),
        "case_ids": [
            str(result["case_id"])
            for result in evaluation.get("results") or []
        ],
        "validation_plan_hash": resolved_plan.plan_hash,
        "training_diagnostics": _sb3_training_diagnostics(model),
    }


def _sb3_training_diagnostics(model) -> dict:
    """Read deterministic optimizer diagnostics without changing RNG state."""

    diagnostics = {
        "optimizer_updates": int(getattr(model, "_n_updates", 0)),
        "logger": {},
        "replay": None,
        "critic_q": None,
    }
    try:
        logger = model.logger
    except (AttributeError, RuntimeError):
        logger = None
    values = dict(getattr(logger, "name_to_value", {}) or {})
    for source, target in (
        ("train/actor_loss", "actor_loss"),
        ("train/critic_loss", "critic_loss"),
        ("train/ent_coef", "ent_coef"),
        ("train/ent_coef_loss", "ent_coef_loss"),
        ("train/learning_rate", "learning_rate"),
        ("train/n_updates", "n_updates"),
    ):
        value = values.get(source)
        if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
            float(value)
        ):
            diagnostics["logger"][target] = float(value)

    replay = getattr(model, "replay_buffer", None)
    size_resolver = getattr(replay, "size", None)
    replay_size = int(size_resolver()) if callable(size_resolver) else 0
    if replay is None or replay_size <= 0:
        return diagnostics
    diagnostics["replay"] = {
        "size": replay_size,
        "capacity": int(getattr(replay, "buffer_size", replay_size)),
    }
    observations = getattr(replay, "observations", None)
    actions = getattr(replay, "actions", None)
    critic = getattr(model, "critic", None)
    policy = getattr(model, "policy", None)
    if observations is None or actions is None or not callable(critic):
        return diagnostics
    if policy is None or not callable(getattr(policy, "obs_to_tensor", None)):
        return diagnostics

    sample_count = min(256, replay_size)
    indices = np.linspace(
        0,
        replay_size - 1,
        num=sample_count,
        dtype=np.int64,
    )
    observations = np.asarray(observations[indices])
    actions = np.asarray(actions[indices])
    if observations.ndim >= 3:
        observations = observations[:, 0]
    if actions.ndim >= 3:
        actions = actions[:, 0]
    try:
        import torch

        observation_tensor, _ = policy.obs_to_tensor(observations)
        action_tensor = torch.as_tensor(
            actions,
            device=observation_tensor.device,
            dtype=observation_tensor.dtype,
        )
        with torch.no_grad():
            heads = critic(observation_tensor, action_tensor)
        if not isinstance(heads, (list, tuple)):
            heads = (heads,)
        flattened = torch.cat(
            [head.detach().reshape(-1) for head in heads],
            dim=0,
        ).cpu().numpy()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return diagnostics
    if flattened.size == 0 or not np.all(np.isfinite(flattened)):
        return diagnostics
    diagnostics["critic_q"] = {
        "sample_count": sample_count,
        "head_count": len(heads),
        "mean": float(np.mean(flattened)),
        "std": float(np.std(flattened)),
        "abs_mean": float(np.mean(np.abs(flattened))),
        "abs_max": float(np.max(np.abs(flattened))),
        "min": float(np.min(flattened)),
        "max": float(np.max(flattened)),
    }
    return diagnostics


def save_resumable_training_state(
    plan,
    model,
    checkpoint_path: str | Path,
    *,
    best_validation=None,
    validation_state=None,
    selected_checkpoint=None,
) -> str:
    """Save SB3 state for a declared restart-episode continuation."""

    config = unified_config(plan)
    path = str(checkpoint_path)
    algorithm = get_algorithm_adapter(config.algorithm_id)
    if algorithm.off_policy:
        model.save_replay_buffer(f"{path}.replay.pkl")
    resume_state = _capture_sb3_resume_state(plan, model)
    state = TrainingCheckpoint(
        config=config,
        transition_count=int(getattr(model, "num_timesteps", 0)),
        update_count=int(getattr(model, "_n_updates", 0)),
        algorithm_state={
            "format": "stable-baselines3",
            "checkpoint_path": path,
            "algorithm_class": type(model).__name__,
            "optimizer_updates": int(getattr(model, "_n_updates", 0)),
        },
        replay_state=(
            {"reference": f"{path}.replay.pkl"}
            if algorithm.off_policy
            else None
        ),
        normalization_state=None,
        coordinator_state=resume_state,
        curriculum_state=(
            {"curriculum_id": config.curriculum_id}
            if config.curriculum_id
            else None
        ),
        best_validation=best_validation,
        rng_state=capture_rng_state(),
        validation_state=validation_state,
        selected_checkpoint=selected_checkpoint,
        resume_mode="restart_episode",
        last_committed_episode_index=resume_state.get(
            "last_committed_episode_index"
        ),
        next_episode_index=resume_state.get("next_episode_index"),
        partial_episodes_discarded=int(
            resume_state.get("partial_episodes_discarded", 0)
        ),
        n_envs=int(config.n_envs),
        vector_backend=str(config.algorithm["vector_backend"]),
        code_commit=os.environ.get("GIT_COMMIT"),
    )
    sidecar = f"{path}.training.ckpt"
    CheckpointManager.save(sidecar, state)
    return sidecar


def prepare_resume_training_state(plan):
    """Load and validate a sidecar before constructing vector workers."""

    config = unified_config(plan)
    if not config.resume_checkpoint:
        return None
    state_path = f"{config.resume_checkpoint}.training.ckpt"
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"restart-episode resume requires sidecar: {state_path}"
        )
    checkpoint = CheckpointManager.load(
        state_path,
        expected_config=config,
        allow_runtime_changes=True,
    )
    _validate_sb3_resume_contract(plan, checkpoint)
    checkpoint.require_exact_validation_resume()
    return checkpoint


def _capture_sb3_resume_state(plan, model) -> dict:
    vector_env = model.get_env()
    try:
        worker_states = vector_env.env_method("training_resume_state")
    except Exception as exc:
        raise RuntimeError(
            "training vector backend cannot expose real coordinator state"
        ) from exc
    active = [
        int(state["active_episode_index"])
        for state in worker_states
        if state.get("active_episode_index") is not None
    ]
    completed = [
        int(state["last_completed_episode_index"])
        for state in worker_states
        if state.get("last_completed_episode_index") is not None
    ]
    next_episode_index = (
        max(active) + 1
        if active
        else max(
            int(state["coordinator"]["next_episode_index"])
            for state in worker_states
        )
    )
    track = plan.track
    config = plan.config
    distribution = track.training_distribution()
    return {
        "managed": True,
        "resume_mode": "restart_episode",
        "base_seed": int(config.training_seed),
        "namespace": track.seed_namespace("training"),
        "worker_states": worker_states,
        "last_committed_episode_index": (
            max(completed) if completed else None
        ),
        "next_episode_index": next_episode_index,
        "active_episode_indexes": active,
        "partial_episodes_discarded": len(active),
        "n_envs": int(config.n_envs),
        "vector_backend": str(config.algorithm["vector_backend"]),
        "track_hash": track.track_hash,
        "reward_spec_id": track.reward_spec_id,
        "distribution_hash": distribution.distribution_hash,
        "policy_contract_hash": _mapping_hash(track.policy_contract),
        "algorithm_id": config.algorithm_id,
        "replay_schema": "stable-baselines3-native-v1",
    }


def _validate_sb3_resume_contract(plan, checkpoint) -> None:
    if checkpoint.resume_mode != "restart_episode":
        raise ValueError("SB3 supports only resume_mode='restart_episode'")
    track = plan.track
    expected = {
        "track_hash": track.track_hash,
        "reward_spec_id": track.reward_spec_id,
        "distribution_hash": track.training_distribution().distribution_hash,
        "policy_contract_hash": _mapping_hash(track.policy_contract),
        "algorithm_id": plan.config.algorithm_id,
        "replay_schema": "stable-baselines3-native-v1",
    }
    state = dict(checkpoint.coordinator_state)
    for name, value in expected.items():
        if state.get(name) != value:
            raise ValueError(
                f"resume checkpoint {name} does not match training contract"
            )


def _mapping_hash(value) -> str:
    return stable_json_hash(value)


def make_learning_curve_callback(
    plan,
    best_checkpoint_path: str | Path | None = None,
):
    """Create the periodic complete-validation checkpoint selector."""

    from stable_baselines3.common.callbacks import BaseCallback

    every = int(plan.config.evaluation["every_transitions"])

    class LearningCurveCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.history = []
            self.training_episode_specs = {}
            self._next_eval = max(1, every)
            self.best_metric_value = None
            self.best_step = None
            self.validator = CompleteValidationCallback(
                plan.track,
                base_seeds=plan.config.validation_seeds,
            )
            self.validation_plan = self.validator.plan

        def state_dict(self):
            return {
                "schema_version": "aiogym.sb3_validation_callback.v1",
                "validation_callback": self.validator.state_dict(),
                "learning_curve_history": list(self.history),
                "training_episode_specs": dict(
                    self.training_episode_specs
                ),
                "next_validation_boundary": int(self._next_eval),
                "best_metric_value": self.best_metric_value,
                "best_step": self.best_step,
            }

        def load_state_dict(self, state):
            if not isinstance(state, dict):
                raise TypeError("SB3 validation state must be a mapping")
            payload = dict(state)
            if payload.pop("schema_version", None) != (
                "aiogym.sb3_validation_callback.v1"
            ):
                raise ValueError("unsupported SB3 validation state")
            self.validator.load_state_dict(
                payload.pop("validation_callback", None)
            )
            history = payload.pop("learning_curve_history", None)
            episode_specs = payload.pop("training_episode_specs", None)
            if not isinstance(history, (list, tuple)):
                raise TypeError("SB3 learning curve history is invalid")
            if not isinstance(episode_specs, dict):
                raise TypeError("SB3 episode spec state is invalid")
            boundary = payload.pop("next_validation_boundary", None)
            if (
                isinstance(boundary, bool)
                or not isinstance(boundary, int)
                or boundary <= 0
            ):
                raise ValueError("next validation boundary is invalid")
            self.history = list(history)
            self.training_episode_specs = dict(episode_specs)
            self._next_eval = boundary
            self.best_metric_value = payload.pop(
                "best_metric_value", None
            )
            self.best_step = payload.pop("best_step", None)
            if payload:
                raise ValueError(
                    "SB3 validation state contains unknown fields: "
                    + ", ".join(sorted(payload))
                )

        def evaluate_checkpoint(
            self,
            *,
            step: int,
            checkpoint_id: str,
            phase: str,
        ):
            """Evaluate one unique validation step and persist a new best."""

            resolved_step = int(step)
            existing_index = next(
                (
                    index
                    for index in range(len(self.history) - 1, -1, -1)
                    if int(
                        self.history[index].get("timesteps", -1)
                    )
                    == resolved_step
                ),
                None,
            )
            if any(
                record.step == resolved_step
                for record in self.validator.selector.records
            ):
                if existing_index is None:
                    raise ValueError(
                        "validation selector/history step mismatch"
                    )
                row = dict(self.history[existing_index])
                row["phase"] = str(phase)
                return row

            row = evaluate_training_policy(
                plan,
                self.model,
                resolved_step,
                phase=str(phase),
                episode_plan=self.validation_plan,
                validation_callback=self.validator,
                checkpoint_id=str(checkpoint_id),
            )
            row["timesteps"] = resolved_step
            if existing_index is None:
                self.history.append(row)
            else:
                self.history[existing_index] = row
            improved = (
                self.validator.selector.best is not None
                and self.validator.selector.best.checkpoint_id
                == str(checkpoint_id)
            )
            if not improved:
                return row

            self.best_metric_value = float(row["metric_value"])
            self.best_step = resolved_step
            if best_checkpoint_path is None:
                return row

            self.model.save(str(best_checkpoint_path))
            selected = selected_checkpoint_manifest(
                best_checkpoint_path,
                self.validator.selector.best.__dict__,
            )
            save_resumable_training_state(
                plan,
                self.model,
                best_checkpoint_path,
                best_validation=row,
                validation_state=self.state_dict(),
                selected_checkpoint=selected,
            )
            return row

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", ()):
                episode_hash = info.get("episode_spec_hash")
                if episode_hash is None:
                    continue
                self.training_episode_specs.setdefault(
                    str(episode_hash),
                    {
                        "episode_index": int(info["episode_index"]),
                        "episode_spec_id": str(info["episode_spec_id"]),
                        "episode_spec_hash": str(episode_hash),
                        "distribution_id": str(info["distribution_id"]),
                        "distribution_hash": str(
                            info["distribution_hash"]
                        ),
                    },
                )
            if every <= 0 or self.num_timesteps < self._next_eval:
                return True
            self.evaluate_checkpoint(
                step=self.num_timesteps,
                checkpoint_id=f"step-{self.num_timesteps}",
                phase="eval",
            )
            self._next_eval += max(1, every)
            return True

    return LearningCurveCallback()


def run_sb3(plan) -> BackendResult:
    """Train SB3 and return the selected checkpoint without final artifacts."""

    config = unified_config(plan)
    if config.resume_mode != "restart_episode":
        raise ValueError(
            "SB3 supports only resume_mode='restart_episode'"
        )
    algorithm = dict(config.algorithm)
    replay = dict(config.replay)
    gamma = float(algorithm["gamma"])
    if not 0.0 < gamma <= 1.0:
        raise ValueError("algorithm.gamma must be in (0, 1]")
    if config.utd_ratio < 0.0:
        raise ValueError("algorithm.utd_ratio must be non-negative")

    try:
        import torch
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "stable-baselines3 and torch are required for training; install "
            "AIO-Gym with `pip install 'aiogym[rl]'`."
        ) from exc

    torch.set_num_threads(max(1, int(algorithm["torch_threads"])))
    resume_checkpoint = prepare_resume_training_state(plan)
    resume_coordinator_state = (
        None
        if resume_checkpoint is None
        else resume_checkpoint.coordinator_state
    )
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    vector_backend = str(algorithm["vector_backend"])
    env_fns = [
        make_training_env(
            plan,
            rank=rank,
            resume_coordinator_state=resume_coordinator_state,
        )
        for rank in range(config.n_envs)
    ]
    env = None
    try:
        env = (
            SubprocVecEnv(
                env_fns,
                start_method=str(algorithm["subproc_start_method"]),
            )
            if vector_backend == "subproc"
            else DummyVecEnv(env_fns)
        )
        env.seed(config.training_seed)
        model = build_algo(
            plan,
            env,
            resume_training_checkpoint=resume_checkpoint,
        )
        starting_step = int(getattr(model, "num_timesteps", 0))
        curve_callback = make_learning_curve_callback(
            plan,
            plan.policy_path,
        )
        selected_checkpoint = None
        if resume_checkpoint is not None:
            curve_callback.load_state_dict(
                resume_checkpoint.validation_state
            )
            if resume_checkpoint.selected_checkpoint is not None:
                selected_checkpoint = _materialize_selected_checkpoint(
                    resume_checkpoint.selected_checkpoint,
                    plan.policy_path,
                )
        if config.resume_checkpoint:
            initial = curve_callback.evaluate_checkpoint(
                step=starting_step,
                checkpoint_id=f"resume-step-{starting_step}",
                phase="resume",
            )
        else:
            initial = evaluate_training_policy(
                plan,
                model,
                starting_step,
                phase="initial",
                episode_plan=curve_callback.validation_plan,
            )
        initial["timesteps"] = starting_step
        if not curve_callback.history or int(
            curve_callback.history[-1].get("timesteps", -1)
        ) != starting_step:
            curve_callback.history.append(initial)

        started_at = time.monotonic()
        remaining_steps = max(
            0,
            int(config.total_transitions) - starting_step,
        )
        if remaining_steps:
            model.learn(
                total_timesteps=remaining_steps,
                progress_bar=False,
                callback=curve_callback,
                reset_num_timesteps=not bool(config.resume_checkpoint),
            )
        train_seconds = time.monotonic() - started_at
        final_step = int(
            getattr(model, "num_timesteps", config.total_transitions)
        )
        collected_transitions = max(0, final_step - starting_step)

        final_curve_point = curve_callback.evaluate_checkpoint(
            step=final_step,
            checkpoint_id=f"final-step-{final_step}",
            phase="final",
        )

        final_checkpoint_path, checkpoint_selection = _save_sb3_checkpoints(
            plan, model, curve_callback, selected_checkpoint
        )
        return _finalize_sb3_result(
            plan=plan,
            config=config,
            algorithm=algorithm,
            replay=replay,
            model=model,
            curve_callback=curve_callback,
            final_curve_point=final_curve_point,
            final_checkpoint_path=final_checkpoint_path,
            final_step=final_step,
            starting_step=starting_step,
            train_seconds=train_seconds,
            collected_transitions=collected_transitions,
            checkpoint_selection=checkpoint_selection,
            vector_backend=vector_backend,
            gamma=gamma,
        )
    finally:
        if env is not None:
            env.close()


def _save_sb3_checkpoints(plan, model, curve_callback, selected_checkpoint):
        final_checkpoint_path = plan.output_dir / f"{plan.run_name}.final.zip"
        model.save(str(final_checkpoint_path))
        best_validation = (
            {
                "step": curve_callback.best_step,
                "metric_value": curve_callback.best_metric_value,
            }
            if curve_callback.best_step is not None
            else None
        )
        save_resumable_training_state(
            plan,
            model,
            final_checkpoint_path,
            best_validation=best_validation,
            validation_state=curve_callback.state_dict(),
            selected_checkpoint=(
                selected_checkpoint_manifest(
                    plan.policy_path,
                    curve_callback.validator.selector.best.__dict__,
                )
                if curve_callback.validator.selector.best is not None
                and plan.policy_path.is_file()
                else selected_checkpoint
            ),
        )
        if curve_callback.best_step is None:
            model.save(str(plan.policy_path))
            save_resumable_training_state(
                plan,
                model,
                plan.policy_path,
                best_validation=best_validation,
                validation_state=curve_callback.state_dict(),
                selected_checkpoint=None,
            )
            checkpoint_selection = "final"
        else:
            checkpoint_selection = "best-validation"
        return final_checkpoint_path, checkpoint_selection


def _finalize_sb3_result(
    *,
    plan,
    config,
    algorithm,
    replay,
    model,
    curve_callback,
    final_curve_point,
    final_checkpoint_path,
    final_step,
    starting_step,
    train_seconds,
    collected_transitions,
    checkpoint_selection,
    vector_backend,
    gamma,
):
        learning_curve = list(curve_callback.history)
        if (
            learning_curve
            and int(learning_curve[-1].get("timesteps", -1))
            == final_step
        ):
            learning_curve[-1] = final_curve_point
        else:
            learning_curve.append(final_curve_point)
        train_freq, gradient_steps = sb3_update_schedule(
            utd_ratio=(
                0.0 if config.algorithm_id == "ppo" else config.utd_ratio
            ),
            n_envs=config.n_envs,
            vector_steps=int(
                algorithm.get("rollout_vector_steps", 1)
            ),
        )
        episode_specs = sorted(
            curve_callback.training_episode_specs.values(),
            key=lambda row: (
                row["episode_index"],
                row["episode_spec_hash"],
            ),
        )
        runtime = {
            "seconds": train_seconds,
            "collected_transitions": collected_transitions,
            "steps_per_second": (
                collected_transitions / train_seconds
                if train_seconds > 0 and collected_transitions
                else None
            ),
            "steps_per_second_per_env": (
                collected_transitions / train_seconds / config.n_envs
                if train_seconds > 0 and collected_transitions
                else None
            ),
        }
        metadata = {
            "vector_backend": vector_backend,
            "subproc_start_method": (
                str(algorithm["subproc_start_method"])
                if vector_backend == "subproc"
                else None
            ),
            "torch_threads": int(algorithm["torch_threads"]),
            "gamma": gamma,
            "learning_rate": float(algorithm["learning_rate"]),
            "batch_size": int(algorithm["batch_size"]),
            "replay_capacity": int(replay.get("capacity", 0)),
            "learning_starts": int(
                replay.get("learning_starts", 0)
            ),
            "train_freq": int(train_freq),
            "gradient_steps": int(gradient_steps),
            "utd_ratio": float(config.utd_ratio),
            "effective_algorithm_kwargs": _effective_algorithm_kwargs(
                config,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                action_shape=tuple(
                    getattr(getattr(model, "action_space", None), "shape", ())
                    or ()
                ),
            ),
            "learning_curve_every": int(
                config.evaluation["every_transitions"]
            ),
            "starting_step": starting_step,
            "final_checkpoint_path": str(final_checkpoint_path),
            "best_checkpoint_path": (
                str(plan.policy_path)
                if curve_callback.best_step is not None
                else None
            ),
            "best_step": curve_callback.best_step,
            "best_metric_value": curve_callback.best_metric_value,
            "environment_transitions": final_step,
            "optimizer_updates": int(getattr(model, "_n_updates", 0)),
            "offline_samples_available": 0,
            "offline_samples_drawn": 0,
            "training_episode_specs": episode_specs,
            "unique_training_episode_specs": len(episode_specs),
        }
        return BackendResult(
            algorithm_id=config.algorithm_id,
            policy_path=plan.policy_path,
            final_step=final_step,
            checkpoint_selection=checkpoint_selection,
            learning_curve=tuple(learning_curve),
            training_metadata=metadata,
            runtime=runtime,
        )


def _effective_algorithm_kwargs(
    config,
    *,
    train_freq,
    gradient_steps,
    action_shape,
):
    """Return the JSON form of the kwargs applied by the SB3 adapter."""

    algorithm = config.as_dict()["algorithm"]
    common = {
        "policy": algorithm["policy"],
        "policy_kwargs": algorithm["policy_kwargs"],
        "seed": config.training_seed,
        "device": config.device,
        "learning_rate": algorithm["learning_rate"],
        "gamma": algorithm["gamma"],
        "tensorboard_log": algorithm["tensorboard_log"],
        "verbose": algorithm["verbose"],
    }
    if config.algorithm_id == "ppo":
        return {
            **common,
            "n_steps": algorithm["n_steps"],
            "batch_size": algorithm["batch_size"],
            "gae_lambda": algorithm["gae_lambda"],
            "clip_range": algorithm["clip_range"],
            "n_epochs": algorithm["n_epochs"],
            "ent_coef": algorithm["ent_coef"],
            "vf_coef": algorithm["vf_coef"],
            "max_grad_norm": algorithm["max_grad_norm"],
        }
    effective = {
        **common,
        "batch_size": algorithm["batch_size"],
        "train_freq": int(train_freq),
        "gradient_steps": int(gradient_steps),
        "buffer_size": config.replay["capacity"],
        "learning_starts": config.replay["learning_starts"],
        "tau": algorithm["tau"],
    }
    if config.algorithm_id == "sac":
        effective.update(
            ent_coef=algorithm["ent_coef"],
            target_entropy=algorithm["target_entropy"],
        )
    else:
        effective.update(
            action_noise=algorithm["action_noise"],
            action_noise_sigma=algorithm["action_noise_sigma"],
            action_noise_shape=[int(size) for size in action_shape],
            policy_delay=algorithm["policy_delay"],
            target_policy_noise=algorithm["target_policy_noise"],
            target_noise_clip=algorithm["target_noise_clip"],
        )
    return effective


def _materialize_selected_checkpoint(manifest, destination: Path):
    """Copy a verified historical best into the current run namespace."""

    source = validate_selected_checkpoint(manifest)
    destination = Path(destination).resolve()
    if source.resolve() == destination:
        return dict(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    for suffix in (".replay.pkl",):
        companion = Path(f"{source}{suffix}")
        if companion.is_file():
            shutil.copy2(companion, Path(f"{destination}{suffix}"))
    updated = selected_checkpoint_manifest(
        destination,
        manifest["selection_record"],
    )
    sidecar = Path(f"{source}.training.ckpt")
    if sidecar.is_file():
        checkpoint = CheckpointManager.load(sidecar)
        CheckpointManager.save(
            f"{destination}.training.ckpt",
            replace(checkpoint, selected_checkpoint=updated),
        )
    return updated


__all__ = [
    "best_device",
    "build_algo",
    "default_n_envs",
    "evaluate_training_policy",
    "make_learning_curve_callback",
    "make_training_env",
    "prepare_resume_training_state",
    "run_sb3",
    "save_resumable_training_state",
    "unified_config",
]
