#!/usr/bin/env python3
"""Train, export, and evaluate Stable-Baselines3 baselines for AIO-Gym.

Rollout parallelism uses SubprocVecEnv by default: one plant per worker process.
For these small MLP policies, CPU is the default on Apple machines because MPS
overhead is usually larger than the network compute. CUDA is still used when
available.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import hashlib
import json
import math
import os
import time

from aiogym._internal.config import parse_seed_list
from aiogym._internal.paths import run_path
from aiogym.controllers import make_controller
from aiogym.env_factory import make_env
from aiogym.evaluation import (
    evaluate_controller,
    rollout_controller,
)
from aiogym.rl.training_config import (
    configure_training_auto_events,
    configure_training_case,
    configure_training_track,
    training_identity,
)
from aiogym.rl.training_artifacts import (
    learning_curve_point,
    result_row,
    rl_payload,
    utc_run_id,
    write_rl_artifacts,
)
from aiogym.rl.algorithm_registry import (
    get_algorithm_adapter,
    load_algorithm_checkpoint,
)
from aiogym.rl.checkpoints import (
    CheckpointManager,
    TrainingCheckpoint,
    capture_rng_state,
    restore_rng_state,
)
from aiogym.rl.coordinator import EpisodeCoordinator
from aiogym.rl.episode_env import make_track_training_env
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.statistics import NormalizedActionWrapper
from aiogym.rl.utd import sb3_update_schedule
from aiogym.rl.validation import (
    CompleteValidationCallback,
    ValidationEpisodePlan,
    evaluate_validation_policy,
)


def make_training_env(
    args,
    rank: int = 0,
    *,
    coordinator: EpisodeCoordinator | None = None,
):
    def _init():
        track = getattr(args, "track_spec", None)
        if track is not None:
            if coordinator is not None:
                resolved_coordinator = coordinator
            else:
                resume_state = getattr(
                    args,
                    "_resume_coordinator_state",
                    None,
                )
                start = (
                    int(resume_state["next_episode_index"])
                    if resume_state is not None
                    else 0
                )
                resolved_coordinator = EpisodeCoordinator(
                    base_seed=args.seed,
                    namespace=track.seed_namespace("training"),
                    next_episode_index=start + rank,
                    stride=max(1, int(getattr(args, "n_envs", 1))),
                )
            env = make_track_training_env(
                track,
                base_seed=args.seed,
                worker_index=rank,
                coordinator=resolved_coordinator,
                info_level="minimal",
            )
            return NormalizedActionWrapper(env)
        env = make_env(
            config={
                "scenario": args.scenario,
                "case": getattr(args, "case", None),
                "reward_spec": args.resolved_reward_spec_id,
                "info_level": "minimal",
                "environment": {
                    "action_mode": args.action_mode,
                    "control_dt": args.control_dt,
                    "episode_steps": args.train_episode_steps,
                    "auto_events": args.auto_events,
                    "randomize": args.randomize,
                    "randomize_setpoints": args.randomize_setpoints,
                    "randomize_plant": args.randomize_plant,
                    "plant_drift": args.plant_drift,
                    "integral_obs": args.integral_obs,
                    "disturbance_obs": args.disturbance_obs,
                    "previous_action_obs": args.previous_action_obs,
                    "normalize_observations": args.normalize_observations,
                    "tracking_error_obs": args.tracking_error_obs,
                    "terminate_on_runaway": args.terminate_on_runaway,
                    "noise": args.noise,
                    "noise_pct": args.noise_pct,
                },
            }
        )
        return NormalizedActionWrapper(env)

    return _init


def default_n_envs():
    return min(16, max(1, (os.cpu_count() or 4) - 2))


def _make_custom_evaluation_env(args):
    return make_env(
        config={
            "scenario": args.scenario,
            "case": getattr(args, "case", None),
            "reward_spec": args.resolved_reward_spec_id,
            "environment": {
                "action_mode": args.action_mode,
                "episode_steps": args.eval_episode_steps,
                "control_dt": args.control_dt,
            },
        }
    )


def best_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_algo(args, env):
    config = unified_config(args)
    if args.resume:
        model = load_algorithm_checkpoint(
            args.algo,
            args.resume,
            env=env,
            device=args.device,
        )
        replay_path = f"{args.resume}.replay.pkl"
        if (
            get_algorithm_adapter(args.algo).off_policy
            and os.path.exists(replay_path)
        ):
            model.load_replay_buffer(replay_path)
        state_path = f"{args.resume}.training.ckpt"
        if os.path.exists(state_path):
            checkpoint = getattr(
                args,
                "_resume_training_checkpoint",
                None,
            ) or CheckpointManager.load(
                state_path,
                expected_config=config,
                allow_runtime_changes=True,
            )
            restore_rng_state(checkpoint.rng_state)
        return model
    return get_algorithm_adapter(args.algo).build(
        config,
        env,
        verbose=args.verbose,
    )


def unified_config(args) -> RLTrainingConfig:
    algorithm = {
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "batch_size": args.batch_size,
        "tensorboard_log": args.tensorboard_log,
        "utd_ratio": 0.0 if args.algo == "ppo" else args.utd_ratio,
        "rollout_vector_steps": args.train_freq,
        "n_steps": args.ppo_n_steps,
    }
    return RLTrainingConfig(
        track_id=args.track or training_identity(args),
        algorithm_id=args.algo,
        training_seed=args.seed,
        total_transitions=args.steps,
        n_envs=args.n_envs,
        device=args.device,
        algorithm=algorithm,
        replay={
            "capacity": args.buffer_size,
            "learning_starts": args.learning_starts,
        },
        evaluation={
            "every_transitions": args.learning_curve_every,
            "episodes": args.learning_curve_episodes,
        },
        checkpointing={},
        curriculum_id=getattr(args, "curriculum_id", None),
    )


def _evaluate_validation_checkpoint(args, checkpoint_path: str):
    if getattr(args, "track_spec", None) is not None:
        controller = make_controller(
            "sb3",
            scenario=args.scenario,
            config={
                "path": checkpoint_path,
                "algo": args.algo,
                "action_mode": args.action_mode,
                "normalized_actions": True,
            },
        )
        seeds = parse_seed_list(
            args.eval_seed_list,
            args.eval_seed,
            args.eval_episodes,
            option="--eval-seed-list",
        )
        validation_plan = ValidationEpisodePlan(
            args.track_spec,
            base_seeds=seeds,
        )
        evaluation = evaluate_validation_policy(
            controller,
            validation_plan,
            include_episodes=True,
        )
        rollouts = []
        if args.save_rollout:
            for case in args.track_spec.resolved_cases("validation"):
                env = make_env(
                    args.scenario,
                    case=case.profile,
                    reward_spec=args.reward_spec,
                )
                try:
                    rollout = rollout_controller(
                        controller,
                        env,
                        seed=seeds[0],
                        max_steps=args.rollout_steps,
                    )
                finally:
                    env.close()
                rollout.update(
                    {
                        "track_id": args.track,
                        "track_split": "validation",
                        "case_id": case.case_id,
                        "resolved_case_hash": case.resolved_case_hash,
                    }
                )
                rollouts.append(rollout)
        return args.track_spec, evaluation, rollouts

    controller = make_controller(
        "sb3",
        scenario=args.scenario,
        config={
            "path": checkpoint_path,
            "algo": args.algo,
            "action_mode": args.action_mode,
            "normalized_actions": True,
        },
    )
    seeds = parse_seed_list(
        args.eval_seed_list,
        args.eval_seed,
        args.eval_episodes,
        option="--eval-seed-list",
    )
    evaluation_env = _make_custom_evaluation_env(args)
    result = evaluate_controller(
        controller,
        evaluation_env,
        episodes=len(seeds),
        seed=seeds[0],
        seed_list=seeds,
        goal_specification=args.goal,
        include_episodes=True,
    )
    rollout = None
    if args.save_rollout:
        rollout_env = _make_custom_evaluation_env(args)
        rollout = rollout_controller(
            controller,
            rollout_env,
            seed=seeds[0],
            max_steps=args.rollout_steps,
        )
        rollout_env.close()
    evaluation = {
        "split": "validation",
        "case_count": 1,
        "results": [result],
        "aggregate": {
            "metric": result["metric"],
            "metric_direction": result["metric_direction"],
            "metric_value": result[result["metric"]],
            "case_count": 1,
            "ranking_eligible": result.get("ranking_eligible", True),
        },
    }
    return None, evaluation, [rollout] if rollout is not None else []


def evaluate_training_policy(
    args,
    model,
    step: int,
    phase: str = "eval",
    *,
    episode_plan=None,
    validation_callback=None,
    checkpoint_id=None,
):
    if getattr(args, "track_spec", None) is not None:
        controller = make_controller(
            "sb3",
            scenario=args.scenario,
            policy=model,
            config={
                "algo": args.algo,
                "action_mode": args.action_mode,
                "name": f"SB3-{args.algo.upper()}",
                "normalized_actions": True,
            },
        )
        seeds = parse_seed_list(
            args.eval_seed_list,
            args.eval_seed,
            args.learning_curve_episodes,
            option="--eval-seed-list",
        )
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
                episode_plan,
                include_episodes=False,
            )
        )
        aggregate = evaluation["aggregate"]
        row = {
            "step": int(step),
            "phase": phase,
            **aggregate,
            "track_id": args.track,
            "track_split": "validation",
            "seed_namespace": evaluation["seed_namespace"],
            "case_values": list(aggregate["case_values"]),
            "validation_plan_hash": (
                episode_plan.plan_hash
                if episode_plan is not None
                else None
            ),
        }
        return row

    controller = make_controller(
        "sb3",
        scenario=args.scenario,
        policy=model,
        config={
            "algo": args.algo,
            "action_mode": args.action_mode,
            "name": f"SB3-{args.algo.upper()}",
            "normalized_actions": True,
        },
    )
    seeds = parse_seed_list(
        args.eval_seed_list,
        args.eval_seed,
        args.learning_curve_episodes,
        option="--eval-seed-list",
    )
    env = _make_custom_evaluation_env(args)
    try:
        result = evaluate_controller(
            controller,
            env,
            episodes=len(seeds),
            seed=seeds[0],
            seed_list=seeds,
            goal_specification=args.goal,
            include_episodes=False,
        )
    finally:
        if hasattr(env, "close"):
            env.close()
    return learning_curve_point(step, result, phase=phase)


def training_metadata(
    args,
    checkpoint_path: str,
    *,
    final_checkpoint_path: str | None = None,
    best_checkpoint_path: str | None = None,
    best_step: int | None = None,
    best_metric_value: float | None = None,
    checkpoint_selection: str = "final",
):
    metadata = {
        "algo": args.algo,
        "scenario": args.scenario,
        "case": args.case,
        "action_mode": args.action_mode,
        "goal": args.goal,
        "reward_spec_id": args.resolved_reward_spec_id,
        "total_timesteps": args.steps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "vec_env": args.vec_env,
        "subproc_start_method": args.subproc_start_method if args.vec_env == "subproc" else None,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "train_episode_steps": args.train_episode_steps,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "utd_ratio": args.utd_ratio,
        "training_config_hash": unified_config(args).config_hash,
        "learning_curve_every": args.learning_curve_every,
        "learning_curve_episodes": args.learning_curve_episodes,
        "save_rollout": args.save_rollout,
        "disturbance_obs": args.disturbance_obs,
        "previous_action_obs": args.previous_action_obs,
        "normalize_observations": args.normalize_observations,
        "tracking_error_obs": args.tracking_error_obs,
        "env_kwargs": {
            "auto_events": args.auto_events,
            "randomize": args.randomize,
            "randomize_setpoints": args.randomize_setpoints,
            "randomize_plant": args.randomize_plant,
            "plant_drift": args.plant_drift,
            "integral_obs": args.integral_obs,
            "disturbance_obs": args.disturbance_obs,
            "previous_action_obs": args.previous_action_obs,
            "normalize_observations": args.normalize_observations,
            "tracking_error_obs": args.tracking_error_obs,
            "terminate_on_runaway": args.terminate_on_runaway,
            "noise": args.noise,
            "noise_pct": args.noise_pct,
            "control_dt": args.control_dt,
            "case": args.case,
        },
        "checkpoint_path": checkpoint_path,
        "checkpoint_selection": checkpoint_selection,
    }
    track = getattr(args, "track_spec", None)
    metadata.update(
        {
            "track_id": track.id if track is not None else None,
            "track_hash": track.track_hash if track is not None else None,
            "goal": getattr(args, "goal", None),
            "reward_spec_id": getattr(
                args,
                "resolved_reward_spec_id",
                None,
            ),
            "policy_scope": getattr(
                args,
                "policy_scope",
                "specialist",
            ),
            "training_seed_namespace": getattr(
                args,
                "training_seed_namespace",
                None,
            ),
            "validation_seed_namespace": getattr(
                args,
                "validation_seed_namespace",
                None,
            ),
            "training_seed_namespace_hash": getattr(
                args,
                "training_seed_namespace_hash",
                None,
            ),
            "validation_seed_namespace_hash": getattr(
                args,
                "validation_seed_namespace_hash",
                None,
            ),
        }
    )
    if final_checkpoint_path is not None:
        metadata["final_checkpoint_path"] = final_checkpoint_path
    if best_checkpoint_path is not None:
        metadata["best_checkpoint_path"] = best_checkpoint_path
    if best_step is not None:
        metadata["best_step"] = int(best_step)
    if best_metric_value is not None:
        metadata["best_metric_value"] = float(best_metric_value)
    return metadata


def save_resumable_training_state(
    args,
    model,
    checkpoint_path: str,
    *,
    best_validation=None,
) -> str:
    """Save SB3 state for a declared restart-episode continuation."""

    adapter = get_algorithm_adapter(args.algo)
    if adapter.off_policy:
        model.save_replay_buffer(f"{checkpoint_path}.replay.pkl")
    config = unified_config(args)
    resume_state = _capture_sb3_resume_state(args, model)
    state = TrainingCheckpoint(
        config=config,
        transition_count=int(getattr(model, "num_timesteps", 0)),
        update_count=int(getattr(model, "_n_updates", 0)),
        algorithm_state={
            "format": "stable-baselines3",
            "checkpoint_path": checkpoint_path,
            "algorithm_class": type(model).__name__,
            "optimizer_updates": int(getattr(model, "_n_updates", 0)),
        },
        replay_state=(
            {"reference": f"{checkpoint_path}.replay.pkl"}
            if adapter.off_policy
            else None
        ),
        normalization_state=None,
        coordinator_state=resume_state,
        curriculum_state=(
            {"curriculum_id": args.curriculum_id}
            if getattr(args, "curriculum_id", None)
            else None
        ),
        best_validation=best_validation,
        rng_state=capture_rng_state(),
        resume_mode="restart_episode",
        last_committed_episode_index=resume_state.get(
            "last_committed_episode_index"
        ),
        next_episode_index=resume_state.get("next_episode_index"),
        partial_episodes_discarded=int(
            resume_state.get("partial_episodes_discarded", 0)
        ),
        n_envs=int(args.n_envs),
        vector_backend=str(args.vec_env),
        code_commit=os.environ.get("GIT_COMMIT"),
    )
    path = f"{checkpoint_path}.training.ckpt"
    CheckpointManager.save(path, state)
    return path


def prepare_resume_training_state(args):
    """Load and validate the sidecar before constructing training envs."""

    if not args.resume:
        args._resume_training_checkpoint = None
        args._resume_coordinator_state = None
        return None
    state_path = f"{args.resume}.training.ckpt"
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"restart-episode resume requires sidecar: {state_path}"
        )
    checkpoint = CheckpointManager.load(
        state_path,
        expected_config=unified_config(args),
        allow_runtime_changes=True,
    )
    _validate_sb3_resume_contract(args, checkpoint)
    args._resume_training_checkpoint = checkpoint
    args._resume_coordinator_state = checkpoint.coordinator_state
    return checkpoint


def _capture_sb3_resume_state(args, model) -> dict:
    if getattr(args, "track_spec", None) is None:
        return {
            "managed": False,
            "resume_mode": "restart_episode",
            "next_episode_index": None,
            "last_committed_episode_index": None,
            "partial_episodes_discarded": 0,
            "n_envs": int(args.n_envs),
            "vector_backend": str(args.vec_env),
        }
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
    distribution = args.track_spec.training_distribution()
    return {
        "managed": True,
        "resume_mode": "restart_episode",
        "base_seed": int(args.seed),
        "namespace": args.training_seed_namespace,
        "worker_states": worker_states,
        "last_committed_episode_index": (
            max(completed) if completed else None
        ),
        "next_episode_index": next_episode_index,
        "active_episode_indexes": active,
        "partial_episodes_discarded": len(active),
        "n_envs": int(args.n_envs),
        "vector_backend": str(args.vec_env),
        "track_hash": args.track_spec.track_hash,
        "reward_spec_id": args.resolved_reward_spec_id,
        "distribution_hash": distribution.distribution_hash,
        "policy_contract_hash": _mapping_hash(
            args.track_spec.policy_contract
        ),
        "algorithm_id": args.algo,
        "replay_schema": "stable-baselines3-native-v1",
    }


def _validate_sb3_resume_contract(args, checkpoint) -> None:
    if checkpoint.resume_mode != "restart_episode":
        raise ValueError(
            "SB3 supports only resume_mode='restart_episode'"
        )
    state = dict(checkpoint.coordinator_state)
    if getattr(args, "track_spec", None) is None:
        return
    expected = {
        "track_hash": args.track_spec.track_hash,
        "reward_spec_id": args.resolved_reward_spec_id,
        "distribution_hash": (
            args.track_spec.training_distribution().distribution_hash
        ),
        "policy_contract_hash": _mapping_hash(
            args.track_spec.policy_contract
        ),
        "algorithm_id": args.algo,
        "replay_schema": "stable-baselines3-native-v1",
    }
    for name, value in expected.items():
        if state.get(name) != value:
            raise ValueError(
                f"resume checkpoint {name} does not match training contract"
            )


def _mapping_hash(value) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def artifact_dir_for(args, run_name: str) -> str:
    return args.artifact_dir or os.path.join(args.out_dir, f"{run_name}_artifacts")


def run_name_for(args, run_id: str | None = None) -> str:
    if args.name:
        return args.name
    stem = (
        f"{args.algo}_{training_identity(args)}_"
        f"training-seed{args.seed}"
    )
    return f"{stem}_{run_id or utc_run_id()}"


def _custom_learning_curve_point_is_better(row, best_metric_value):
    if not bool(row.get("ranking_eligible", True)):
        return False
    value = float(row["metric_value"])
    if not math.isfinite(value):
        return False
    if best_metric_value is None:
        return True
    if row.get("metric_direction") == "maximize":
        return value > best_metric_value
    return value < best_metric_value


def make_learning_curve_callback(args, best_checkpoint_path: str | None = None):
    from stable_baselines3.common.callbacks import BaseCallback

    class LearningCurveCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.history = []
            self._next_eval = max(1, int(args.learning_curve_every))
            self.best_metric_value = None
            self.best_step = None
            self.validator = (
                CompleteValidationCallback(
                    args.track_spec,
                    base_seeds=parse_seed_list(
                        args.eval_seed_list,
                        args.eval_seed,
                        args.learning_curve_episodes,
                        option="--eval-seed-list",
                    ),
                )
                if getattr(args, "track_spec", None) is not None
                else None
            )
            self.validation_plan = (
                self.validator.plan
                if self.validator is not None
                else None
            )

        def _on_step(self) -> bool:
            if args.learning_curve_every <= 0 or self.num_timesteps < self._next_eval:
                return True
            row = evaluate_training_policy(
                args,
                self.model,
                self.num_timesteps,
                episode_plan=self.validation_plan,
                validation_callback=self.validator,
                checkpoint_id=f"step-{self.num_timesteps}",
            )
            row["timesteps"] = self.num_timesteps
            self.history.append(row)
            improved = (
                self.validator is not None
                and self.validator.selector.best is not None
                and self.validator.selector.best.checkpoint_id
                == f"step-{self.num_timesteps}"
            )
            if self.validator is None:
                improved = _custom_learning_curve_point_is_better(
                    row,
                    self.best_metric_value,
                )
            if best_checkpoint_path is not None and improved:
                self.best_metric_value = float(row["metric_value"])
                self.best_step = int(self.num_timesteps)
                self.model.save(best_checkpoint_path)
                save_resumable_training_state(
                    args,
                    self.model,
                    f"{best_checkpoint_path}.zip",
                    best_validation=row,
                )
            self._next_eval += max(1, int(args.learning_curve_every))
            return True

    return LearningCurveCallback()


def export_onnx(model, obs_dim: int, path: str):
    import torch

    class DeterministicPolicy(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            return self.policy._predict(obs, deterministic=True)

    policy = model.policy
    policy.eval()
    device = next(policy.parameters()).device
    dummy = torch.zeros(1, obs_dim, device=device)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    export_options = {
        "input_names": ["obs"],
        "output_names": ["action"],
        "opset_version": 17,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_options["dynamo"] = False
    torch.onnx.export(
        DeterministicPolicy(policy),
        dummy,
        path,
        **export_options,
    )


def require_onnx_export_dependencies():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as ex:
        raise SystemExit(
            "torch is required for ONNX export; install AIO-Gym "
            "with `pip install 'aiogym[rl]'`."
        ) from ex
    if importlib.util.find_spec("onnx") is None:
        raise SystemExit(
            "onnx is required for ONNX export; install AIO-Gym "
            "with `pip install 'aiogym[rl]'`."
        )


def _run_backend(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument(
        "--track",
        default=None,
        help=(
            "official benchmark track ID; when no custom selectors are "
            "provided, the default official track is used"
        ),
    )
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--case", default=None, help="custom specialist Case v2 ID")
    ap.add_argument("--goal", choices=["regulation", "economic"], default=None)
    ap.add_argument("--reward-spec", default=None)
    ap.add_argument(
        "--policy-scope",
        choices=["generalist", "specialist"],
        default=None,
    )
    ap.add_argument("--algo", default="sac", choices=["sac", "ppo", "td3"])
    ap.add_argument("--action-mode", default=None, choices=["actuator", "setpoint"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=default_n_envs())
    ap.add_argument("--vec-env", default="subproc", choices=["subproc", "dummy"],
                    help="parallel rollout backend; subproc gives one process per env")
    ap.add_argument("--subproc-start-method", default="fork", choices=["fork", "forkserver", "spawn"],
                    help="multiprocessing start method for SubprocVecEnv")
    ap.add_argument("--train-episode-steps", type=int, default=None,
                    help="override case episode length; case/default owns it when omitted")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--control-dt", type=float, default=None,
                    help="override case control interval; case/default owns it when omitted")
    ap.add_argument(
        "--auto-events",
        action="store_true",
        default=None,
        help="enable generic automatically generated within-episode events",
    )
    ap.add_argument(
        "--randomize",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument(
        "--randomize-setpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument(
        "--randomize-plant",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument(
        "--plant-drift",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument(
        "--integral-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument(
        "--disturbance-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include current disturbances in the policy observation; case-owned when omitted",
    )
    ap.add_argument(
        "--previous-action-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include the previous applied action; case-owned when omitted",
    )
    ap.add_argument(
        "--normalize-observations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="scale observations with fixed physical bounds; case-owned when omitted",
    )
    ap.add_argument(
        "--tracking-error-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replace setpoints with normalized tracking errors; case-owned when omitted",
    )
    ap.add_argument(
        "--terminate-on-runaway",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument(
        "--noise",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    ap.add_argument("--noise-pct", type=float, default=None)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor; use a value near 1 for long-horizon tracking",
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--buffer-size", type=int, default=300000)
    ap.add_argument("--learning-starts", type=int, default=100)
    ap.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="vector-environment rollout steps between optimizer phases",
    )
    ap.add_argument(
        "--utd-ratio",
        type=float,
        default=1.0,
        help="optimizer updates per collected transition (SAC/TD3)",
    )
    ap.add_argument(
        "--gradient-steps",
        type=int,
        default=None,
        help="deprecated compatibility field; derived from UTD and n-envs",
    )
    ap.add_argument("--ppo-n-steps", type=int, default=2048)
    ap.add_argument("--device", default=None,
                    help="SB3 policy device; defaults to CUDA if available, otherwise CPU. Use 'mps' explicitly if desired.")
    ap.add_argument("--torch-threads", type=int, default=2,
                    help="torch intra-op threads; keep low so SubprocVecEnv workers get CPU time")
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--tensorboard-log", default=None)
    ap.add_argument("--out-dir", default=str(run_path("rl", "sb3")))
    ap.add_argument("--name", default=None, help="stable run name; defaults to a timestamped name")
    ap.add_argument("--artifact-dir", default=None,
                    help="standard benchmark artifact directory; defaults to <out-dir>/<name>_artifacts")
    ap.add_argument("--eval-episodes", type=int, default=1)
    ap.add_argument("--eval-episode-steps", type=int, default=None,
                    help="override evaluation episode length; case owns it when omitted")
    ap.add_argument("--eval-seed", type=int, default=9000)
    ap.add_argument("--eval-seed-list", default=None)
    ap.add_argument("--learning-curve-every", type=int, default=10000,
                    help="evaluate every N timesteps; 0 records only initial/final points (default: 10000)")
    ap.add_argument("--learning-curve-episodes", type=int, default=1)
    ap.add_argument(
        "--save-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save a final tracking rollout for report plots (default: enabled)",
    )
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument("--onnx", action="store_true", help="export deterministic policy to ONNX after training")
    ap.add_argument("--onnx-path", default=None, help="optional ONNX export path; defaults to checkpoint basename + .onnx")
    ap.add_argument("--resume", default=None, help="resume from an SB3 checkpoint")
    args = ap.parse_args(argv)
    if not 0.0 < args.gamma <= 1.0:
        ap.error("--gamma must be in (0, 1]")
    if args.utd_ratio < 0.0:
        ap.error("--utd-ratio must be non-negative")
    if args.gradient_steps is not None:
        ap.error("--gradient-steps is replaced by explicit --utd-ratio")
    configure_training_track(args)
    configure_training_auto_events(args)
    if args.track_spec is None:
        configure_training_case(args)
    if args.onnx:
        require_onnx_export_dependencies()

    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        import torch
    except ModuleNotFoundError as ex:
        raise SystemExit(
            "stable-baselines3 and torch are required for training; install "
            "AIO-Gym with `pip install 'aiogym[rl]'`."
        ) from ex
    args.device = args.device or best_device()
    args.train_freq, args.gradient_steps = sb3_update_schedule(
        utd_ratio=0.0 if args.algo == "ppo" else args.utd_ratio,
        n_envs=args.n_envs,
        vector_steps=args.train_freq,
    )
    torch.set_num_threads(max(1, int(args.torch_threads)))
    prepare_resume_training_state(args)

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = run_name_for(args)
    checkpoint_path = os.path.join(args.out_dir, run_name)

    env_fns = [
        make_training_env(args, rank=rank)
        for rank in range(args.n_envs)
    ]
    env = (
        SubprocVecEnv(
            env_fns,
            start_method=args.subproc_start_method,
        )
        if args.vec_env == "subproc"
        else DummyVecEnv(env_fns)
    )
    env.seed(args.seed)
    model = build_algo(args, env)
    starting_step = int(getattr(model, "num_timesteps", 0))
    print(
        f"training {args.algo.upper()} | {args.n_envs} {args.vec_env} envs | "
        f"device={args.device} | action={args.action_mode} | "
        f"track={args.track or 'custom'} | goal={args.goal} | "
        f"reward={args.reward_spec}"
    )
    curve_callback = make_learning_curve_callback(args, checkpoint_path)
    initial_curve_point = evaluate_training_policy(
        args,
        model,
        starting_step,
        phase="resume" if args.resume else "initial",
        episode_plan=curve_callback.validation_plan,
    )
    initial_curve_point["timesteps"] = starting_step
    curve_callback.history.append(initial_curve_point)
    t0 = time.time()
    remaining_steps = max(0, int(args.steps) - starting_step)
    if remaining_steps:
        model.learn(
            total_timesteps=remaining_steps,
            progress_bar=False,
            callback=curve_callback,
            reset_num_timesteps=not bool(args.resume),
        )
    train_seconds = time.time() - t0
    final_step = int(getattr(model, "num_timesteps", args.steps))
    collected_transitions = max(0, final_step - starting_step)
    if (
        curve_callback.history
        and int(curve_callback.history[-1].get("timesteps", -1)) == final_step
    ):
        final_curve_point = dict(curve_callback.history[-1])
        final_curve_point["phase"] = "final"
    else:
        final_curve_point = evaluate_training_policy(
            args,
            model,
            final_step,
            phase="final",
            episode_plan=curve_callback.validation_plan,
        )
    final_curve_point["timesteps"] = final_step
    final_checkpoint_path = f"{checkpoint_path}_final"
    model.save(final_checkpoint_path)
    final_checkpoint_zip = f"{final_checkpoint_path}.zip"
    best_validation = (
        {
            "step": curve_callback.best_step,
            "metric_value": curve_callback.best_metric_value,
        }
        if curve_callback.best_step is not None
        else None
    )
    save_resumable_training_state(
        args,
        model,
        final_checkpoint_zip,
        best_validation=best_validation,
    )
    if curve_callback.best_step is None:
        model.save(checkpoint_path)
        checkpoint_selection = "final"
    else:
        checkpoint_selection = "best-validation"
    checkpoint_zip = f"{checkpoint_path}.zip"
    if checkpoint_selection == "final":
        save_resumable_training_state(
            args,
            model,
            checkpoint_zip,
            best_validation=best_validation,
        )
    onnx_path = None
    if args.onnx:
        onnx_path = args.onnx_path or f"{checkpoint_path}.onnx"
        export_model = model
        if checkpoint_selection == "best-validation":
            algorithm_class = type(model)
            export_model = algorithm_class.load(
                checkpoint_zip,
                device=args.device,
            )
        export_onnx(export_model, env.observation_space.shape[0], onnx_path)
    env.close()

    evaluation_spec, evaluation, rollouts = _evaluate_validation_checkpoint(
        args,
        checkpoint_zip,
    )
    training = training_metadata(
        args,
        checkpoint_zip,
        final_checkpoint_path=final_checkpoint_zip,
        best_checkpoint_path=(
            checkpoint_zip if curve_callback.best_step is not None else None
        ),
        best_step=curve_callback.best_step,
        best_metric_value=curve_callback.best_metric_value,
        checkpoint_selection=checkpoint_selection,
    )
    if onnx_path is not None:
        training["onnx_path"] = onnx_path
    learning_curve = list(curve_callback.history)
    if learning_curve and int(learning_curve[-1].get("timesteps", -1)) == final_step:
        learning_curve[-1] = final_curve_point
    else:
        learning_curve.append(final_curve_point)
    results = evaluation["results"]
    aggregate = evaluation["aggregate"]
    evaluation_metadata = (
        evaluation_spec.metadata()
        if evaluation_spec is not None
        else {
            "scenario": args.scenario,
            "case": args.case,
            "goal": args.goal,
            "reward_spec_id": args.resolved_reward_spec_id,
        }
    )
    artifact_payload = rl_payload(
        kind="sb3_train_eval",
        scenario=args.scenario,
        goal=args.goal,
        action_mode=args.action_mode,
        training=training,
        evaluation=evaluation_metadata,
        results=results,
        rows=[
            result_row(
                result,
                scenario=args.scenario,
                action_mode=args.action_mode,
                controller=f"SB3-{args.algo.upper()}",
                run_case_id=(
                    f"{args.track}:{result.get('case_id')}:sb3_{args.algo}"
                    if args.track_spec is not None
                    else (
                        f"{args.goal}:{args.scenario}:"
                        f"sb3_{args.algo}"
                    )
                ),
            )
            for result in results
        ],
        learning_curve=learning_curve,
        rollouts=rollouts,
        extra={
            "track_evaluation": (
                evaluation
                if args.track_spec is not None
                else None
            ),
            "training_runtime": {
                "seconds": train_seconds,
                "collected_transitions": collected_transitions,
                "steps_per_second": (
                    collected_transitions / train_seconds
                    if train_seconds > 0 and collected_transitions
                    else None
                ),
                "steps_per_second_per_env": (
                    collected_transitions / train_seconds / args.n_envs
                    if train_seconds > 0 and collected_transitions
                    else None
                ),
            },
        },
    )
    write_rl_artifacts(artifact_dir_for(args, run_name), artifact_payload)

    metric = aggregate["metric"]
    print(f"saved selected checkpoint {checkpoint_zip} ({checkpoint_selection})")
    print(f"saved final checkpoint {final_checkpoint_zip}")
    if onnx_path is not None:
        print(f"exported onnx {onnx_path}")
    print(f"saved artifacts {artifact_dir_for(args, run_name)}")
    if train_seconds > 0 and collected_transitions:
        print(
            f"train throughput {collected_transitions / train_seconds:.1f} "
            f"steps/s ({args.n_envs} envs x "
            f"{collected_transitions / train_seconds / args.n_envs:.1f}/env/s)"
        )
    print(
        f"validation {metric}={aggregate['metric_value']:.3f} "
        f"cases={aggregate['case_count']} "
        f"eligible={aggregate['ranking_eligible']}"
    )
