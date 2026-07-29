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
import math
import os
import time

from aiogym._internal.config import parse_seed_list
from aiogym._internal.paths import run_path
from aiogym.benchmarks import CaseMixtureEnv, evaluate_policy_on_track
from aiogym.controllers import make_controller
from aiogym.env import AIOGymEnv
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


def make_training_env(args, rank: int = 0):
    def _init():
        track = getattr(args, "track_spec", None)
        if track is not None:
            env = CaseMixtureEnv(
                track,
                split="training",
                worker_index=rank,
                training=True,
            )
            env.reset(seed=args.seed)
            return env
        env = AIOGymEnv(
            args.scenario,
            case=getattr(args, "case", None),
            reward_spec=args.resolved_reward_spec_id,
            action_mode=args.action_mode,
            control_dt=args.control_dt,
            episode_steps=args.train_episode_steps,
            auto_events=args.auto_events,
            randomize=args.randomize,
            randomize_setpoints=args.randomize_setpoints,
            randomize_plant=args.randomize_plant,
            plant_drift=args.plant_drift,
            integral_obs=args.integral_obs,
            disturbance_obs=args.disturbance_obs,
            previous_action_obs=args.previous_action_obs,
            normalize_observations=args.normalize_observations,
            tracking_error_obs=args.tracking_error_obs,
            terminate_on_runaway=args.terminate_on_runaway,
            noise=args.noise,
            noise_pct=args.noise_pct,
        )
        env.reset(seed=args.seed + rank)
        return env

    return _init


def default_n_envs():
    return min(16, max(1, (os.cpu_count() or 4) - 2))


def best_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_algo(args, env):
    try:
        from stable_baselines3 import PPO, SAC, TD3
    except ModuleNotFoundError as ex:
        raise SystemExit(
            "stable-baselines3 is required for training; install AIO-Gym "
            "with `pip install 'aiogym[rl]'`."
        ) from ex

    algo = args.algo.lower()
    common = dict(
        policy="MlpPolicy",
        env=env,
        verbose=args.verbose,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tensorboard_log=args.tensorboard_log,
    )
    if algo == "sac":
        return SAC(
            batch_size=args.batch_size,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            **common,
        )
    if algo == "td3":
        return TD3(
            batch_size=args.batch_size,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            **common,
        )
    if algo == "ppo":
        return PPO(
            n_steps=args.ppo_n_steps,
            batch_size=args.batch_size,
            **common,
        )
    raise ValueError(f"unsupported SB3 algorithm: {args.algo}")


def evaluate_checkpoint(args, checkpoint_path: str):
    if getattr(args, "track_spec", None) is not None:
        controller = make_controller(
            "sb3",
            scenario=args.scenario,
            config={
                "path": checkpoint_path,
                "algo": args.algo,
                "action_mode": args.action_mode,
            },
        )
        seeds = parse_seed_list(
            args.eval_seed_list,
            args.eval_seed,
            args.eval_episodes,
            option="--eval-seed-list",
        )
        evaluation = evaluate_policy_on_track(
            controller,
            args.track_spec,
            split="test",
            base_seeds=seeds,
            include_episodes=True,
        )
        rollouts = []
        if args.save_rollout:
            for case in args.track_spec.resolved_cases("test"):
                env = AIOGymEnv(
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
                        "track_split": "test",
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
        },
    )
    seeds = parse_seed_list(
        args.eval_seed_list,
        args.eval_seed,
        args.eval_episodes,
        option="--eval-seed-list",
    )
    evaluation_env = AIOGymEnv(
        args.scenario,
        case=getattr(args, "case", None),
        reward_spec=args.resolved_reward_spec_id,
        action_mode=args.action_mode,
        episode_steps=args.eval_episode_steps,
        control_dt=args.control_dt,
    )
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
        rollout_env = AIOGymEnv(
            args.scenario,
            case=getattr(args, "case", None),
            reward_spec=args.resolved_reward_spec_id,
            action_mode=args.action_mode,
            episode_steps=args.eval_episode_steps,
            control_dt=args.control_dt,
        )
        rollout = rollout_controller(
            controller,
            rollout_env,
            seed=seeds[0],
            max_steps=args.rollout_steps,
        )
        rollout_env.close()
    evaluation = {
        "split": "test",
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


def evaluate_training_policy(args, model, step: int, phase: str = "eval"):
    if getattr(args, "track_spec", None) is not None:
        controller = make_controller(
            "sb3",
            scenario=args.scenario,
            policy=model,
            config={
                "algo": args.algo,
                "action_mode": args.action_mode,
                "name": f"SB3-{args.algo.upper()}",
            },
        )
        seeds = parse_seed_list(
            args.eval_seed_list,
            args.eval_seed,
            args.learning_curve_episodes,
            option="--eval-seed-list",
        )
        evaluation = evaluate_policy_on_track(
            controller,
            args.track_spec,
            split="validation",
            base_seeds=seeds,
            include_episodes=False,
        )
        aggregate = evaluation["aggregate"]
        row = {
            "step": int(step),
            "phase": phase,
            **aggregate,
            "track_id": args.track,
            "track_split": "validation",
            "seed_namespace": evaluation["seed_namespace"],
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
        },
    )
    seeds = parse_seed_list(
        args.eval_seed_list,
        args.eval_seed,
        args.learning_curve_episodes,
        option="--eval-seed-list",
    )
    env = AIOGymEnv(
        args.scenario,
        case=getattr(args, "case", None),
        reward_spec=args.resolved_reward_spec_id,
        action_mode=args.action_mode,
        episode_steps=args.eval_episode_steps,
        control_dt=args.control_dt,
    )
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
            "test_seed_namespace": getattr(
                args,
                "test_seed_namespace",
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
            "test_seed_namespace_hash": getattr(
                args,
                "test_seed_namespace_hash",
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


def _learning_curve_point_is_better(row, best_metric_value):
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

        def _on_step(self) -> bool:
            if args.learning_curve_every <= 0 or self.num_timesteps < self._next_eval:
                return True
            row = evaluate_training_policy(args, self.model, self.num_timesteps)
            row["timesteps"] = self.num_timesteps
            self.history.append(row)
            if (
                best_checkpoint_path is not None
                and _learning_curve_point_is_better(row, self.best_metric_value)
            ):
                self.best_metric_value = float(row["metric_value"])
                self.best_step = int(self.num_timesteps)
                self.model.save(best_checkpoint_path)
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
    torch.onnx.export(
        DeterministicPolicy(policy),
        dummy,
        path,
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
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


def main(argv=None, prog=None):
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
        "--gradient-steps",
        type=int,
        default=1,
        help="optimizer steps per phase; with N environments, N gives about one update per collected transition",
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
    args = ap.parse_args(argv)
    if not 0.0 < args.gamma <= 1.0:
        ap.error("--gamma must be in (0, 1]")
    configure_training_track(args)
    configure_training_auto_events(args)
    if args.track_spec is None:
        configure_training_case(args)
    if args.onnx:
        require_onnx_export_dependencies()

    try:
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        import torch
    except ModuleNotFoundError as ex:
        raise SystemExit(
            "stable-baselines3 and torch are required for training; install "
            "AIO-Gym with `pip install 'aiogym[rl]'`."
        ) from ex
    args.device = args.device or best_device()
    torch.set_num_threads(max(1, int(args.torch_threads)))

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = run_name_for(args)
    checkpoint_path = os.path.join(args.out_dir, run_name)

    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    vec_env_kwargs = {"start_method": args.subproc_start_method} if args.vec_env == "subproc" else None
    env = make_vec_env(
        make_training_env(args),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )
    model = build_algo(args, env)
    print(
        f"training {args.algo.upper()} | {args.n_envs} {args.vec_env} envs | "
        f"device={args.device} | action={args.action_mode} | "
        f"track={args.track or 'custom'} | goal={args.goal} | "
        f"reward={args.reward_spec}"
    )
    curve_callback = make_learning_curve_callback(args, checkpoint_path)
    initial_curve_point = evaluate_training_policy(args, model, 0, phase="initial")
    initial_curve_point["timesteps"] = 0
    curve_callback.history.append(initial_curve_point)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, progress_bar=False, callback=curve_callback)
    train_seconds = time.time() - t0
    final_step = int(getattr(model, "num_timesteps", args.steps))
    if (
        curve_callback.history
        and int(curve_callback.history[-1].get("timesteps", -1)) == final_step
    ):
        final_curve_point = dict(curve_callback.history[-1])
        final_curve_point["phase"] = "final"
    else:
        final_curve_point = evaluate_training_policy(
            args, model, final_step, phase="final"
        )
    final_curve_point["timesteps"] = final_step
    final_checkpoint_path = f"{checkpoint_path}_final"
    model.save(final_checkpoint_path)
    final_checkpoint_zip = f"{final_checkpoint_path}.zip"
    if curve_callback.best_step is None:
        model.save(checkpoint_path)
        checkpoint_selection = "final"
    else:
        checkpoint_selection = "best-validation"
    checkpoint_zip = f"{checkpoint_path}.zip"
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

    evaluation_spec, evaluation, rollouts = evaluate_checkpoint(
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
                "steps_per_second": args.steps / train_seconds if train_seconds > 0 else None,
                "steps_per_second_per_env": (args.steps / train_seconds / args.n_envs) if train_seconds > 0 else None,
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
    if train_seconds > 0:
        print(
            f"train throughput {args.steps / train_seconds:.1f} steps/s "
            f"({args.n_envs} envs x {args.steps / train_seconds / args.n_envs:.1f}/env/s)"
        )
    print(
        f"eval {metric}={aggregate['metric_value']:.3f} "
        f"cases={aggregate['case_count']} "
        f"eligible={aggregate['ranking_eligible']}"
    )


if __name__ == "__main__":
    main()
