#!/usr/bin/env python3
"""Internal RLPD execution kernel for the unified training adapter.

Pipeline:
  1. load an immutable Dataset v2 prior by path and verify its identity
  2. BC sanity warm start and offline critic pretraining
  3. vectorized online collection with canonical 50/50 replay sampling
  3. rank RLPD vs PID vs MPC by the same composite KPI score
     (tracking + excess-energy + safety) under dynamic disturbed
     conditions, so "RL beats MPC" is apples-to-apples
  4. save a checkpoint and export ONNX

"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from aiogym._internal.paths import run_path
from aiogym.controllers import make_controller
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
from aiogym.rl.online_collection import VectorOnlineCollector
from aiogym.rl.episode_env import (
    make_track_episode_sampler,
    make_track_training_base_env,
    make_track_training_env,
)
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.coordinator import EpisodeCoordinator
from aiogym.rl.checkpoints import validate_resume_config
from aiogym.rl.validation import (
    CompleteValidationCallback,
    ValidationEpisodePlan,
    evaluate_validation_policy,
)


def make_training_env(
    track,
    *,
    base_seed: int,
    worker_index: int = 0,
    n_envs: int = 1,
):
    coordinator = EpisodeCoordinator(
        base_seed=base_seed,
        namespace=track.seed_namespace("training"),
        next_episode_index=worker_index,
        stride=n_envs,
    )
    return make_track_training_env(
        track,
        base_seed=base_seed,
        worker_index=worker_index,
        coordinator=coordinator,
    )


def unified_config(args, dataset_id: str) -> RLTrainingConfig:
    return RLTrainingConfig(
        track_id=args.track or training_identity(args),
        algorithm_id="rlpd",
        training_seed=args.seed,
        total_transitions=args.online_steps,
        n_envs=args.n_envs,
        device=args.device,
        algorithm={
            "utd_ratio": float(args.utd),
            "n_critics": args.n_critics,
            "batch_size": args.batch_size,
            "offline_fraction": args.offline_fraction,
            "bc_steps": args.bc_steps,
            "pretrain_updates": args.pretrain_updates,
        },
        replay={
            "online_capacity": args.online_capacity,
            "source_stratified": args.dataset is not None,
        },
        evaluation={"every_transitions": args.eval_every},
        checkpointing={},
        dataset_id=dataset_id,
    )


def checkpoint_state(agent, config, *, collector=None, args=None):
    state = agent.state_dict()
    state["training_config"] = config.as_dict()
    state["training_config_hash"] = config.config_hash
    if collector is not None:
        resume_contract = collector.resume_state()
        if getattr(args, "track_spec", None) is not None:
            resume_contract.update(
                {
                    "track_hash": args.track_spec.track_hash,
                    "reward_spec_id": args.resolved_reward_spec_id,
                    "distribution_hash": (
                        args.track_spec.training_distribution().distribution_hash
                    ),
                    "policy_contract_hash": _stable_mapping_hash(
                        args.track_spec.policy_contract
                    ),
                    "algorithm_id": "rlpd",
                    "replay_schema": "aiogym.rlpd_replay.v1",
                }
            )
        state["resume_contract"] = resume_contract
    return state


def artifact_dir_for(args, base: str) -> str:
    return args.artifact_dir or f"{base}_artifacts"


def output_base_for(args, run_id: str | None = None) -> str:
    if args.out:
        return args.out
    stem = (
        f"rlpd_{training_identity(args)}_"
        f"training-seed{getattr(args, 'seed', 0)}"
    )
    return str(
        run_path("rl", "rlpd", f"{stem}_{run_id or utc_run_id()}")
    )


def _aggregate_evaluation_result(evaluation):
    aggregate = evaluation["aggregate"]
    metric = str(aggregate["metric"])
    return {
        "metric": metric,
        "metric_direction": aggregate["metric_direction"],
        metric: float(aggregate["metric_value"]),
        f"{metric}_std": float(np.std(aggregate.get("case_values", [0.0]))),
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


def _run_backend(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--track", default=None)
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--case", default=None, help="custom specialist Case v2 ID")
    ap.add_argument("--goal", choices=["regulation", "economic"], default=None)
    ap.add_argument("--reward-spec", default=None)
    ap.add_argument(
        "--policy-scope",
        choices=["generalist", "specialist"],
        default=None,
    )
    ap.add_argument("--control-dt", type=float, default=None,
                    help="override case control interval; case/default owns it when omitted")
    ap.add_argument("--episode-steps", type=int, default=None,
                    help="override case episode length; case/default owns it when omitted")
    ap.add_argument(
        "--dataset",
        default=None,
        help="Dataset v2 directory used as the immutable prior",
    )
    ap.add_argument(
        "--dataset-id",
        default=None,
        help="optional expected dataset ID; mismatch is rejected",
    )
    ap.add_argument(
        "--auto-events",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable generic automatically generated within-episode events",
    )
    ap.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--randomize-setpoints", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--randomize-plant", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--plant-drift", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--integral-obs", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument(
        "--disturbance-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include current disturbances; case-owned when omitted",
    )
    ap.add_argument(
        "--previous-action-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include the previous action; case-owned when omitted",
    )
    ap.add_argument(
        "--normalize-observations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="scale observations with fixed bounds; case-owned when omitted",
    )
    ap.add_argument(
        "--tracking-error-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replace setpoints with tracking errors; case-owned when omitted",
    )
    ap.add_argument("--terminate-on-runaway", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--noise", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--noise-pct", type=float, default=None)
    ap.add_argument("--action-mode", default=None, choices=["actuator", "setpoint"])
    ap.add_argument("--bc-steps", type=int, default=4000)
    ap.add_argument("--pretrain-updates", type=int, default=5000)
    ap.add_argument("--online-steps", type=int, default=30000)
    ap.add_argument("--utd", type=int, default=5)
    ap.add_argument("--n-critics", type=int, default=5)
    ap.add_argument("--n-envs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--online-capacity", type=int, default=1_000_000)
    ap.add_argument("--offline-fraction", type=float, default=0.5)
    ap.add_argument(
        "--allow-offline-ratio-variant",
        action="store_true",
        help="mark a non-50/50 replay ratio as an explicit RLPD variant",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--baseline-episodes", type=int, default=16)
    ap.add_argument("--validation-episodes", type=int, default=12)
    ap.add_argument(
        "--validation-seed-list",
        default=None,
        help="fixed comma-separated validation seeds",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="stable output basename; defaults to a timestamped path")
    ap.add_argument("--artifact-dir", default=None,
                    help="standard benchmark artifact directory; defaults to <out>_artifacts")
    ap.add_argument("--save-rollout", action="store_true")
    ap.add_argument("--rollout-steps", type=int, default=None)
    args = ap.parse_args(argv)
    if args.dataset is None:
        ap.error("--dataset is required")
    if args.n_envs <= 0:
        ap.error("--n-envs must be positive")
    if args.online_steps <= 0:
        ap.error("--online-steps must be positive")
    if min(
        args.baseline_episodes,
        args.validation_episodes,
    ) <= 0:
        ap.error("evaluation episode counts must be positive")
    if args.batch_size <= 0 or args.batch_size % 2:
        ap.error("--batch-size must be a positive even integer")
    if args.offline_fraction != 0.5 and not args.allow_offline_ratio_variant:
        ap.error(
            "canonical RLPD uses --offline-fraction 0.5; pass "
            "--allow-offline-ratio-variant for an explicit ablation"
        )
    locked_validation_seeds = (
        tuple(
            int(part.strip())
            for part in args.validation_seed_list.split(",")
            if part.strip()
        )
        if args.validation_seed_list
        else tuple(range(5000, 5000 + args.validation_episodes))
    )
    if (
        not locked_validation_seeds
        or len(set(locked_validation_seeds))
        != len(locked_validation_seeds)
        or min(locked_validation_seeds) < 0
    ):
        ap.error("--validation-seed-list must contain unique non-negative seeds")
    args.validation_episodes = len(locked_validation_seeds)
    configure_training_track(args)
    configure_training_auto_events(args)
    if args.track_spec is None and args.plant_drift is None:
        args.plant_drift = args.randomize_plant
    if args.track_spec is None:
        configure_training_case(
            args,
            episode_steps_attr="episode_steps",
            eval_episode_steps_attr=None,
        )

    import torch
    from aiogym.rl.rlpd import RLPD

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    training_sampler = (
        make_track_episode_sampler(args.track_spec)
        if args.track_spec is not None
        else None
    )

    def mkenv(mode=None):
        if args.track_spec is not None:
            if mode is not None and mode != args.action_mode:
                raise ValueError(
                    "official track action_mode cannot be overridden"
                )
            return make_training_env(
                args.track_spec,
                base_seed=args.seed,
                n_envs=args.n_envs,
            )
        from aiogym.env_factory import make_env

        return make_env(
            config={
                "scenario": args.scenario,
                "case": args.case,
                "reward_spec": args.resolved_reward_spec_id,
                "environment": {
                    "action_mode": mode or args.action_mode,
                    "control_dt": args.control_dt,
                    "episode_steps": args.episode_steps,
                },
            }
        )

    validation_callback = (
        CompleteValidationCallback(
            args.track_spec,
            base_seeds=locked_validation_seeds,
        )
        if args.track_spec is not None
        else None
    )

    def eval_result(agent, episodes=None, seed=5000, mode=None, include_episodes=False):
        requested_seeds = (
            locked_validation_seeds
            if episodes is None
            else tuple(range(seed, seed + episodes))
        )
        episodes = len(requested_seeds)
        if args.track_spec is not None:
            if mode is not None and mode != args.action_mode:
                raise ValueError(
                    "official track action_mode cannot be overridden"
                )
            validation_plan = ValidationEpisodePlan(
                args.track_spec,
                base_seeds=requested_seeds,
            )
            evaluation = evaluate_validation_policy(
                agent,
                validation_plan,
                include_episodes=include_episodes,
            )
            return _aggregate_evaluation_result(evaluation)
        return evaluate_controller(
            agent,
            mkenv(mode or args.action_mode),
            episodes=episodes,
            seed=seed,
            goal_specification=args.goal,
            include_episodes=include_episodes,
        )

    env = mkenv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()

    # baselines (the bar to beat) — FIXED-SP PID / fixed-model MPC on an actuator env,
    # the static operating point the supervisory RL must beat by adapting its setpoints.
    if args.track_spec is not None or getattr(
        args,
        "resolved_reward_spec_id",
        None,
    ) is not None:
        metric = (
            "regulation_cost_rate"
            if args.track_spec is not None and args.goal == "regulation"
            else "profit_rate"
            if args.track_spec is not None
            else "regulation_cost"
            if args.goal == "regulation"
            else "profit"
        )
        direction = "minimize" if args.goal == "regulation" else "maximize"
    else:
        metric = "regulation_cost" if args.goal == "regulation" else "profit"
        direction = "minimize" if args.goal == "regulation" else "maximize"

    def improvement(candidate, baseline):
        return candidate - baseline if direction == "maximize" else baseline - candidate

    def is_better(candidate, baseline):
        return improvement(candidate, baseline) > 0
    pid_controller = make_controller("pid", scenario=args.scenario)
    mpc_controller = make_controller("mpc", scenario=args.scenario)
    if args.track_spec is not None:
        pid = eval_result(pid_controller, episodes=args.baseline_episodes)
        mpc = eval_result(mpc_controller, episodes=args.baseline_episodes)
    else:
        pid = evaluate_controller(
            pid_controller,
            mkenv("actuator"),
            episodes=args.baseline_episodes,
            goal_specification=args.goal,
        )
        mpc = evaluate_controller(
            mpc_controller,
            mkenv("actuator"),
            episodes=args.baseline_episodes,
            goal_specification=args.goal,
        )
    print(f"[baseline] PID {metric}={pid[metric]:.1f}   MPC {metric}={mpc[metric]:.1f}")

    rlpd = RLPD(
        obs_dim,
        act_dim,
        n_critics=args.n_critics,
        utd=args.utd,
        batch=args.batch_size,
        device=args.device,
        online_capacity=args.online_capacity,
        offline_fraction=args.offline_fraction,
        canonical=not args.allow_offline_ratio_variant,
        seed=args.seed,
    )
    offline_replay = rlpd.load_dataset(
        args.dataset,
        stratify=True,
        verify_checksums=True,
    )
    if (
        args.dataset_id is not None
        and offline_replay.dataset_id != args.dataset_id
    ):
        ap.error(
            f"--dataset-id {args.dataset_id!r} does not match "
            f"manifest {offline_replay.dataset_id!r}"
        )
    print(
        f"[offline] dataset={offline_replay.dataset_id} "
        f"hash={offline_replay.dataset_hash[:12]} "
        f"transitions={len(offline_replay)} "
        f"strata={len(offline_replay.strata)}"
    )
    resolved_dataset_id = offline_replay.dataset_id
    training_config = unified_config(args, resolved_dataset_id)
    resume_contract = None
    if args.resume:
        resume_state = torch.load(
            args.resume,
            map_location=args.device,
            weights_only=False,
        )
        try:
            previous_config = RLTrainingConfig.from_mapping(
                resume_state["training_config"]
            )
            validate_resume_config(previous_config, training_config)
            _validate_rlpd_resume_contract(
                args,
                resume_state.get("resume_contract"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            ap.error(str(exc))
        resume_contract = resume_state["resume_contract"]
        rlpd.load_state_dict(resume_state)
        if args.dataset is not None:
            accounting = rlpd.accounting()
            if accounting["dataset_id"] != offline_replay.dataset_id:
                ap.error("resume checkpoint dataset_id does not match --dataset")
            if accounting["dataset_hash"] != offline_replay.dataset_hash:
                ap.error("resume checkpoint dataset hash does not match --dataset")
        print(
            f"[resume] online_transitions="
            f"{rlpd.environment_transitions} updates={rlpd.gradient_updates}"
        )

    # 1b) BC warm-start (regulation only): start near PID. For economic goals the
    # optimum is FAR from the PID setpoint, so BC-to-PID is a bad init (and imperfect
    # clones run a nonlinear CSTR away → runaway) — skip it with --bc-steps 0 and let
    # pretrain update the actor (original RLPD offline pretrain).
    if args.bc_steps > 0 and not args.resume:
        print(f"[bc] warm-starting actor ({args.bc_steps} steps)...")
        rlpd.bc_warmstart(args.bc_steps)
        bc_result = eval_result(rlpd)
        bc0 = bc_result[metric]
        print(f"[bc] done  {metric}={bc0:.1f}  (PID {pid[metric]:.1f} / MPC {mpc[metric]:.1f})")

    # 2a) offline pretrain. With BC: critic-only (hold the warm-started actor). Without
    # BC: full actor+critic (learn a policy from the PID prior data).
    pretrain_actor = args.bc_steps == 0
    pretrain_updates = 0 if args.resume else args.pretrain_updates
    print(
        f"[pretrain] {pretrain_updates} offline updates "
        f"(actor={pretrain_actor})..."
    )
    t0 = time.time()
    for _ in range(pretrain_updates):
        rlpd.update(actor=pretrain_actor)
    pretrain_result = eval_result(rlpd)
    pre = pretrain_result[metric]
    print(f"[pretrain] done in {time.time()-t0:.0f}s  {metric}={pre:.1f}")

    # 2b) online learning (symmetric offline+online sampling), best-checkpoint by KPI
    base = output_base_for(args)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    best = -1e18 if direction == "maximize" else 1e18
    best_path = base + "_best.pt"
    hist = []
    if args.bc_steps > 0 and not args.resume:
        hist.append(learning_curve_point(0, bc_result, phase="bc"))
    hist.append(learning_curve_point(0, pretrain_result, phase="pretrain"))
    t0 = time.time()
    collector_env_fns = (
        [
            lambda: make_track_training_base_env(
                args.track_spec,
                sampler=training_sampler,
            )
            for _ in range(args.n_envs)
        ]
        if args.track_spec is not None
        else [mkenv for _ in range(args.n_envs)]
    )
    collector = VectorOnlineCollector(
        collector_env_fns,
        rlpd.online,
        base_seed=args.seed,
        namespace=args.training_seed_namespace,
        sampler=training_sampler,
        track=args.track_spec,
        coordinator_state=resume_contract,
    )
    collector.online_transitions = rlpd.environment_transitions
    start_step = rlpd.environment_transitions
    while rlpd.environment_transitions < args.online_steps:
        remaining = args.online_steps - rlpd.environment_transitions
        collected = min(args.n_envs, remaining)
        collector.collect(rlpd, collected)
        rlpd.environment_transitions += collected
        for _ in range(collected):
            rlpd.update()
        step = rlpd.environment_transitions
        crossed_eval = (
            args.eval_every > 0
            and (
                step % args.eval_every < collected
                or step == args.online_steps
            )
        )
        if crossed_eval:
            checkpoint_id = f"step-{step}"
            if validation_callback is not None:
                online_evaluation = validation_callback.evaluate(
                    rlpd,
                    checkpoint_id=checkpoint_id,
                    step=step,
                )
                online_result = _aggregate_evaluation_result(
                    online_evaluation
                )
                improved = (
                    validation_callback.selector.best is not None
                    and validation_callback.selector.best.checkpoint_id
                    == checkpoint_id
                )
            else:
                online_result = eval_result(rlpd)
                improved = is_better(online_result[metric], best)
            ret, std = online_result[metric], online_result.get(f"{metric}_std", 0.0)
            if improved:
                best = ret
                torch.save(
                    checkpoint_state(
                        rlpd,
                        training_config,
                        collector=collector,
                        args=args,
                    ),
                    best_path,
                )
            hist.append(learning_curve_point(step, online_result, phase="online"))
            sps = (step - start_step) / max(time.time() - t0, 1e-12)
            print(f"[online] step {step:6d}  RLPD {metric}={ret:8.1f}±{std:.1f}  "
                  f"(PID {pid[metric]:.1f} / MPC {mpc[metric]:.1f})  best={best:.1f}  {sps:.0f} steps/s")
    collector_accounting = collector.accounting()
    run_accounting = rlpd.accounting()
    collector.close()

    selected_by_validation = os.path.exists(best_path)
    if selected_by_validation:                         # restore the best checkpoint for the final policy
        rlpd.load_state_dict(
            torch.load(
                best_path,
                map_location=args.device,
                weights_only=False,
            )
        )
    if args.track_spec is not None:
        final_validation_plan = validation_callback.plan
        final_evaluation = evaluate_validation_policy(
            rlpd,
            final_validation_plan,
            include_episodes=True,
        )
        final_result = _aggregate_evaluation_result(final_evaluation)
        final_results = final_evaluation["results"]
        evaluation_metadata = args.track_spec.metadata()
    else:
        final_result = eval_result(
            rlpd,
            episodes=args.validation_episodes,
            seed=5000,
            include_episodes=True,
        )
        final_results = [final_result]
        final_evaluation = None
        evaluation_metadata = {
            "scenario": args.scenario,
            "case": args.case,
            "goal": args.goal,
            "reward_spec_id": args.resolved_reward_spec_id,
        }
    final, final_std = final_result[metric], final_result.get(f"{metric}_std", 0.0)
    final_point = learning_curve_point(args.online_steps, final_result, phase="final")
    if not hist or hist[-1].get("phase") != "final":
        hist.append(final_point)
    result = {
        "scenario": args.scenario,
        "goal": args.goal,
        "reward_spec_id": args.resolved_reward_spec_id,
        "metric": metric,
        "PID": {metric: pid[metric]}, "MPC": {metric: mpc[metric]},
        "RLPD": {metric: final, "std": final_std, "best": best},
        "history": hist,
        "beats_pid": is_better(final, pid[metric]),
        "beats_mpc": is_better(final, mpc[metric]),
        "margin_vs_mpc": improvement(final, mpc[metric]),
        "margin_vs_pid": improvement(final, pid[metric]),
    }
    print(json.dumps({k: result[k] for k in ("scenario", "metric", "beats_pid", "beats_mpc",
                                             "margin_vs_mpc", "margin_vs_pid")}, indent=2))

    torch.save(
        checkpoint_state(
            rlpd,
            training_config,
            collector=collector,
            args=args,
        ),
        base + ".pt",
    )
    rlpd.save_onnx(base + ".onnx")
    rollouts = []
    if args.save_rollout:
        if args.track_spec is not None:
            for case in args.track_spec.resolved_cases("validation"):
                from aiogym.env_factory import make_env

                rollout = rollout_controller(
                    rlpd,
                    make_env(
                        args.scenario,
                        case=case.profile,
                        reward_spec=args.reward_spec,
                    ),
                    seed=5000,
                    max_steps=args.rollout_steps,
                )
                rollout.update(
                    {
                        "track_id": args.track,
                        "track_split": "validation",
                        "case_id": case.case_id,
                    }
                )
                rollouts.append(rollout)
        else:
            rollouts.append(rollout_controller(
                rlpd,
                mkenv(args.action_mode),
                seed=5000,
                max_steps=args.rollout_steps,
            ))
    training = {
        "algo": "rlpd",
        "scenario": args.scenario,
        "case": args.case,
        "action_mode": args.action_mode,
        "track_id": args.track,
        "track_hash": getattr(args, "track_hash", None),
        "goal": getattr(args, "goal", None),
        "reward_spec_id": getattr(args, "resolved_reward_spec_id", None),
        "policy_scope": getattr(args, "policy_scope", "specialist"),
        "training_seed_namespace": args.training_seed_namespace,
        "validation_seed_namespace": args.validation_seed_namespace,
        "training_seed_namespace_hash": args.training_seed_namespace_hash,
        "validation_seed_namespace_hash": args.validation_seed_namespace_hash,
        "seed": args.seed,
        "dataset_id": run_accounting["dataset_id"],
        "dataset_hash": run_accounting["dataset_hash"],
        "dataset_path": args.dataset,
        "offline_transitions": run_accounting[
            "offline_transitions"
        ],
        "online_transitions": collector_accounting[
            "online_transitions"
        ],
        "online_replay_transitions": run_accounting[
            "online_replay_transitions"
        ],
        "offline_samples": run_accounting["offline_samples"],
        "online_samples": run_accounting["online_samples"],
        "sampled_offline_fraction": run_accounting[
            "sampled_offline_fraction"
        ],
        "source_stratified": args.dataset is not None,
        "offline_fraction": args.offline_fraction,
        "rlpd_variant": (
            "canonical-50-50"
            if not args.allow_offline_ratio_variant
            else "offline-ratio-ablation"
        ),
        "n_envs": args.n_envs,
        "training_config_hash": training_config.config_hash,
        "bc_steps": args.bc_steps,
        "pretrain_updates": args.pretrain_updates,
        "online_steps": args.online_steps,
        "eval_every": args.eval_every,
        "utd": args.utd,
        "n_critics": args.n_critics,
        "checkpoint_path": base + ".pt",
        "checkpoint_selection": (
            "best-validation"
            if selected_by_validation
            else "final-no-validation"
        ),
        "onnx_path": base + ".onnx",
        "onnx_normalized_actions": True,
    }
    standard_payload = rl_payload(
        kind="rlpd_train_eval",
        scenario=args.scenario,
        goal=args.goal,
        action_mode=args.action_mode,
        training=training,
        evaluation=evaluation_metadata,
        results=final_results,
        rows=[
            result_row(
                case_result,
                args.scenario,
                args.action_mode,
                controller="RLPD",
                run_case_id=(
                    f"{args.track}:{case_result.get('case_id')}:rlpd"
                    if args.track_spec is not None
                    else f"{args.goal}:{args.scenario}:rlpd"
                ),
            )
            for case_result in final_results
        ],
        learning_curve=hist,
        rollouts=rollouts,
        extra={
            "training_runtime": {
                "online_seconds": time.time() - t0,
            },
            "replay_accounting": run_accounting,
            "collector_accounting": collector_accounting,
            "rl_comparison": result,
            "track_evaluation": final_evaluation,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_rl_artifacts(artifact_dir_for(args, base), standard_payload)
    print(f"saved {base}.pt / .onnx / .json")
    print(f"saved artifacts {artifact_dir_for(args, base)}")


def _validate_rlpd_resume_contract(args, state) -> None:
    if not isinstance(state, dict):
        raise ValueError(
            "RLPD checkpoint is missing restart-episode coordinator state"
        )
    if state.get("resume_mode") != "restart_episode":
        raise ValueError("RLPD supports only restart_episode resume")
    if getattr(args, "track_spec", None) is None:
        return
    expected = {
        "track_hash": args.track_spec.track_hash,
        "reward_spec_id": args.resolved_reward_spec_id,
        "distribution_hash": (
            args.track_spec.training_distribution().distribution_hash
        ),
        "policy_contract_hash": _stable_mapping_hash(
            args.track_spec.policy_contract
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
