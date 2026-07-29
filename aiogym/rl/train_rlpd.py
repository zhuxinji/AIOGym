#!/usr/bin/env python3
"""Train RLPD on the native AIO-Gym env and beat PID / MPC on the gym's own KPI.

Pipeline (the offline->online story):
  1. roll out the existing PID controller -> an offline "historian" dataset
  2. offline-pretrain RLPD, then keep learning online (symmetric sampling)
  3. rank RLPD vs PID vs MPC by the same composite KPI score
     (tracking + excess-energy + safety) under dynamic disturbed
     conditions, so "RL beats MPC" is apples-to-apples
  4. save a checkpoint and export ONNX

    python -m aiogym.rl.train_rlpd --scenario cascade --online-steps 30000
"""
from __future__ import annotations
import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

from aiogym._internal.paths import run_path
from aiogym.benchmarks import CaseMixtureEnv, evaluate_policy_on_track
from aiogym.controllers import build_context, make_controller, validate_action
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


def collect_offline(env, agent, episodes, seed=1000):
    """Prior data. Supervisory (RL-on-PID) env: roll the default-SP action (= fixed-SP
    PID) + exploration noise. Actuator env: roll the PID controller directly."""
    supervisory = getattr(env, "layout", None) is not None
    a0 = env.default_sp_action() if supervisory else None
    data = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        if agent is not None:
            agent.reset(seed=seed + ep)
        done = False
        info = {}
        while not done:
            if supervisory:
                a = np.clip(a0 + np.random.normal(0, 0.15, a0.shape), 0, 1).astype(np.float32)
            else:
                a = validate_action(agent.act(obs, build_context(env, info)), env, agent.name)
            o2, r, term, trunc, info = env.step(a)
            data.append((obs, a, r, o2, float(term)))
            obs = o2
            done = term or trunc
    return data


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


def main(argv=None, prog=None):
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
    ap.add_argument("--offline-episodes", type=int, default=40)
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
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="stable output basename; defaults to a timestamped path")
    ap.add_argument("--artifact-dir", default=None,
                    help="standard benchmark artifact directory; defaults to <out>_artifacts")
    ap.add_argument("--save-rollout", action="store_true")
    ap.add_argument("--rollout-steps", type=int, default=None)
    args = ap.parse_args(argv)
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
    from aiogym.rl import RLPD

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def mkenv(mode=None):
        if args.track_spec is not None:
            if mode is not None and mode != args.action_mode:
                raise ValueError(
                    "official track action_mode cannot be overridden"
                )
            return CaseMixtureEnv(
                args.track_spec,
                split="training",
                training=True,
            )
        from aiogym.env import AIOGymEnv

        return AIOGymEnv(
            args.scenario,
            case=args.case,
            reward_spec=args.resolved_reward_spec_id,
            action_mode=mode or args.action_mode,
            control_dt=args.control_dt,
            episode_steps=args.episode_steps,
        )

    def eval_result(agent, episodes=12, seed=5000, mode=None, include_episodes=False):
        if args.track_spec is not None:
            if mode is not None and mode != args.action_mode:
                raise ValueError(
                    "official track action_mode cannot be overridden"
                )
            evaluation = evaluate_policy_on_track(
                agent,
                args.track_spec,
                split="validation",
                base_seeds=range(seed, seed + episodes),
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
        pid = eval_result(pid_controller, episodes=16)
        mpc = eval_result(mpc_controller, episodes=16)
    else:
        pid = evaluate_controller(
            pid_controller,
            mkenv("actuator"),
            episodes=16,
            goal_specification=args.goal,
        )
        mpc = evaluate_controller(
            mpc_controller,
            mkenv("actuator"),
            episodes=16,
            goal_specification=args.goal,
        )
    print(f"[baseline] PID {metric}={pid[metric]:.1f}   MPC {metric}={mpc[metric]:.1f}")

    # 1) offline historian from PID (fixed nominal PID is a fine prior)
    print(f"[offline] collecting {args.offline_episodes} PID episodes...")
    offline = collect_offline(mkenv(), make_controller("pid", scenario=args.scenario), args.offline_episodes)
    print(f"[offline] {len(offline)} transitions")

    rlpd = RLPD(obs_dim, act_dim, n_critics=args.n_critics, utd=args.utd, batch=256)
    rlpd.load_offline(offline)

    # 1b) BC warm-start (regulation only): start near PID. For economic goals the
    # optimum is FAR from the PID setpoint, so BC-to-PID is a bad init (and imperfect
    # clones run a nonlinear CSTR away → runaway) — skip it with --bc-steps 0 and let
    # pretrain update the actor (original RLPD offline pretrain).
    if args.bc_steps > 0:
        print(f"[bc] warm-starting actor ({args.bc_steps} steps)...")
        rlpd.bc_warmstart(args.bc_steps)
        bc_result = eval_result(rlpd)
        bc0 = bc_result[metric]
        print(f"[bc] done  {metric}={bc0:.1f}  (PID {pid[metric]:.1f} / MPC {mpc[metric]:.1f})")

    # 2a) offline pretrain. With BC: critic-only (hold the warm-started actor). Without
    # BC: full actor+critic (learn a policy from the PID prior data).
    pretrain_actor = args.bc_steps == 0
    print(f"[pretrain] {args.pretrain_updates} offline updates (actor={pretrain_actor})...")
    t0 = time.time()
    for i in range(args.pretrain_updates):
        rlpd.update(actor=pretrain_actor)
    pretrain_result = eval_result(rlpd)
    pre = pretrain_result[metric]
    print(f"[pretrain] done in {time.time()-t0:.0f}s  {metric}={pre:.1f}")

    # 2b) online learning (symmetric offline+online sampling), best-checkpoint by KPI
    base = output_base_for(args)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    best = -1e18 if direction == "maximize" else 1e18
    best_path = base + "_best.pt"
    obs, _ = env.reset(seed=args.seed)
    hist = []
    if args.bc_steps > 0:
        hist.append(learning_curve_point(0, bc_result, phase="bc"))
    hist.append(learning_curve_point(0, pretrain_result, phase="pretrain"))
    t0 = time.time()
    for step in range(1, args.online_steps + 1):
        a = rlpd.act(obs, deterministic=False)
        o2, r, term, trunc, _ = env.step(a)
        rlpd.push(obs, a, r, o2, term)
        obs = o2 if not (term or trunc) else env.reset()[0]
        rlpd.update()
        if step % args.eval_every == 0:
            online_result = eval_result(rlpd)
            ret, std = online_result[metric], online_result.get(f"{metric}_std", 0.0)
            if is_better(ret, best):                    # keep the peak — off-policy RL can collapse late
                best = ret
                torch.save(rlpd.state_dict(), best_path)
            hist.append(learning_curve_point(step, online_result, phase="online"))
            sps = step / (time.time() - t0)
            print(f"[online] step {step:6d}  RLPD {metric}={ret:8.1f}±{std:.1f}  "
                  f"(PID {pid[metric]:.1f} / MPC {mpc[metric]:.1f})  best={best:.1f}  {sps:.0f} steps/s")

    selected_by_validation = os.path.exists(best_path)
    if selected_by_validation:                         # restore the best checkpoint for the final policy
        rlpd.load_state_dict(torch.load(best_path))
    if args.track_spec is not None:
        final_evaluation = evaluate_policy_on_track(
            rlpd,
            args.track_spec,
            split="test",
            base_seeds=range(5000, 5024),
            include_episodes=True,
        )
        final_result = _aggregate_evaluation_result(final_evaluation)
        final_results = final_evaluation["results"]
        evaluation_metadata = args.track_spec.metadata()
    else:
        final_result = eval_result(
            rlpd,
            episodes=24,
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

    torch.save(rlpd.state_dict(), base + ".pt")
    rlpd.save_onnx(base + ".onnx")
    rollouts = []
    if args.save_rollout:
        if args.track_spec is not None:
            for case in args.track_spec.resolved_cases("test"):
                from aiogym.env import AIOGymEnv

                rollout = rollout_controller(
                    rlpd,
                    AIOGymEnv(
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
                        "track_split": "test",
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
        "test_seed_namespace": args.test_seed_namespace,
        "training_seed_namespace_hash": args.training_seed_namespace_hash,
        "validation_seed_namespace_hash": args.validation_seed_namespace_hash,
        "test_seed_namespace_hash": args.test_seed_namespace_hash,
        "seed": args.seed,
        "offline_episodes": args.offline_episodes,
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
            "rl_comparison": result,
            "track_evaluation": final_evaluation,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_rl_artifacts(artifact_dir_for(args, base), standard_payload)
    print(f"saved {base}.pt / .onnx / .json")
    print(f"saved artifacts {artifact_dir_for(args, base)}")


if __name__ == "__main__":
    main()
