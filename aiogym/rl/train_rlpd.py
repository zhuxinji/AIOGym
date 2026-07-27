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
from aiogym._internal.vocabulary import OBJECTIVE_NAMES
from aiogym.controllers import build_context, make_controller, validate_action
from aiogym.evaluation import (
    evaluate_controller,
    metric_direction,
    metric_for_reward_mode,
    primary_metric_for_objective,
    rollout_controller,
)
from aiogym.rl.training_config import (
    configure_training_auto_events,
    configure_training_objective,
    configure_training_task,
    training_identity,
    training_protocol,
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
    return str(
        run_path("rl", "rlpd", f"{training_identity(args)}_{run_id or utc_run_id()}")
    )


def main(argv=None, prog=None):
    ap = argparse.ArgumentParser(prog=prog)
    ap.add_argument("--scenario", default="cascade")
    ap.add_argument("--task", default=None, help="named scenario task profile")
    ap.add_argument(
        "--objective",
        default=None,
        choices=OBJECTIVE_NAMES,
        help="training and evaluation objective; task-owned when available, otherwise kpi",
    )
    ap.add_argument("--control-dt", type=float, default=None,
                    help="override task control interval; task/default owns it when omitted")
    ap.add_argument("--episode-steps", type=int, default=None,
                    help="override task episode length; task/default owns it when omitted")
    ap.add_argument("--tracking-q-y", type=float, default=None,
                    help="override scalar tracking weight Q; task-owned when omitted")
    ap.add_argument("--tracking-r-move", type=float, default=None,
                    help="override move weight R; task-owned when omitted")
    ap.add_argument("--offline-episodes", type=int, default=40)
    ap.add_argument(
        "--auto-events",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable generic automatically generated within-episode events",
    )
    ap.add_argument("--randomize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--randomize-setpoints", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--randomize-plant", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--plant-drift", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--integral-obs", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--disturbance-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include current disturbances; task-owned when omitted",
    )
    ap.add_argument(
        "--previous-action-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="include the previous action; task-owned when omitted",
    )
    ap.add_argument(
        "--normalize-observations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="scale observations with fixed bounds; task-owned when omitted",
    )
    ap.add_argument(
        "--tracking-error-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="replace setpoints with tracking errors; task-owned when omitted",
    )
    ap.add_argument("--terminate-on-runaway", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--noise", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--noise-pct", type=float, default=0.01)
    ap.add_argument("--action-mode", default="actuator", choices=["actuator", "setpoint"])
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
    configure_training_objective(args)
    configure_training_auto_events(args)
    if args.plant_drift is None:
        args.plant_drift = args.randomize_plant
    configure_training_task(
        args,
        episode_steps_attr="episode_steps",
        eval_episode_steps_attr=None,
    )

    import torch
    from aiogym.rl import RLPD

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    def protocol(mode=None):
        # RL uses the requested action_mode (setpoint = supervisory RL-on-PID); baselines
        # run on an actuator-mode env (they ARE fixed-SP controllers).
        return training_protocol(
            args,
            action_mode=mode,
            episode_steps_attr="episode_steps",
        )

    def mkenv(mode=None):
        return protocol(mode).make_env()

    def eval_result(agent, episodes=12, seed=5000, mode=None, include_episodes=False):
        eval_protocol = protocol(mode or args.action_mode)
        return evaluate_controller(
            agent,
            eval_protocol.make_env(),
            episodes=episodes,
            seed=seed,
            protocol=eval_protocol,
            include_episodes=include_episodes,
        )

    env = mkenv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # baselines (the bar to beat) — FIXED-SP PID / fixed-model MPC on an actuator env,
    # the static operating point the supervisory RL must beat by adapting its setpoints.
    metric = (
        metric_for_reward_mode(args.resolved_reward_mode)
        if args.objective in {"economic", "tracking", "kpi"}
        else primary_metric_for_objective(args.objective)
    )
    direction = metric_direction(metric)

    def improvement(candidate, baseline):
        return candidate - baseline if direction == "maximize" else baseline - candidate

    def is_better(candidate, baseline):
        return improvement(candidate, baseline) > 0
    baseline_protocol = protocol("actuator")
    pid_controller = make_controller("pid", scenario=args.scenario)
    mpc_controller = make_controller("mpc", scenario=args.scenario)
    pid = evaluate_controller(pid_controller, baseline_protocol.make_env(),
                              episodes=16, protocol=baseline_protocol)
    mpc = evaluate_controller(mpc_controller, baseline_protocol.make_env(),
                              episodes=16, protocol=baseline_protocol)
    print(f"[baseline] PID {metric}={pid[metric]:.1f}   MPC {metric}={mpc[metric]:.1f}")

    # 1) offline historian from PID (fixed nominal PID is a fine prior)
    print(f"[offline] collecting {args.offline_episodes} PID episodes...")
    offline = collect_offline(mkenv(), make_controller("pid", scenario=args.scenario), args.offline_episodes)
    print(f"[offline] {len(offline)} transitions")

    rlpd = RLPD(obs_dim, act_dim, n_critics=args.n_critics, utd=args.utd, batch=256)
    rlpd.load_offline(offline)

    # 1b) BC warm-start (tracking only): start near PID. For economic objectives the
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

    if os.path.exists(best_path):                       # restore the best checkpoint for the final policy
        rlpd.load_state_dict(torch.load(best_path))
    eval_protocol = protocol(args.action_mode)
    final_result = evaluate_controller(
        rlpd,
        eval_protocol.make_env(),
        episodes=24,
        seed=5000,
        protocol=eval_protocol,
        include_episodes=True,
    )
    final, final_std = final_result[metric], final_result.get(f"{metric}_std", 0.0)
    final_point = learning_curve_point(args.online_steps, final_result, phase="final")
    if not hist or hist[-1].get("phase") != "final":
        hist.append(final_point)
    result = {
        "scenario": args.scenario,
        "objective": args.objective,
        "resolved_reward_mode": args.resolved_reward_mode,
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
        rollouts.append(rollout_controller(
            rlpd,
            eval_protocol.make_env(),
            seed=5000,
            max_steps=args.rollout_steps,
            protocol=eval_protocol,
        ))
    training = {
        "algo": "rlpd",
        "scenario": args.scenario,
        "task": args.task,
        "action_mode": args.action_mode,
        "objective": args.objective,
        "resolved_reward_mode": args.resolved_reward_mode,
        "seed": args.seed,
        "offline_episodes": args.offline_episodes,
        "bc_steps": args.bc_steps,
        "pretrain_updates": args.pretrain_updates,
        "online_steps": args.online_steps,
        "eval_every": args.eval_every,
        "utd": args.utd,
        "n_critics": args.n_critics,
        "checkpoint_path": base + ".pt",
        "onnx_path": base + ".onnx",
    }
    standard_payload = rl_payload(
        kind="rlpd_train_eval",
        scenario=args.scenario,
        objective=eval_protocol.objective,
        action_mode=args.action_mode,
        training=training,
        protocol=eval_protocol.metadata(),
        results=[final_result, pid, mpc],
        rows=[
            result_row(final_result, args.scenario, args.action_mode, controller="RLPD",
                       suite_case=f"{eval_protocol.objective}:{args.scenario}:rlpd"),
            result_row(pid, args.scenario, "actuator", controller="PID",
                       suite_case=f"{eval_protocol.objective}:{args.scenario}:pid"),
            result_row(mpc, args.scenario, "actuator", controller="MPC",
                       suite_case=f"{eval_protocol.objective}:{args.scenario}:mpc"),
        ],
        learning_curve=hist,
        rollouts=rollouts,
        extra={
            "training_runtime": {
                "online_seconds": time.time() - t0,
            },
            "rl_comparison": result,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_rl_artifacts(artifact_dir_for(args, base), standard_payload)
    print(f"saved {base}.pt / .onnx / .json")
    print(f"saved artifacts {artifact_dir_for(args, base)}")


if __name__ == "__main__":
    main()
