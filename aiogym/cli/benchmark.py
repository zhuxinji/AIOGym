#!/usr/bin/env python3
"""Benchmark CLI for official Tracks and explicit scenario/Case runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from aiogym._internal.config import parse_seed_list
from aiogym._internal.serialization import jsonable
from aiogym._internal.vocabulary import GOAL_NAMES
from aiogym.benchmarks import evaluate_policy_on_track, list_tracks, load_track
from aiogym.catalog import list_cases, list_scenarios
from aiogym.controllers import make_controller
from aiogym.env import AIOGymEnv
from aiogym.evaluation import evaluate_controller
from aiogym.models import make_model
from aiogym.models.cases import case_identity, load_case
from aiogym.rewards import get_reward_spec


DEFAULT_CONTROLLERS = ("pid", "mpc", "oracle")


def _controller_names(raw: str | None) -> tuple[str, ...]:
    names = tuple(
        part.strip().lower()
        for part in (raw or ",".join(DEFAULT_CONTROLLERS)).split(",")
        if part.strip()
    )
    if not names:
        raise ValueError("--controllers must include at least one controller")
    return names


def _controller_config(args, name: str) -> dict:
    if name == "sb3":
        if not args.sb3_path:
            raise ValueError("controller 'sb3' requires --sb3-path")
        return {
            "path": args.sb3_path,
            "algo": args.sb3_algo,
            "action_mode": args.policy_action_mode,
        }
    if name == "onnx":
        if not args.onnx_path:
            raise ValueError("controller 'onnx' requires --onnx-path")
        return {
            "path": args.onnx_path,
            "action_mode": args.policy_action_mode,
        }
    config = {}
    if args.controller_profile:
        config["profile"] = args.controller_profile
    return config


def _make_controller(args, name: str, scenario: str):
    return make_controller(
        name,
        model=make_model(scenario),
        scenario=scenario,
        config=_controller_config(args, name),
    )


def _direct_case_result(args, controller_name: str, seeds: tuple[int, ...]):
    profile = (
        load_case(args.case, scenario=args.target)
        if args.case is not None
        else None
    )
    env = AIOGymEnv(
        args.target,
        case=profile,
        reward_spec=args.reward_spec,
        action_mode=(
            args.policy_action_mode
            if controller_name in {"sb3", "onnx"}
            else "actuator"
        ),
        episode_steps=args.episode_steps,
        control_dt=args.control_dt,
    )
    controller = _make_controller(args, controller_name, args.target)
    try:
        result = evaluate_controller(
            controller,
            env,
            episodes=len(seeds),
            seed_list=seeds,
            goal_specification=args.goal,
            include_episodes=args.include_episodes,
        )
    finally:
        env.close()
    identity = case_identity(profile)
    result.update(
        {
            "case_id": identity["name"],
            "resolved_case_hash": identity["profile_hash"],
            "policy_scope": "specialist",
        }
    )
    return result


def _track_result(args, controller_name: str, seeds: tuple[int, ...]):
    track = load_track(args.target)
    controller = _make_controller(args, controller_name, track.scenario)
    return evaluate_policy_on_track(
        controller,
        track,
        split=args.split,
        base_seeds=seeds,
        include_episodes=args.include_episodes,
    )


def _print_result(controller_name: str, payload: dict) -> None:
    aggregate = payload.get("aggregate")
    if aggregate:
        print(
            f"{controller_name:14s} "
            f"{aggregate['metric']}={aggregate['metric_value']:.6g} "
            f"eligible={aggregate['ranking_eligible']}"
        )
        return
    print(
        f"{controller_name:14s} "
        f"{payload['metric']}={float(payload[payload['metric']]):.6g} "
        f"eligible={payload.get('ranking_eligible', True)}"
    )


def _write_payload(path: str, payload: dict, *, overwrite: bool) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(jsonable(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")


def build_parser(prog=None):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Evaluate an official Track or one explicit Scenario/Case pair."
        ),
    )
    parser.add_argument(
        "target",
        metavar="TRACK_OR_SCENARIO",
        help=(
            f"Track ({', '.join(list_tracks())}) or scenario "
            f"({', '.join(list_scenarios())})"
        ),
    )
    parser.add_argument(
        "case",
        nargs="?",
        metavar="CASE",
        help="Case ID/name for a direct scenario run",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--goal", choices=GOAL_NAMES)
    parser.add_argument("--reward-spec")
    parser.add_argument("--controllers")
    parser.add_argument("--controller-profile")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--seed-list")
    parser.add_argument("--episode-steps", type=int)
    parser.add_argument("--control-dt", type=float)
    parser.add_argument("--sb3-path")
    parser.add_argument("--sb3-algo", default="sac", choices=("sac", "ppo", "td3"))
    parser.add_argument("--onnx-path")
    parser.add_argument(
        "--policy-action-mode",
        choices=("actuator", "setpoint"),
        default="setpoint",
    )
    parser.add_argument("--include-episodes", action="store_true")
    parser.add_argument("--output", help="write the complete JSON result to this path")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv=None, prog=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(prog)
    if not raw_args:
        parser.print_help()
        return 0
    args = parser.parse_args(raw_args)
    tracks = set(list_tracks())
    scenarios = set(list_scenarios())
    is_track = args.target in tracks
    if not is_track and args.target not in scenarios:
        parser.error(f"unknown Track or scenario: {args.target}")
    if is_track:
        if args.case is not None:
            parser.error("Track evaluation does not accept a separate CASE")
        forbidden = {
            "--goal": args.goal,
            "--reward-spec": args.reward_spec,
            "--episode-steps": args.episode_steps,
            "--control-dt": args.control_dt,
        }
        conflicts = [name for name, value in forbidden.items() if value is not None]
        if conflicts:
            parser.error(
                "Track owns goal, reward and environment contract; remove "
                + ", ".join(conflicts)
            )
    else:
        if args.case is not None:
            available = {
                case_id.split("/", 1)[1]
                for case_id in list_cases(args.target)
            }
            if args.case not in available and args.case not in set(
                list_cases(args.target)
            ):
                parser.error(f"unknown Case for {args.target}: {args.case}")
        if args.reward_spec is None:
            args.goal = args.goal or "regulation"
            args.reward_spec = f"{args.goal}-v1"
        try:
            reward_spec = get_reward_spec(args.reward_spec)
        except ValueError as exc:
            parser.error(str(exc))
        if args.goal is not None and reward_spec.goal != args.goal:
            parser.error("--goal conflicts with --reward-spec")
        args.goal = reward_spec.goal

    try:
        seeds = tuple(parse_seed_list(args.seed_list, args.seed, args.episodes))
        controllers = _controller_names(args.controllers)
        results = {
            name: (
                _track_result(args, name, seeds)
                if is_track
                else _direct_case_result(args, name, seeds)
            )
            for name in controllers
        }
    except (FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    for name, payload in results.items():
        _print_result(name, payload)
    output = {
        "benchmark": "track" if is_track else "case",
        "target": args.target,
        "case": args.case,
        "controllers": list(controllers),
        "seed_list": list(seeds),
        "results": results,
    }
    if args.output:
        try:
            _write_payload(args.output, output, overwrite=args.overwrite)
        except (FileExistsError, OSError) as exc:
            parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
