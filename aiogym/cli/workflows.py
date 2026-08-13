"""Thin command-line adapters for collect, train, evaluate, and compare."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiogym.core.io import jsonable


def _environment_arguments(parser):
    parser.add_argument("scenario")
    parser.add_argument("--reward")
    parser.add_argument("--parameters", type=Path)
    parser.add_argument("--benchmark")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--noise", choices=("on", "off"), default="off")
    parser.add_argument("--delay", choices=("on", "off"), default="off")
    parser.add_argument("--fault", choices=("on", "off"), default="off")


def _parser(command):
    parser = argparse.ArgumentParser(prog=f"aiogym {command}")
    _environment_arguments(parser)
    if command == "collect":
        parser.add_argument("--controller", default="random")
        parser.add_argument("--episodes", type=int, default=1)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--max-steps", type=int)
        parser.add_argument("--output", required=True, type=Path)
    elif command == "train":
        parser.add_argument("algorithm", choices=("sac", "ppo", "td3", "ddpg"))
        parser.add_argument("--steps", required=True, type=int)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--algorithm-kwargs", type=Path)
        parser.add_argument("--record-every", type=int, default=500)
        parser.add_argument("--output", required=True, type=Path)
    elif command == "evaluate":
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--controller")
        source.add_argument("--checkpoint", type=Path)
        parser.add_argument("--algorithm", choices=("sac", "ppo", "td3", "ddpg"))
        parser.add_argument("--seeds", nargs="+", required=True, type=int)
        parser.add_argument("--max-steps", type=int)
        parser.add_argument("--output", required=True, type=Path)
    else:
        parser.add_argument(
            "--controllers",
            nargs="+",
            required=True,
            choices=("hold", "mpc", "pid", "random"),
        )
        parser.add_argument("--seeds", nargs="+", required=True, type=int)
        parser.add_argument("--max-steps", type=int)
        parser.add_argument("--output", type=Path)
    return parser


def _json_object_file(path, parser, label):
    if path is None:
        return None
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        parser.error(f"could not read {label} {path}: {error}")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        parser.error(
            f"{label} is not valid JSON: {error.msg} "
            f"(line {error.lineno}, column {error.colno})"
        )
    if not isinstance(value, dict):
        parser.error(f"{label} must be a JSON object")
    return value


def main(command, argv=None):
    parser = _parser(command)
    args = parser.parse_args(argv)
    parameters = _json_object_file(args.parameters, parser, "parameters")
    env = None
    try:
        import aiogym

        env = aiogym.make_env(
            args.scenario,
            reward=args.reward,
            parameters=parameters,
            benchmark=args.benchmark,
            randomize=args.randomize,
            noise=args.noise == "on",
            delay=args.delay == "on",
            fault=args.fault == "on",
        )
        if command == "collect":
            result = aiogym.collect(
                env=env,
                policy=args.controller,
                episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
                output=args.output,
            )
        elif command == "train":
            algorithm_kwargs = _json_object_file(
                args.algorithm_kwargs,
                parser,
                "algorithm kwargs",
            )
            result = aiogym.train(
                env=env,
                algorithm=args.algorithm,
                steps=args.steps,
                seed=args.seed,
                algorithm_kwargs=algorithm_kwargs,
                record_every=args.record_every,
                output=args.output,
            )
        elif command == "evaluate":
            if args.checkpoint is not None:
                if args.algorithm is None:
                    parser.error("--algorithm is required with --checkpoint")
                policy = aiogym.load_policy(
                    args.checkpoint,
                    algorithm=args.algorithm,
                    env=env,
                )
            else:
                if args.algorithm is not None:
                    parser.error("--algorithm is only valid with --checkpoint")
                policy = args.controller
            result = aiogym.evaluate(
                env=env,
                policy=policy,
                seeds=args.seeds,
                max_steps=args.max_steps,
                output=args.output,
            )
        else:
            if len(set(args.controllers)) != len(args.controllers):
                parser.error("--controllers must not contain duplicates")
            result = aiogym.compare_policies(
                env=env,
                policies={name: name for name in args.controllers},
                seeds=args.seeds,
                max_steps=args.max_steps,
                output=args.output,
            )
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    finally:
        if env is not None:
            env.close()
    print(json.dumps(jsonable(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = ["main"]
