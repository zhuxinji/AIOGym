"""CLI adapters for collect, train, and evaluate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiogym.core import jsonable


def _common(parser):
    parser.add_argument("task")
    parser.add_argument("--plant", type=Path)
    parser.add_argument("--preset")
    parser.add_argument("--output", required=True, type=Path)


def _parser(command):
    parser = argparse.ArgumentParser(prog=f"aiogym {command}")
    if command == "collect":
        _common(parser)
        parser.add_argument("--controller", default="random")
        parser.add_argument("--episodes", type=int, default=1)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--max-steps", type=int)
        parser.add_argument("--resume", action="store_true")
    elif command == "train":
        parser.add_argument("task", nargs="?")
        parser.add_argument("algorithm", nargs="?")
        parser.add_argument("--config", type=Path)
        parser.add_argument("--plant", type=Path)
        parser.add_argument("--preset")
        parser.add_argument("--steps", type=int)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--eval-seeds", nargs="+", type=int)
        parser.add_argument("--algorithm-kwargs", default=None)
        parser.add_argument("--output", type=Path)
        parser.add_argument("--force", action="store_true")
    else:
        _common(parser)
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--controller")
        source.add_argument("--checkpoint", type=Path)
        parser.add_argument("--algorithm", choices=("sac", "ppo", "td3", "ddpg"))
        parser.add_argument("--seeds", nargs="+", type=int, default=(0,))
        parser.add_argument("--max-steps", type=int)
        parser.add_argument("--force", action="store_true")
    return parser


def _train_arguments(args, parser):
    values = {}
    if args.config:
        values = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            parser.error("training config must be a JSON object")
    overrides = {
        "task": args.task,
        "algorithm": args.algorithm,
        "plant": args.plant,
        "preset": args.preset,
        "steps": args.steps,
        "seed": args.seed,
        "eval_seeds": args.eval_seeds,
        "output": args.output,
    }
    values.update({key: value for key, value in overrides.items() if value is not None})
    if args.algorithm_kwargs is not None:
        values["algorithm_kwargs"] = json.loads(args.algorithm_kwargs)
    values["overwrite"] = args.force
    required = {"task", "algorithm", "steps", "output"}
    missing = sorted(required - set(values))
    if missing:
        parser.error("missing training values: " + ", ".join(missing))
    return values


def main(command, argv=None):
    parser = _parser(command)
    args = parser.parse_args(argv)
    try:
        if command == "collect":
            from aiogym import collect

            result = collect(
                task=args.task,
                plant=args.plant,
                preset=args.preset,
                policy=args.controller,
                episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
                output=args.output,
                resume=args.resume,
            )
        elif command == "train":
            from aiogym import train

            result = train(**_train_arguments(args, parser))
        else:
            from aiogym import evaluate

            policy = args.controller
            if args.checkpoint:
                if not args.algorithm:
                    parser.error("--algorithm is required with --checkpoint")
                from aiogym.workflows import load_checkpoint

                policy = load_checkpoint(args.checkpoint, algorithm=args.algorithm)
            result = evaluate(
                policy,
                task=args.task,
                plant=args.plant,
                preset=args.preset,
                seeds=args.seeds,
                max_steps=args.max_steps,
                output=args.output,
                overwrite=args.force,
            )
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(jsonable(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = ["main"]
