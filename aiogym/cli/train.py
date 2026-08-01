"""Config-first unified training command."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.runner import run_experiment, run_seed_sweep
from aiogym._internal.validation import seed_sequence
from aiogym._internal.config import parse_csv_ints


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train the algorithm declared by one immutable config.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", default=None)
    group.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    config = RLTrainingConfig.load(args.config)
    output = dict(config.output)
    if args.output is not None:
        output["directory"] = args.output
    config = replace(
        config,
        training_seed=(
            config.training_seed if args.seed is None else args.seed
        ),
        device=config.device if args.device is None else args.device,
        output=output,
        resume_checkpoint=(
            config.resume_checkpoint
            if args.resume is None
            else args.resume
        ),
    )
    if args.seeds is not None:
        if config.resume_checkpoint:
            raise ValueError(
                "--resume cannot be combined with --seeds; multi-seed "
                "sweeps are independent fresh runs"
            )
        if args.seed is not None:
            raise ValueError("--seed and --seeds are mutually exclusive")
        result = run_seed_sweep(
            config,
            _seed_csv(args.seeds),
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = run_experiment(config, overwrite=bool(args.overwrite))
    payload = {
        **result.as_dict(),
        "next_command": (
            "aiogym evaluate "
            f"--checkpoint {result.policy_path} "
            f"--track {result.track_id} "
            f"--algorithm {result.algorithm_id}"
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _seed_csv(value: str) -> tuple[int, ...]:
    return seed_sequence(
        "--seeds", parse_csv_ints(str(value), option="--seeds")
    )


__all__ = ["build_parser", "main"]
