"""Config-first unified training command."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.profiles import build_training_config
from aiogym.rl.runner import run_experiment, run_seed_sweep
from aiogym._internal.validation import nonnegative_int, seed_sequence
from aiogym._internal.config import parse_csv_ints


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Train from a guided profile or immutable config.",
    )
    parser.add_argument("target", nargs="?")
    parser.add_argument("algorithm", nargs="?")
    parser.add_argument("--config")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", default=None)
    group.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    requested_target = None
    profile_id = None
    if args.config is not None:
        if any(
            value is not None
            for value in (
                args.target,
                args.algorithm,
                args.profile,
                args.dataset,
            )
        ):
            parser.error(
                "--config is mutually exclusive with TARGET, ALGORITHM, "
                "--profile, and --dataset"
            )
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
    else:
        if args.target is None or args.algorithm is None:
            parser.error("guided mode requires TARGET and ALGORITHM")
        requested_target = args.target
        selected_profile = args.profile or "quick"
        seed = nonnegative_int(
            "--seed",
            0 if args.seed is None else args.seed,
        )
        config, profile = build_training_config(
            args.target,
            args.algorithm,
            selected_profile,
            training_seed=seed,
            device=args.device,
            output_directory=args.output,
            dataset_path=args.dataset,
            resume_checkpoint=args.resume,
        )
        profile_id = profile.id
        _print_resolution(args.target, config, profile)
    if args.seeds is not None and args.seed is not None:
        raise ValueError("--seed and --seeds are mutually exclusive")
    if args.seeds is not None and config.resume_checkpoint:
        raise ValueError(
            "--resume cannot be combined with --seeds; multi-seed "
            "sweeps are independent fresh runs"
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "requested_target": requested_target,
                    "profile_id": profile_id,
                    "track_id": config.track_id,
                    "reward_spec_id": _track_reward(config.track_id),
                    "algorithm_id": config.algorithm_id,
                    "budget_unit": config.budget_unit,
                    "budget_value": config.budget_value,
                    "config_hash": config.config_hash,
                    "seeds": (
                        list(_seed_csv(args.seeds))
                        if args.seeds is not None
                        else [config.training_seed]
                    ),
                    "config": config.as_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.seeds is not None:
        result = run_seed_sweep(
            config,
            _seed_csv(args.seeds),
            overwrite=bool(args.overwrite),
        )
        print(
            json.dumps(
                {
                    **result,
                    "requested_target": requested_target,
                    "profile_id": profile_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    result = run_experiment(config, overwrite=bool(args.overwrite))
    payload = {
        **result.as_dict(),
        "requested_target": requested_target,
        "profile_id": profile_id,
        "next_command": f"aiogym evaluate {_run_result_reference(result)}",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _seed_csv(value: str) -> tuple[int, ...]:
    return seed_sequence(
        "--seeds", parse_csv_ints(str(value), option="--seeds")
    )


def _track_reward(track_id: str) -> str:
    from aiogym.benchmarks.tracks.registry import load_track

    return load_track(track_id, validate_policy_contract=False).reward_spec_id


def _run_result_reference(result) -> str:
    from pathlib import Path

    resolved = Path(result.resolved_config_path)
    name = resolved.name
    if name.endswith(".resolved.json"):
        name = name.removesuffix(".resolved.json")
    elif name.endswith(".json"):
        name = name.removesuffix(".json")
    return str(resolved.with_name(f"{name}.run-result.json"))


def _print_resolution(target, config, profile) -> None:
    print(f"Requested target: {target}", file=sys.stderr)
    print(f"Resolved Track:   {config.track_id}", file=sys.stderr)
    print(f"Reward:           {_track_reward(config.track_id)}", file=sys.stderr)
    print(f"Algorithm:        {config.algorithm_id}", file=sys.stderr)
    print(f"Profile:          {profile.id}", file=sys.stderr)
    print(
        f"Budget:           {config.budget_value} {config.budget_unit}",
        file=sys.stderr,
    )
    if profile.selector == "quick":
        print(
            "Profile quick-v1 is intended for tutorials and smoke validation. "
            "No verified baseline profile is registered; use --config for "
            "standard experiments.",
            file=sys.stderr,
        )


__all__ = ["build_parser", "main"]
