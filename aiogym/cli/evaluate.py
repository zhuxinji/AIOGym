"""Validation-only checkpoint evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiogym.benchmarks.evaluation import evaluate_policy_on_track
from aiogym.benchmarks.tracks.registry import load_track
from aiogym.controllers.checkpoints import (
    SUPPORTED_POLICY_ALGORITHMS,
    checkpoint_sha256,
    learned_policy_spec_for_track,
    load_policy_checkpoint,
)
from aiogym._internal.config import parse_csv_ints
from aiogym._internal.serialization import write_json_artifact
from aiogym._internal.validation import seed_sequence


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Evaluate a frozen checkpoint on validation only.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--track", required=True)
    parser.add_argument(
        "--algorithm",
        choices=SUPPORTED_POLICY_ALGORITHMS,
        required=True,
    )
    parser.add_argument(
        "--sha256",
        dest="checkpoint_sha256",
        default=None,
        help=(
            "expected lowercase checkpoint digest; when omitted, pin the "
            "file's current digest"
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default="5000")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main(argv=None, prog: str | None = None) -> int:
    args = build_parser(prog).parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    track = load_track(args.track)
    digest = args.checkpoint_sha256 or checkpoint_sha256(checkpoint)
    policy_spec = learned_policy_spec_for_track(
        checkpoint,
        args.algorithm,
        digest,
        track,
    )
    controller = load_policy_checkpoint(
        policy_spec,
        device=args.device,
    )
    seeds = seed_sequence(
        "--seeds", parse_csv_ints(args.seeds, option="--seeds")
    )
    result = evaluate_policy_on_track(
        controller,
        track,
        base_seeds=seeds,
        include_episodes=True,
    )
    if args.output:
        write_json_artifact(
            args.output,
            result,
            overwrite=args.overwrite,
        )
    print(
        json.dumps(
            {
                "track_id": result["track_id"],
                "track_hash": result["track_hash"],
                "split": result["split"],
                "seed_namespace": result["seed_namespace"],
                "base_seeds": result["base_seeds"],
                "case_count": result["case_count"],
                "aggregate": result["aggregate"],
                "checkpoint_sha256": digest,
                "output": args.output,
                "next_command": "aiogym final-test --config FILE",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = ["build_parser", "main"]
