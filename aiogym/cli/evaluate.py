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
from aiogym.rl.run_reference import load_run_reference


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Evaluate a frozen checkpoint on validation only.",
    )
    parser.add_argument("run", nargs="?")
    parser.add_argument("--checkpoint")
    parser.add_argument("--track")
    parser.add_argument(
        "--algorithm",
        choices=SUPPORTED_POLICY_ALGORITHMS,
        required=False,
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
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    run_reference = None
    if args.run is not None:
        if any(
            value is not None
            for value in (
                args.checkpoint,
                args.track,
                args.algorithm,
                args.checkpoint_sha256,
            )
        ):
            parser.error(
                "RUN is mutually exclusive with --checkpoint, --track, "
                "--algorithm, and --sha256"
            )
        run_reference = load_run_reference(args.run)
        checkpoint = run_reference.policy_path
        track = load_track(run_reference.track_id)
        algorithm_id = run_reference.algorithm_id
        digest = run_reference.policy_sha256
    else:
        if not args.checkpoint or not args.track or not args.algorithm:
            parser.error(
                "provide RUN or --checkpoint FILE --track TRACK --algorithm ALGO"
            )
        checkpoint = Path(args.checkpoint)
        track = load_track(args.track)
        algorithm_id = args.algorithm
        digest = args.checkpoint_sha256 or checkpoint_sha256(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    policy_spec = learned_policy_spec_for_track(
        checkpoint,
        algorithm_id,
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
                "run_reference": (
                    None
                    if run_reference is None
                    else str(run_reference.manifest_path)
                ),
                "output": args.output,
                "next_command": "aiogym final-test --config FILE",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = ["build_parser", "main"]
