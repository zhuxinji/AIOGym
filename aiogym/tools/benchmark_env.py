"""CLI for establishing simulator throughput baselines."""
from __future__ import annotations

import argparse
import json

from aiogym.benchmarks import load_track
from aiogym.rl.episode_env import make_track_training_env
from aiogym.rl.profiler import profile_environments


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--track",
        default="quadruple-regulation-generalist-v1",
    )
    parser.add_argument("--n-envs", default="1,2,4,8,16")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args(argv)
    widths = tuple(int(value) for value in args.n_envs.split(","))
    if any(value <= 0 for value in widths):
        parser.error("--n-envs values must be positive")
    track = load_track(args.track)
    reports = []
    for width in widths:
        report = profile_environments(
            lambda worker: make_track_training_env(
                track,
                base_seed=args.seed,
                worker_index=worker,
                info_level="minimal",
                profile_timing=True,
            ),
            n_envs=width,
            transitions=args.steps,
            seed=args.seed,
        )
        reports.append(report.as_dict())
    print(
        json.dumps(
            {
                "track_id": track.id,
                "seed": args.seed,
                "profiles": reports,
                "component_timing_note": (
                    "dispatch overhead is synchronous Python dispatch; use "
                    "the trainer profiler to measure subprocess IPC"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
