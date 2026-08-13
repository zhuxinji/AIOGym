#!/usr/bin/env python3
"""Scenario-oriented AIO-Gym 0.7 command-line interface."""
from __future__ import annotations

import argparse
import json
import sys


_WORKFLOWS = {"collect", "compare", "evaluate", "train"}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="aiogym",
        description="Run Scenario, Benchmark, and Reward workflows.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    listing = commands.add_parser("list", help="list registered resources")
    resources = listing.add_subparsers(dest="resource", metavar="RESOURCE")
    for name in (
        "scenarios",
        "rewards",
        "benchmarks",
        "parameters",
        "controllers",
        "algorithms",
    ):
        item = resources.add_parser(name)
        if name in {"rewards", "benchmarks", "parameters"}:
            item.add_argument("--scenario")
    help_text = {
        "collect": "collect an episode-oriented Dataset",
        "train": "train one SB3 policy",
        "evaluate": "evaluate one policy on explicit seeds",
        "compare": "compare built-in controllers on identical seeds",
    }
    for name, description in help_text.items():
        commands.add_parser(name, help=description, add_help=False)
    return parser


def _list(args):
    import aiogym

    if args.resource == "scenarios":
        values = aiogym.list_scenarios()
    elif args.resource == "rewards":
        if not args.scenario:
            raise ValueError("--scenario is required for rewards")
        values = aiogym.list_rewards(args.scenario)
    elif args.resource == "benchmarks":
        if not args.scenario:
            raise ValueError("--scenario is required for benchmarks")
        benchmark_ids = aiogym.list_benchmarks(args.scenario)
        from aiogym.core.registry import get_scenario

        scenario = get_scenario(args.scenario)
        model = scenario.make_model(None)
        rows = []
        for benchmark_id in benchmark_ids:
            benchmark = scenario.benchmarks[benchmark_id]
            noise = benchmark.measurement_noise
            noise_label = (
                "none"
                if noise is None
                else f"std={noise['std']:g},bias_std={noise['bias_std']:g}"
            )
            rows.append(
                (
                    benchmark_id,
                    str(benchmark.make_episode(model).horizon),
                    scenario.default_reward,
                    "scenario-defaults",
                    noise_label,
                    ",".join(
                        name for name, _direction in benchmark.ranking_metrics
                    ),
                )
            )
        headings = (
            "ID",
            "HORIZON",
            "REWARD",
            "PARAMETERS",
            "MEASUREMENT_NOISE",
            "RANKING_METRICS",
        )
        widths = [
            max(len(headings[index]), *(len(row[index]) for row in rows))
            for index in range(len(headings))
        ]
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(headings)))
        for row in rows:
            print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
        return 0
    elif args.resource == "parameters":
        if not args.scenario:
            raise ValueError("--scenario is required for parameters")
        rows = aiogym.list_parameters(args.scenario)
        rendered = [
            (row["name"], json.dumps(row["default"]), row["unit"])
            for row in rows
        ]
        name_width = max(
            (len(name) for name, _, _ in rendered), default=len("NAME")
        )
        default_width = max(
            (len(default) for _, default, _ in rendered), default=len("DEFAULT")
        )
        print(f"{'NAME':<{name_width}}  {'DEFAULT':<{default_width}}  UNIT")
        for name, default, unit in rendered:
            print(f"{name:<{name_width}}  {default:<{default_width}}  {unit}")
        return 0
    elif args.resource == "controllers":
        values = ("hold", "mpc", "pid", "random")
    elif args.resource == "algorithms":
        from aiogym.workflows.train import ALGORITHMS

        values = ALGORITHMS
    else:
        raise ValueError(
            "choose one of: scenarios, rewards, benchmarks, "
            "parameters, controllers, algorithms"
        )
    print("\n".join(values))
    return 0


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in _WORKFLOWS:
        from .workflows import main as workflow_main

        return workflow_main(raw[0], raw[1:])
    parser = build_parser()
    args = parser.parse_args(raw)
    if args.command == "list":
        try:
            return _list(args)
        except (KeyError, ValueError) as error:
            parser.error(str(error))
    parser.print_help()
    return 0


if __name__ == "__main__":
    main()
