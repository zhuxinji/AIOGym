#!/usr/bin/env python3
"""Scenario-oriented AIO-Gym command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_WORKFLOWS = {"collect", "compare", "evaluate", "train"}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="aiogym",
        description="Run Scenario, Benchmark, and Reward workflows.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    status = commands.add_parser("status", help="inspect training run status")
    status.add_argument("paths", nargs="*", help="run directories or status.json files; omit to discover recent runs under ./runs")
    status.add_argument("--limit", type=int, default=10, help="number of recent runs when paths are omitted (default: 10)")
    status.add_argument("--json", action="store_true", help="emit structured status and progress instead of a table")
    listing = commands.add_parser("list", help="list registered resources")
    resources = listing.add_subparsers(dest="resource", metavar="RESOURCE")
    resource_help = {
        "scenarios": "list registered Scenario ids",
        "rewards": "list Reward ids for one Scenario",
        "benchmarks": "list fixed Benchmarks for one Scenario",
        "parameters": "list model parameter defaults and units",
        "controllers": "list built-in controller ids",
        "algorithms": "list registered training algorithm ids",
    }
    for name, description in resource_help.items():
        item = resources.add_parser(name, help=description)
        if name in {"rewards", "benchmarks", "parameters"}:
            item.add_argument("--scenario", help="registered Scenario id")
    help_text = {
        "collect": "collect an episode-oriented Dataset",
        "train": "train algorithms across independent seeds",
        "evaluate": "evaluate one policy on explicit seeds",
        "compare": "compare controllers and checkpoints on identical seeds",
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
            rows.append(
                (
                    benchmark_id,
                    str(benchmark.make_episode(model, 0).horizon),
                    benchmark.reward_id,
                    "scenario-defaults",
                    ",".join(name for name, _direction in benchmark.ranking_metrics),
                )
            )
        headings = (
            "ID",
            "HORIZON",
            "REWARD",
            "PARAMETERS",
            "RANKING_METRICS",
        )
        widths = [
            max(len(headings[index]), *(len(row[index]) for row in rows))
            for index in range(len(headings))
        ]
        print(
            "  ".join(
                value.ljust(widths[index]) for index, value in enumerate(headings)
            )
        )
        for row in rows:
            print(
                "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            )
        return 0
    elif args.resource == "parameters":
        if not args.scenario:
            raise ValueError("--scenario is required for parameters")
        rows = aiogym.list_parameters(args.scenario)
        rendered = [
            (row["name"], json.dumps(row["default"]), row["unit"]) for row in rows
        ]
        name_width = max((len(name) for name, _, _ in rendered), default=len("NAME"))
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
        values = aiogym.list_algorithms()
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
    if args.command == "status":
        from ._train_job import print_training_status, training_progress

        try:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            paths = args.paths
            if not paths:
                paths = sorted(
                    Path("runs").glob("*/training/**/status.json"),
                    key=lambda path: max(path.stat().st_mtime,
                        (path.parent / "training_curve.json").stat().st_mtime
                        if (path.parent / "training_curve.json").is_file() else 0),
                    reverse=True,
                )[:args.limit]
            jobs = [training_progress(path) for path in paths]
            if args.json:
                print(json.dumps(jobs, indent=2))
            else:
                print_training_status(jobs)
        except (OSError, KeyError, ValueError) as error:
            parser.error(str(error))
        return 0
    if args.command == "list":
        try:
            return _list(args)
        except (KeyError, ValueError) as error:
            parser.error(str(error))
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
