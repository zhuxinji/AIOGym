#!/usr/bin/env python3
"""Scenario-oriented AIO-Gym command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_WORKFLOWS = {"collect", "evaluate", "train"}


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
    listing = commands.add_parser("list", help="list built-in resources")
    resources = listing.add_subparsers(dest="resource", metavar="RESOURCE")
    resource_help = {
        "scenarios": "list built-in Scenario ids",
        "rewards": "list Reward goals and success criteria",
        "benchmarks": "list fixed Benchmark protocols and ranking metrics",
        "parameters": "list parameter defaults, units and allowed values",
        "states": "list physical state definitions",
        "actions": "list action ranges and physical interpretations",
        "observations": "list observation sources and normalization rules",
        "outputs": "list controlled output definitions",
        "safety_rules": "list safety conditions and effects",
        "info": "list the complete environment configuration",
        "controllers": "list built-in controller ids",
        "algorithms": "list built-in training algorithm ids",
    }
    for name, description in resource_help.items():
        item = resources.add_parser(name, help=description)
        if name not in {"scenarios", "controllers", "algorithms"}:
            item.add_argument("--scenario", required=True, help="built-in Scenario id")
        if name == "info":
            item.add_argument("--benchmark", help="fixed Benchmark id; describes its template without reset")
        item.add_argument("--json", action="store_true", help="emit structured JSON instead of a table")
    help_text = {
        "collect": "collect an episode-oriented Dataset",
        "train": "train algorithms across independent seeds",
        "evaluate": "evaluate one or more policies on identical seeds",
    }
    for name, description in help_text.items():
        commands.add_parser(name, help=description, add_help=False)
    return parser


def _display_value(value):
    if value is None:
        return "Not provided"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _print_information(value):
    if value is None:
        print("Not provided")
        return
    if isinstance(value, dict):
        if value and all(isinstance(row, dict) for row in value.values()):
            rows = [{"name": name, **row} for name, row in value.items()]
        else:
            rows = [{"field": name, "value": item} for name, item in value.items()]
    else:
        rows = list(value)
    if not rows:
        print("None")
        return
    if isinstance(rows[0], str):
        print("\n".join(rows))
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if "default" in columns and all(row.get("value") == row.get("default") for row in rows):
        columns = [key for key in columns if key != "value"]
    rendered = [[_display_value(row.get(key)) for key in columns] for row in rows]
    headings = [key.upper() for key in columns]
    widths = [max(len(headings[i]), *(len(row[i]) for row in rendered)) for i in range(len(columns))]
    for row in [headings, *rendered]:
        print("  ".join(item.ljust(width) for item, width in zip(row, widths)).rstrip())


def _list(args):
    import aiogym

    if args.resource is None:
        raise ValueError("choose a resource; run 'aiogym list --help'")
    if args.resource == "controllers":
        value = ("hold", "mpc", "pid", "random")
    elif args.resource in {"scenarios", "algorithms"}:
        value = getattr(aiogym, f"list_{args.resource}")()
    elif args.resource == "info":
        value = aiogym.list_info(args.scenario, benchmark=args.benchmark)
    else:
        value = getattr(aiogym, f"list_{args.resource}")(args.scenario)
    if args.json:
        print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    elif args.resource == "info":
        for name, section in value.items():
            print(name.replace("_", " ").title())
            _print_information(section)
            print()
    else:
        _print_information(value)
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
