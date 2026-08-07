#!/usr/bin/env python3
"""Five-command AIO-Gym 0.2 command-line interface."""
from __future__ import annotations

import argparse
import sys


_WORKFLOWS = {"collect", "evaluate", "train"}
_REMOVED = {
    "artifacts",
    "benchmark",
    "describe",
    "final-test",
    "run",
    "tune",
}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="aiogym",
        description="Design plants and run Task-oriented process-control workflows.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    listing = commands.add_parser("list", help="list registered resources")
    resources = listing.add_subparsers(dest="resource", metavar="RESOURCE")
    for name in (
        "scenarios",
        "tasks",
        "plants",
        "conditions",
        "controllers",
        "algorithms",
    ):
        item = resources.add_parser(name)
        if name in {"tasks", "plants", "conditions"}:
            item.add_argument("--scenario")
        if name == "conditions":
            item.add_argument("--plant")
    help_text = {
        "design": "validate plants and run design studies",
        "collect": "collect an episode-oriented Dataset v4",
        "train": "train one SB3 policy",
        "evaluate": "evaluate one policy on explicit seeds",
    }
    for name, description in help_text.items():
        commands.add_parser(name, help=description, add_help=False)
    return parser


def _list(args):
    import aiogym

    if args.resource == "scenarios":
        values = aiogym.list_scenarios()
    elif args.resource == "tasks":
        values = aiogym.list_tasks(args.scenario)
    elif args.resource == "plants":
        if not args.scenario:
            raise ValueError("--scenario is required for plants")
        values = aiogym.list_plants(args.scenario)
    elif args.resource == "conditions":
        if not args.scenario:
            raise ValueError("--scenario is required for conditions")
        values = aiogym.list_conditions(args.scenario, args.plant)
    elif args.resource == "controllers":
        values = ("hold", "mpc", "pid", "random", "sb3")
    elif args.resource == "algorithms":
        from aiogym.workflows.train import ALGORITHMS

        values = ALGORITHMS
    else:
        raise ValueError(
            "choose one of: scenarios, tasks, plants, conditions, "
            "controllers, algorithms"
        )
    print("\n".join(values))
    return 0


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] in _REMOVED:
        raise SystemExit(
            f"aiogym {raw[0]} was removed in 0.2; use design, collect, train, "
            "or evaluate (see docs/migration-v0.2.md)"
        )
    if raw and raw[0] == "design":
        from .design import main as design_main

        return design_main(raw[1:], prog="aiogym design")
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
