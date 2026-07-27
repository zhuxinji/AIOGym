#!/usr/bin/env python3
"""Unified command-line entry point for AIO-Gym."""
from __future__ import annotations

import argparse
import sys

from aiogym.catalog import (
    list_controllers,
    list_scenarios,
    list_suites,
    list_tasks,
)


def _benchmark(argv):
    from aiogym.cli.benchmark import main

    return main(argv, prog="aiogym benchmark suite")


def _train_sb3(argv):
    from aiogym.rl.train_sb3 import main

    return main(argv, prog="aiogym train sb3")


def _train_rlpd(argv):
    from aiogym.rl.train_rlpd import main

    return main(argv, prog="aiogym train rlpd")


def _artifact_report(argv):
    from aiogym.cli.artifact_commands import report_main

    return report_main(argv, prog="aiogym artifacts report")


def _artifact_check(argv):
    from aiogym.cli.artifact_commands import artifact_check_main

    return artifact_check_main(argv, prog="aiogym artifacts check")


def _print_items(items):
    for item in items:
        print(item)


def _list_scenarios(_args):
    _print_items(list_scenarios())


def _list_tasks(args):
    _print_items(list_tasks(args.scenario))


def _list_suites(_args):
    _print_items(list_suites())


def _list_controllers(_args):
    _print_items(list_controllers())


def _add_delegate(subparsers, name, help_text, handler):
    parser = subparsers.add_parser(
        name,
        help=help_text,
        description=f"Pass options through to the {help_text} command.",
        add_help=False,
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    parser.set_defaults(handler=lambda args: handler(args.arguments))
    return parser


def build_parser():
    parser = argparse.ArgumentParser(
        prog="aiogym",
        description="Discover resources and run AIO-Gym workflows.",
    )
    parser.set_defaults(selected_parser=parser)
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    list_parser = commands.add_parser("list", help="list canonical resource IDs")
    list_parser.set_defaults(selected_parser=list_parser)
    list_commands = list_parser.add_subparsers(dest="resource", metavar="RESOURCE")

    scenarios = list_commands.add_parser("scenarios", help="registered process scenarios")
    scenarios.set_defaults(handler=_list_scenarios, selected_parser=scenarios)

    tasks = list_commands.add_parser("tasks", help="bundled task profiles")
    tasks.add_argument("--scenario", default=None, help="only list tasks for one scenario")
    tasks.set_defaults(handler=_list_tasks, selected_parser=tasks)

    suites = list_commands.add_parser("suites", help="bundled benchmark suites")
    suites.set_defaults(handler=_list_suites, selected_parser=suites)

    controllers = list_commands.add_parser("controllers", help="registered controllers")
    controllers.set_defaults(handler=_list_controllers, selected_parser=controllers)

    benchmark = commands.add_parser("benchmark", help="run benchmarks")
    benchmark.set_defaults(selected_parser=benchmark)
    benchmark_commands = benchmark.add_subparsers(dest="benchmark_command", metavar="COMMAND")
    _add_delegate(
        benchmark_commands,
        "suite",
        "single-task or multi-task benchmark",
        _benchmark,
    )

    train = commands.add_parser("train", help="train reinforcement-learning agents")
    train.set_defaults(selected_parser=train)
    train_commands = train.add_subparsers(dest="train_command", metavar="BACKEND")
    _add_delegate(train_commands, "sb3", "Stable-Baselines3 trainer", _train_sb3)
    _add_delegate(train_commands, "rlpd", "RLPD trainer", _train_rlpd)

    artifacts = commands.add_parser("artifacts", help="inspect benchmark artifacts")
    artifacts.set_defaults(selected_parser=artifacts)
    artifact_commands = artifacts.add_subparsers(dest="artifact_command", metavar="COMMAND")
    _add_delegate(artifact_commands, "report", "artifact report", _artifact_report)
    _add_delegate(artifact_commands, "check", "artifact validator", _artifact_check)

    return parser


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    delegated_commands = {
        ("benchmark", "suite"): _benchmark,
        ("train", "sb3"): _train_sb3,
        ("train", "rlpd"): _train_rlpd,
        ("artifacts", "report"): _artifact_report,
        ("artifacts", "check"): _artifact_check,
    }
    for route, handler in delegated_commands.items():
        if tuple(raw_args[:len(route)]) == route:
            return handler(raw_args[len(route):])

    parser = build_parser()
    args = parser.parse_args(raw_args)
    handler = getattr(args, "handler", None)
    if handler is None:
        args.selected_parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    main()
