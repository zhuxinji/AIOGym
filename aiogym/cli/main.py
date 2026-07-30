#!/usr/bin/env python3
"""Unified command-line entry point for AIO-Gym."""
from __future__ import annotations

import argparse
import sys

from aiogym.catalog import (
    list_cases,
    list_controllers,
    list_scenarios,
)
from aiogym.benchmarks import list_tracks


def _benchmark(argv):
    from aiogym.cli.benchmark import main

    return main(argv, prog="aiogym benchmark")


def _train(argv):
    from aiogym.cli.train import main

    return main(argv, prog="aiogym train")


def _evaluate(argv):
    from aiogym.cli.evaluate import main

    return main(argv, prog="aiogym evaluate")


def _tune(argv):
    from aiogym.cli.tune import main

    return main(argv, prog="aiogym tune")


def _collect(argv):
    from aiogym.datasets.collect import main

    return main(argv, prog="aiogym collect")


def _final_test(argv):
    from aiogym.cli.final_test import main

    return main(argv, prog="aiogym final-test")


def _artifact_report(argv):
    from aiogym.cli.artifact_commands import report_main

    return report_main(argv, prog="aiogym artifacts report")


def _artifact_check(argv):
    from aiogym.cli.artifact_commands import artifact_check_main

    return artifact_check_main(argv, prog="aiogym artifacts check")


def _artifact_compact(argv):
    from aiogym.cli.artifact_commands import compact_main

    return compact_main(argv, prog="aiogym artifacts compact")


def _print_items(items):
    for item in items:
        print(item)


def _list_scenarios(_args):
    _print_items(list_scenarios())


def _list_cases(args):
    _print_items(list_cases(args.scenario))


def _list_tracks(_args):
    _print_items(list_tracks())


def _list_controllers(_args):
    _print_items(list_controllers())


def _list_algorithms(_args):
    from aiogym.rl import list_algorithms

    _print_items(list_algorithms())


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

    cases = list_commands.add_parser("cases", help="bundled Case v2 profiles")
    cases.add_argument("--scenario", default=None, help="only list cases for one scenario")
    cases.set_defaults(handler=_list_cases, selected_parser=cases)

    tracks = list_commands.add_parser("tracks", help="official benchmark tracks")
    tracks.set_defaults(handler=_list_tracks, selected_parser=tracks)

    controllers = list_commands.add_parser("controllers", help="registered controllers")
    controllers.set_defaults(handler=_list_controllers, selected_parser=controllers)

    algorithms = list_commands.add_parser(
        "algorithms",
        help="stable training algorithms",
    )
    algorithms.set_defaults(handler=_list_algorithms, selected_parser=algorithms)

    benchmark = commands.add_parser("benchmark", help="run benchmarks")
    benchmark.set_defaults(selected_parser=benchmark)
    _add_delegate(
        commands,
        "collect",
        "Dataset v2 collector",
        _collect,
    )
    _add_delegate(
        commands,
        "final-test",
        "locked one-shot final test",
        _final_test,
    )
    _add_delegate(
        commands,
        "evaluate",
        "validation-only checkpoint evaluation",
        _evaluate,
    )
    _add_delegate(
        commands,
        "tune",
        "validation-only hyperparameter tuning",
        _tune,
    )

    _add_delegate(
        commands,
        "train",
        "config-first reinforcement-learning training",
        _train,
    )

    artifacts = commands.add_parser("artifacts", help="inspect benchmark artifacts")
    artifacts.set_defaults(selected_parser=artifacts)
    artifact_commands = artifacts.add_subparsers(dest="artifact_command", metavar="COMMAND")
    _add_delegate(artifact_commands, "report", "artifact report", _artifact_report)
    _add_delegate(artifact_commands, "check", "artifact validator", _artifact_check)
    _add_delegate(
        artifact_commands,
        "compact",
        "artifact compactor",
        _artifact_compact,
    )

    return parser


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    delegated_commands = {
        ("benchmark",): _benchmark,
        ("train",): _train,
        ("evaluate",): _evaluate,
        ("tune",): _tune,
        ("collect",): _collect,
        ("final-test",): _final_test,
        ("artifacts", "report"): _artifact_report,
        ("artifacts", "check"): _artifact_check,
        ("artifacts", "compact"): _artifact_compact,
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
