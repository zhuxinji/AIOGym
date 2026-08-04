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


def _list_tracks(args):
    from aiogym.cli.describe import render_track_catalog

    print(render_track_catalog(ids=args.ids, json_output=args.json_output))


def _list_rewards(args):
    from aiogym.cli.describe import render_reward_catalog

    print(render_reward_catalog(json_output=args.json_output))


def _list_profiles(args):
    from aiogym.cli.describe import render_profile_catalog

    try:
        output = render_profile_catalog(
            kind=args.kind,
            target=args.target,
            algorithm=args.algorithm,
            json_output=args.json_output,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        args.selected_parser.error(str(exc))
    print(output)


def _describe_track(args):
    from aiogym.cli.describe import render_track_description

    try:
        output = render_track_description(
            args.target,
            json_output=args.json_output,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        args.selected_parser.error(str(exc))
    print(output)


def _describe_reward(args):
    from aiogym.cli.describe import render_reward_description

    try:
        output = render_reward_description(
            args.reward,
            json_output=args.json_output,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        args.selected_parser.error(str(exc))
    print(output)


def _describe_profile(args):
    from aiogym.cli.describe import render_profile_description

    try:
        output = render_profile_description(
            args.profile,
            target=args.target,
            algorithm=args.algorithm,
            json_output=args.json_output,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        args.selected_parser.error(str(exc))
    print(output)


def _list_controllers(_args):
    _print_items(list_controllers())


def _list_algorithms(_args):
    from aiogym.rl.config import list_algorithms

    _print_items(list_algorithms())


_COMMANDS = {
    "benchmark": _benchmark,
    "collect": _collect,
    "evaluate": _evaluate,
    "final-test": _final_test,
    "train": _train,
    "tune": _tune,
}

_ARTIFACT_COMMANDS = {
    "check": _artifact_check,
    "compact": _artifact_compact,
    "report": _artifact_report,
}

_COMMAND_HELP = {
    "benchmark": "run an official Track benchmark",
    "collect": "collect a Dataset v2 bundle",
    "evaluate": "evaluate one checkpoint on validation seeds",
    "final-test": "run a locked one-shot final test",
    "train": "train from one resolved configuration",
    "tune": "tune on validation seeds",
}


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
    track_output = tracks.add_mutually_exclusive_group()
    track_output.add_argument(
        "--ids",
        action="store_true",
        help="print canonical IDs only",
    )
    track_output.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="print structured JSON",
    )
    tracks.set_defaults(handler=_list_tracks, selected_parser=tracks)

    rewards = list_commands.add_parser(
        "rewards",
        help="canonical reward specifications",
    )
    rewards.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="print structured JSON",
    )
    rewards.set_defaults(handler=_list_rewards, selected_parser=rewards)

    profiles = list_commands.add_parser(
        "profiles",
        help="versioned training and collection profiles",
    )
    profiles.add_argument(
        "--kind",
        choices=("training", "collection"),
    )
    profiles.add_argument("--target")
    profiles.add_argument("--algorithm")
    profiles.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="print structured JSON",
    )
    profiles.set_defaults(handler=_list_profiles, selected_parser=profiles)

    controllers = list_commands.add_parser("controllers", help="registered controllers")
    controllers.set_defaults(handler=_list_controllers, selected_parser=controllers)

    algorithms = list_commands.add_parser(
        "algorithms",
        help="stable training algorithms",
    )
    algorithms.set_defaults(handler=_list_algorithms, selected_parser=algorithms)

    describe = commands.add_parser(
        "describe",
        help="describe a canonical resource",
    )
    describe.set_defaults(selected_parser=describe)
    describe_commands = describe.add_subparsers(
        dest="describe_resource",
        metavar="RESOURCE",
    )

    describe_track = describe_commands.add_parser(
        "track",
        help="describe an official Track",
    )
    describe_track.add_argument("target")
    describe_track.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
    )
    describe_track.set_defaults(
        handler=_describe_track,
        selected_parser=describe_track,
    )

    describe_reward = describe_commands.add_parser(
        "reward",
        help="describe a canonical reward",
    )
    describe_reward.add_argument("reward")
    describe_reward.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
    )
    describe_reward.set_defaults(
        handler=_describe_reward,
        selected_parser=describe_reward,
    )

    describe_profile = describe_commands.add_parser(
        "profile",
        help="describe a versioned profile",
    )
    describe_profile.add_argument("profile")
    describe_profile.add_argument("--target")
    describe_profile.add_argument("--algorithm")
    describe_profile.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
    )
    describe_profile.set_defaults(
        handler=_describe_profile,
        selected_parser=describe_profile,
    )

    for name, help_text in _COMMAND_HELP.items():
        delegated = commands.add_parser(name, help=help_text, add_help=False)
        delegated.set_defaults(selected_parser=delegated)

    artifacts = commands.add_parser("artifacts", help="inspect benchmark artifacts")
    artifacts.set_defaults(selected_parser=artifacts)
    artifact_commands = artifacts.add_subparsers(dest="artifact_command", metavar="COMMAND")
    for name in _ARTIFACT_COMMANDS:
        delegated = artifact_commands.add_parser(name, add_help=False)
        delegated.set_defaults(selected_parser=delegated)

    return parser


def _dispatch(raw_args):
    if not raw_args:
        return None
    if raw_args[0] in _COMMANDS:
        return _COMMANDS[raw_args[0]](raw_args[1:])
    if raw_args[0] == "artifacts" and len(raw_args) > 1:
        handler = _ARTIFACT_COMMANDS.get(raw_args[1])
        if handler is not None:
            return handler(raw_args[2:])
    return None


def main(argv=None):
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and (
        raw_args[0] in _COMMANDS
        or (
            raw_args[0] == "artifacts"
            and len(raw_args) > 1
            and raw_args[1] in _ARTIFACT_COMMANDS
        )
    ):
        return _dispatch(raw_args)

    parser = build_parser()
    args = parser.parse_args(raw_args)
    handler = getattr(args, "handler", None)
    if handler is None:
        args.selected_parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    main()
