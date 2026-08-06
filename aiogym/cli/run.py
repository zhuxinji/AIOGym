"""Run one complete experiment from one declaration."""
from __future__ import annotations

import argparse
import json

from aiogym.experiments import ExperimentSpec, run_experiment_spec


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run one experiment config into one result directory.",
    )
    parser.add_argument("config", help="aiogym.experiment.v1 JSON file")
    parser.add_argument("--output", help="override the experiment output directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    try:
        spec = ExperimentSpec.load(args.config)
        if args.output is not None:
            spec = spec.with_output(args.output)
        result = run_experiment_spec(
            spec,
            dry_run=bool(args.dry_run),
            overwrite=bool(args.overwrite),
        )
    except (FileExistsError, FileNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = ["build_parser", "main"]
