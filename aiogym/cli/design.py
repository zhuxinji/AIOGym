"""PlantConfig design commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiogym.core import get_scenario, jsonable, write_json
from aiogym.workflows import load_plant, study, sweep, validate_plant


def _parser(prog):
    parser = argparse.ArgumentParser(prog=prog)
    commands = parser.add_subparsers(dest="action", required=True)
    new = commands.add_parser("new", help="write a built-in PlantConfig template")
    new.add_argument("scenario")
    new.add_argument("output", type=Path)
    new.add_argument("--force", action="store_true")
    validate = commands.add_parser("validate", help="validate and resolve a PlantConfig")
    validate.add_argument("plant", type=Path)
    run = commands.add_parser("run", help="run the scenario design study")
    run.add_argument("plant", type=Path)
    run.add_argument("--condition")
    run.add_argument("--controller", default="pid")
    run.add_argument("--output", type=Path)
    run.add_argument("--samples", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--force", action="store_true")
    scan = commands.add_parser("sweep", help="sweep one dotted PlantConfig field")
    scan.add_argument("plant", type=Path)
    scan.add_argument("--parameter", required=True)
    scan.add_argument("--values", nargs="+", required=True, type=float)
    scan.add_argument("--condition")
    scan.add_argument("--controller", default="pid")
    scan.add_argument("--output", type=Path)
    scan.add_argument("--samples", type=int)
    scan.add_argument("--seed", type=int)
    scan.add_argument("--force", action="store_true")
    return parser


def main(argv=None, *, prog="aiogym design"):
    parser = _parser(prog)
    args = parser.parse_args(argv)
    try:
        if args.action == "new":
            import aiogym.scenarios  # noqa: F401

            plugin = get_scenario(args.scenario)
            source = plugin.default_plant
            if callable(source):
                result = source()
            else:
                result = plugin.built_in_plants[source]()
            write_json(args.output, result, overwrite=args.force)
            result = {"plant": str(args.output.resolve()), **result}
        elif args.action == "validate":
            resolved = validate_plant(args.plant)
            result = {
                "valid": True,
                "plant": resolved.config.as_dict(),
                "parameters": dict(resolved.parameters),
                "provenance": dict(resolved.provenance),
            }
        elif args.action == "run":
            result = study(
                args.plant,
                condition=args.condition,
                controller=args.controller,
                robustness_samples=args.samples,
                seed=args.seed,
                output=args.output,
                overwrite=args.force,
            )
        else:
            result = sweep(
                load_plant(args.plant),
                parameter=args.parameter,
                values=args.values,
                condition=args.condition,
                controller=args.controller,
                robustness_samples=args.samples,
                seed=args.seed,
                output=args.output,
                overwrite=args.force,
            )
    except (FileExistsError, FileNotFoundError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(jsonable(result), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
