#!/usr/bin/env python3
"""Benchmark canonical controllers on an official validation Track."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from aiogym._internal.serialization import jsonable
from aiogym.benchmarks.evaluation import evaluate_policy_on_track
from aiogym.benchmarks.tracks.registry import list_tracks, load_track
from aiogym.controllers.checkpoints import (
    SUPPORTED_POLICY_ALGORITHMS,
    checkpoint_sha256,
    learned_policy_spec_for_track,
    load_policy_checkpoint,
)
from aiogym.controllers.registry import make_controller
from aiogym.models.registry import make_model


BENCHMARK_CONFIG_SCHEMA_VERSION = "aiogym.benchmark_run.v1"
DEFAULT_CONTROLLERS = ("pid", "mpc")
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "track_id",
        "controllers",
        "seeds",
        "checkpoint",
        "algorithm_id",
        "checkpoint_sha256",
        "checkpoint_name",
        "device",
        "include_episodes",
        "output",
    }
)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Compare controllers on one official validation Track.",
    )
    parser.add_argument("track", nargs="?", choices=list_tracks())
    parser.add_argument("--config")
    parser.add_argument(
        "--controllers",
        default=",".join(DEFAULT_CONTROLLERS),
        help="comma-separated baseline IDs (default: pid,mpc)",
    )
    parser.add_argument("--seeds", default="9000")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--algorithm",
        dest="algorithm_id",
        choices=SUPPORTED_POLICY_ALGORITHMS,
    )
    parser.add_argument("--sha256", dest="checkpoint_sha256")
    parser.add_argument("--name", dest="checkpoint_name")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-episodes", action="store_true")
    parser.add_argument("--output")
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv=None, prog: str | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser(prog)
    if not raw_args:
        parser.print_help()
        return 0
    args = parser.parse_args(raw_args)
    if args.config:
        if args.track is not None:
            parser.error("--config and positional TRACK are mutually exclusive")
        try:
            _apply_config(args, _load_config(args.config))
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            parser.error(str(exc))
    if args.track is None:
        parser.error("provide --config FILE or an official TRACK")
    try:
        track = load_track(args.track)
        seeds = _parse_seeds(args.seeds)
        controllers = _baseline_controllers(args.controllers, track.scenario)
        if args.checkpoint:
            if not args.algorithm_id:
                raise ValueError("--checkpoint requires --algorithm")
            name = args.checkpoint_name or args.algorithm_id
            if name in controllers:
                raise ValueError(
                    f"checkpoint name {name!r} conflicts with a baseline"
                )
            digest = (
                args.checkpoint_sha256
                or checkpoint_sha256(args.checkpoint)
            )
            spec = learned_policy_spec_for_track(
                args.checkpoint,
                args.algorithm_id,
                digest,
                track,
            )
            controllers[name] = load_policy_checkpoint(
                spec,
                device=args.device,
            )
        elif args.algorithm_id or args.checkpoint_sha256 or args.checkpoint_name:
            raise ValueError(
                "--algorithm, --sha256, and --name require --checkpoint"
            )
        results = {
            name: evaluate_policy_on_track(
                controller,
                track,
                base_seeds=seeds,
                include_episodes=args.include_episodes,
            )
            for name, controller in controllers.items()
        }
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    for name, payload in results.items():
        aggregate = payload["aggregate"]
        print(
            f"{name:14s} "
            f"official_score={aggregate['official_score']:.6g} "
            f"{aggregate['metric']}={aggregate['metric_value']:.6g} "
            f"eligible={aggregate['ranking_eligible']}"
        )
    output = {
        "schema_version": "aiogym.benchmark_result.v1",
        "track_id": track.id,
        "track_hash": track.track_hash,
        "split": "validation",
        "controllers": list(controllers),
        "base_seeds": list(seeds),
        "results": results,
    }
    if args.output:
        try:
            _write_payload(args.output, output, overwrite=args.overwrite)
        except (FileExistsError, OSError) as exc:
            parser.error(str(exc))
    return 0


def _baseline_controllers(raw: str, scenario: str) -> dict:
    names = tuple(
        part.strip().lower()
        for part in str(raw).split(",")
        if part.strip()
    )
    if not names:
        raise ValueError("--controllers must include at least one baseline")
    unknown = set(names) - set(DEFAULT_CONTROLLERS)
    if unknown:
        raise ValueError(
            "official benchmark baselines are: "
            + ", ".join(DEFAULT_CONTROLLERS)
        )
    model = make_model(scenario)
    return {
        name: make_controller(name, model=model, scenario=scenario)
        for name in names
    }


def _parse_seeds(raw) -> tuple[int, ...]:
    if isinstance(raw, (list, tuple)):
        seeds = tuple(int(seed) for seed in raw)
    else:
        seeds = tuple(
            int(part.strip())
            for part in str(raw).split(",")
            if part.strip()
        )
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain unique non-negative integers")
    return seeds


def _load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        declaration = json.load(stream)
    if not isinstance(declaration, Mapping):
        raise TypeError("benchmark config must be a mapping")
    unknown = set(declaration) - _CONFIG_FIELDS
    if unknown:
        raise ValueError(
            "unknown benchmark config fields: "
            + ", ".join(sorted(unknown))
        )
    if declaration.get("schema_version") != BENCHMARK_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported benchmark config schema")
    if not isinstance(declaration.get("track_id"), str):
        raise ValueError("benchmark config requires track_id")
    return dict(declaration)


def _apply_config(args, declaration: dict) -> None:
    args.track = declaration["track_id"]
    if "controllers" in declaration:
        controllers = declaration["controllers"]
        args.controllers = (
            ",".join(controllers)
            if isinstance(controllers, list)
            else str(controllers)
        )
    if "seeds" in declaration:
        args.seeds = declaration["seeds"]
    for field in (
        "checkpoint",
        "algorithm_id",
        "checkpoint_sha256",
        "checkpoint_name",
        "device",
        "include_episodes",
        "output",
    ):
        if field in declaration:
            setattr(args, field, declaration[field])


def _write_payload(path: str, payload: dict, *, overwrite: bool) -> None:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["BENCHMARK_CONFIG_SCHEMA_VERSION", "build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
