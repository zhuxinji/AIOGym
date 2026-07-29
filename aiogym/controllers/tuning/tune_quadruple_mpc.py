#!/usr/bin/env python3
"""Grid-search Case-specific quadruple-tank linear MPC parameters."""
from __future__ import annotations

import argparse
import itertools
import json
from time import perf_counter

from aiogym.controllers.tuning.evaluation import evaluate_specialist


CASE_PROFILES = {
    "minimum-phase": "quadruple-minimum-phase",
    "nonminimum-phase": "quadruple-nonminimum-phase",
    "zero-boundary-stress": "quadruple-zero-boundary",
}


def _csv(raw: str, cast):
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=sorted(CASE_PROFILES))
    parser.add_argument("--profile", default=None)
    parser.add_argument("--horizons", default="10", help="comma-separated integers")
    parser.add_argument("--move-weights", default="0", help="comma-separated floats")
    parser.add_argument(
        "--steady-input-weights",
        default="0",
        help="comma-separated steady-state input regularization weights",
    )
    parser.add_argument("--error-weight", type=float, default=1.0)
    parser.add_argument(
        "--ranking-metric",
        choices=("tracking_error_cost", "tracking_cost"),
        default="tracking_cost",
    )
    parser.add_argument("--cv-scales", default="1", help="comma-separated scalar scales")
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--episode-steps", type=int, default=None)
    args = parser.parse_args()

    profile = args.profile or CASE_PROFILES[args.case]
    grid = itertools.product(
        _csv(args.horizons, int),
        _csv(args.move_weights, float),
        _csv(args.steady_input_weights, float),
        _csv(args.cv_scales, float),
    )
    rows = []
    for horizon, move, steady_input_weight, cv_scale in grid:
        parameters = {
            "P": horizon,
            "move_supp": move,
            "steady_input_weight": steady_input_weight,
            "cv_scale": [cv_scale, cv_scale],
        }
        started = perf_counter()
        row = evaluate_specialist(
            scenario="quadruple",
            case=args.case,
            controller="mpc",
            seeds=[args.seed],
            controller_config={
                "profile": profile,
                "case": args.case,
                "policy_scope": "specialist",
                "goal": "regulation",
                "reward_spec": "regulation-v1",
                "parameters": parameters,
            },
            environment=(
                {"episode_steps": args.episode_steps}
                if args.episode_steps
                else {}
            ),
        )
        result = {
            **parameters,
            "case": args.case,
            "tracking_error_cost": row.get("tracking_error_cost"),
            "tracking_cost": row.get("tracking_cost"),
            "constraint_violation_count": row.get("constraint_violation_count"),
            "controller_fallback_count": row.get("controller_fallback_count"),
            "runtime_seconds_per_step": row.get("runtime_seconds_per_step"),
            "wall_seconds": perf_counter() - started,
        }
        rows.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    feasible = [
        row for row in rows
        if not row["constraint_violation_count"] and not row["controller_fallback_count"]
    ]
    ranked = feasible or rows
    best = min(ranked, key=lambda row: float(row[args.ranking_metric]))
    print(json.dumps({
        "best": best,
        "candidate_count": len(rows),
        "ranking_metric": args.ranking_metric,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
