#!/usr/bin/env python3
"""Tune the extraction PID baseline on the tracking benchmark."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import datetime, timezone

from aiogym._internal.config import parse_seed_list, resolve_auto_events
from aiogym._internal.paths import run_path
from aiogym.controllers.tuning.evaluation import evaluate_specialist


def parse_float_list(raw: str):
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("grid lists must contain at least one value")
    return values


def extraction_pid_config(kp: float, ki: float, kd: float, gas_hold: float):
    return {
        "loops": [{"u_index": 0, "y_index": 4, "pid": [kp, ki, kd]}],
        "demand_u_index": None,
        "holds": [{"u_index": 1, "value": gas_hold}],
    }


def evaluate_config(environment: dict, seeds: list[int], config: dict):
    return evaluate_specialist(
        scenario="extraction",
        case="tracking-specialist",
        controller="pid",
        seeds=seeds,
        environment=environment,
        controller_config={
            "case": "tracking-specialist",
            "policy_scope": "specialist",
            "goal": "regulation",
            "reward_spec": "regulation-v1",
            "parameters": config,
        },
    )


def compact_result(params: dict, result: dict):
    return {
        **params,
        "tracking_iae": result["tracking_iae"],
        "tracking_ise": result["tracking_ise"],
        "tracking_itae": result["tracking_itae"],
        "tracking_overshoot": result["tracking_overshoot"],
        "track": result["track"],
        "profit": result["profit"],
        "energy_kwh": result["energy_kwh"],
        "constraint_violation_count": result["constraint_violation_count"],
        "constraint_violation_severity": result["constraint_violation_severity"],
        "runtime_seconds_per_step": result["runtime_seconds_per_step"],
    }


def rank_key(row: dict):
    return (
        row["constraint_violation_count"],
        row["constraint_violation_severity"],
        row["tracking_iae"],
        row["tracking_itae"],
        row["energy_kwh"],
    )


def recommended_row(rows: list[dict], relative_iae_tolerance: float):
    best_iae = min(row["tracking_iae"] for row in rows)
    limit = best_iae * (1.0 + relative_iae_tolerance)
    candidates = [row for row in rows if row["tracking_iae"] <= limit]
    return sorted(candidates, key=lambda row: (
        row["constraint_violation_count"],
        row["constraint_violation_severity"],
        row["energy_kwh"],
        row["tracking_iae"],
        row["tracking_itae"],
    ))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-steps", type=int, default=80)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=9700)
    ap.add_argument("--seed-list", default=None)
    ap.add_argument("--control-dt", type=float, default=0.5)
    ap.add_argument("--auto-events", action="store_true", default=None)
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--randomize-plant", action="store_true")
    ap.add_argument("--plant-drift", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--kp", default="0,10,25,50,75,100,150,200,400,700,900")
    ap.add_argument("--ki", default="0,0.25,0.5,1,2,5,10")
    ap.add_argument("--kd", default="0")
    ap.add_argument("--gas-hold", default="0,0.25,0.5,1.0")
    ap.add_argument("--recommendation-relative-iae-tolerance", type=float, default=0.0005)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", default=str(run_path("tuning", "tune_extraction_pid_tracking.json")))
    args = ap.parse_args()
    args.auto_events = resolve_auto_events(
        default=False,
        auto_events=args.auto_events,
    )

    seeds = parse_seed_list(args.seed_list, args.seed, args.episodes)
    environment = {
        "episode_steps": args.episode_steps,
        "control_dt": args.control_dt,
        "auto_events": args.auto_events,
        "randomize": args.randomize,
        "randomize_plant": args.randomize_plant,
        "plant_drift": args.plant_drift,
        "noise": args.noise,
    }
    grids = {
        "kp": parse_float_list(args.kp),
        "ki": parse_float_list(args.ki),
        "kd": parse_float_list(args.kd),
        "gas_hold": parse_float_list(args.gas_hold),
    }

    rows = []
    keys = list(grids)
    for values in itertools.product(*(grids[key] for key in keys)):
        params = dict(zip(keys, values))
        config = extraction_pid_config(
            params["kp"],
            params["ki"],
            params["kd"],
            params["gas_hold"],
        )
        result = evaluate_config(environment, seeds, config)
        row = compact_result(params, result)
        row["config"] = config
        rows.append(row)
        print(
            f"kp={params['kp']:7.1f} ki={params['ki']:6.1f} kd={params['kd']:4.1f} "
            f"gas={params['gas_hold']:4.2f} iae={row['tracking_iae']:.6f} "
            f"itae={row['tracking_itae']:.6f} "
            f"energy={row['energy_kwh']:.6f} safety={row['constraint_violation_count']:.1f}"
        )

    rows.sort(key=rank_key)
    best = rows[0]
    recommended = recommended_row(rows, args.recommendation_relative_iae_tolerance)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": "extraction",
        "goal": "regulation",
        "reward_spec": "regulation-v1",
        "policy_scope": "specialist",
        "metric_order": [
            "constraint_violation_count",
            "constraint_violation_severity",
            "tracking_iae",
            "tracking_itae",
            "energy_kwh",
        ],
        "environment": environment,
        "seeds": seeds,
        "grid": grids,
        "best": best,
        "recommended": recommended,
        "recommendation_relative_iae_tolerance": args.recommendation_relative_iae_tolerance,
        "top": rows[:args.top],
        "rows": rows,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print("\nTOP")
    for row in rows[:args.top]:
        print(
            f"iae={row['tracking_iae']:.6f} itae={row['tracking_itae']:.6f} "
            f"energy={row['energy_kwh']:.6f} "
            f"kp={row['kp']:.1f} ki={row['ki']:.1f} kd={row['kd']:.1f} gas={row['gas_hold']:.2f}"
        )
    print(
        "\nRECOMMENDED "
        f"iae={recommended['tracking_iae']:.6f} itae={recommended['tracking_itae']:.6f} "
        f"energy={recommended['energy_kwh']:.6f} "
        f"kp={recommended['kp']:.1f} ki={recommended['ki']:.2f} "
        f"kd={recommended['kd']:.1f} gas={recommended['gas_hold']:.2f}"
    )
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
