#!/usr/bin/env python3
"""Reproducible random/refinement search for quadruple benchmark PID profiles."""
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from aiogym.controllers.configs import load_controller_config
from aiogym.controllers.tuning.evaluation import evaluate_specialist


FAMILIES = {
    "minimum-phase": {
        "cases": ["minimum-phase"],
        "baseline_profile": "quadruple-minimum-phase-benchmark",
        "bias": 0.3,
        "kp_bounds": (0.03, 2.0),
        "ki_bounds": (0.0003, 0.1),
        "topologies": ["direct"],
    },
    "nonminimum-phase": {
        "cases": ["nonminimum-phase"],
        "baseline_profile": "quadruple-nonminimum-phase-benchmark",
        "bias": 0.315,
        "kp_bounds": (0.001, 0.3),
        "ki_bounds": (0.000001, 0.01),
        "topologies": ["paper", "cross"],
    },
    "zero-boundary": {
        "cases": ["zero-boundary-stress"],
        "baseline_profile": "quadruple-zero-boundary",
        "bias": 0.3,
        "kp_bounds": (0.01, 1.0),
        "ki_bounds": (0.00001, 0.1),
        "topologies": ["direct"],
    },
    "disturbance-rejection": {
        "cases": ["disturbance-rejection"],
        "baseline_profile": "quadruple-disturbance-rejection",
        "bias": 0.3,
        "kp_bounds": (0.01, 0.5),
        "ki_bounds": (0.0001, 0.05),
        "topologies": ["direct"],
    },
}


def _log_uniform(rng: random.Random, bounds: tuple[float, float]) -> float:
    lo, hi = (math.log(value) for value in bounds)
    return math.exp(rng.uniform(lo, hi))


def _loops(topology: str, gains: list[float], bias: float) -> list[dict]:
    kp1, ki1, kp2, ki2 = gains
    if topology == "cross":
        pairs = [(0, 1, False), (1, 0, False)]
    elif topology == "paper":
        pairs = [(0, 0, False), (1, 1, True)]
    else:
        pairs = [(0, 0, False), (1, 1, False)]
    return [
        {
            "u_index": u_index,
            "y_index": y_index,
            "pid": [kp, ki, 0.0],
            "bias": bias,
            **({"reverse": True} if reverse else {}),
        }
        for (u_index, y_index, reverse), (kp, ki) in zip(
            pairs, ((kp1, ki1), (kp2, ki2))
        )
    ]


def _evaluate(case: str, loops: list[dict]) -> dict:
    result = evaluate_specialist(
        scenario="quadruple",
        case=case,
        controller="pid",
        seeds=[9000],
        controller_config={
            "case": case,
            "policy_scope": "specialist",
            "goal": "regulation",
            "reward_spec": "regulation-v1",
            "parameters": {"loops": loops},
        },
    )
    return {
        "case": case,
        "goal": "regulation",
        "metric": result["metric"],
        "metric_value": result[result["metric"]],
        "tracking_cost": result.get("tracking_cost"),
        "tracking_error_cost": result.get("tracking_error_cost"),
        "tracking_mse": result.get("tracking_mse"),
        "tracking_iae": result.get("tracking_iae"),
        "constraint_violation_count": result.get("constraint_violation_count", 0.0),
        "constraint_violation_severity": result.get("constraint_violation_severity", 0.0),
    }


def _loss(result: dict) -> float:
    return float(result["metric_value"])


def _candidate(topology: str, gains: list[float], bias: float, baseline: list[dict]) -> dict:
    loops = _loops(topology, gains, bias)
    results = [_evaluate(case, loops) for case in baseline["cases"]]
    relative_losses = [
        _loss(result) / max(1e-12, _loss(reference))
        for result, reference in zip(results, baseline["results"])
    ]
    return {
        "topology": topology,
        "gains": gains,
        "loops": loops,
        "results": results,
        "relative_losses": relative_losses,
        "score": sum(relative_losses) / len(relative_losses),
        "constraint_violation_count": sum(
            float(result["constraint_violation_count"]) for result in results
        ),
        "constraint_violation_severity": sum(
            float(result["constraint_violation_severity"]) for result in results
        ),
    }


def _rank_key(candidate: dict) -> tuple[float, float, float]:
    return (
        candidate["constraint_violation_count"],
        candidate["constraint_violation_severity"],
        candidate["score"],
    )


def _baseline(family: dict) -> dict:
    parameters = load_controller_config(
        "pid", "quadruple", profile=family["baseline_profile"]
    )["parameters"]
    loops = parameters["loops"]
    results = [_evaluate(case, loops) for case in family["cases"]]
    return {"cases": family["cases"], "loops": loops, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--refine", type=int, default=60)
    parser.add_argument("--search-seed", type=int, default=20260715)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    family = FAMILIES[args.family]
    rng = random.Random(args.search_seed)
    baseline = _baseline(family)
    rows = []
    best = None

    for index in range(args.samples):
        topology = rng.choice(family["topologies"])
        gains = [
            _log_uniform(rng, family["kp_bounds"]),
            _log_uniform(rng, family["ki_bounds"]),
            _log_uniform(rng, family["kp_bounds"]),
            _log_uniform(rng, family["ki_bounds"]),
        ]
        candidate = _candidate(topology, gains, family["bias"], baseline)
        rows.append(candidate)
        if best is None or _rank_key(candidate) < _rank_key(best):
            best = candidate
            print(f"sample={index + 1} best_score={best['score']:.6f} topology={topology} gains={gains}", flush=True)

    assert best is not None
    for index in range(args.refine):
        bounds = [
            family["kp_bounds"], family["ki_bounds"],
            family["kp_bounds"], family["ki_bounds"],
        ]
        gains = [
            min(hi, max(lo, value * math.exp(rng.uniform(-0.55, 0.55))))
            for value, (lo, hi) in zip(best["gains"], bounds)
        ]
        candidate = _candidate(best["topology"], gains, family["bias"], baseline)
        rows.append(candidate)
        if _rank_key(candidate) < _rank_key(best):
            best = candidate
            print(f"refine={index + 1} best_score={best['score']:.6f} gains={gains}", flush=True)

    rows.sort(key=_rank_key)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": "quadruple",
        "family": args.family,
        "search_seed": args.search_seed,
        "samples": args.samples,
        "refine": args.refine,
        "baseline_profile": family["baseline_profile"],
        "baseline": baseline,
        "best": best,
        "top": rows[: args.top],
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"baseline": baseline["results"], "best": best}, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
