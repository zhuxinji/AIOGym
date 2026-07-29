"""Evaluate one policy checkpoint on every case in a track split."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from aiogym.evaluation import evaluate_controller
from aiogym.evaluation.provenance import track_provenance
from aiogym.evaluation.results import build_evaluation_report

from .tracks import TrackSpec, load_track


def evaluate_policy_on_track(
    controller,
    track: TrackSpec | str | Mapping[str, Any],
    *,
    split: str = "test",
    base_seeds: Sequence[int] = (9000,),
    include_episodes: bool = True,
    env_factory=None,
    evaluate_fn=None,
) -> dict[str, Any]:
    """Evaluate the same controller instance on all declared split cases."""

    resolved_track = (
        track
        if isinstance(track, TrackSpec)
        else load_track(track)
    )
    seeds = tuple(int(seed) for seed in base_seeds)
    if not seeds:
        raise ValueError("track evaluation requires at least one base seed")
    if env_factory is None:
        from aiogym.env import AIOGymEnv

        env_factory = lambda case: AIOGymEnv(
            resolved_track.scenario,
            case=case.profile,
            reward_spec=resolved_track.reward_spec_id,
        )
    evaluator = evaluate_fn or evaluate_controller
    results = []
    for case in resolved_track.resolved_cases(split):
        env = env_factory(case)
        try:
            horizon_seconds = float(
                env.control_dt * env.episode_steps
            )
            result = evaluator(
                controller,
                env,
                episodes=len(seeds),
                seed=seeds[0],
                seed_list=seeds,
                goal_specification=resolved_track.goal,
                seed_namespace=resolved_track.seed_namespace(split),
                include_episodes=include_episodes,
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        result.update(
            {
                "track_id": resolved_track.id,
                "track_hash": resolved_track.track_hash,
                "track_split": split,
                "case_id": case.case_id,
                "resolved_case_hash": case.resolved_case_hash,
                "case_group": case.group,
                "pair_id": case.pair_id,
                "pair_role": case.condition,
                "base_case_id": case.base_case_id or case.case_id,
                "policy_scope": resolved_track.policy_scope,
                "case_horizon_seconds": horizon_seconds,
            }
        )
        gate = dict(result.get("safety_gate") or {})
        result["provenance"] = track_provenance(
            resolved_track,
            case=case,
            controller=controller,
            custom_overrides={},
            eligible=bool(result.get("ranking_eligible", True)),
            eligibility_reasons=list(gate.get("reasons") or ()),
        )
        results.append(result)
    aggregate = aggregate_track_results(results, resolved_track)
    report = build_evaluation_report(results)
    return {
        "track_id": resolved_track.id,
        "track_hash": resolved_track.track_hash,
        "split": split,
        "seed_namespace": resolved_track.seed_namespace(split),
        "base_seeds": list(seeds),
        "case_count": len(results),
        "results": results,
        "aggregate": aggregate,
        "report": report,
        "report_views": report["views"],
        "provenance": track_provenance(
            resolved_track,
            controller=controller,
            custom_overrides={},
            eligible=bool(aggregate["ranking_eligible"]),
            eligibility_reasons=[
                reason
                for result in results
                for reason in dict(
                    result.get("safety_gate") or {}
                ).get("reasons", ())
            ],
        ),
    }


def aggregate_track_results(
    results: Sequence[Mapping[str, Any]],
    track: TrackSpec,
) -> dict[str, Any]:
    """Build the validation-selection rate without consulting test results."""

    if not results:
        raise ValueError("track aggregation requires at least one case result")
    if track.goal == "regulation":
        source_metric = "regulation_cost"
        metric = "regulation_cost_rate"
        direction = "minimize"
    else:
        source_metric = "profit"
        metric = "profit_rate"
        direction = "maximize"
    values = []
    eligible = True
    for result in results:
        horizon = float(result.get("case_horizon_seconds", 0.0))
        if horizon <= 0.0:
            raise ValueError("track case horizon must be positive")
        values.append(float(result[source_metric]) / horizon)
        eligible = eligible and bool(result.get("ranking_eligible", True))
    return {
        "metric": metric,
        "metric_direction": direction,
        "metric_value": float(np.mean(values)),
        "case_values": values,
        "case_count": len(values),
        "ranking_eligible": eligible,
    }


__all__ = ["aggregate_track_results", "evaluate_policy_on_track"]
