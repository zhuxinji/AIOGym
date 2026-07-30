"""Evaluate one policy checkpoint on every case in a track split."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from aiogym.evaluation import evaluate_controller
from aiogym.evaluation.provenance import track_provenance
from aiogym.evaluation.results import build_evaluation_report

from .ranking import rank_track_results
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
    episode_plan=None,
    safety_gate_spec=None,
) -> dict[str, Any]:
    """Evaluate the same controller instance on all declared split cases."""

    resolved_track = (
        track
        if isinstance(track, TrackSpec)
        else load_track(track)
    )
    if resolved_track.official and safety_gate_spec is not None:
        raise ValueError(
            "official Track safety gate cannot be overridden"
        )
    resolved_safety_gate = (
        resolved_track.safety_gate_spec()
        if safety_gate_spec is None
        else safety_gate_spec
    )
    seeds = tuple(int(seed) for seed in base_seeds)
    if not seeds:
        raise ValueError("track evaluation requires at least one base seed")
    if episode_plan is not None:
        if split != "validation":
            raise ValueError(
                "fixed ValidationEpisodePlan may be used only on validation"
            )
        if episode_plan.track.id != resolved_track.id:
            raise ValueError("episode plan track does not match evaluation track")
        if episode_plan.track.track_hash != resolved_track.track_hash:
            raise ValueError("episode plan track hash does not match")
        if tuple(episode_plan.base_seeds) != seeds:
            raise ValueError("episode plan seeds do not match base_seeds")
    if env_factory is None:
        from aiogym._environment.builder import (
            build_track_case_environment,
        )

        env_factory = lambda case: build_track_case_environment(
            resolved_track,
            case,
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
                episode_specs=(
                    episode_plan.episode_specs(case.case_id)
                    if episode_plan is not None
                    else None
                ),
                safety_gate_spec=resolved_safety_gate,
            )
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        if bool(
            dict(result.get("controller") or {}).get(
                "environment_context_access",
                False,
            )
        ):
            raise ValueError(
                "official Track policies cannot access the environment "
                "through ControllerContext"
            )
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
    for result, utility, score in zip(
        results,
        aggregate["case_utilities"],
        aggregate["case_scores"],
    ):
        result["ranking_utility"] = utility
        result["official_score"] = score
        result["ranking_spec_id"] = resolved_track.ranking_spec_id
    report = build_evaluation_report(results)
    return {
        "track_id": resolved_track.id,
        "track_hash": resolved_track.track_hash,
        "split": split,
        "seed_namespace": resolved_track.seed_namespace(split),
        "base_seeds": list(seeds),
        "episode_plan_hash": (
            episode_plan.plan_hash if episode_plan is not None else None
        ),
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
    for result in results:
        horizon = float(result.get("case_horizon_seconds", 0.0))
        if horizon <= 0.0:
            raise ValueError("track case horizon must be positive")
        values.append(float(result[source_metric]) / horizon)
    ranking = rank_track_results(results, track)
    return {
        "metric": metric,
        "metric_direction": direction,
        "metric_value": float(np.mean(values)),
        "case_values": values,
        "case_count": len(values),
        **ranking,
    }


__all__ = ["aggregate_track_results", "evaluate_policy_on_track"]
