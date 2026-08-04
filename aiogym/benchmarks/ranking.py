"""Execution of versioned Track ranking declarations."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .anchors import AnchorSet, load_anchor_set
from .anchors.quality import validate_anchor_gap


RANKING_EPSILON = 1e-12


def case_utility(result: Mapping[str, Any], *, goal: str) -> float:
    horizon = float(result.get("case_horizon_seconds", 0.0))
    if horizon <= 0.0:
        raise ValueError("track case horizon must be positive")
    if goal == "regulation":
        utility = -float(result["regulation_cost"]) / horizon
    elif goal == "economic":
        utility = float(result["profit"]) / horizon
    else:
        raise ValueError(f"unsupported ranking goal {goal!r}")
    if not math.isfinite(utility):
        raise ValueError("track case utility must be finite")
    return utility


def fixed_anchor_score(
    utility: float,
    *,
    bad_utility: float,
    reference_utility: float,
) -> float:
    return max(
        0.0,
        fixed_anchor_margin(
            utility,
            bad_utility=bad_utility,
            reference_utility=reference_utility,
        ),
    )


def fixed_anchor_margin(
    utility: float,
    *,
    bad_utility: float,
    reference_utility: float,
) -> float:
    """Return the unclipped percentage margin between fixed anchors."""

    validate_anchor_gap(bad_utility, reference_utility)
    return (
        100.0
        * (float(utility) - float(bad_utility))
        / (float(reference_utility) - float(bad_utility))
    )


def weighted_geometric_mean(
    scores: Sequence[float],
    weights: Sequence[float],
    *,
    epsilon: float = RANKING_EPSILON,
) -> float:
    values = np.asarray(scores, dtype=np.float64)
    resolved_weights = np.asarray(weights, dtype=np.float64)
    if (
        values.ndim != 1
        or resolved_weights.shape != values.shape
        or values.size == 0
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(resolved_weights))
        or np.any(values < 0.0)
        or np.any(resolved_weights <= 0.0)
    ):
        raise ValueError("geometric ranking requires finite non-negative scores")
    resolved_weights = resolved_weights / np.sum(resolved_weights)
    return float(
        np.exp(
            np.sum(
                resolved_weights
                * np.log(np.maximum(values, float(epsilon)))
            )
        )
    )


def rank_track_results(
    results: Sequence[Mapping[str, Any]],
    track,
) -> dict[str, Any]:
    declaration = track.ranking_declaration
    case_rule = declaration["case_score"]
    aggregation = declaration["track_aggregation"]
    split = _single_split(results)
    declared_cases = track.resolved_cases(split)
    if [row.case_id for row in declared_cases] != [
        str(result["case_id"]) for result in results
    ]:
        raise ValueError("track results do not match declared case order")
    weights = [float(case.weight) for case in declared_cases]
    raw_utilities = [
        case_utility(result, goal=track.goal) for result in results
    ]
    anchors: AnchorSet | None = None
    if case_rule == "fixed-anchor-v1":
        anchors = load_anchor_set(declaration["anchor_id"], track=track)
        anchor_margins = [
            fixed_anchor_margin(
                utility,
                bad_utility=anchors.case(
                    str(result["resolved_case_hash"])
                ).bad_utility,
                reference_utility=anchors.case(
                    str(result["resolved_case_hash"])
                ).reference_utility,
            )
            for utility, result in zip(raw_utilities, results)
        ]
        scores = [max(0.0, margin) for margin in anchor_margins]
    elif case_rule == "diagnostic-only-v1":
        scores = list(raw_utilities)
        anchor_margins = []
    else:
        raise ValueError(f"unsupported case ranking rule {case_rule!r}")
    eligible = all(
        bool(result.get("ranking_eligible", True)) for result in results
    )
    gated_scores = [
        float(score) if bool(result.get("ranking_eligible", True)) else 0.0
        for score, result in zip(scores, results)
    ]
    if not eligible:
        official_score = 0.0
    elif aggregation == "weighted-geometric-mean-v1":
        official_score = weighted_geometric_mean(gated_scores, weights)
    elif aggregation == "arithmetic-mean-v1":
        normalized_weights = np.asarray(weights, dtype=np.float64)
        normalized_weights /= np.sum(normalized_weights)
        official_score = float(
            np.sum(normalized_weights * np.asarray(gated_scores))
        )
    else:
        raise ValueError(
            f"unsupported Track aggregation rule {aggregation!r}"
        )
    return {
        "ranking_spec_id": track.ranking_spec_id,
        "case_score_rule": case_rule,
        "track_aggregation": aggregation,
        "ranking_epsilon": RANKING_EPSILON,
        "case_utilities": raw_utilities,
        "case_anchor_margins": anchor_margins,
        "case_scores": gated_scores,
        "case_weights": weights,
        "official_score": float(official_score),
        "ranking_eligible": eligible,
        "anchor_id": anchors.anchor_id if anchors is not None else None,
        "anchor_artifact_hash": (
            anchors.artifact_hash if anchors is not None else None
        ),
    }


def _single_split(results: Sequence[Mapping[str, Any]]) -> str:
    splits = {str(result.get("track_split", "")) for result in results}
    if len(splits) != 1 or "" in splits:
        raise ValueError("track results must declare one common split")
    return splits.pop()


__all__ = [
    "RANKING_EPSILON",
    "case_utility",
    "fixed_anchor_margin",
    "fixed_anchor_score",
    "rank_track_results",
    "weighted_geometric_mean",
]
