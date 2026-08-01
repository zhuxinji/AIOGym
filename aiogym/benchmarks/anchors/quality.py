"""Auditable quality contract for official fixed ranking anchors."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import statistics
from typing import Any


@dataclass(frozen=True)
class AnchorQualityPolicy:
    id: str = "fixed-anchor-quality-v1"
    min_absolute_gap: float = 1e-8
    min_relative_gap: float = 1e-4
    min_signal_to_noise: float = 5.0
    utility_scale_floor: float = 1e-8


DEFAULT_ANCHOR_QUALITY_POLICY = AnchorQualityPolicy()


def anchor_quality_statistics(
    bad_samples: Sequence[float],
    reference_samples: Sequence[float],
    *,
    bad_eligible: Sequence[bool] | None = None,
    reference_eligible: Sequence[bool] | None = None,
    policy: AnchorQualityPolicy = DEFAULT_ANCHOR_QUALITY_POLICY,
) -> dict[str, Any]:
    """Compute the hash-protected paired-sample anchor audit block."""

    bad = _samples("bad_samples", bad_samples)
    reference = _samples("reference_samples", reference_samples)
    if len(bad) != len(reference):
        raise ValueError("anchor sample sequences must have equal length")
    bad_ok = _eligibility("bad_eligible", bad_eligible, len(bad))
    reference_ok = _eligibility(
        "reference_eligible",
        reference_eligible,
        len(reference),
    )
    bad_mean = statistics.fmean(bad)
    reference_mean = statistics.fmean(reference)
    bad_std = statistics.stdev(bad) if len(bad) > 1 else 0.0
    reference_std = (
        statistics.stdev(reference) if len(reference) > 1 else 0.0
    )
    gap = reference_mean - bad_mean
    scale = max(
        abs(reference_mean),
        abs(bad_mean),
        policy.utility_scale_floor,
    )
    relative_gap = gap / scale
    standard_error = math.sqrt(
        bad_std * bad_std / len(bad)
        + reference_std * reference_std / len(reference)
    )
    signal_to_noise = (
        math.inf
        if standard_error == 0.0 and gap > 0.0
        else gap / standard_error if standard_error > 0.0 else -math.inf
    )
    reasons = []
    if gap <= 0.0:
        reasons.append("reference utility must exceed bad utility")
    if gap < policy.min_absolute_gap:
        reasons.append("absolute gap is below the quality threshold")
    if relative_gap < policy.min_relative_gap:
        reasons.append("relative gap is below the quality threshold")
    if signal_to_noise < policy.min_signal_to_noise:
        reasons.append("signal-to-noise ratio is below the quality threshold")
    if not all(bad_ok):
        reasons.append("bad baseline is ineligible on at least one seed")
    if not all(reference_ok):
        reasons.append("reference baseline is ineligible on at least one seed")
    return {
        "policy_id": policy.id,
        "bad_samples": list(bad),
        "reference_samples": list(reference),
        "bad_eligible": list(bad_ok),
        "reference_eligible": list(reference_ok),
        "sample_count": len(bad),
        "bad_mean": bad_mean,
        "reference_mean": reference_mean,
        "bad_std": bad_std,
        "reference_std": reference_std,
        "bad_min": min(bad),
        "bad_max": max(bad),
        "reference_min": min(reference),
        "reference_max": max(reference),
        "absolute_gap": gap,
        "relative_gap": relative_gap,
        "standard_error": standard_error,
        "signal_to_noise": (
            "infinity" if math.isinf(signal_to_noise) else signal_to_noise
        ),
        "passed": not reasons,
        "failure_reasons": reasons,
    }


def validate_anchor_quality(
    quality: Mapping[str, Any],
    *,
    policy: AnchorQualityPolicy = DEFAULT_ANCHOR_QUALITY_POLICY,
) -> dict[str, Any]:
    """Recompute and verify one stored quality block."""

    if not isinstance(quality, Mapping):
        raise TypeError("anchor quality must be a mapping")
    if quality.get("policy_id") != policy.id:
        raise ValueError("unknown anchor quality policy")
    expected = anchor_quality_statistics(
        quality.get("bad_samples", ()),
        quality.get("reference_samples", ()),
        bad_eligible=quality.get("bad_eligible"),
        reference_eligible=quality.get("reference_eligible"),
        policy=policy,
    )
    if dict(quality) != expected:
        raise ValueError("anchor quality statistics do not match samples")
    return expected


def validate_anchor_gap(
    bad_utility: float,
    reference_utility: float,
    *,
    policy: AnchorQualityPolicy = DEFAULT_ANCHOR_QUALITY_POLICY,
) -> tuple[float, float]:
    """Reject a deterministic denominator that cannot support ranking."""

    bad = float(bad_utility)
    reference = float(reference_utility)
    if not math.isfinite(bad) or not math.isfinite(reference):
        raise ValueError("anchor utilities must be finite")
    gap = reference - bad
    scale = max(abs(reference), abs(bad), policy.utility_scale_floor)
    relative_gap = gap / scale
    if gap < policy.min_absolute_gap or relative_gap < policy.min_relative_gap:
        raise ValueError(
            "degenerate ranking anchor: gap does not meet "
            f"{policy.id} (absolute_gap={gap:.6g}, "
            f"relative_gap={relative_gap:.6g})"
        )
    return gap, relative_gap


def _samples(name: str, values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a numeric sequence")
    resolved = tuple(float(value) for value in values)
    if not resolved or any(not math.isfinite(value) for value in resolved):
        raise ValueError(f"{name} must contain finite values")
    return resolved


def _eligibility(
    name: str,
    values: Sequence[bool] | None,
    count: int,
) -> tuple[bool, ...]:
    if values is None:
        return (True,) * count
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a boolean sequence")
    if len(values) != count or any(not isinstance(value, bool) for value in values):
        raise ValueError(f"{name} must contain one boolean per sample")
    return tuple(values)


__all__ = [
    "AnchorQualityPolicy",
    "DEFAULT_ANCHOR_QUALITY_POLICY",
    "anchor_quality_statistics",
    "validate_anchor_gap",
    "validate_anchor_quality",
]
