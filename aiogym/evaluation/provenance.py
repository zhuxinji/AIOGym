"""Stable provenance records for track-first evaluation artifacts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

from aiogym.rewards import get_reward_spec


ARTIFACT_PROVENANCE_SCHEMA_VERSION = "aiogym.artifact_provenance.v1"


def stable_hash(value: Any) -> str:
    """Hash one JSON-compatible value with stable ordering."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reward_spec_hash(reward_spec_id: str) -> str:
    spec = get_reward_spec(reward_spec_id)
    return stable_hash(
        {
            "id": spec.id,
            "version": spec.version,
            "goal": spec.goal,
            "term_weights": dict(spec.term_weights),
            "cost_weights": dict(spec.cost_weights),
            "terminal_failure_cost_rate": spec.terminal_failure_cost_rate,
            "metadata": dict(spec.metadata),
        }
    )


def seed_namespace_hash(namespace: str) -> str:
    return hashlib.sha256(str(namespace).encode("utf-8")).hexdigest()


def package_version() -> str:
    try:
        return version("aiogym")
    except PackageNotFoundError:
        return "0.1.0"


def code_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def controller_access_level(controller: Any) -> str:
    """Classify information access without relying on implementation details."""

    explicit = getattr(controller, "access_level", None)
    if explicit:
        return str(explicit)
    name = str(
        getattr(controller, "name", None)
        or getattr(controller, "__name__", None)
        or controller.__class__.__name__
    ).lower()
    if "oracle" in name or "mpc" in name:
        return "model-based"
    if "pid" in name:
        return "measurement-and-setpoint"
    return "policy-observation"


def eligibility_record(
    *,
    eligible: bool,
    reasons: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    resolved_reasons = [str(reason) for reason in (reasons or ())]
    return {
        "status": "eligible" if eligible else "ineligible",
        "eligible": bool(eligible),
        "reasons": resolved_reasons,
    }


def track_provenance(
    track,
    *,
    case=None,
    controller: Any = None,
    training_seed: int | None = None,
    custom_overrides: Mapping[str, Any] | None = None,
    eligible: bool = True,
    eligibility_reasons: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build the complete required provenance for a track or track case."""

    namespaces = {
        split: track.seed_namespace(split)
        for split in ("training", "validation", "test")
    }
    access_level = (
        controller_access_level(controller)
        if controller is not None
        else None
    )
    reasons = [str(reason) for reason in (eligibility_reasons or ())]
    return {
        "schema_version": ARTIFACT_PROVENANCE_SCHEMA_VERSION,
        "track_id": track.id,
        "track_hash": track.track_hash,
        "case_id": getattr(case, "case_id", None),
        "resolved_case_hash": getattr(case, "resolved_case_hash", None),
        "goal": track.goal,
        "reward_spec_id": track.reward_spec_id,
        "reward_spec_hash": reward_spec_hash(track.reward_spec_id),
        "scorecard_spec_id": track.scorecard_spec_id,
        "ranking_spec_id": track.ranking_spec_id,
        "policy_scope": track.policy_scope,
        "training_seed": (
            None if training_seed is None else int(training_seed)
        ),
        "training_seed_namespace": namespaces["training"],
        "validation_seed_namespace": namespaces["validation"],
        "test_seed_namespace": namespaces["test"],
        "training_seed_namespace_hash": seed_namespace_hash(
            namespaces["training"]
        ),
        "validation_seed_namespace_hash": seed_namespace_hash(
            namespaces["validation"]
        ),
        "test_seed_namespace_hash": seed_namespace_hash(namespaces["test"]),
        "controller_access_level": access_level,
        "model_access_level": access_level,
        "code_commit": code_commit(),
        "package_version": package_version(),
        "custom_override_hash": stable_hash(dict(custom_overrides or {})),
        "eligibility": eligibility_record(
            eligible=eligible,
            reasons=reasons,
        ),
        "eligibility_status": "eligible" if eligible else "ineligible",
        "eligibility_reasons": reasons,
    }


__all__ = [
    "ARTIFACT_PROVENANCE_SCHEMA_VERSION",
    "code_commit",
    "controller_access_level",
    "eligibility_record",
    "package_version",
    "reward_spec_hash",
    "seed_namespace_hash",
    "stable_hash",
    "track_provenance",
]
