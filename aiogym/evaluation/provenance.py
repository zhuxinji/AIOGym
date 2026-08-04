"""Stable provenance records for track-first evaluation artifacts."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

from aiogym.rewards.registry import get_reward_spec
from aiogym._internal.serialization import stable_json_hash


ARTIFACT_PROVENANCE_SCHEMA_VERSION = "aiogym.artifact_provenance.v1"


def stable_hash(value: Any) -> str:
    """Hash one JSON-compatible value with stable ordering."""

    return stable_json_hash(value, default=str)


def reward_spec_hash(reward_spec_id: str) -> str:
    return get_reward_spec(reward_spec_id).spec_hash


def seed_namespace_hash(namespace: str) -> str:
    return hashlib.sha256(str(namespace).encode("utf-8")).hexdigest()


def package_version() -> str:
    from aiogym import __version__

    return __version__


def _installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def runtime_environment(
    *,
    vector_backend: str | None = None,
    multiprocessing_start_method: str | None = None,
) -> dict[str, Any]:
    """Return JSON-safe runtime metadata without importing optional packages."""

    torch = sys.modules.get("torch")
    cuda_available = None
    cuda_runtime = None
    device_name = None
    torch_threads = None
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        cuda_runtime = getattr(getattr(torch, "version", None), "cuda", None)
        torch_threads = int(torch.get_num_threads())
        if cuda_available:
            device_name = str(torch.cuda.get_device_name(0))
    resolved_start_method = multiprocessing_start_method
    if resolved_start_method is None:
        resolved_start_method = multiprocessing.get_start_method(allow_none=True)
    payload = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "aiogym_version": package_version(),
        "numpy_version": _installed_version("numpy"),
        "gymnasium_version": _installed_version("gymnasium"),
        "torch_version": _installed_version("torch"),
        "stable_baselines3_version": _installed_version("stable-baselines3"),
        "casadi_version": _installed_version("casadi"),
        "onnx_version": _installed_version("onnx"),
        "onnxruntime_version": _installed_version("onnxruntime"),
        "cuda_available": cuda_available,
        "cuda_runtime": cuda_runtime,
        "device_name": device_name,
        "torch_threads": torch_threads,
        "vector_backend": vector_backend,
        "multiprocessing_start_method": resolved_start_method,
    }
    json.dumps(payload, allow_nan=False)
    return payload


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
    if bool(getattr(controller, "allow_environment_context", False)):
        return "environment-context"
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
    include_test_split: bool = True,
) -> dict[str, Any]:
    """Build the complete required provenance for a track or track case."""

    splits = (
        ("training", "validation", "test")
        if include_test_split
        else ("training", "validation")
    )
    namespaces = {
        split: track.seed_namespace(split) for split in splits
    }
    access_level = (
        controller_access_level(controller)
        if controller is not None
        else None
    )
    reasons = [str(reason) for reason in (eligibility_reasons or ())]
    provenance = {
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
        "training_seed_namespace_hash": seed_namespace_hash(
            namespaces["training"]
        ),
        "validation_seed_namespace_hash": seed_namespace_hash(
            namespaces["validation"]
        ),
        "controller_access_level": access_level,
        "model_access_level": access_level,
        "code_commit": code_commit(),
        "package_version": package_version(),
        "runtime_environment": runtime_environment(),
        "custom_override_hash": stable_hash(dict(custom_overrides or {})),
        "eligibility": eligibility_record(
            eligible=eligible,
            reasons=reasons,
        ),
        "eligibility_status": "eligible" if eligible else "ineligible",
        "eligibility_reasons": reasons,
    }
    if include_test_split:
        provenance.update(
            {
                "test_seed_namespace": namespaces["test"],
                "test_seed_namespace_hash": seed_namespace_hash(
                    namespaces["test"]
                ),
            }
        )
    return provenance


__all__ = [
    "ARTIFACT_PROVENANCE_SCHEMA_VERSION",
    "code_commit",
    "controller_access_level",
    "eligibility_record",
    "package_version",
    "reward_spec_hash",
    "runtime_environment",
    "seed_namespace_hash",
    "stable_hash",
    "track_provenance",
]
