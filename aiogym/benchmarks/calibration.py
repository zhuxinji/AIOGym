"""Offline generation of reviewable fixed-anchor candidate manifests."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from collections.abc import Sequence
from typing import Any

from aiogym._environment.builder import build_track_case_environment
from aiogym.controllers.registry import make_controller
from aiogym.evaluation.execution import evaluate_controller
from aiogym.models.cases import case_controller_config
from aiogym._internal.validation import seed_sequence

from .anchors import ANCHOR_SCHEMA_VERSION, anchor_artifact_hash
from .anchors.quality import anchor_quality_statistics
from .ranking import case_utility


def calibrate_anchor_manifest(
    track,
    *,
    anchor_id: str,
    bad_controller_id: str,
    reference_controller_id: str,
    base_seeds: Sequence[int],
) -> dict[str, Any]:
    """Evaluate frozen baselines without mutating official anchor files."""

    seeds = seed_sequence("anchor calibration seeds", base_seeds)
    unique_cases = {}
    for split in ("validation", "test"):
        for case in track.resolved_cases(split):
            unique_cases.setdefault(case.resolved_case_hash, (split, case))
    cases = {}
    for case_hash, (split, case) in unique_cases.items():
        utilities = {}
        frozen_controllers = {}
        for controller_id in (
            bad_controller_id,
            reference_controller_id,
        ):
            env = build_track_case_environment(track, case)
            config = case_controller_config(
                case.profile,
                controller_id,
            )
            controller = make_controller(
                controller_id,
                model=env.model,
                scenario=track.scenario,
                config=config,
            )
            samples = []
            eligibility = []
            ineligibility_reasons = []
            try:
                horizon = float(env.control_dt * env.episode_steps)
                for seed in seeds:
                    result = evaluate_controller(
                        controller,
                        env,
                        episodes=1,
                        seed_list=(seed,),
                        goal_specification=track.goal,
                        seed_namespace=track.seed_namespace(split),
                        include_episodes=False,
                        safety_gate_spec=track.safety_gate_spec(),
                    )
                    samples.append(
                        case_utility(
                            {
                                **result,
                                "case_horizon_seconds": horizon,
                            },
                            goal=track.goal,
                        )
                    )
                    eligibility.append(
                        bool(result.get("ranking_eligible", False))
                    )
                    ineligibility_reasons.append(
                        list(
                            dict(result.get("safety_gate") or {}).get(
                                "reasons",
                                (),
                            )
                        )
                    )
            finally:
                env.close()
            utilities[controller_id] = samples
            utilities[f"{controller_id}:eligible"] = eligibility
            utilities[f"{controller_id}:reasons"] = ineligibility_reasons
            frozen_controllers[controller_id] = _frozen_controller_metadata(
                controller_id,
                controller,
                config,
            )
        quality = anchor_quality_statistics(
            utilities[bad_controller_id],
            utilities[reference_controller_id],
            bad_eligible=utilities[f"{bad_controller_id}:eligible"],
            reference_eligible=utilities[
                f"{reference_controller_id}:eligible"
            ],
        )
        if not quality["passed"]:
            safety_details = []
            for controller_id in (
                bad_controller_id,
                reference_controller_id,
            ):
                reasons = utilities[f"{controller_id}:reasons"]
                nonempty = [
                    {"seed": seed, "reasons": row}
                    for seed, row in zip(seeds, reasons)
                    if row
                ]
                if nonempty:
                    safety_details.append(
                        f"{controller_id} safety={nonempty}"
                    )
            raise ValueError(
                f"anchor quality failed for {case.case_id!r}: "
                + "; ".join(
                    [*quality["failure_reasons"], *safety_details]
                )
            )
        bad_utility = quality["bad_mean"]
        reference_utility = quality["reference_mean"]
        cases[case_hash] = {
            "case_id": case.case_id,
            "resolved_case_hash": case_hash,
            "bad_utility": bad_utility,
            "reference_utility": reference_utility,
            "bad_controller": frozen_controllers[bad_controller_id],
            "reference_controller": frozen_controllers[
                reference_controller_id
            ],
            "quality": quality,
        }
    payload = {
        "schema_version": ANCHOR_SCHEMA_VERSION,
        "id": str(anchor_id),
        "track_id": track.id,
        "track_hash": track.track_hash,
        "ranking_spec_id": track.ranking_spec_id,
        "goal": track.goal,
        "bad_controller": {
            "id": str(bad_controller_id),
            "config_scope": "per-case",
        },
        "reference_controller": {
            "id": str(reference_controller_id),
            "config_scope": "per-case",
        },
        "evaluation_seeds": list(seeds),
        "calibration": {
            "generator": "aiogym.benchmarks.calibration.calibrate_anchor_manifest",
            "paired_seed_evaluation": True,
            "python_version": platform.python_version(),
            "dependency_versions": _dependency_versions(),
        },
        "cases": dict(sorted(cases.items())),
    }
    payload["artifact_hash"] = anchor_artifact_hash(payload)
    return payload


def _frozen_controller_metadata(
    controller_id: str,
    controller,
    source_config,
) -> dict:
    metadata = dict(controller.metadata())
    canonical = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return {
        "id": str(controller_id),
        "controller_api_version": str(
            metadata.get("api", "unknown")
        ),
        "source_config": dict(source_config),
        "controller_config_hash": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
    }


def _dependency_versions() -> dict[str, str]:
    resolved = {}
    for distribution in ("aiogym", "numpy", "gymnasium", "casadi"):
        try:
            resolved[distribution] = version(distribution)
        except PackageNotFoundError:
            resolved[distribution] = "not-installed"
    return resolved


__all__ = ["calibrate_anchor_manifest"]
