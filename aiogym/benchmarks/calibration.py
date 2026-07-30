"""Offline generation of reviewable fixed-anchor candidate manifests."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from aiogym._environment.builder import build_track_case_environment
from aiogym.controllers import make_controller
from aiogym.evaluation import evaluate_controller
from aiogym.models.cases import case_controller_config

from .anchors import ANCHOR_SCHEMA_VERSION, anchor_artifact_hash
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

    seeds = tuple(int(seed) for seed in base_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("anchor calibration seeds must be non-empty and unique")
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
            try:
                result = evaluate_controller(
                    controller,
                    env,
                    episodes=len(seeds),
                    seed_list=seeds,
                    goal_specification=track.goal,
                    seed_namespace=track.seed_namespace(split),
                    include_episodes=False,
                    safety_gate_spec=track.safety_gate_spec(),
                )
                horizon = float(env.control_dt * env.episode_steps)
            finally:
                env.close()
            if not result.get("ranking_eligible", False):
                raise ValueError(
                    f"anchor controller {controller_id!r} is unsafe on "
                    f"case {case.case_id!r}"
                )
            utility = case_utility(
                {
                    **result,
                    "case_horizon_seconds": horizon,
                },
                goal=track.goal,
            )
            utilities[controller_id] = utility
            frozen_controllers[controller_id] = _frozen_controller_metadata(
                controller_id,
                controller,
                config,
            )
        bad_utility = utilities[bad_controller_id]
        reference_utility = utilities[reference_controller_id]
        if reference_utility <= bad_utility:
            raise ValueError(
                f"reference utility does not exceed bad utility for "
                f"{case.case_id!r}: {reference_utility} <= {bad_utility}"
            )
        cases[case_hash] = {
            "case_id": case.case_id,
            "resolved_case_hash": case_hash,
            "bad_utility": bad_utility,
            "reference_utility": reference_utility,
            "bad_controller": frozen_controllers[bad_controller_id],
            "reference_controller": frozen_controllers[
                reference_controller_id
            ],
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


__all__ = ["calibrate_anchor_manifest"]
