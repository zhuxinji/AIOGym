"""Immutable fixed-anchor manifests for official benchmark ranking."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ANCHOR_SCHEMA_VERSION = "aiogym.ranking_anchors.v1"
BUILTIN_ANCHOR_DIR = Path(__file__).with_name("builtin")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def anchor_artifact_hash(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("artifact_hash", None)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CaseAnchors:
    case_id: str
    resolved_case_hash: str
    bad_utility: float
    reference_utility: float
    bad_controller: Mapping[str, Any]
    reference_controller: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.case_id or not self.resolved_case_hash:
            raise ValueError("anchor case identity must be non-empty")
        if not math.isfinite(self.bad_utility) or not math.isfinite(
            self.reference_utility
        ):
            raise ValueError("anchor utilities must be finite")
        if self.reference_utility <= self.bad_utility:
            raise ValueError(
                f"reference utility must exceed bad utility for "
                f"{self.case_id!r}"
            )


@dataclass(frozen=True)
class AnchorSet:
    anchor_id: str
    track_id: str
    track_hash: str
    ranking_spec_id: str
    goal: str
    bad_controller: Mapping[str, Any]
    reference_controller: Mapping[str, Any]
    evaluation_seeds: tuple[int, ...]
    cases: Mapping[str, CaseAnchors]
    artifact_hash: str

    def case(self, resolved_case_hash: str) -> CaseAnchors:
        try:
            return self.cases[str(resolved_case_hash)]
        except KeyError as exc:
            raise KeyError(
                "anchor set has no entry for resolved case hash "
                f"{resolved_case_hash!r}"
            ) from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": ANCHOR_SCHEMA_VERSION,
            "id": self.anchor_id,
            "track_id": self.track_id,
            "track_hash": self.track_hash,
            "ranking_spec_id": self.ranking_spec_id,
            "goal": self.goal,
            "bad_controller": deepcopy(dict(self.bad_controller)),
            "reference_controller": deepcopy(
                dict(self.reference_controller)
            ),
            "evaluation_seeds": list(self.evaluation_seeds),
            "cases": {
                key: {
                    "case_id": row.case_id,
                    "resolved_case_hash": row.resolved_case_hash,
                    "bad_utility": row.bad_utility,
                    "reference_utility": row.reference_utility,
                    "bad_controller": deepcopy(
                        dict(row.bad_controller)
                    ),
                    "reference_controller": deepcopy(
                        dict(row.reference_controller)
                    ),
                }
                for key, row in sorted(self.cases.items())
            },
            "artifact_hash": self.artifact_hash,
        }


def load_anchor_set(
    source: str | Path | Mapping[str, Any],
    *,
    track=None,
) -> AnchorSet:
    if isinstance(source, Mapping):
        payload = deepcopy(dict(source))
    else:
        path = Path(source)
        if isinstance(source, str) and "/" not in source and not path.suffix:
            path = BUILTIN_ANCHOR_DIR / f"{source}.json"
        if not path.is_file():
            raise FileNotFoundError(f"ranking anchor set not found: {source}")
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    if payload.get("schema_version") != ANCHOR_SCHEMA_VERSION:
        raise ValueError("unsupported ranking anchor schema")
    expected_hash = anchor_artifact_hash(payload)
    if payload.get("artifact_hash") != expected_hash:
        raise ValueError("ranking anchor artifact hash mismatch")
    cases = {}
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Mapping) or not raw_cases:
        raise ValueError("ranking anchor set requires cases")
    for key, value in raw_cases.items():
        row = CaseAnchors(
            case_id=str(value["case_id"]),
            resolved_case_hash=str(value["resolved_case_hash"]),
            bad_utility=float(value["bad_utility"]),
            reference_utility=float(value["reference_utility"]),
            bad_controller=deepcopy(dict(value["bad_controller"])),
            reference_controller=deepcopy(
                dict(value["reference_controller"])
            ),
        )
        if str(key) != row.resolved_case_hash:
            raise ValueError("anchor case key must equal resolved_case_hash")
        cases[str(key)] = row
    seeds = tuple(int(seed) for seed in payload.get("evaluation_seeds", ()))
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("anchor evaluation seeds must be non-empty and unique")
    anchors = AnchorSet(
        anchor_id=str(payload["id"]),
        track_id=str(payload["track_id"]),
        track_hash=str(payload["track_hash"]),
        ranking_spec_id=str(payload["ranking_spec_id"]),
        goal=str(payload["goal"]),
        bad_controller=deepcopy(dict(payload["bad_controller"])),
        reference_controller=deepcopy(dict(payload["reference_controller"])),
        evaluation_seeds=seeds,
        cases=cases,
        artifact_hash=expected_hash,
    )
    if track is not None:
        if anchors.track_id != track.id:
            raise ValueError("ranking anchor Track ID mismatch")
        if anchors.track_hash != track.track_hash:
            raise ValueError("ranking anchor Track hash mismatch")
        if anchors.ranking_spec_id != track.ranking_spec_id:
            raise ValueError("ranking anchor spec ID mismatch")
        if anchors.goal != track.goal:
            raise ValueError("ranking anchor goal mismatch")
    return anchors


__all__ = [
    "ANCHOR_SCHEMA_VERSION",
    "BUILTIN_ANCHOR_DIR",
    "AnchorSet",
    "CaseAnchors",
    "anchor_artifact_hash",
    "load_anchor_set",
]
