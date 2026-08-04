"""Statically review fixed-anchor candidates without running controllers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiogym._internal.serialization import atomic_write_json
from aiogym.benchmarks.anchors import (
    anchor_artifact_hash,
    load_anchor_set,
)
from aiogym.benchmarks.tracks import (
    audit_split_isolation,
    load_track,
)


REVIEW_SCHEMA_VERSION = "aiogym.anchor_candidate_review.v1"


def review_anchor_candidate(path: str | Path) -> dict:
    """Validate one candidate and return compact audit evidence."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    track = load_track(str(payload["track_id"]))
    failures = []
    actual_artifact_hash = anchor_artifact_hash(payload)
    if payload.get("artifact_hash") != actual_artifact_hash:
        failures.append("artifact hash mismatch")
    if payload.get("id") != track.ranking_declaration["anchor_id"]:
        failures.append("anchor ID does not match Track ranking declaration")
    split_audit = audit_split_isolation(track)
    if track.id.endswith("-v2") and not split_audit["passed"]:
        failures.append("v2 validation/test split isolation failed")

    anchors = None
    try:
        anchors = load_anchor_set(payload, track=track)
    except (KeyError, TypeError, ValueError) as exc:
        failures.append(str(exc))

    expected_cases = {
        case.resolved_case_hash: case.case_id
        for split in ("validation", "test")
        for case in track.resolved_cases(split)
    }
    raw_cases = dict(payload.get("cases") or {})
    missing = sorted(set(expected_cases) - set(raw_cases))
    extra = sorted(set(raw_cases) - set(expected_cases))
    if missing:
        failures.append("missing resolved Case hashes")
    if extra:
        failures.append("unexpected resolved Case hashes")

    cases = []
    for case_hash, raw in sorted(raw_cases.items()):
        quality = dict(raw.get("quality") or {})
        bad = list(quality.get("bad_eligible") or ())
        reference = list(quality.get("reference_eligible") or ())
        row_failures = list(quality.get("failure_reasons") or ())
        if not all(reference):
            row_failures.append("reference baseline is not safety eligible")
        if float(raw.get("reference_utility", 0.0)) <= float(
            raw.get("bad_utility", 0.0)
        ):
            row_failures.append("reference utility does not exceed bad utility")
        cases.append(
            {
                "case_id": str(raw.get("case_id", expected_cases.get(case_hash, ""))),
                "resolved_case_hash": str(case_hash),
                "bad_utility": raw.get("bad_utility"),
                "reference_utility": raw.get("reference_utility"),
                "absolute_gap": quality.get("absolute_gap"),
                "relative_gap": quality.get("relative_gap"),
                "bad_eligible": bad,
                "reference_eligible": reference,
                "failure_reasons": row_failures,
            }
        )
        failures.extend(
            f"{case_hash}: {reason}" for reason in row_failures
        )

    dependency_versions = dict(
        (payload.get("calibration") or {}).get("dependency_versions") or {}
    )
    if not dependency_versions:
        failures.append("dependency provenance is missing")
    return {
        "path": str(source.resolve()),
        "track_id": track.id,
        "track_hash": track.track_hash,
        "anchor_id": str(payload.get("id", "")),
        "ranking_spec_id": str(payload.get("ranking_spec_id", "")),
        "artifact_hash": str(payload.get("artifact_hash", "")),
        "artifact_hash_valid": (
            payload.get("artifact_hash") == actual_artifact_hash
        ),
        "evaluation_seeds": list(payload.get("evaluation_seeds") or ()),
        "dependency_versions": dependency_versions,
        "split_isolation": split_audit,
        "missing_case_hashes": missing,
        "extra_case_hashes": extra,
        "case_count": len(cases),
        "cases": cases,
        "schema_valid": anchors is not None,
        "failure_reasons": sorted(set(failures)),
        "passed": not failures,
    }


def review_anchor_candidates(source: str | Path) -> dict:
    """Review every candidate JSON in a directory or one explicit file."""

    root = Path(source)
    paths = (root,) if root.is_file() else tuple(sorted(root.glob("*.json")))
    if not paths:
        raise FileNotFoundError(f"no anchor candidates found: {root}")
    candidates = [review_anchor_candidate(path) for path in paths]
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "source": str(root.resolve()),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "passed": all(row["passed"] for row in candidates),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = review_anchor_candidates(args.source)
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REVIEW_SCHEMA_VERSION",
    "review_anchor_candidate",
    "review_anchor_candidates",
]
