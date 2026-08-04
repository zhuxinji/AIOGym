"""Static audit helpers for benchmark split isolation."""
from __future__ import annotations

from .registry import load_track
from .schema import TrackSpec


def audit_split_isolation(track) -> dict:
    """Report validation/test resolved identities without running episodes."""

    resolved = track if isinstance(track, TrackSpec) else load_track(track)
    validation = tuple(resolved.resolved_cases("validation"))
    test = tuple(resolved.resolved_cases("test"))
    validation_hashes = {case.resolved_case_hash for case in validation}
    test_hashes = {case.resolved_case_hash for case in test}
    overlapping = tuple(sorted(validation_hashes & test_hashes))
    return {
        "track_id": resolved.id,
        "track_hash": resolved.track_hash,
        "validation_cases": [
            {
                "case_id": case.case_id,
                "resolved_case_hash": case.resolved_case_hash,
            }
            for case in validation
        ],
        "test_cases": [
            {
                "case_id": case.case_id,
                "resolved_case_hash": case.resolved_case_hash,
            }
            for case in test
        ],
        "overlapping_hashes": list(overlapping),
        "passed": not overlapping,
    }


def require_split_isolation(track) -> None:
    """Reject a protocol whose fixed validation and test profiles overlap."""

    report = audit_split_isolation(track)
    if report["passed"]:
        return
    overlaps = ", ".join(report["overlapping_hashes"])
    raise ValueError(
        f"Track {report['track_id']!r} validation/test resolved Case "
        f"hashes overlap: {overlaps}"
    )


__all__ = ["audit_split_isolation", "require_split_isolation"]
