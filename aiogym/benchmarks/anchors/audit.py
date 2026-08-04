"""Audit built-in fixed-anchor manifests and their quality metadata."""
from __future__ import annotations

import json

from aiogym.benchmarks.anchors import load_anchor_set
from aiogym.benchmarks.tracks.registry import list_tracks, load_track


def audit_builtin_anchors() -> list[dict]:
    rows = []
    for track_id in list_tracks():
        track = load_track(track_id)
        anchor_id = track.ranking_declaration.get("anchor_id")
        if not anchor_id:
            continue
        anchors = load_anchor_set(anchor_id, track=track)
        for case in sorted(anchors.cases.values(), key=lambda row: row.case_id):
            quality = case.quality
            rows.append(
                {
                    "anchor_id": anchors.anchor_id,
                    "artifact_hash": anchors.artifact_hash,
                    "track_id": track.id,
                    "track_hash": track.track_hash,
                    "case_id": case.case_id,
                    "resolved_case_hash": case.resolved_case_hash,
                    "bad_utility": case.bad_utility,
                    "reference_utility": case.reference_utility,
                    "absolute_gap": quality["absolute_gap"],
                    "relative_gap": quality["relative_gap"],
                    "signal_to_noise": quality["signal_to_noise"],
                    "quality_passed": quality["passed"],
                }
            )
    return rows


def main() -> int:
    print(json.dumps(audit_builtin_anchors(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_builtin_anchors"]
