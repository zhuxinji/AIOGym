"""Repository maintenance script for reviewable fixed-anchor candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiogym.benchmarks.calibration import calibrate_anchor_manifest
from aiogym.benchmarks.tracks.audit import require_split_isolation
from aiogym.benchmarks.tracks.registry import load_track


OFFICIAL_CALIBRATIONS = (
    (
        "cascade-economic-specialist-v1",
        "cascade-economic-anchors-v2",
        "hold",
        "oracle",
    ),
    (
        "cascade-regulation-generalist-v1",
        "cascade-regulation-anchors-v2",
        "hold",
        "mpc",
    ),
    (
        "cascade-recirculating-regulation-generalist-v1",
        "cascade-recirculating-regulation-v1-anchors-v3",
        "hold",
        "mpc",
    ),
    (
        "quadruple-regulation-generalist-v1",
        "quadruple-regulation-anchors-v2",
        "hold",
        "pid",
    ),
)
V2_REGULATION_CALIBRATIONS = (
    (
        "quadruple-regulation-generalist-v2",
        "quadruple-regulation-anchors-v3",
        "hold",
        "pid",
    ),
    (
        "cascade-regulation-generalist-v2",
        "cascade-regulation-anchors-v3",
        "hold",
        "mpc",
    ),
    (
        "cascade-recirculating-regulation-generalist-v2",
        "cascade-recirculating-regulation-anchors-v3",
        "hold",
        "mpc",
    ),
)
CALIBRATION_SUITES = {
    "v1": OFFICIAL_CALIBRATIONS,
    "v1-cascade-recirculating": OFFICIAL_CALIBRATIONS[2:3],
    "v2-quadruple": V2_REGULATION_CALIBRATIONS[:1],
    "v2-cascade": V2_REGULATION_CALIBRATIONS[1:2],
    "v2-cascade-recirculating": V2_REGULATION_CALIBRATIONS[2:],
    "v2-regulation": V2_REGULATION_CALIBRATIONS,
    "all": (*OFFICIAL_CALIBRATIONS, *V2_REGULATION_CALIBRATIONS),
}
OFFICIAL_CALIBRATION_SEEDS = (9000, 9001, 9002, 9003, 9004)


def generate_official_anchor_candidates(
    output_dir: str | Path,
    *,
    suite: str = "v1",
) -> tuple[Path, ...]:
    """Write candidates to a review directory without touching built-ins."""

    try:
        calibrations = CALIBRATION_SUITES[str(suite)]
    except KeyError as exc:
        choices = ", ".join(sorted(CALIBRATION_SUITES))
        raise ValueError(
            f"unknown anchor calibration suite {suite!r}; choose: {choices}"
        ) from exc
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    candidate_paths = tuple(
        destination / f"{anchor_id}.json"
        for _, anchor_id, _, _ in calibrations
    )
    existing = [path for path in candidate_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "anchor candidate already exists: "
            + ", ".join(str(path) for path in existing)
        )
    outputs = []
    for track_id, anchor_id, bad_id, reference_id in calibrations:
        output = destination / f"{anchor_id}.json"
        track = load_track(track_id)
        if track_id.endswith("-v2"):
            require_split_isolation(track)
        payload = calibrate_anchor_manifest(
            track,
            anchor_id=anchor_id,
            bad_controller_id=bad_id,
            reference_controller_id=reference_id,
            base_seeds=OFFICIAL_CALIBRATION_SEEDS,
        )
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        outputs.append(output)
    return tuple(outputs)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=tuple(CALIBRATION_SUITES),
        default="v1",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    for output in generate_official_anchor_candidates(
        args.output_dir,
        suite=args.suite,
    ):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_SUITES",
    "OFFICIAL_CALIBRATIONS",
    "OFFICIAL_CALIBRATION_SEEDS",
    "V2_REGULATION_CALIBRATIONS",
    "generate_official_anchor_candidates",
]
