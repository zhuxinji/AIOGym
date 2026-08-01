"""Generate reviewable v2 candidates for all official fixed anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aiogym.benchmarks.calibration import calibrate_anchor_manifest
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
        "cascade-recirculating-regulation-anchors-v2",
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
OFFICIAL_CALIBRATION_SEEDS = (9000, 9001, 9002, 9003, 9004)


def generate_official_anchor_candidates(output_dir: str | Path) -> tuple[Path, ...]:
    """Write candidates to a review directory without touching built-ins."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for track_id, anchor_id, bad_id, reference_id in OFFICIAL_CALIBRATIONS:
        output = destination / f"{anchor_id}.json"
        if output.exists():
            raise FileExistsError(f"anchor candidate already exists: {output}")
        payload = calibrate_anchor_manifest(
            load_track(track_id),
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
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    for output in generate_official_anchor_candidates(args.output_dir):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OFFICIAL_CALIBRATIONS",
    "OFFICIAL_CALIBRATION_SEEDS",
    "generate_official_anchor_candidates",
]
