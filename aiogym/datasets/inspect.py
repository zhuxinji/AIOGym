"""CLI for inspecting Dataset v2 metadata and quality."""
from __future__ import annotations

import argparse
import json

from .quality import build_quality_report
from .reader import DatasetReader


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect an AIO-Gym dataset.")
    parser.add_argument("path")
    parser.add_argument("--quality", action="store_true")
    args = parser.parse_args(argv)
    reader = DatasetReader(args.path)
    payload = (
        build_quality_report(reader)
        if args.quality
        else {
            "dataset_id": reader.manifest["dataset_id"],
            "schema_version": reader.manifest["schema_version"],
            "split": reader.manifest["split"],
            "episodes": len(reader),
            "transitions": reader.transition_count,
            "collectors": sorted(
                {
                    record["collector_id"]
                    for record in reader.metadata_records()
                }
            ),
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
