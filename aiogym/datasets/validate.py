"""CLI for checksum and schema validation of Dataset v2."""
from __future__ import annotations

import argparse
import json

from .reader import validate_dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate an AIO-Gym dataset.")
    parser.add_argument("path")
    args = parser.parse_args(argv)
    report = validate_dataset(args.path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
