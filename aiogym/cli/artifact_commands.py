#!/usr/bin/env python3
"""Command handlers for inspecting benchmark artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aiogym.evaluation import check_benchmark_artifacts, render_benchmark_report


def report_main(argv=None, prog=None):
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Render report.md from a standard benchmark artifact directory."
    )
    ap.add_argument("artifact_dir", help="standard benchmark artifact directory")
    ap.add_argument("--out", default=None, help="report path; defaults to <artifact_dir>/report.md")
    ap.add_argument("--stdout", action="store_true", help="also print the report")
    args = ap.parse_args(argv)

    artifact_dir = Path(args.artifact_dir)
    out_path = Path(args.out) if args.out else artifact_dir / "report.md"
    text = render_benchmark_report(artifact_dir, out_path=out_path)
    if args.stdout:
        print(text)
    print(f"saved report {out_path}")


def artifact_check_main(argv=None, prog=None):
    ap = argparse.ArgumentParser(
        prog=prog,
        description="Validate a standard benchmark artifact directory."
    )
    ap.add_argument("artifact_dir", help="standard benchmark artifact directory")
    ap.add_argument("--json", action="store_true", help="print structured JSON")
    args = ap.parse_args(argv)

    result = check_benchmark_artifacts(args.artifact_dir)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status = "OK" if result["ok"] else "FAIL"
        print(f"{status} {result['artifact_dir']}")
        for row in result["checks"]:
            mark = "OK" if row["ok"] else "FAIL"
            print(f"  {mark:4s} {row['name']}: {row['message']} ({row['path']})")
    if not result["ok"]:
        raise SystemExit(1)


def main(argv=None):
    commands = {
        "report": report_main,
        "check": artifact_check_main,
    }
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m aiogym.cli.artifact_commands {report,check} ...")
        return
    command = args[0]
    if command not in commands:
        choices = ", ".join(commands)
        raise SystemExit(f"unknown artifacts command {command!r}; choose one of: {choices}")
    commands[command](args[1:])


if __name__ == "__main__":
    main()
