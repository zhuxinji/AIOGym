#!/usr/bin/env python3
"""Reject incomplete or contaminated AIO-Gym release archives."""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


REQUIRED_PATHS = (
    "aiogym/__init__.py",
    "aiogym/_environment/env.py",
    "aiogym/controllers/export.py",
    "aiogym/cli/describe.py",
    "aiogym/datasets/profiles.py",
    "aiogym/rl/run_claim.py",
    "aiogym/rl/run_reference.py",
)
FORBIDDEN_PARTS = {
    "__MACOSX",
    ".DS_Store",
    "__pycache__",
    ".pytest_cache",
}
FORBIDDEN_ROOTS = ("runs/", "datasets/", "checkpoints/", "aiogym.egg-info/")
FORBIDDEN_SUFFIXES = (".pyc",)
LEGACY_PATH = "aiogym/_environment/runtime.py"


def _archive_names(path: Path) -> list[str]:
    if path.suffix in {".whl", ".zip"}:
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    if path.name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:gz") as archive:
            return archive.getnames()
    raise ValueError(f"unsupported release archive: {path}")


def _matches_member(name: str, expected: str) -> bool:
    normalized = str(PurePosixPath(name))
    return normalized == expected or normalized.endswith("/" + expected)


def _logical_path(name: str) -> str:
    parts = PurePosixPath(name).parts
    if parts and parts[0].startswith("aiogym-"):
        parts = parts[1:]
    return "/".join(parts)


def check_archive(path: str | Path) -> list[str]:
    """Return every contract violation found in one release archive."""

    archive_path = Path(path)
    names = _archive_names(archive_path)
    is_sdist = archive_path.name.endswith((".tar.gz", ".tgz"))
    errors: list[str] = []
    for required in REQUIRED_PATHS:
        if not any(_matches_member(name, required) for name in names):
            errors.append(f"missing required path: {required}")
    for name in names:
        parts = PurePosixPath(name).parts
        leaf = parts[-1] if parts else ""
        logical = _logical_path(name)
        if leaf.startswith("._"):
            errors.append(f"forbidden path: {name}")
            continue
        if any(part in FORBIDDEN_PARTS for part in parts):
            errors.append(f"forbidden path: {name}")
            continue
        setuptools_sdist_manifest = (
            is_sdist and logical == "aiogym.egg-info/SOURCES.txt"
        )
        if not setuptools_sdist_manifest and (
            logical.startswith(FORBIDDEN_ROOTS) or logical.startswith(
            "aiogym/tests/"
            )
        ):
            errors.append(f"forbidden path: {name}")
            continue
        if leaf.endswith(FORBIDDEN_SUFFIXES):
            errors.append(f"forbidden path: {name}")
    if any(_matches_member(name, LEGACY_PATH) for name in names):
        errors.append(f"legacy module is forbidden: {LEGACY_PATH}")
    return sorted(set(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args(argv)
    failed = False
    for archive in args.archives:
        try:
            errors = check_archive(archive)
        except (OSError, tarfile.TarError, zipfile.BadZipFile, ValueError) as exc:
            errors = [str(exc)]
        if errors:
            failed = True
            print(f"FAIL {archive}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {archive}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
