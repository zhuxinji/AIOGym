"""Create a small text/JSON-only experiment review bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from aiogym._internal.serialization import file_sha256


BUNDLE_SCHEMA_VERSION = "aiogym.review_bundle.v1"
_EXCLUDED_SUFFIXES = {".zip", ".pt", ".pkl", ".npy", ".npz"}
_MAX_TEXT_BYTES = 5_000_000
_LOG_TAIL_LINES = 300


def make_review_bundle(
    source: str | Path,
    output: str | Path,
    *,
    runtime_report: str | Path | None = None,
    review_json: str | Path | None = None,
    review_markdown: str | Path | None = None,
    log: str | Path | None = None,
    overwrite: bool = False,
) -> dict:
    source_path = Path(source)
    output_path = Path(output)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"review bundle already exists: {output_path}")
    payload = _read_json(source_path)
    runs = payload.get("runs") if "runs" in payload else [payload]
    paths = [source_path]
    for run in runs:
        config = _reference_path(run.get("resolved_config_path"), source_path.parent)
        artifact_dir = _reference_path(run.get("artifact_dir"), source_path.parent)
        policy = _reference_path(run.get("policy_path"), source_path.parent)
        if config is not None:
            paths.append(config)
            run_result = config.with_name(
                config.name.replace(".resolved.json", ".run-result.json")
            )
            paths.append(run_result)
        if artifact_dir is not None:
            paths.extend(
                (artifact_dir / "benchmark.json", artifact_dir / "report.md")
            )
        if policy is not None:
            paths.append(Path(f"{policy}.selection.json"))
    paths.extend(
        Path(value)
        for value in (runtime_report, review_json, review_markdown)
        if value is not None
    )
    included = []
    skipped = []
    entries = {}
    used_names = set()
    for path in _unique_paths(paths):
        if not path.is_file():
            skipped.append({"path": str(path), "reason": "missing"})
            continue
        reason = _exclusion_reason(path)
        if reason is not None:
            skipped.append({"path": str(path), "reason": reason})
            continue
        arcname = _unique_arcname(path, used_names)
        data = path.read_bytes()
        entries[arcname] = data
        included.append(
            {
                "path": str(path),
                "archive_path": arcname,
                "sha256": file_sha256(path),
                "bytes": len(data),
            }
        )
    if log is not None:
        log_path = Path(log)
        if log_path.is_file():
            lines = log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            data = ("\n".join(lines[-_LOG_TAIL_LINES:]) + "\n").encode()
            entries["logs/training.tail.log"] = data
            included.append(
                {
                    "path": str(log_path),
                    "archive_path": "logs/training.tail.log",
                    "sha256": _bytes_sha256(data),
                    "bytes": len(data),
                    "tail_lines": min(len(lines), _LOG_TAIL_LINES),
                }
            )
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source": str(source_path),
        "included": included,
        "skipped": skipped,
        "exclusion_policy": {
            "binary_checkpoints": True,
            "replay_buffers": True,
            "tensorboard_events": True,
            "rollout_arrays": True,
            "large_files_over_bytes": _MAX_TEXT_BYTES,
        },
    }
    entries["MANIFEST.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output_path,
        "w" if overwrite else "x",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(name, data)
    return {**manifest, "bundle": str(output_path)}


def _exclusion_reason(path: Path) -> str | None:
    lower = path.name.lower()
    if path.suffix.lower() in _EXCLUDED_SUFFIXES:
        return "binary or array artifact"
    if lower.startswith("events.out.tfevents"):
        return "TensorBoard event"
    if "rollout" in lower and path.suffix.lower() not in {".json", ".md"}:
        return "rollout array"
    if path.stat().st_size > _MAX_TEXT_BYTES:
        return "file exceeds review size limit"
    if path.suffix.lower() not in {".json", ".md", ".txt", ".log"}:
        return "non-text review artifact"
    return None


def _unique_arcname(path: Path, used: set[str]) -> str:
    base = f"artifacts/{path.parent.name}/{path.name}"
    candidate = base
    counter = 2
    while candidate in used:
        candidate = f"artifacts/{path.parent.name}/{counter}-{path.name}"
        counter += 1
    used.add(candidate)
    return candidate


def _unique_paths(paths):
    seen = set()
    for path in paths:
        resolved = Path(path).resolve()
        if resolved not in seen:
            seen.add(resolved)
            yield resolved


def _reference_path(value, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("review bundle source must be a JSON mapping")
    return value


def _bytes_sha256(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-report")
    parser.add_argument("--review-json")
    parser.add_argument("--review-markdown")
    parser.add_argument("--log")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = make_review_bundle(
        args.source,
        args.output,
        runtime_report=args.runtime_report,
        review_json=args.review_json,
        review_markdown=args.review_markdown,
        log=args.log,
        overwrite=args.overwrite,
    )
    print(report["bundle"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
