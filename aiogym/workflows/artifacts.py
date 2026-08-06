"""Shared Run artifact writer for design, collect, train, and evaluate."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from aiogym.core import file_sha256, stable_hash, write_json, write_text


RUN_SCHEMA_VERSION = "aiogym.run.v1"


def write_run_bundle(
    result: dict[str, Any],
    output: str | Path,
    *,
    workflow: str,
    report_markdown: str,
    overwrite: bool = False,
) -> dict[str, str]:
    directory = Path(output)
    targets = {
        "result": directory / "report.json",
        "report": directory / "report.md",
        "manifest": directory / "manifest.json",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing Run artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    directory.mkdir(parents=True, exist_ok=True)
    write_json(targets["result"], result, overwrite=overwrite)
    write_text(targets["report"], report_markdown.rstrip() + "\n", overwrite=overwrite)
    files = {
        "report.json": {
            "sha256": file_sha256(targets["result"]),
            "bytes": targets["result"].stat().st_size,
        },
        "report.md": {
            "sha256": file_sha256(targets["report"]),
            "bytes": targets["report"].stat().st_size,
        },
    }
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "workflow": workflow,
        "status": "complete",
        "run_hash": stable_hash(result),
        "task_id": result.get("task_id"),
        "task_hash": result.get("task_hash"),
        "plant_id": result.get("plant_id"),
        "plant_hash": result.get("plant_hash"),
        "seed": result.get("seed"),
        "seeds": result.get("seeds"),
        "preset": result.get("preset"),
        "policy": result.get("policy"),
        "files": files,
    }
    write_json(targets["manifest"], manifest, overwrite=overwrite)
    return {name: str(path) for name, path in targets.items()}
__all__ = ["RUN_SCHEMA_VERSION", "write_run_bundle"]
