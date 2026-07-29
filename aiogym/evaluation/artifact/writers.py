"""Standard benchmark artifact file writers."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from ..._internal.identifiers import canonicalize_artifact_ids
from ..._internal.serialization import write_json as _write_json
from ...models import collect_model_metadata
from .paths import resolve_artifact_path
from .tables import (
    _artifact_scenarios,
    _write_learning_curve_csv,
    _write_summary_csv,
)


def _write_benchmark_artifacts(out_dir: Path, payload: Mapping[str, Any]) -> dict[str, str]:
    artifacts = {"benchmark": str(out_dir / "benchmark.json")}
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    summary_path = data_dir / "summary.csv"
    rows = list(payload.get("rows", []))
    _write_summary_csv(summary_path, rows)
    artifacts["summary_csv"] = str(summary_path)

    model_metadata_artifacts = _write_model_metadata_artifacts(
        data_dir, payload
    )
    artifacts.update(model_metadata_artifacts)

    if payload.get("training"):
        training_path = data_dir / "training.json"
        _write_json(training_path, payload.get("training", {}))
        artifacts["training"] = str(training_path)
    if payload.get("learning_curve"):
        curve_json = data_dir / "learning_curve.json"
        curve_csv = data_dir / "learning_curve.csv"
        _write_json(curve_json, payload.get("learning_curve", []))
        _write_learning_curve_csv(curve_csv, payload.get("learning_curve", []))
        artifacts["learning_curve"] = str(curve_json)
        artifacts["learning_curve_csv"] = str(curve_csv)

    if payload.get("rollouts"):
        rollout_path = data_dir / "rollouts.json.gz"
        _write_json_gz(rollout_path, payload.get("rollouts", []))
        artifacts["rollouts"] = str(rollout_path)
    return artifacts


def _write_json_gz(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))


def _write_model_metadata_artifacts(
    data_dir: Path, payload: Mapping[str, Any]
) -> dict[str, str]:
    scenarios = _artifact_scenarios(payload)
    if not scenarios:
        return {}
    metadata = collect_model_metadata(scenarios)
    artifacts = {}
    models_dir = data_dir / "models"
    if len(metadata) == 1:
        _clear_json_files(models_dir)
        _, model_metadata = next(iter(metadata.items()))
        path = data_dir / "model_metadata.json"
        _write_json(path, canonicalize_artifact_ids(model_metadata))
        artifacts["model_metadata"] = str(path)
        return artifacts
    single_path = data_dir / "model_metadata.json"
    if single_path.exists():
        single_path.unlink()
    models_dir.mkdir(parents=True, exist_ok=True)
    _clear_json_files(models_dir)
    artifact_root = data_dir.parent
    manifest = {"scenarios": list(metadata), "models": {}}
    for scenario, model_metadata in metadata.items():
        path = models_dir / f"{scenario}.json"
        _write_json(path, canonicalize_artifact_ids(model_metadata))
        manifest["models"][scenario] = str(path.relative_to(artifact_root))
    manifest_path = models_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    artifacts["model_metadata_dir"] = str(models_dir)
    artifacts["model_metadata_manifest"] = str(manifest_path)
    return artifacts


def _clear_json_files(path: Path) -> None:
    if not path.exists():
        return
    for child in path.glob("*.json"):
        child.unlink()


def write_benchmark_artifacts(
    out_dir: str | Path, payload: Mapping[str, Any]
) -> dict[str, str]:
    """Write the standard artifact directory for a benchmark payload."""

    return _write_benchmark_artifacts(
        Path(out_dir), canonicalize_artifact_ids(dict(payload))
    )


def finalize_benchmark_artifacts(
    out_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    create_plots: bool = False,
    markdown_report: bool = False,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Write one canonical benchmark payload and its derived outputs."""

    root = Path(out_dir)
    if replace_existing:
        _clear_managed_artifacts(root)
    root.mkdir(parents=True, exist_ok=True)
    data = canonicalize_artifact_ids(dict(payload))
    data["artifacts"] = write_benchmark_artifacts(root, data)
    benchmark_path = root / "benchmark.json"
    persisted_data = dict(data)
    persisted_data.pop("rollouts", None)
    _write_json(benchmark_path, persisted_data)
    if create_plots:
        from .plotting import plot_results

        plot_results(root)
        with benchmark_path.open() as stream:
            persisted_data = json.load(stream)
    if markdown_report:
        from .report import render_benchmark_report

        report_path = root / "report.md"
        render_benchmark_report(root, out_path=report_path)
        persisted_data.setdefault("artifacts", {})
        persisted_data["artifacts"]["markdown_report"] = str(report_path)
        _write_json(benchmark_path, persisted_data)
    data["artifacts"] = persisted_data["artifacts"]
    return data


def compact_benchmark_artifacts(out_dir: str | Path) -> dict[str, Any]:
    """Rewrite one current run into the canonical compact layout."""

    root = Path(out_dir)
    benchmark_path = root / "benchmark.json"
    with benchmark_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    artifacts = dict(payload.get("artifacts") or {})
    raw_rollout_path = artifacts.get("rollouts")
    rollout_path = resolve_artifact_path(
        root,
        raw_rollout_path,
        "rollouts/rollouts.json",
    )
    if rollout_path.exists():
        opener = gzip.open if rollout_path.suffix == ".gz" else open
        with opener(rollout_path, "rt", encoding="utf-8") as stream:
            payload["rollouts"] = json.load(stream)
    payload.pop("artifacts", None)
    return finalize_benchmark_artifacts(
        root,
        payload,
        create_plots=True,
        markdown_report=True,
        replace_existing=True,
    )


def _clear_managed_artifacts(root: Path) -> None:
    """Remove only files and directories owned by the artifact writer."""

    managed = (
        "benchmark.json",
        "report.md",
        "data",
        "config",
        "metadata",
        "summary",
        "results",
        "rollouts",
        "figures",
        "training",
    )
    for name in managed:
        path = root / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
