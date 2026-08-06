"""One-file experiment workflow built on the stable training lifecycle."""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from aiogym._internal.serialization import (
    atomic_write_json,
    stable_json_hash,
)
from aiogym._internal.validation import seed_sequence
from aiogym.rl.config import RLTrainingConfig
from aiogym.rl.runner import RunResult, run_experiment, run_seed_sweep


EXPERIMENT_SCHEMA_VERSION = "aiogym.experiment.v1"
EXPERIMENT_RESULT_SCHEMA_VERSION = "aiogym.experiment_result.v1"
EXPERIMENT_METRICS_SCHEMA_VERSION = "aiogym.experiment_metrics.v1"
_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "description",
        "output_dir",
        "seeds",
        "training",
    }
)
_RESERVED_TRAINING_OUTPUT_FIELDS = frozenset(
    {"directory", "name", "artifact_dir"}
)


@dataclass(frozen=True)
class ExperimentSpec:
    """A complete experiment declaration with one output directory."""

    name: str
    description: str
    output_dir: Path
    seeds: tuple[int, ...]
    training: RLTrainingConfig
    schema_version: str = EXPERIMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXPERIMENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported experiment schema: {self.schema_version!r}"
            )
        if not _file_safe_name(self.name):
            raise ValueError(
                "experiment name must be a non-empty file-safe name"
            )
        if not isinstance(self.description, str):
            raise TypeError("experiment description must be a string")
        output_dir = Path(self.output_dir)
        if output_dir in {Path(""), Path(".")}:
            raise ValueError("experiment output_dir must name a directory")
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "seeds", seed_sequence("seeds", self.seeds))
        if not isinstance(self.training, RLTrainingConfig):
            raise TypeError("experiment training must be an RLTrainingConfig")
        reserved = _RESERVED_TRAINING_OUTPUT_FIELDS & set(
            self.training.output
        )
        if reserved:
            raise ValueError(
                "experiment owns output paths; remove training.output fields: "
                + ", ".join(sorted(reserved))
            )

    @property
    def spec_hash(self) -> str:
        return stable_json_hash(self.as_dict(), ensure_ascii=False)

    @property
    def internal_dir(self) -> Path:
        return self.output_dir / "internal"

    @property
    def training_dir(self) -> Path:
        return self.internal_dir / "training"

    def training_config(self) -> RLTrainingConfig:
        output = {
            **dict(self.training.output),
            "directory": str(self.training_dir),
            "name": "model",
        }
        return replace(
            self.training,
            training_seed=self.seeds[0],
            output=output,
        )

    def with_output(self, output_dir: str | Path) -> "ExperimentSpec":
        return replace(self, output_dir=Path(output_dir))

    def as_dict(self) -> dict[str, Any]:
        training = self.training.as_dict()
        training.pop("training_seed", None)
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "output_dir": str(self.output_dir),
            "seeds": list(self.seeds),
            "training": training,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentSpec":
        if not isinstance(value, Mapping):
            raise TypeError("experiment config must be a mapping")
        data = dict(value)
        unknown = set(data) - _FIELDS
        if unknown:
            raise ValueError(
                "unknown experiment config fields: "
                + ", ".join(sorted(unknown))
            )
        missing = _FIELDS - set(data)
        if missing:
            raise ValueError(
                "missing experiment config fields: "
                + ", ".join(sorted(missing))
            )
        seeds = seed_sequence("seeds", data["seeds"])
        training = data["training"]
        if not isinstance(training, Mapping):
            raise TypeError("experiment training must be a mapping")
        training_data = dict(training)
        if "training_seed" in training_data:
            raise ValueError(
                "experiment seeds belong at top level; remove "
                "training.training_seed"
            )
        training_data["training_seed"] = seeds[0]
        return cls(
            schema_version=data["schema_version"],
            name=data["name"],
            description=data["description"],
            output_dir=Path(data["output_dir"]),
            seeds=seeds,
            training=RLTrainingConfig.from_mapping(training_data),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentSpec":
        source = Path(path)
        with source.open(encoding="utf-8") as stream:
            value = json.load(stream)
        return cls.from_mapping(value)


def load_training_config(path: str | Path) -> RLTrainingConfig:
    """Load either an ExperimentSpec or a low-level training config."""

    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise TypeError("configuration must be a mapping")
    if value.get("schema_version") == EXPERIMENT_SCHEMA_VERSION:
        return ExperimentSpec.from_mapping(value).training_config()
    return RLTrainingConfig.from_mapping(value)


def run_experiment_spec(
    spec: ExperimentSpec,
    *,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Execute one ExperimentSpec and commit its public result last."""

    if not isinstance(spec, ExperimentSpec):
        raise TypeError("spec must be an ExperimentSpec")
    config = spec.training_config()
    resolved = {
        "dry_run": bool(dry_run),
        "schema_version": spec.schema_version,
        "name": spec.name,
        "description": spec.description,
        "spec_hash": spec.spec_hash,
        "output_dir": str(spec.output_dir),
        "seeds": list(spec.seeds),
        "track_id": config.track_id,
        "algorithm_id": config.algorithm_id,
        "training": config.as_dict(),
        "public_artifacts": {
            "run": str(spec.output_dir / "run.json"),
            "metrics": str(spec.output_dir / "metrics.json"),
            "report": str(spec.output_dir / "report.md"),
        },
        "internal_dir": str(spec.internal_dir),
    }
    if dry_run:
        return resolved

    run_path = spec.output_dir / "run.json"
    public_paths = (
        run_path,
        spec.output_dir / "metrics.json",
        spec.output_dir / "report.md",
        spec.internal_dir / "resolved-experiment.json",
    )
    occupied = tuple(path for path in public_paths if path.exists())
    if occupied and not overwrite:
        raise FileExistsError(
            "experiment output already exists: "
            + ", ".join(str(path) for path in occupied)
        )
    spec.internal_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        spec.internal_dir / "resolved-experiment.json",
        spec.as_dict(),
        overwrite=overwrite,
    )

    if len(spec.seeds) == 1:
        result = run_experiment(config, overwrite=overwrite)
        runs = [result.as_dict()]
        internal_result = runs[0]
        internal_result_path = _run_result_path(result)
        validation_summary = _single_validation_summary(result)
        training_seed_statistics = {}
    else:
        internal_result = run_seed_sweep(
            config,
            spec.seeds,
            overwrite=overwrite,
        )
        runs = list(internal_result["runs"])
        internal_result_path = Path(internal_result["summary_path"])
        validation_summary = dict(
            internal_result.get("validation_summary") or {}
        )
        training_seed_statistics = dict(
            internal_result.get("training_seed_statistics") or {}
        )

    metrics = {
        "schema_version": EXPERIMENT_METRICS_SCHEMA_VERSION,
        "experiment": spec.name,
        "spec_hash": spec.spec_hash,
        "track_id": config.track_id,
        "algorithm_id": config.algorithm_id,
        "training_seeds": list(spec.seeds),
        "validation_summary": validation_summary,
        "training_seed_statistics": training_seed_statistics,
        "runs": [
            {
                "training_seed": row["training_seed"],
                "validation": dict(row.get("validation") or {}),
            }
            for row in runs
        ],
    }
    metrics_path = atomic_write_json(
        spec.output_dir / "metrics.json",
        metrics,
        overwrite=overwrite,
    )
    report_path = spec.output_dir / "report.md"
    _atomic_write_text(
        report_path,
        _render_report(spec, metrics),
        overwrite=overwrite,
    )
    public_result = {
        "schema_version": EXPERIMENT_RESULT_SCHEMA_VERSION,
        "status": "complete",
        "name": spec.name,
        "description": spec.description,
        "spec_hash": spec.spec_hash,
        "track_id": config.track_id,
        "algorithm_id": config.algorithm_id,
        "training_seeds": list(spec.seeds),
        "output_dir": str(spec.output_dir),
        "models": [
            {
                "training_seed": row["training_seed"],
                "path": _portable_path(row["policy_path"], spec.output_dir),
                "sha256": row["policy_sha256"],
            }
            for row in runs
        ],
        "metrics": metrics_path.name,
        "report": report_path.name,
        "internal": {
            "resolved_experiment": "internal/resolved-experiment.json",
            "training_result": _portable_path(
                internal_result_path, spec.output_dir
            ),
        },
    }
    atomic_write_json(run_path, public_result, overwrite=overwrite)
    return public_result


def _single_validation_summary(result: RunResult) -> dict[str, Any]:
    aggregate = dict((result.validation or {}).get("aggregate") or {})
    if not aggregate:
        return {}
    return {
        "metric": aggregate.get("metric"),
        "metric_direction": aggregate.get("metric_direction"),
        "mean_metric_value": aggregate.get("metric_value"),
        "mean_official_score": aggregate.get("official_score"),
        "eligible_runs": int(bool(aggregate.get("ranking_eligible", True))),
        "run_count": 1,
    }


def _run_result_path(result: RunResult) -> Path:
    resolved = Path(result.resolved_config_path)
    name = resolved.name.removesuffix(".resolved.json")
    return resolved.with_name(f"{name}.run-result.json")


def _render_report(spec: ExperimentSpec, metrics: Mapping[str, Any]) -> str:
    summary = dict(metrics.get("validation_summary") or {})
    lines = [
        f"# {spec.name}",
        "",
        spec.description or "AIO-Gym experiment.",
        "",
        "- Status: complete",
        f"- Track: `{metrics['track_id']}`",
        f"- Algorithm: `{metrics['algorithm_id']}`",
        "- Training seeds: "
        + ", ".join(str(seed) for seed in metrics["training_seeds"]),
    ]
    if summary:
        lines.extend(
            [
                f"- Validation metric: `{summary.get('metric')}`",
                f"- Mean metric value: {summary.get('mean_metric_value')}",
                f"- Mean official score: {summary.get('mean_official_score')}",
                f"- Ranking-eligible runs: {summary.get('eligible_runs')}/{summary.get('run_count')}",
            ]
        )
    lines.extend(
        [
            "",
            "Detailed checkpoints, resolved configurations, and provenance are "
            "kept under `internal/`.",
            "",
        ]
    )
    return "\n".join(lines)


def _portable_path(value: str | Path, root: Path) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(path)


def _atomic_write_text(path: Path, value: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"text target already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_safe_name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and Path(value).name == value
    )


__all__ = [
    "EXPERIMENT_METRICS_SCHEMA_VERSION",
    "EXPERIMENT_RESULT_SCHEMA_VERSION",
    "EXPERIMENT_SCHEMA_VERSION",
    "ExperimentSpec",
    "load_training_config",
    "run_experiment_spec",
]
