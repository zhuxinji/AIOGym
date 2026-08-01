"""Resolve immutable training configuration into executable resources."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from aiogym.benchmarks.tracks.registry import load_track
from aiogym.benchmarks.tracks.schema import TrackSpec
from aiogym.datasets.writer import manifest_with_hash
from aiogym.datasets.writer import load_manifest

from .config import RLTrainingConfig, resolve_training_defaults
from .validation import ValidationEpisodePlan


@dataclass(frozen=True)
class ResolvedTrainingPlan:
    config: RLTrainingConfig
    track: TrackSpec
    validation_plan: ValidationEpisodePlan
    output_dir: Path
    run_name: str
    artifact_dir: Path
    policy_path: Path
    dataset_path: Path | None
    dataset_id: str | None
    dataset_hash: str | None
    replace_existing: bool = False

    @property
    def resolved_config_path(self) -> Path:
        return self.output_dir / f"{self.run_name}.resolved.json"


def resolve_training_plan(
    config: RLTrainingConfig,
) -> ResolvedTrainingPlan:
    if not isinstance(config, RLTrainingConfig):
        raise TypeError("config must be an RLTrainingConfig")
    config = resolve_training_defaults(config)
    track = load_track(config.track_id)
    validation_plan = ValidationEpisodePlan(
        track,
        base_seeds=config.validation_seeds,
    )
    output = dict(config.output)
    output_dir = Path(str(output.get("directory", "runs")))
    run_name = str(
        output.get("name")
        or (
            f"{config.algorithm_id}-{track.id}-"
            f"seed{config.training_seed}"
        )
    )
    if not run_name or run_name in {".", ".."}:
        raise ValueError("output.name must be a non-empty file-safe name")
    artifact_dir = Path(
        str(
            output.get(
                "artifact_dir",
                output_dir / f"{run_name}_artifacts",
            )
        )
    )
    suffix = ".pt" if config.algorithm_id in {"bc", "rlpd"} else ".zip"
    policy_path = output_dir / f"{run_name}{suffix}"
    dataset_path = (
        None if config.dataset_path is None else Path(config.dataset_path)
    )
    dataset_id = config.dataset_id
    dataset_hash = config.dataset_hash
    if config.algorithm_id in {"bc", "rlpd"}:
        if dataset_path is None:
            raise ValueError(
                f"{config.algorithm_id} requires dataset_path"
            )
        manifest = load_manifest(dataset_path)
        actual_id = str(manifest["dataset_id"])
        actual_hash = str(manifest_with_hash(manifest)["manifest_hash"])
        if dataset_id is not None and dataset_id != actual_id:
            raise ValueError("configured dataset_id does not match manifest")
        if dataset_hash is not None and dataset_hash != actual_hash:
            raise ValueError(
                "configured dataset_hash does not match manifest"
            )
        dataset_id = actual_id
        dataset_hash = actual_hash
    elif any(
        value is not None
        for value in (dataset_path, dataset_id, dataset_hash)
    ):
        raise ValueError(
            f"{config.algorithm_id} does not accept an offline dataset"
        )
    resolved_config = replace(
        config,
        dataset_path=(
            None if dataset_path is None else str(dataset_path)
        ),
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
        output={
            **dict(config.output),
            "directory": str(output_dir),
            "name": run_name,
            "artifact_dir": str(artifact_dir),
        },
    )
    return ResolvedTrainingPlan(
        config=resolved_config,
        track=track,
        validation_plan=validation_plan,
        output_dir=output_dir,
        run_name=run_name,
        artifact_dir=artifact_dir,
        policy_path=policy_path,
        dataset_path=dataset_path,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
    )


__all__ = ["ResolvedTrainingPlan", "resolve_training_plan"]
