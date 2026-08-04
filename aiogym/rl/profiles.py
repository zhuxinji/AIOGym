"""Versioned guided-training profiles loaded from package resources."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
from types import MappingProxyType
from typing import Any, Mapping

from aiogym.benchmarks.tracks.registry import (
    preferred_track_selector,
    resolve_track_selector,
)

from .config import RLTrainingConfig, list_algorithms


TRAINING_PROFILE_SCHEMA_VERSION = "aiogym.training_profile.v1"
_RESOURCE_ROOT = "resources/profiles/training"


@dataclass(frozen=True)
class TrainingProfile:
    id: str
    selector: str
    description: str
    track_id: str
    algorithm_id: str
    source: str
    config_template: Mapping[str, Any]


def _resource_jsons(root):
    for scenario in sorted(root.iterdir(), key=lambda item: item.name):
        if not scenario.is_dir():
            continue
        for resource in sorted(scenario.iterdir(), key=lambda item: item.name):
            if resource.name.endswith(".json"):
                yield resource


@lru_cache(maxsize=1)
def _profiles() -> tuple[TrainingProfile, ...]:
    root = files("aiogym").joinpath(_RESOURCE_ROOT)
    loaded = []
    for resource in _resource_jsons(root):
        data = json.loads(resource.read_text(encoding="utf-8"))
        required = {
            "schema_version",
            "id",
            "selector",
            "description",
            "track_id",
            "algorithm_id",
            "config_template",
        }
        if set(data) != required:
            raise ValueError(f"invalid training profile fields: {resource}")
        if data["schema_version"] != TRAINING_PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported training profile schema: {resource}")
        template = data["config_template"]
        if not isinstance(template, dict):
            raise TypeError(f"training profile config_template must be a mapping: {resource}")
        if template.get("track_id") != data["track_id"]:
            raise ValueError(f"training profile track identity mismatch: {resource}")
        if template.get("algorithm_id") != data["algorithm_id"]:
            raise ValueError(f"training profile algorithm identity mismatch: {resource}")
        relative = f"{_RESOURCE_ROOT}/{resource.parent.name}/{resource.name}"
        loaded.append(
            TrainingProfile(
                id=str(data["id"]),
                selector=str(data["selector"]),
                description=str(data["description"]),
                track_id=str(data["track_id"]),
                algorithm_id=str(data["algorithm_id"]),
                source=f"package:aiogym/{relative}",
                config_template=MappingProxyType(copy.deepcopy(template)),
            )
        )
    return tuple(loaded)


def list_training_profiles(
    *,
    track_id: str | None = None,
    algorithm_id: str | None = None,
) -> tuple[TrainingProfile, ...]:
    canonical_track = None if track_id is None else resolve_track_selector(track_id)
    canonical_algorithm = None if algorithm_id is None else str(algorithm_id).lower()
    return tuple(
        profile
        for profile in _profiles()
        if (canonical_track is None or profile.track_id == canonical_track)
        and (canonical_algorithm is None or profile.algorithm_id == canonical_algorithm)
    )


def build_training_config(
    target: str,
    algorithm_id: str,
    profile: str,
    *,
    training_seed: int,
    device: str | None,
    output_directory: str | None,
    dataset_path: str | None,
    resume_checkpoint: str | None,
) -> tuple[RLTrainingConfig, TrainingProfile]:
    track_id = resolve_track_selector(target)
    algorithm = str(algorithm_id).lower()
    if algorithm not in list_algorithms():
        raise ValueError("algorithm must be one of: " + ", ".join(list_algorithms()))
    matches = tuple(
        candidate
        for candidate in list_training_profiles(track_id=track_id, algorithm_id=algorithm)
        if profile in {candidate.selector, candidate.id}
    )
    if not matches:
        available = ", ".join(
            f"{preferred_track_selector(row.track_id)}/{row.algorithm_id}/{row.id}"
            for row in _profiles()
        )
        raise ValueError(
            f"no training profile for {target}/{algorithm}/{profile}; "
            f"available combinations: {available}; use --config for a custom run"
        )
    selected = matches[0]
    offline = algorithm in {"bc", "rlpd"}
    if offline and dataset_path is None:
        raise ValueError(f"{algorithm} requires --dataset")
    if not offline and dataset_path is not None:
        raise ValueError(f"{algorithm} does not accept --dataset")
    declaration = copy.deepcopy(dict(selected.config_template))
    selector = preferred_track_selector(track_id) or track_id
    slug = selector.replace(":", "-")
    declaration.update(
        {
            "track_id": track_id,
            "algorithm_id": algorithm,
            "training_seed": training_seed,
            "device": device or "cpu",
            "output": {
                "directory": output_directory or "runs",
                "name": f"{slug}-{algorithm}-{selected.selector}-seed{training_seed}",
            },
            "resume_checkpoint": resume_checkpoint,
            "dataset_path": dataset_path,
            "dataset_id": None,
            "dataset_hash": None,
        }
    )
    return RLTrainingConfig.from_mapping(declaration), selected


__all__ = [
    "TRAINING_PROFILE_SCHEMA_VERSION",
    "TrainingProfile",
    "build_training_config",
    "list_training_profiles",
]
