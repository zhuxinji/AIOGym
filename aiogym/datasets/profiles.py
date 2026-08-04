"""Versioned collection profiles loaded from package resources."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
from types import MappingProxyType
from typing import Any, Mapping

from aiogym._internal.validation import nonnegative_int, positive_int
from aiogym.benchmarks.tracks.registry import (
    preferred_track_selector,
    resolve_track_selector,
)

from .config import COLLECTION_CONFIG_SCHEMA_VERSION, DatasetCollectionConfig


COLLECTION_PROFILE_SCHEMA_VERSION = "aiogym.collection_profile.v1"
_RESOURCE_ROOT = "resources/profiles/collection"


@dataclass(frozen=True)
class CollectionProfile:
    id: str
    selector: str
    description: str
    track_id: str
    source: str
    default_transitions: int
    default_workers: int
    collectors: tuple[Mapping[str, Any], ...]


@lru_cache(maxsize=1)
def _profiles() -> tuple[CollectionProfile, ...]:
    root = files("aiogym").joinpath(_RESOURCE_ROOT)
    loaded = []
    for scenario in sorted(root.iterdir(), key=lambda item: item.name):
        if not scenario.is_dir():
            continue
        for resource in sorted(scenario.iterdir(), key=lambda item: item.name):
            if not resource.name.endswith(".json"):
                continue
            data = json.loads(resource.read_text(encoding="utf-8"))
            required = {
                "schema_version",
                "id",
                "selector",
                "description",
                "track_id",
                "config_template",
            }
            if set(data) != required:
                raise ValueError(f"invalid collection profile fields: {resource}")
            if data["schema_version"] != COLLECTION_PROFILE_SCHEMA_VERSION:
                raise ValueError(f"unsupported collection profile schema: {resource}")
            template = data["config_template"]
            if not isinstance(template, dict):
                raise TypeError(f"collection profile config_template must be a mapping: {resource}")
            if template.get("track_id") != data["track_id"]:
                raise ValueError(f"collection profile track identity mismatch: {resource}")
            relative = f"{_RESOURCE_ROOT}/{scenario.name}/{resource.name}"
            loaded.append(
                CollectionProfile(
                    id=str(data["id"]),
                    selector=str(data["selector"]),
                    description=str(data["description"]),
                    track_id=str(data["track_id"]),
                    source=f"package:aiogym/{relative}",
                    default_transitions=positive_int(
                        "target_transitions", template["target_transitions"]
                    ),
                    default_workers=positive_int("workers", template["workers"]),
                    collectors=tuple(
                        MappingProxyType(copy.deepcopy(row))
                        for row in template["collectors"]
                    ),
                )
            )
    return tuple(loaded)


def list_collection_profiles(
    *, track_id: str | None = None
) -> tuple[CollectionProfile, ...]:
    canonical = None if track_id is None else resolve_track_selector(track_id)
    return tuple(
        profile
        for profile in _profiles()
        if canonical is None or profile.track_id == canonical
    )


def build_collection_config(
    target: str,
    profile: str,
    *,
    base_seed: int,
    transitions: int | None,
    workers: int | None,
    output: str | None,
    dataset_id: str | None,
) -> tuple[DatasetCollectionConfig, CollectionProfile]:
    track_id = resolve_track_selector(target)
    matches = tuple(
        candidate
        for candidate in list_collection_profiles(track_id=track_id)
        if profile in {candidate.selector, candidate.id}
    )
    if not matches:
        available = ", ".join(
            f"{preferred_track_selector(row.track_id)}/{row.id}"
            for row in _profiles()
        )
        raise ValueError(
            f"no collection profile for {target}/{profile}; available "
            f"combinations: {available}; use --config for a custom collection"
        )
    selected = matches[0]
    seed = nonnegative_int("base_seed", base_seed)
    resolved_transitions = positive_int(
        "target_transitions",
        selected.default_transitions if transitions is None else transitions,
    )
    resolved_workers = positive_int(
        "workers", selected.default_workers if workers is None else workers
    )
    selector = preferred_track_selector(track_id) or track_id
    slug = selector.replace(":", "-")
    default_name = f"{slug}-{selected.selector}-seed{seed}"
    declaration = {
        "schema_version": COLLECTION_CONFIG_SCHEMA_VERSION,
        "track_id": track_id,
        "dataset_id": dataset_id or default_name,
        "split": "training",
        "base_seed": seed,
        "target_transitions": resolved_transitions,
        "workers": resolved_workers,
        "collectors": [copy.deepcopy(dict(row)) for row in selected.collectors],
        "output": output or f"datasets/{default_name}",
        "complete_episode_overshoot": True,
    }
    return DatasetCollectionConfig(declaration), selected


__all__ = [
    "COLLECTION_PROFILE_SCHEMA_VERSION",
    "CollectionProfile",
    "build_collection_config",
    "list_collection_profiles",
]
