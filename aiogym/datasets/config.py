"""Validated configuration for deterministic Dataset v2 collection."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COLLECTION_CONFIG_SCHEMA_VERSION = "aiogym.dataset_collection.v1"
_FIELDS = frozenset(
    {
        "schema_version",
        "track_id",
        "dataset_id",
        "split",
        "base_seed",
        "target_transitions",
        "workers",
        "collectors",
        "output",
        "complete_episode_overshoot",
    }
)
_COLLECTOR_FIELDS = frozenset({"id", "weight", "options"})


@dataclass(frozen=True)
class CollectorAllocation:
    collector_id: str
    weight: float
    options: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.collector_id,
            "weight": self.weight,
            "options": copy.deepcopy(dict(self.options)),
        }


@dataclass(frozen=True, init=False)
class DatasetCollectionConfig:
    """Immutable collection declaration with a stable content hash."""

    _canonical_json: str

    def __init__(self, declaration: Mapping[str, Any]) -> None:
        if not isinstance(declaration, Mapping):
            raise TypeError("dataset collection config must be a mapping")
        data = copy.deepcopy(dict(declaration))
        _validate(data)
        object.__setattr__(
            self,
            "_canonical_json",
            json.dumps(
                data,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "DatasetCollectionConfig":
        source = Path(path)
        with source.open(encoding="utf-8") as stream:
            return cls(json.load(stream))

    @property
    def declaration(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(
            self._canonical_json.encode("utf-8")
        ).hexdigest()

    @property
    def track_id(self) -> str:
        return str(self.declaration["track_id"])

    @property
    def dataset_id(self) -> str:
        return str(self.declaration["dataset_id"])

    @property
    def split(self) -> str:
        return str(self.declaration["split"])

    @property
    def base_seed(self) -> int:
        return int(self.declaration["base_seed"])

    @property
    def target_transitions(self) -> int:
        return int(self.declaration["target_transitions"])

    @property
    def workers(self) -> int:
        return int(self.declaration["workers"])

    @property
    def output(self) -> Path:
        return Path(self.declaration["output"])

    @property
    def collectors(self) -> tuple[CollectorAllocation, ...]:
        return tuple(
            CollectorAllocation(
                collector_id=str(row["id"]),
                weight=float(row["weight"]),
                options=copy.deepcopy(dict(row.get("options") or {})),
            )
            for row in self.declaration["collectors"]
        )

    def as_dict(self) -> dict[str, Any]:
        return self.declaration


def _validate(data: Mapping[str, Any]) -> None:
    unknown = set(data) - _FIELDS
    if unknown:
        raise ValueError(
            "unknown dataset collection config fields: "
            + ", ".join(sorted(unknown))
        )
    required = _FIELDS - {"complete_episode_overshoot"}
    missing = required - set(data)
    if missing:
        raise ValueError(
            "dataset collection config is missing fields: "
            + ", ".join(sorted(missing))
        )
    if data["schema_version"] != COLLECTION_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported dataset collection config schema")
    for name in ("track_id", "dataset_id", "output"):
        if not isinstance(data[name], str) or not data[name]:
            raise ValueError(f"{name} must be a non-empty string")
    if data["split"] != "training":
        raise ValueError("dataset collection is restricted to training split")
    _non_negative_integer("base_seed", data["base_seed"])
    _positive_integer("target_transitions", data["target_transitions"])
    _positive_integer("workers", data["workers"])
    if data.get("complete_episode_overshoot", True) is not True:
        raise ValueError(
            "v1 collection requires complete_episode_overshoot=true"
        )
    collectors = data["collectors"]
    if not isinstance(collectors, list) or not collectors:
        raise ValueError("collectors must be a non-empty list")
    seen = set()
    total = 0.0
    for index, row in enumerate(collectors):
        if not isinstance(row, Mapping):
            raise TypeError(f"collectors[{index}] must be a mapping")
        unknown_fields = set(row) - _COLLECTOR_FIELDS
        if unknown_fields:
            raise ValueError(
                f"collectors[{index}] has unknown fields: "
                + ", ".join(sorted(unknown_fields))
            )
        collector_id = row.get("id")
        if not isinstance(collector_id, str) or not collector_id:
            raise ValueError(f"collectors[{index}].id must be non-empty")
        if collector_id in seen:
            raise ValueError(f"duplicate collector ID {collector_id!r}")
        seen.add(collector_id)
        try:
            weight = float(row.get("weight"))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"collectors[{index}].weight must be numeric"
            ) from exc
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"collectors[{index}].weight must be finite and positive"
            )
        total += weight
        options = row.get("options", {})
        if not isinstance(options, Mapping):
            raise TypeError(f"collectors[{index}].options must be a mapping")
    if total <= 0.0:
        raise ValueError("collector weights must have positive total")
    json.dumps(data, allow_nan=False)


def _non_negative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_integer(name: str, value: Any) -> int:
    result = _non_negative_integer(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


__all__ = [
    "COLLECTION_CONFIG_SCHEMA_VERSION",
    "CollectorAllocation",
    "DatasetCollectionConfig",
]
