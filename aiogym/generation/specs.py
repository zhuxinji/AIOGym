"""Immutable schemas for training distributions and resolved episodes."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .seed_tree import SEED_COMPONENTS


DISTRIBUTION_SCHEMA_VERSION = "aiogym.distribution.v1"
EPISODE_SPEC_SCHEMA_VERSION = "aiogym.episode_spec.v1"

_DISTRIBUTION_FIELDS = frozenset(
    {
        "schema_version",
        "distribution_id",
        "scenario_id",
        "goal",
        "control_dt",
        "episode_steps",
        "plant_distribution",
        "initial_state_distribution",
        "reference_distribution",
        "disturbance_distribution",
        "sensor_distribution",
        "actuator_distribution",
        "economic_context_distribution",
        "mixture_weights",
        "curriculum_id",
    }
)
_DISTRIBUTION_MAPPING_FIELDS = (
    "plant_distribution",
    "initial_state_distribution",
    "reference_distribution",
    "disturbance_distribution",
    "sensor_distribution",
    "actuator_distribution",
    "mixture_weights",
)
_EPISODE_PAYLOAD_FIELDS = frozenset(
    {
        "schema_version",
        "distribution_id",
        "distribution_hash",
        "base_seed",
        "component_seeds",
        "scenario_id",
        "goal",
        "control_dt",
        "episode_steps",
        "plant_parameters",
        "initial_state",
        "reference_schedule",
        "disturbance_schedule",
        "sensor_model",
        "actuator_model",
        "economic_context",
        "difficulty_tags",
    }
)
def _canonical_json(value: Mapping[str, Any]) -> str:
    normalized = _json_value(value, path="spec")
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256(canonical_json: str) -> str:
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, init=False)
class DistributionSpec:
    """Versioned description of a probability distribution over episodes."""

    _canonical_json: str

    def __init__(self, declaration: Mapping[str, Any]) -> None:
        if not isinstance(declaration, Mapping):
            raise TypeError("DistributionSpec declaration must be a mapping")
        resolved = deepcopy(dict(declaration))
        _validate_distribution(resolved)
        object.__setattr__(self, "_canonical_json", _canonical_json(resolved))

    @property
    def declaration(self) -> dict[str, Any]:
        return json.loads(self._canonical_json)

    @property
    def schema_version(self) -> str:
        return str(self.declaration["schema_version"])

    @property
    def distribution_id(self) -> str:
        return str(self.declaration["distribution_id"])

    @property
    def scenario_id(self) -> str:
        return str(self.declaration["scenario_id"])

    @property
    def goal(self) -> str:
        return str(self.declaration["goal"])

    @property
    def control_dt(self) -> float:
        return float(self.declaration["control_dt"])

    @property
    def episode_steps(self) -> int:
        return int(self.declaration["episode_steps"])

    @property
    def distribution_hash(self) -> str:
        return _sha256(self._canonical_json)

    def as_dict(self) -> dict[str, Any]:
        return self.declaration

    def metadata(self) -> dict[str, Any]:
        return {
            **self.declaration,
            "distribution_hash": self.distribution_hash,
        }


@dataclass(frozen=True, init=False)
class EpisodeSpec:
    """Fully resolved, deterministic inputs for exactly one episode."""

    _payload_json: str
    _episode_spec_id: str
    _resolved_hash: str

    def __init__(self, declaration: Mapping[str, Any]) -> None:
        if not isinstance(declaration, Mapping):
            raise TypeError("EpisodeSpec declaration must be a mapping")
        resolved = deepcopy(dict(declaration))
        supplied_id = resolved.pop("episode_spec_id", None)
        supplied_hash = resolved.pop("resolved_hash", None)
        _validate_episode_payload(resolved)
        payload_json = _canonical_json(resolved)
        resolved_hash = _sha256(payload_json)
        episode_spec_id = (
            f"{resolved['distribution_id']}:episode:{resolved_hash[:16]}"
        )
        if supplied_hash is not None and supplied_hash != resolved_hash:
            raise ValueError(
                "EpisodeSpec resolved_hash does not match its resolved payload"
            )
        if supplied_id is not None and supplied_id != episode_spec_id:
            raise ValueError(
                "EpisodeSpec episode_spec_id does not match its resolved payload"
            )
        object.__setattr__(self, "_payload_json", payload_json)
        object.__setattr__(self, "_episode_spec_id", episode_spec_id)
        object.__setattr__(self, "_resolved_hash", resolved_hash)

    @classmethod
    def from_distribution(
        cls,
        distribution: DistributionSpec,
        *,
        base_seed: int,
        component_seeds: Mapping[str, int],
        plant_parameters: Mapping[str, Any],
        initial_state: Sequence[float],
        reference_schedule: Sequence[Mapping[str, Any]] = (),
        disturbance_schedule: Sequence[Mapping[str, Any]] = (),
        sensor_model: Mapping[str, Any] | None = None,
        actuator_model: Mapping[str, Any] | None = None,
        economic_context: Mapping[str, Any] | None = None,
        difficulty_tags: Sequence[str] = (),
    ) -> "EpisodeSpec":
        """Create a resolved episode linked to a validated distribution."""

        if not isinstance(distribution, DistributionSpec):
            raise TypeError("distribution must be a DistributionSpec")
        return cls(
            {
                "schema_version": EPISODE_SPEC_SCHEMA_VERSION,
                "distribution_id": distribution.distribution_id,
                "distribution_hash": distribution.distribution_hash,
                "base_seed": base_seed,
                "component_seeds": dict(component_seeds),
                "scenario_id": distribution.scenario_id,
                "goal": distribution.goal,
                "control_dt": distribution.control_dt,
                "episode_steps": distribution.episode_steps,
                "plant_parameters": dict(plant_parameters),
                "initial_state": list(initial_state),
                "reference_schedule": list(reference_schedule),
                "disturbance_schedule": list(disturbance_schedule),
                "sensor_model": dict(sensor_model or {}),
                "actuator_model": dict(actuator_model or {}),
                "economic_context": dict(economic_context or {}),
                "difficulty_tags": list(difficulty_tags),
            }
        )

    @property
    def payload(self) -> dict[str, Any]:
        return json.loads(self._payload_json)

    @property
    def episode_spec_id(self) -> str:
        return self._episode_spec_id

    @property
    def resolved_hash(self) -> str:
        return self._resolved_hash

    @property
    def distribution_id(self) -> str:
        return str(self.payload["distribution_id"])

    @property
    def distribution_hash(self) -> str:
        return str(self.payload["distribution_hash"])

    @property
    def base_seed(self) -> int:
        return int(self.payload["base_seed"])

    @property
    def component_seeds(self) -> dict[str, int]:
        return dict(self.payload["component_seeds"])

    @property
    def scenario_id(self) -> str:
        return str(self.payload["scenario_id"])

    @property
    def goal(self) -> str:
        return str(self.payload["goal"])

    @property
    def control_dt(self) -> float:
        return float(self.payload["control_dt"])

    @property
    def episode_steps(self) -> int:
        return int(self.payload["episode_steps"])

    @property
    def initial_state(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.payload["initial_state"])

    @property
    def plant_parameters(self) -> dict[str, Any]:
        return deepcopy(self.payload["plant_parameters"])

    @property
    def reference_schedule(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            deepcopy(event) for event in self.payload["reference_schedule"]
        )

    @property
    def disturbance_schedule(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            deepcopy(event)
            for event in self.payload["disturbance_schedule"]
        )

    @property
    def sensor_model(self) -> dict[str, Any]:
        return deepcopy(self.payload["sensor_model"])

    @property
    def actuator_model(self) -> dict[str, Any]:
        return deepcopy(self.payload["actuator_model"])

    @property
    def economic_context(self) -> dict[str, Any]:
        return deepcopy(self.payload["economic_context"])

    @property
    def difficulty_tags(self) -> tuple[str, ...]:
        return tuple(self.payload["difficulty_tags"])

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.payload,
            "episode_spec_id": self.episode_spec_id,
            "resolved_hash": self.resolved_hash,
        }

    def metadata(self) -> dict[str, Any]:
        return self.as_dict()


def _validate_distribution(declaration: Mapping[str, Any]) -> None:
    if not isinstance(declaration, Mapping):
        raise TypeError("DistributionSpec declaration must be a mapping")
    unknown = set(declaration) - _DISTRIBUTION_FIELDS
    if unknown:
        raise ValueError(
            "unknown DistributionSpec fields: " + ", ".join(sorted(unknown))
        )
    required = _DISTRIBUTION_FIELDS - {
        "economic_context_distribution",
        "curriculum_id",
    }
    missing = sorted(required - set(declaration))
    if missing:
        raise ValueError(
            "DistributionSpec is missing required fields: "
            + ", ".join(missing)
        )
    if declaration["schema_version"] != DISTRIBUTION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported DistributionSpec schema: "
            f"{declaration['schema_version']!r}"
        )
    for field in ("distribution_id", "scenario_id"):
        _require_non_empty_string(field, declaration[field])
    _validate_goal(declaration["goal"])
    _positive_float("control_dt", declaration["control_dt"])
    _positive_integer("episode_steps", declaration["episode_steps"])
    for field in _DISTRIBUTION_MAPPING_FIELDS:
        if not isinstance(declaration[field], Mapping):
            raise TypeError(f"DistributionSpec {field} must be a mapping")
    context = declaration.get("economic_context_distribution")
    if context is not None and not isinstance(context, Mapping):
        raise TypeError(
            "DistributionSpec economic_context_distribution must be "
            "a mapping or None"
        )
    curriculum_id = declaration.get("curriculum_id")
    if curriculum_id is not None:
        _require_non_empty_string("curriculum_id", curriculum_id)
    weights = declaration["mixture_weights"]
    if not weights:
        raise ValueError("DistributionSpec mixture_weights must not be empty")
    total = 0.0
    for name, value in weights.items():
        _require_non_empty_string("mixture weight name", name)
        weight = _non_negative_float(f"mixture weight {name!r}", value)
        total += weight
    if total <= 0:
        raise ValueError(
            "DistributionSpec mixture_weights must have positive total weight"
        )
    _canonical_json(declaration)


def _validate_episode_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("EpisodeSpec declaration must be a mapping")
    unknown = set(payload) - _EPISODE_PAYLOAD_FIELDS
    if unknown:
        raise ValueError(
            "unknown EpisodeSpec fields: " + ", ".join(sorted(unknown))
        )
    missing = sorted(_EPISODE_PAYLOAD_FIELDS - set(payload))
    if missing:
        raise ValueError(
            "EpisodeSpec is missing required fields: " + ", ".join(missing)
        )
    if payload["schema_version"] != EPISODE_SPEC_SCHEMA_VERSION:
        raise ValueError(
            "unsupported EpisodeSpec schema: "
            f"{payload['schema_version']!r}"
        )
    for field in ("distribution_id", "scenario_id"):
        _require_non_empty_string(field, payload[field])
    _require_sha256("distribution_hash", payload["distribution_hash"])
    _non_negative_integer("base_seed", payload["base_seed"])
    _validate_goal(payload["goal"])
    _positive_float("control_dt", payload["control_dt"])
    episode_steps = _positive_integer(
        "episode_steps", payload["episode_steps"]
    )
    seeds = payload["component_seeds"]
    if not isinstance(seeds, Mapping) or not seeds:
        raise TypeError("EpisodeSpec component_seeds must be a non-empty mapping")
    missing_seeds = sorted(set(SEED_COMPONENTS) - set(seeds))
    unknown_seeds = sorted(set(seeds) - set(SEED_COMPONENTS))
    if missing_seeds or unknown_seeds:
        details = []
        if missing_seeds:
            details.append("missing: " + ", ".join(missing_seeds))
        if unknown_seeds:
            details.append("unknown: " + ", ".join(unknown_seeds))
        raise ValueError(
            "EpisodeSpec component_seeds must match the v1 seed tree ("
            + "; ".join(details)
            + ")"
        )
    for name, value in seeds.items():
        _require_non_empty_string("component seed name", name)
        _non_negative_integer(f"component seed {name!r}", value)
    for field in (
        "plant_parameters",
        "sensor_model",
        "actuator_model",
        "economic_context",
    ):
        if not isinstance(payload[field], Mapping):
            raise TypeError(f"EpisodeSpec {field} must be a mapping")
    state = payload["initial_state"]
    if (
        isinstance(state, (str, bytes))
        or not isinstance(state, Sequence)
        or not state
    ):
        raise ValueError(
            "EpisodeSpec initial_state must be a non-empty numeric sequence"
        )
    for index, value in enumerate(state):
        _finite_float(f"initial_state[{index}]", value)
    for field in ("reference_schedule", "disturbance_schedule"):
        schedule = payload[field]
        if isinstance(schedule, (str, bytes)) or not isinstance(
            schedule, Sequence
        ):
            raise TypeError(f"EpisodeSpec {field} must be a sequence")
        previous_step = -1
        for index, event in enumerate(schedule):
            if not isinstance(event, Mapping):
                raise TypeError(
                    f"EpisodeSpec {field}[{index}] must be a mapping"
                )
            if "at_step" in event:
                step = _non_negative_integer(
                    f"{field}[{index}].at_step", event["at_step"]
                )
                if step >= episode_steps:
                    raise ValueError(
                        f"EpisodeSpec {field}[{index}].at_step must be "
                        "less than episode_steps"
                    )
                if step < previous_step:
                    raise ValueError(
                        f"EpisodeSpec {field} must be ordered by at_step"
                    )
                previous_step = step
    tags = payload["difficulty_tags"]
    if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
        raise TypeError("EpisodeSpec difficulty_tags must be a sequence")
    for tag in tags:
        _require_non_empty_string("difficulty tag", tag)
    if len(set(tags)) != len(tags):
        raise ValueError("EpisodeSpec difficulty_tags must be unique")
    _canonical_json(payload)


def _json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} mapping keys must be non-empty strings")
            normalized[key] = _json_value(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


def _validate_goal(value: Any) -> str:
    if value not in {"regulation", "economic"}:
        raise ValueError("goal must be one of: economic, regulation")
    return str(value)


def _require_non_empty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_sha256(name: str, value: Any) -> str:
    text = _require_non_empty_string(name, value)
    invalid_character = any(
        character not in "0123456789abcdef" for character in text
    )
    if len(text) != 64 or invalid_character:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_float(name: str, value: Any) -> float:
    result = _finite_float(name, value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _non_negative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_integer(name: str, value: Any) -> int:
    result = _non_negative_integer(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result
