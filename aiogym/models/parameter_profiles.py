"""Versioned physical-parameter metadata for process models.

Profiles are deliberately separate from executable model parameters.  Loading a
profile never mutates a model; callers must explicitly pass ``model_params`` if
they want to change numerical values.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .._internal.identifiers import (
    canonical_scenario_id,
    internal_scenario_id,
)


PARAMETER_PROFILE_SCHEMA_VERSION = "aiogym.parameter_profile.v1"
_PROFILE_DIR = Path(__file__).with_name("parameters")


def list_parameter_profiles() -> tuple[str, ...]:
    """Return scenarios with a bundled parameter metadata profile."""

    return tuple(sorted(
        canonical_scenario_id(path.stem)
        for path in _PROFILE_DIR.glob("*.json")
    ))


def load_parameter_profile(
    source: str | Path | Mapping[str, Any],
    *,
    scenario: str | None = None,
) -> dict[str, Any]:
    """Load and validate a bundled, file-based, or in-memory profile."""

    if isinstance(source, Mapping):
        data = copy.deepcopy(dict(source))
    else:
        path = Path(source)
        if isinstance(source, str) and not path.exists() and path.parent == Path("."):
            path = _PROFILE_DIR / f"{internal_scenario_id(source)}.json"
        if not path.is_file():
            raise FileNotFoundError(f"parameter profile not found: {source}")
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    validate_parameter_profile(data, expected_scenario=scenario)
    return data


def validate_parameter_profile(
    profile: Mapping[str, Any],
    *,
    expected_scenario: str | None = None,
) -> None:
    """Validate the stable parameter-profile surface."""

    if not isinstance(profile, Mapping):
        raise TypeError("parameter profile must be a mapping")
    required = ("schema_version", "scenario", "status", "parameters", "references")
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"parameter profile is missing required fields: {', '.join(missing)}")
    if profile["schema_version"] != PARAMETER_PROFILE_SCHEMA_VERSION:
        raise ValueError(f"unsupported parameter profile schema: {profile['schema_version']!r}")
    if not isinstance(profile["scenario"], str) or not profile["scenario"]:
        raise ValueError("parameter profile scenario must be a non-empty string")
    if not isinstance(profile["status"], str) or not profile["status"]:
        raise ValueError("parameter profile status must be a non-empty string")
    if (
        expected_scenario is not None
        and internal_scenario_id(profile["scenario"])
        != internal_scenario_id(expected_scenario)
    ):
        raise ValueError(
            f"expected parameter profile for {expected_scenario!r}, got {profile['scenario']!r}"
        )
    if not isinstance(profile["parameters"], Mapping):
        raise TypeError("parameter profile parameters must be a mapping")
    if not isinstance(profile["references"], list):
        raise TypeError("parameter profile references must be a list")
    for name, row in profile["parameters"].items():
        if not isinstance(name, str) or not isinstance(row, Mapping):
            raise TypeError("each parameter profile entry must map a name to metadata")
        unknown = set(row) - {
            "value", "unit", "bounds", "valid_range", "uncertainty", "source",
            "source_locator", "status", "description",
        }
        if unknown:
            raise ValueError(f"parameter {name!r} has unknown metadata: {', '.join(sorted(unknown))}")


def model_physical_metadata(scenario: str, *, dt_micro: float) -> dict[str, Any]:
    """Return non-mutating physical and solver metadata for a model."""

    try:
        profile = load_parameter_profile(scenario, scenario=scenario)
    except FileNotFoundError:
        profile = {
            "schema_version": PARAMETER_PROFILE_SCHEMA_VERSION,
            "scenario": scenario,
            "status": "unprofiled",
            "parameters": {},
            "references": [],
        }
    solver = copy.deepcopy(profile.get("solver", {}))
    solver.setdefault("method", "rk4")
    solver.setdefault("max_step", float(dt_micro))
    return {
        "parameter_profile_schema": profile["schema_version"],
        "parameter_status": profile["status"],
        "fidelity": profile.get("fidelity", "legacy-benchmark"),
        "time_unit": profile.get("time_unit", "s"),
        "validity_domain": copy.deepcopy(profile.get("validity_domain", {})),
        "nominal_operating_point": copy.deepcopy(profile.get("nominal_operating_point", {})),
        "references": copy.deepcopy(profile.get("references", [])),
        "solver": solver,
    }


def enrich_parameter_schema(scenario: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Merge provenance metadata into executable parameter rows."""

    rows = copy.deepcopy(dict(schema))
    try:
        profile = load_parameter_profile(scenario, scenario=scenario)
    except FileNotFoundError:
        return rows
    metadata = profile.get("parameters", {})
    default_status = profile.get("status", "unverified")
    for name, row in rows.items():
        row.update(copy.deepcopy(metadata.get(name, {})))
        row.setdefault("status", default_status)
    return rows
