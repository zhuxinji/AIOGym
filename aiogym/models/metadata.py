"""Structured model metadata validation and JSON export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

from .._internal.identifiers import internal_scenario_id
from .._internal.serialization import jsonable as _jsonable
from .._internal.identifiers import canonical_scenario_ids
from .registry import MODELS, make_model

MODEL_METADATA_SCHEMA_VERSION = "aiogym.model_metadata.v1"

REQUIRED_MODEL_METADATA_FIELDS = (
    "schema_version",
    "scenario",
    "name",
    "summary",
    "states",
    "actions",
    "state_vector",
    "action_vector",
    "dynamics_disturbances",
    "parameters",
    "physical_metadata",
    "solver",
    "disturbances",
    "disturbance_defaults",
    "constraints",
    "plant_regime",
    "economic_config",
    "supervisory_layout",
    "dt_micro",
    "energy_scored",
)


def iter_model_metadata(scenarios: Iterable[str] | None = None):
    """Yield validated structured metadata for registered models."""

    for scenario in scenarios or canonical_scenario_ids(tuple(MODELS)):
        metadata = _jsonable(make_model(scenario).metadata())
        metadata["schema_version"] = MODEL_METADATA_SCHEMA_VERSION
        validate_model_metadata(metadata, expected_scenario=scenario)
        yield scenario, metadata


def collect_model_metadata(
    scenarios: Iterable[str] | None = None,
) -> dict[str, dict]:
    """Return validated structured metadata keyed by scenario name."""

    return {
        scenario: metadata
        for scenario, metadata in iter_model_metadata(scenarios)
    }


def validate_model_metadata(
    metadata: Mapping,
    expected_scenario: str | None = None,
) -> None:
    """Validate the stable structured metadata used by benchmark artifacts."""

    missing = [
        field for field in REQUIRED_MODEL_METADATA_FIELDS
        if field not in metadata
    ]
    if missing:
        raise ValueError(
            f"model metadata is missing required fields: {', '.join(missing)}"
        )
    if metadata["schema_version"] != MODEL_METADATA_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported model metadata schema: {metadata['schema_version']!r}"
        )
    if (
        expected_scenario is not None
        and internal_scenario_id(metadata["scenario"])
        != internal_scenario_id(expected_scenario)
    ):
        raise ValueError(
            f"expected scenario {expected_scenario!r}, "
            f"got {metadata['scenario']!r}"
        )
    if not isinstance(metadata["states"], list) or not metadata["states"]:
        raise ValueError(
            f"{metadata['scenario']} model metadata must include at least one state"
        )
    if not isinstance(metadata["actions"], list):
        raise ValueError(
            f"{metadata['scenario']} model metadata actions must be a list"
        )
    for row in metadata["states"]:
        _require_metadata_fields(
            metadata["scenario"], "state", row, ("name", "unit", "bounds")
        )
        _require_metadata_bounds(metadata["scenario"], "state", row)
    for row in metadata["actions"]:
        _require_metadata_fields(
            metadata["scenario"],
            "action",
            row,
            ("name", "kind", "index", "unit", "bounds"),
        )
        _require_metadata_bounds(metadata["scenario"], "action", row)
    if "controlled_outputs" in metadata:
        if not isinstance(metadata["controlled_outputs"], list):
            raise ValueError(
                f"{metadata['scenario']} controlled_outputs must be a list"
            )
        for row in metadata["controlled_outputs"]:
            _require_metadata_fields(
                metadata["scenario"],
                "controlled output",
                row,
                ("name", "unit", "bounds"),
            )
        if (
            "controlled_output_vector" in metadata
            and len(metadata["controlled_outputs"])
            != int(metadata["controlled_output_vector"]["length"])
        ):
            raise ValueError(
                f"{metadata['scenario']} controlled output count does not "
                "match controlled_output_vector length"
            )
    if "setpoints" in metadata:
        if not isinstance(metadata["setpoints"], list):
            raise ValueError(
                f"{metadata['scenario']} setpoints must be a list"
            )
        for row in metadata["setpoints"]:
            _require_metadata_fields(
                metadata["scenario"],
                "setpoint",
                row,
                ("name", "unit", "bounds"),
            )
        if (
            "setpoint_vector" in metadata
            and len(metadata["setpoints"])
            != int(metadata["setpoint_vector"]["length"])
        ):
            raise ValueError(
                f"{metadata['scenario']} setpoint count does not match "
                "setpoint_vector length"
            )
    if len(metadata["states"]) != int(metadata["state_vector"]["length"]):
        raise ValueError(
            f"{metadata['scenario']} state count does not match "
            "state_vector length"
        )
    if len(metadata["actions"]) != int(metadata["action_vector"]["length"]):
        raise ValueError(
            f"{metadata['scenario']} action count does not match "
            "action_vector length"
        )
    if not isinstance(metadata["parameters"], dict) or not metadata["parameters"]:
        raise ValueError(
            f"{metadata['scenario']} model metadata must include parameters"
        )
    for name, row in metadata["parameters"].items():
        _require_metadata_fields(
            metadata["scenario"],
            f"parameter {name}",
            row,
            ("value", "unit", "bounds"),
        )
    if not isinstance(metadata["physical_metadata"], dict):
        raise ValueError(
            f"{metadata['scenario']} physical_metadata must be a mapping"
        )
    _require_metadata_fields(
        metadata["scenario"], "physical metadata", metadata["physical_metadata"],
        ("parameter_status", "fidelity", "time_unit", "references", "solver"),
    )
    if not isinstance(metadata["solver"], dict):
        raise ValueError(f"{metadata['scenario']} solver must be a mapping")
    _require_metadata_fields(
        metadata["scenario"],
        "solver",
        metadata["solver"],
        ("method", "max_step"),
    )
    if not isinstance(metadata["disturbances"], list):
        raise ValueError(f"{metadata['scenario']} disturbances must be a list")
    if not isinstance(metadata["constraints"], list) or not metadata["constraints"]:
        raise ValueError(
            f"{metadata['scenario']} model metadata must include constraints"
        )
    if not isinstance(metadata["plant_regime"], dict) or not metadata["plant_regime"]:
        raise ValueError(
            f"{metadata['scenario']} model metadata must include plant_regime"
        )
    if (
        not isinstance(metadata["economic_config"], dict)
        or not metadata["economic_config"]
    ):
        raise ValueError(
            f"{metadata['scenario']} model metadata must include economic_config"
        )


def export_model_metadata(
    out_dir: str | Path,
    scenarios: Iterable[str] | None = None,
    write_manifest: bool = True,
) -> dict:
    """Write one structured metadata JSON file per scenario."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = collect_model_metadata(scenarios)
    written = {}
    for scenario, model_metadata in metadata.items():
        path = out / f"{scenario}.json"
        _write_metadata_json(path, model_metadata)
        written[scenario] = str(path)
    manifest = {
        "schema_version": MODEL_METADATA_SCHEMA_VERSION,
        "scenarios": list(metadata),
        "models": written,
    }
    if write_manifest:
        manifest_path = out / "manifest.json"
        _write_metadata_json(manifest_path, manifest)
        manifest["manifest"] = str(manifest_path)
    return manifest


def _require_metadata_fields(
    scenario: str,
    kind: str,
    row: Mapping,
    fields: Iterable[str],
) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise ValueError(f"{scenario} {kind} is missing fields: {', '.join(missing)}")


def _require_metadata_bounds(scenario: str, kind: str, row: Mapping) -> None:
    bounds = row.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError(f"{scenario} {kind} {row.get('name', '')!r} must expose [low, high] bounds")


def _write_metadata_json(path: Path, data: Mapping) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
