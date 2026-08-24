"""Versioned calibration records for the experimental Three-Tank rig."""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._integrity import content_digest


CALIBRATION_SCHEMA_VERSION = "aiogym.three_tank.calibration.v3"
CALIBRATION_PARAMETER_UNITS = {
    "area_1": "m2",
    "area_2": "m2",
    "area_3": "m2",
    "cv_V12": "m2.5/s",
    "cv_V23": "m2.5/s",
    "cv_V34": "m2.5/s",
    "pump_flow_max": "m3/s",
    "sensor_delay": "s",
    "actuator_delay": "s",
}
_POSITIVE_PARAMETERS = {
    "area_1",
    "area_2",
    "area_3",
    "cv_V12",
    "cv_V23",
    "cv_V34",
    "pump_flow_max",
}
_NONNEGATIVE_PARAMETERS = {"sensor_delay", "actuator_delay"}


def calibration_template(*, rig_id: str) -> dict[str, Any]:
    """Return an explicitly unmeasured record; it is never a fake calibration."""
    rig = str(rig_id).strip()
    if not rig:
        raise ValueError("rig_id must be non-empty")
    record = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "calibration_id": "unmeasured",
        "rig_id": rig,
        "status": "unmeasured",
        "parameters": {
            name: {
                "unit": unit,
                "estimate": None,
                "standard_uncertainty": None,
                "source_dataset": None,
            }
            for name, unit in CALIBRATION_PARAMETER_UNITS.items()
        },
        "channels": {},
        "fit_dataset_ids": [],
        "validation_dataset_ids": [],
        "validation_metrics": {},
    }
    record["calibration_hash"] = content_digest(record)
    return record


def validate_calibration(
    value: Mapping[str, Any], *, require_measured: bool = False
) -> dict[str, Any]:
    record = dict(value)
    parameter_sources = []
    required = {
        "schema_version",
        "calibration_id",
        "rig_id",
        "status",
        "parameters",
        "channels",
        "fit_dataset_ids",
        "validation_dataset_ids",
        "validation_metrics",
    }
    allowed = required | {"calibration_hash"}
    missing = sorted(required - set(record))
    unknown = sorted(set(record) - allowed)
    if missing or unknown:
        raise ValueError(
            f"calibration fields are invalid: missing={missing}, unknown={unknown}"
        )
    if record["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported three-tank calibration schema_version")
    for name in ("calibration_id", "rig_id", "status"):
        if not str(record[name]).strip():
            raise ValueError(f"calibration {name} must be non-empty")
    parameters = record["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError("calibration parameters must be a mapping")
    missing = set(CALIBRATION_PARAMETER_UNITS) - set(parameters)
    if missing:
        raise ValueError(f"calibration parameters missing: {sorted(missing)}")
    for name, expected_unit in CALIBRATION_PARAMETER_UNITS.items():
        row = parameters[name]
        parameter_fields = {
            "unit",
            "estimate",
            "standard_uncertainty",
            "source_dataset",
        }
        if not isinstance(row, Mapping) or set(row) != parameter_fields:
            raise ValueError(f"calibration parameter {name} fields are invalid")
        if row["unit"] != expected_unit:
            raise ValueError(f"calibration parameter {name} has wrong unit")
        estimate = row["estimate"]
        uncertainty = row["standard_uncertainty"]
        if estimate is not None and not math.isfinite(float(estimate)):
            raise ValueError(f"calibration parameter {name} estimate must be finite")
        if estimate is not None:
            estimate = float(estimate)
            if name in _POSITIVE_PARAMETERS and estimate <= 0.0:
                raise ValueError(f"calibration parameter {name} must be positive")
            if name in _NONNEGATIVE_PARAMETERS and estimate < 0.0:
                raise ValueError(
                    f"calibration parameter {name} must be non-negative"
                )
        if uncertainty is not None and (
            not math.isfinite(float(uncertainty)) or float(uncertainty) < 0.0
        ):
            raise ValueError(
                f"calibration parameter {name} uncertainty must be non-negative"
            )
        if require_measured and (estimate is None or uncertainty is None):
            raise ValueError(f"calibration parameter {name} is not measured")
        if require_measured and not str(row["source_dataset"]).strip():
            raise ValueError(f"calibration parameter {name} has no source_dataset")
        if require_measured:
            parameter_sources.append(str(row["source_dataset"]).strip())
    if require_measured and record["status"] != "validated":
        raise ValueError("measured calibration must have status='validated'")
    fit_ids = _dataset_ids(record, "fit_dataset_ids")
    validation_ids = _dataset_ids(record, "validation_dataset_ids")
    if require_measured:
        if record["calibration_id"] == "unmeasured":
            raise ValueError("validated calibration_id cannot be 'unmeasured'")
        if not fit_ids or not validation_ids:
            raise ValueError(
                "measured calibration requires fit and validation dataset IDs"
            )
        if set(fit_ids) & set(validation_ids):
            raise ValueError("fit and validation dataset IDs must be disjoint")
        if not set(parameter_sources) <= set(fit_ids):
            raise ValueError(
                "calibration parameter source_dataset must reference a fit dataset"
            )
        metrics = record["validation_metrics"]
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("measured calibration requires validation_metrics")
        try:
            finite_metrics = all(math.isfinite(float(value)) for value in metrics.values())
        except (TypeError, ValueError) as exc:
            raise ValueError("calibration validation_metrics must be finite numbers") from exc
        if not finite_metrics:
            raise ValueError("calibration validation_metrics must be finite numbers")
    declared = record["calibration_hash"] if "calibration_hash" in record else None
    payload = {key: item for key, item in record.items() if key != "calibration_hash"}
    resolved_hash = content_digest(payload)
    if declared is not None and str(declared) != resolved_hash:
        raise ValueError("declared calibration_hash does not match calibration")
    record["calibration_hash"] = resolved_hash
    return record


def _dataset_ids(record: Mapping[str, Any], name: str) -> list[str]:
    values = record[name]
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"calibration {name} must be a list")
    resolved = [str(value).strip() for value in values]
    if any(not value for value in resolved) or len(set(resolved)) != len(resolved):
        raise ValueError(f"calibration {name} must contain unique non-empty IDs")
    return resolved


def load_calibration(path: str | Path, *, require_measured: bool = False):
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_calibration(payload, require_measured=require_measured)


__all__ = [
    "CALIBRATION_PARAMETER_UNITS",
    "CALIBRATION_SCHEMA_VERSION",
    "calibration_template",
    "load_calibration",
    "validate_calibration",
]
