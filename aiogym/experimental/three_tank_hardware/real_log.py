"""Versioned records for experimental Three-Tank hardware runs."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from ._integrity import content_digest


REAL_LOG_SCHEMA_VERSION = "aiogym.three_tank.real-step.v3"
MEASUREMENT_NAMES = (
    "tank_1_level",
    "tank_2_level",
    "tank_3_level",
    "tank_1_temperature",
    "tank_2_temperature",
    "tank_3_temperature",
)
ACTUATOR_NAMES = ("pump_P101", "valve_V12", "valve_V23", "valve_V34", "heater_H1")
FLOW_MEASUREMENT_NAMES = ("V12_flow", "V23_flow", "V34_flow")


def _finite_vector(name: str, values: Sequence[float], length: int) -> list[float]:
    resolved = [float(value) for value in values]
    if len(resolved) != length or not all(math.isfinite(value) for value in resolved):
        raise ValueError(f"{name} must contain {length} finite values")
    return resolved


def build_real_step_record(
    *,
    run_id: str,
    episode_id: str,
    sequence: int,
    source_monotonic_time_s: float,
    received_monotonic_time_s: float,
    wall_time_utc: str,
    measurement: Sequence[float],
    flow_measurement: Sequence[float],
    reference: Sequence[float],
    transition_disturbance: Mapping[str, float],
    disturbance: Mapping[str, float],
    commanded_action: Sequence[float],
    applied_action: Sequence[float],
    calibration_id: str,
    calibration_hash: str,
    scenario_id: str,
    benchmark_id: str,
    reward_id: str,
    backend_version: str,
    hardware_mode: str,
    residual_authority: float,
    safety: Mapping[str, Any] | None = None,
    raw_channels: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one self-identifying hardware record without inventing raw data."""
    if isinstance(sequence, bool) or int(sequence) != sequence or int(sequence) < 0:
        raise ValueError("sequence must be a non-negative integer")
    for name, value in {
        "run_id": run_id,
        "episode_id": episode_id,
        "wall_time_utc": wall_time_utc,
        "calibration_id": calibration_id,
        "calibration_hash": calibration_hash,
        "scenario_id": scenario_id,
        "benchmark_id": benchmark_id,
        "reward_id": reward_id,
        "backend_version": backend_version,
        "hardware_mode": hardware_mode,
    }.items():
        if not str(value).strip():
            raise ValueError(f"{name} must be non-empty")
    timestamp = float(source_monotonic_time_s)
    if not math.isfinite(timestamp) or timestamp < 0.0:
        raise ValueError("source_monotonic_time_s must be finite and non-negative")
    received_timestamp = float(received_monotonic_time_s)
    if not math.isfinite(received_timestamp) or received_timestamp < timestamp:
        raise ValueError(
            "received_monotonic_time_s must be finite and no earlier than the source"
        )
    authority = float(residual_authority)
    if not math.isfinite(authority) or not 0.0 <= authority <= 1.0:
        raise ValueError("residual_authority must belong to [0, 1]")
    record = {
        "schema_version": REAL_LOG_SCHEMA_VERSION,
        "run_id": str(run_id),
        "episode_id": str(episode_id),
        "sequence": int(sequence),
        "source_monotonic_time_s": timestamp,
        "received_monotonic_time_s": received_timestamp,
        "wall_time_utc": str(wall_time_utc),
        "measurement_names": list(MEASUREMENT_NAMES),
        "measurement": _finite_vector("measurement", measurement, 6),
        "flow_measurement_names": list(FLOW_MEASUREMENT_NAMES),
        "flow_measurement_m3s": _nonnegative_vector(
            "flow_measurement", flow_measurement, 3
        ),
        "reference": _finite_vector("reference", reference, 6),
        "transition_disturbance": _finite_mapping(
            "transition_disturbance", transition_disturbance
        ),
        "disturbance": _finite_mapping("disturbance", disturbance),
        "commanded_action": _finite_vector("commanded_action", commanded_action, 2),
        "actuator_names": list(ACTUATOR_NAMES),
        "applied_action": _finite_vector("applied_action", applied_action, 5),
        "calibration_id": str(calibration_id),
        "calibration_hash": str(calibration_hash),
        "scenario_id": str(scenario_id),
        "benchmark_id": str(benchmark_id),
        "reward_id": str(reward_id),
        "backend_version": str(backend_version),
        "hardware_mode": str(hardware_mode),
        "residual_authority": authority,
        "safety": {} if safety is None else dict(safety),
        "raw_channels": {} if raw_channels is None else dict(raw_channels),
    }
    record["record_hash"] = content_digest(record)
    return record


def validate_real_step_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    if "record_hash" not in record:
        raise ValueError("real-step record_hash must be present")
    required = {
        "schema_version",
        "run_id",
        "episode_id",
        "sequence",
        "source_monotonic_time_s",
        "received_monotonic_time_s",
        "wall_time_utc",
        "measurement_names",
        "measurement",
        "flow_measurement_names",
        "flow_measurement_m3s",
        "reference",
        "transition_disturbance",
        "disturbance",
        "commanded_action",
        "actuator_names",
        "applied_action",
        "calibration_id",
        "calibration_hash",
        "scenario_id",
        "benchmark_id",
        "reward_id",
        "backend_version",
        "hardware_mode",
        "residual_authority",
        "safety",
        "raw_channels",
        "record_hash",
    }
    if set(record) != required:
        missing = sorted(required - set(record))
        unknown = sorted(set(record) - required)
        raise ValueError(
            f"real-step record fields are invalid: missing={missing}, unknown={unknown}"
        )
    if record["schema_version"] != REAL_LOG_SCHEMA_VERSION:
        raise ValueError("unsupported three-tank real-step schema_version")
    rebuilt = build_real_step_record(
        run_id=record["run_id"],
        episode_id=record["episode_id"],
        sequence=record["sequence"],
        source_monotonic_time_s=record["source_monotonic_time_s"],
        received_monotonic_time_s=record["received_monotonic_time_s"],
        wall_time_utc=record["wall_time_utc"],
        measurement=record["measurement"],
        flow_measurement=record["flow_measurement_m3s"],
        reference=record["reference"],
        transition_disturbance=record["transition_disturbance"],
        disturbance=record["disturbance"],
        commanded_action=record["commanded_action"],
        applied_action=record["applied_action"],
        calibration_id=record["calibration_id"],
        calibration_hash=record["calibration_hash"],
        scenario_id=record["scenario_id"],
        benchmark_id=record["benchmark_id"],
        reward_id=record["reward_id"],
        backend_version=record["backend_version"],
        hardware_mode=record["hardware_mode"],
        residual_authority=record["residual_authority"],
        safety=record["safety"],
        raw_channels=record["raw_channels"],
    )
    for name, expected in (
        ("measurement_names", list(MEASUREMENT_NAMES)),
        ("flow_measurement_names", list(FLOW_MEASUREMENT_NAMES)),
        ("actuator_names", list(ACTUATOR_NAMES)),
    ):
        if record[name] != expected:
            raise ValueError(f"{name} does not match the real-step channel contract")
    declared = record["record_hash"]
    if not str(declared).strip():
        raise ValueError("real-step record_hash must be present")
    if declared != rebuilt["record_hash"]:
        raise ValueError("declared record_hash does not match real-step record")
    return rebuilt


def _finite_mapping(name: str, values: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    resolved = {str(key): float(value) for key, value in values.items()}
    if any(not key for key in resolved) or not all(
        math.isfinite(value) for value in resolved.values()
    ):
        raise ValueError(f"{name} must contain non-empty names and finite values")
    return resolved


def _nonnegative_vector(name, values, length):
    resolved = _finite_vector(name, values, length)
    if any(value < 0.0 for value in resolved):
        raise ValueError(f"{name} must be non-negative")
    return resolved


class RealLogWriter:
    """Append validated records to JSONL while enforcing sequence continuity."""

    def __init__(self, path: str | Path, *, overwrite: bool = False):
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"real log already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._next_sequence = 0

    def append(self, value: Mapping[str, Any]) -> None:
        record = validate_real_step_record(value)
        if record["sequence"] != self._next_sequence:
            raise ValueError(
                f"expected sequence {self._next_sequence}, got {record['sequence']}"
            )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
        self._next_sequence += 1


__all__ = [
    "ACTUATOR_NAMES",
    "FLOW_MEASUREMENT_NAMES",
    "MEASUREMENT_NAMES",
    "REAL_LOG_SCHEMA_VERSION",
    "RealLogWriter",
    "build_real_step_record",
    "validate_real_step_record",
]
