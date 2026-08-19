from __future__ import annotations

import json

import pytest

from aiogym.experimental.three_tank_hardware.real_log import (
    RealLogWriter,
    build_real_step_record,
    validate_real_step_record,
)


def _record(sequence=0):
    return build_real_step_record(
        run_id="thermal-startup-001",
        episode_id="episode-001",
        sequence=sequence,
        source_monotonic_time_s=float(sequence),
        received_monotonic_time_s=float(sequence) + 0.1,
        wall_time_utc="2026-08-08T00:00:00Z",
        measurement=[0.18, 22.0, 0.18, 22.0, 0.18, 22.0, 20.5],
        flow_measurement=[3.0 / 60000.0] * 3,
        reference=[0.18, 22.0, 0.18, 22.0, 0.18, 22.0],
        transition_disturbance={"t_makeup": 20.0},
        disturbance={"t_makeup": 20.1},
        commanded_action=[0.0, 0.0, 0.0, 0.0, 0.0],
        applied_action=[0.5, 0.5, 0.5, 0.5, 0.2],
        calibration_id="calibration-001",
        calibration_hash="calibration-hash",
        scenario_id="three_tank",
        benchmark_id="tracking",
        reward_id="regulation",
        backend_version="hardware-v2",
        hardware_mode="shadow",
        safety={"mode": "shadow", "interlock": False},
        raw_channels={"LT301": 1234},
    )


def test_real_step_record_round_trips_and_rejects_tampering():
    record = _record()
    assert record["schema_version"].endswith(".v6")
    assert record["reward_id"] == "regulation"
    assert validate_real_step_record(record) == record
    record["measurement"][2] = 0.3
    with pytest.raises(ValueError, match="record_hash"):
        validate_real_step_record(record)
    record = _record()
    record.pop("record_hash")
    with pytest.raises(ValueError, match="must be present"):
        validate_real_step_record(record)


def test_real_log_writer_requires_contiguous_sequence(tmp_path):
    path = tmp_path / "real.jsonl"
    writer = RealLogWriter(path)
    writer.append(_record(0))
    with pytest.raises(ValueError, match="expected sequence 1"):
        writer.append(_record(2))
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["sequence"] == 0
