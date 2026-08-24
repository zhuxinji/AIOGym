from __future__ import annotations

import pytest

from aiogym.experimental.three_tank_hardware.calibration import (
    calibration_template,
    validate_calibration,
)


def test_calibration_template_is_explicitly_unmeasured_and_hashed():
    record = calibration_template(rig_id="lab-three-tank")
    assert record["status"] == "unmeasured"
    assert record["parameters"]["pump_flow_max"]["unit"] == "m3/s"
    assert "heater_efficiency" not in record["parameters"]
    assert len(record["calibration_hash"]) == 64
    assert validate_calibration(record) == record
    with pytest.raises(ValueError, match="not measured"):
        validate_calibration(record, require_measured=True)


def test_calibration_hash_rejects_silent_edits():
    record = calibration_template(rig_id="lab-three-tank")
    record["parameters"]["area_1"]["estimate"] = 0.09
    with pytest.raises(ValueError, match="calibration_hash"):
        validate_calibration(record)


@pytest.mark.parametrize(
    ("name", "estimate", "message"),
    [
        ("area_1", 0.0, "positive"),
        ("sensor_delay", -0.1, "non-negative"),
    ],
)
def test_calibration_rejects_nonphysical_estimates(name, estimate, message):
    record = calibration_template(rig_id="lab-three-tank")
    record.pop("calibration_hash")
    record["parameters"][name]["estimate"] = estimate
    with pytest.raises(ValueError, match=message):
        validate_calibration(record)


def test_measured_calibration_requires_disjoint_evidence_and_metrics():
    record = calibration_template(rig_id="lab-three-tank")
    record.update(
        calibration_id="calibration-001",
        status="validated",
        fit_dataset_ids=["fit-1"],
        validation_dataset_ids=["fit-1"],
        validation_metrics={"level_rmse": 0.1},
    )
    record.pop("calibration_hash")
    for row in record["parameters"].values():
        row.update(estimate=1.0, standard_uncertainty=0.01, source_dataset="fit-1")
    with pytest.raises(ValueError, match="disjoint"):
        validate_calibration(record, require_measured=True)
