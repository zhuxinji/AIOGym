from __future__ import annotations

from aiogym.core import OperatingCondition
from aiogym.workflows import study, sweep


CUSTOM_CONDITION = {
    "id": "custom-short-test",
    "initial_state": [0.20, 18.0, 0.21, 19.0, 0.22, 20.0],
    "reference": [0.20, 0.21, 0.22, 24.0, 25.0, 26.0],
    "control_dt": 2.0,
    "horizon": 10,
    "disturbances": {
        "t_amb": 18.0,
        "pump_flow_factor": 1.0,
        "heater_efficiency": 1.0,
        "heat_loss_factor": 1.0,
    },
    "observation": "controlled-output",
}


def test_study_binds_every_check_to_the_requested_condition():
    expected = OperatingCondition.from_mapping(CUSTOM_CONDITION)
    result = study(
        "lab-three-tank-v1",
        condition=CUSTOM_CONDITION,
        robustness_samples=1,
        seed=4,
    )

    assert result["condition_id"] == expected.id
    assert result["condition_hash"] == expected.condition_hash
    operation = result["study_context"]["operation"]
    assert operation["target_levels_m"] == CUSTOM_CONDITION["reference"][:3]
    assert operation["target_temperatures_degC"] == CUSTOM_CONDITION["reference"][3:]
    assert operation["initial_levels_m"] == CUSTOM_CONDITION["initial_state"][0::2]
    assert operation["initial_temperatures_degC"] == CUSTOM_CONDITION["initial_state"][1::2]
    assert operation["ambient_temperature_degC"] == 18.0
    assert operation["control_dt_s"] == 2.0
    assert operation["duration_s"] == 20.0

    assert result["dynamic"]["steps"] == 10
    assert result["dynamic"]["duration_s"] == 20.0
    assert result["dynamic"]["reference"] == CUSTOM_CONDITION["reference"]
    assert result["dynamic"]["initial_state"] == CUSTOM_CONDITION["initial_state"]
    assert result["dynamic"]["disturbance"]["t_amb"] == 18.0

    steady = next(
        row for row in result["checks"] if row["name"] == "steady_state_feasibility"
    )
    assert steady["metrics"]["reference"] == CUSTOM_CONDITION["reference"]
    assert steady["metrics"]["state"] == [
        0.20,
        24.0,
        0.21,
        25.0,
        0.22,
        26.0,
    ]
    assert all(
        case["condition_hash"] == expected.condition_hash
        for case in result["robustness"]["cases"]
    )
    assert all(
        case["steady"]["metrics"]["reference"] == CUSTOM_CONDITION["reference"]
        for case in result["robustness"]["cases"]
    )


def test_short_condition_reports_insufficient_assessment_horizon():
    result = study(
        "lab-three-tank-v1",
        condition=CUSTOM_CONDITION,
        robustness_samples=0,
        seed=0,
    )
    check = next(
        row
        for row in result["checks"]
        if row["name"] == "insufficient_assessment_horizon"
    )
    assert not check["passed"]
    assert check["metrics"] == {
        "available_duration_s": 20.0,
        "maximum_heatup_time_s": 2700.0,
    }
    assert "insufficient_assessment_horizon" in result["dynamic"]["reasons"]
    assert result["verdict"] == "FAIL"


def test_sweep_forwards_the_same_condition_and_controller():
    expected_hash = OperatingCondition.from_mapping(CUSTOM_CONDITION).condition_hash
    result = sweep(
        "lab-three-tank-v1",
        parameter="plant.heaters.0.power_w",
        values=(1900.0, 2000.0),
        condition=CUSTOM_CONDITION,
        controller="pid",
        robustness_samples=0,
        seed=0,
    )
    for candidate in result["candidates"]:
        study_result = candidate["result"]
        assert study_result["condition_hash"] == expected_hash
        assert study_result["dynamic"]["steps"] == 10
        assert study_result["dynamic"]["controller"]["id"] == "pid"
