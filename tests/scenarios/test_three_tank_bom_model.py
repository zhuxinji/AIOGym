from __future__ import annotations

import numpy as np
import pytest

import aiogym
from aiogym.scenarios.three_tank.model import (
    BOM_CONFIGURATION,
    NOMINAL_FLOW_M3S,
    ThreeTankModel,
)
from aiogym.scenarios.three_tank.episodes import BENCHMARKS


def test_bom_geometry_reservoir_and_equipment_are_fixed():
    model = ThreeTankModel()
    assert BOM_CONFIGURATION["process_tanks"]["dimensions_m"] == [0.3, 0.3, 0.5]
    assert model.parameter("area") == [0.09, 0.09, 0.09]
    assert model.parameter("height_max") == [0.5, 0.5, 0.5]
    assert BOM_CONFIGURATION["process_tanks"]["effective_capacity_m3"] == 0.045
    assert BOM_CONFIGURATION["reservoir"]["effective_capacity_m3"] == 0.18
    assert model.parameter("pump_power_max") == 370.0
    assert model.parameter("pump_flow_max") == pytest.approx(25.0 / 60000.0)
    assert model.parameter("nominal_level") == pytest.approx(0.225)
    assert NOMINAL_FLOW_M3S == pytest.approx(3.0 / 60000.0)


def test_public_actions_match_installed_bom_only():
    env = aiogym.make_env("three_tank", reward="regulation")
    try:
        assert [row["name"] for row in env.model.action_schema()] == [
            "pump_P101",
            "valve_V12",
            "valve_V23",
            "valve_V34",
        ]
        assert env.action_space.shape == (4,)
    finally:
        env.close()


def test_default_operating_point_is_a_model_equilibrium():
    model = ThreeTankModel()
    equilibrium = model.nominal_steady_state()
    assert equilibrium["feasible"]
    derivative = model.dynamics(
        equilibrium["state"],
        equilibrium["action"],
        model.default_disturbances(),
    )
    assert max(abs(float(value)) for value in derivative) < 1e-10


def test_45_litre_inventory_preserves_nominal_hydraulic_safety_margin():
    model = ThreeTankModel()
    levels = model.initial_state()
    assert levels == pytest.approx([0.225, 0.225, 0.225])
    trips = model.parameter("high_level_trip")
    assert all(level < trips[index] for index, level in enumerate(levels))


def test_disturbance_benchmark_starts_from_nominal_steady_state():
    model = ThreeTankModel()
    steady = ThreeTankModel().nominal_steady_state()["state"]
    episode = BENCHMARKS["disturbance-rejection"].make_episode(model, 0)
    assert episode.initial_state == pytest.approx(steady)
    assert episode.horizon == 1800
    steps = tuple(episode.disturbance_schedule)
    assert 600 <= steps[0] <= 850
    assert 400 <= steps[1] - steps[0] <= 650
    assert len(steps) == 2
    assert 0.40 <= episode.disturbance_schedule[steps[0]]["pump_flow_factor"] <= 0.70
    assert 0.50 <= episode.disturbance_schedule[steps[0]]["v23_flow_factor"] <= 0.70
    assert episode.disturbance_schedule[steps[1]] == {
        "pump_flow_factor": 1.0,
        "v23_flow_factor": 1.0,
    }


def test_state_and_output_are_the_three_liquid_levels():
    model = ThreeTankModel()
    assert len(model.initial_state()) == 3
    assert model.outputs(model.initial_state()) == pytest.approx(
        [0.225, 0.225, 0.225]
    )


def test_observation_measurement_round_trip_preserves_state_and_output_order():
    model = ThreeTankModel()
    state = [0.10, 0.20, 0.30]
    reference = model.outputs(state)
    observation = model.observation(
        state,
        reference,
        model.default_action(),
        model.default_disturbances(),
    )
    observation_with_other_previous_action = model.observation(
        state,
        reference,
        [0.0] * 4,
        model.default_disturbances(),
    )

    measurement = model.measurement_from_observation(
        observation,
        model.default_disturbances(),
    )

    assert measurement["x"] == pytest.approx(state)
    assert measurement["y"] == pytest.approx(model.outputs(state))
    assert observation_with_other_previous_action == pytest.approx(observation)
    assert len(observation) == 6
    assert [row["kind"] for row in model.observation_schema()] == [
        *("measurement",) * 3,
        *("reference",) * 3,
    ]


def test_v34_and_passive_overflows_return_to_reservoir():
    model = ThreeTankModel()
    state = [0.48, 0.30, 0.30]
    derivative = model.dynamics(state, [0.0, 0.0, 0.0, 0.0])
    assert derivative[0] < 0.0
    assert derivative[1] == pytest.approx(0.0)
    assert derivative[2] == pytest.approx(0.0)

    open_v34 = model.dynamics(
        model.initial_state(), [0.0, 0.0, 0.0, 1.0]
    )
    assert open_v34[2] < 0.0


def test_reservoir_interlock_gates_pump():
    model = ThreeTankModel()
    action = [1.0] * 4
    dry = model.step_info(
        model.initial_state(),
        action,
        {**model.default_disturbances(), "reservoir_available": 0.0},
    )
    assert not dry["P101_enabled"]
    assert dry["P101_flow_m3s"] == 0.0

def test_physical_io_declares_three_process_measurement_sets_and_v34():
    io = ThreeTankModel().physical_io_schema()
    names = {row["name"] for row in io["measurements"]}
    assert {
        "LT101",
        "LT201",
        "LT301",
    } == names - {"FT12", "FT23", "FT34"}
    assert {"FT12", "FT23", "FT34"} <= names
    assert [row["name"] for row in io["actuators"]] == [
        "P101",
        "V12",
        "V23",
        "V34",
    ]
    assert io["boundary"]["effective_capacity_m3"] == 0.18


def test_environment_no_longer_accepts_a_plant_selector():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        aiogym.make_env("three_tank", reward="regulation", plant="anything")


def test_three_tank_environment_applies_requested_action_without_slew():
    env = aiogym.make_env("three_tank", benchmark="tracking")
    try:
        env.reset(seed=8)
        requested = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        _, _, _, _, info = env.step(requested)
        assert env.unwrapped.model.action_slew_limits() is None
        assert info["applied_action"] == pytest.approx(requested)
    finally:
        env.close()


def test_tracking_steady_state_action_matches_nominal_and_disturbance_context():
    model = ThreeTankModel()
    reference = model.default_setpoint_vector()
    nominal = model.tracking_steady_state_action(reference)
    assert nominal == pytest.approx(model.default_action())

    level = model.parameter("nominal_level")
    hydraulic = model.tracking_steady_state_action([level] * 3)
    assert hydraulic is not None
    assert hydraulic[-1] == pytest.approx(nominal[-1])

    reduced_capacity = model.tracking_steady_state_action(
        reference,
        {**model.default_disturbances(), "pump_flow_factor": 0.85},
    )
    assert reduced_capacity is not None
    assert reduced_capacity[0] > nominal[0]
    assert reduced_capacity[1:4] == pytest.approx(nominal[1:4])

    reduced_v23 = model.tracking_steady_state_action(
        reference,
        {**model.default_disturbances(), "v23_flow_factor": 0.8},
    )
    assert reduced_v23 is not None
    assert reduced_v23[2] > nominal[2]
    assert reduced_v23[:2] == pytest.approx(nominal[:2])
    assert reduced_v23[3:] == pytest.approx(nominal[3:])
