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
    assert model.parameter("reservoir_volume") == pytest.approx(0.18)
    assert model.parameter("reservoir_ua_loss") == pytest.approx(120.0)
    assert model.parameter("pump_power_max") == 370.0
    assert model.parameter("pump_flow_max") == pytest.approx(25.0 / 60000.0)
    assert model.parameter("heater_power") == 2000.0
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
            "heater_H1",
        ]
        assert env.action_space.shape == (5,)
        assert BOM_CONFIGURATION["reserved_heater_ports"] == [2, 3]
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


def test_45_litre_inventory_preserves_two_kw_heater_and_safety_margin():
    model = ThreeTankModel()
    levels = [model.initial_state()[index] for index in (0, 2, 4)]
    assert levels == pytest.approx([0.225, 0.225, 0.225])
    trips = model.parameter("low_level_trip")
    assert all(level > trips[index] for index, level in enumerate(levels))
    assert model.parameter("heater_power") == 2000.0


def test_disturbance_benchmark_starts_from_nominal_steady_state():
    model = ThreeTankModel()
    steady = ThreeTankModel().nominal_steady_state()["state"]
    episode = BENCHMARKS["disturbance-rejection"].make_episode(model, 0)
    assert episode.initial_state == pytest.approx(steady)
    assert episode.horizon == 3600
    assert episode.disturbance_schedule == {
        700: {"pump_flow_factor": 0.50, "v23_flow_factor": 0.50},
        1200: {"pump_flow_factor": 1.0, "v23_flow_factor": 1.0},
        1700: {"heat_loss_factor": 3.0},
        2300: {"heat_loss_factor": 1.0},
    }


def test_reservoir_is_a_seventh_dynamic_state_not_a_controlled_output():
    model = ThreeTankModel()
    assert len(model.initial_state()) == 7
    assert model.outputs(model.initial_state()) == pytest.approx(
        [0.225, 20.0, 0.225, 20.0, 0.225, 20.0]
    )
    state = model.nominal_steady_state()["state"]
    action = model.default_action()
    nominal = np.asarray(model.dynamics(state, action, model.default_disturbances()))
    warm_return_state = list(state)
    warm_return_state[5] += 1.0
    warm_return = np.asarray(
        model.dynamics(
            warm_return_state,
            action,
            model.default_disturbances(),
        )
    )
    assert nominal[6] == pytest.approx(0.0, abs=1e-12)
    assert warm_return[6] > nominal[6]


def test_makeup_temperature_affects_reservoir_during_inventory_increase():
    model = ThreeTankModel()
    state = model.nominal_steady_state()["state"]
    filling_action = [1.0, 0.0, 0.0, 0.0, 0.0]
    cold = np.asarray(
        model.dynamics(
            state,
            filling_action,
            {**model.default_disturbances(), "t_makeup": 15.0},
        )
    )
    warm = np.asarray(
        model.dynamics(
            state,
            filling_action,
            {**model.default_disturbances(), "t_makeup": 25.0},
        )
    )
    assert warm[:6] == pytest.approx(cold[:6])
    assert warm[6] > cold[6]


def test_observation_measurement_round_trip_preserves_state_and_output_order():
    model = ThreeTankModel()
    state = [0.10, 21.0, 0.20, 22.0, 0.30, 23.0, 20.5]
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
        [0.0] * 5,
        model.default_disturbances(),
    )

    measurement = model.measurement_from_observation(
        observation,
        model.default_disturbances(),
    )

    assert measurement["x"] == pytest.approx(state)
    assert measurement["y"] == pytest.approx(model.outputs(state))
    assert observation_with_other_previous_action == pytest.approx(observation)
    assert len(observation) == 13
    assert [row["kind"] for row in model.observation_schema()] == [
        *("measurement",) * 7,
        *("reference",) * 6,
    ]


def test_v34_and_passive_overflows_return_to_reservoir():
    model = ThreeTankModel()
    state = [0.48, 20.0, 0.30, 20.0, 0.30, 20.0, 20.0]
    derivative = model.dynamics(state, [0.0, 0.0, 0.0, 0.0, 0.0])
    assert derivative[0] < 0.0
    assert derivative[2] == pytest.approx(0.0)
    assert derivative[4] == pytest.approx(0.0)

    open_v34 = model.dynamics(
        model.initial_state(), [0.0, 0.0, 0.0, 1.0, 0.0]
    )
    assert open_v34[4] < 0.0


def test_hardware_interlocks_gate_pump_and_h1():
    model = ThreeTankModel()
    action = [1.0] * 5
    dry = model.step_info(
        model.initial_state(),
        action,
        {**model.default_disturbances(), "reservoir_available": 0.0},
    )
    assert not dry["P101_enabled"]
    assert dry["P101_flow_m3s"] == 0.0

    low_h1 = model.step_info(
        [0.05, 20.0, 0.24, 20.0, 0.24, 20.0, 20.0],
        action,
        model.default_disturbances(),
    )
    assert not low_h1["H1_enabled"]
    assert low_h1["H1_electric_power_w"] == 0.0


def test_physical_io_declares_three_process_measurement_sets_and_v34():
    io = ThreeTankModel().physical_io_schema()
    names = {row["name"] for row in io["measurements"]}
    assert {
        "LT101",
        "LT201",
        "LT301",
        "TT101",
        "TT201",
        "TT301",
        "TT401",
    } <= names
    assert {"FT12", "FT23", "FT34"} <= names
    assert [row["name"] for row in io["actuators"]] == [
        "P101",
        "V12",
        "V23",
        "V34",
        "H1",
    ]
    assert io["boundary"]["effective_capacity_m3"] == 0.18


def test_environment_no_longer_accepts_a_plant_selector():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        aiogym.make_env("three_tank", reward="regulation", plant="anything")


def test_three_tank_environment_applies_requested_action_without_slew():
    env = aiogym.make_env("three_tank", benchmark="tracking")
    try:
        env.reset(seed=8)
        requested = np.asarray([1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
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
    hydraulic = model.tracking_steady_state_action([level, 20.0] * 3)
    assert hydraulic is not None
    assert hydraulic[-1] == pytest.approx(0.0)

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
