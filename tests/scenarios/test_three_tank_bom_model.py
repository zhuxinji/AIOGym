from __future__ import annotations

from collections import Counter

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
    assert BOM_CONFIGURATION["pump"]["rated_max_flow_m3s"] == pytest.approx(
        8.0 / 3600.0
    )
    assert BOM_CONFIGURATION["pump"]["rated_max_head_m"] == 12.0
    assert model.parameter("cv_bypass") == pytest.approx([4.0e-5] * 3)
    assert model.parameter("flow_observation_scale") == pytest.approx(
        [10.0 / 60000.0] * 4
    )
    for flowmeter in BOM_CONFIGURATION["flowmeters"]:
        assert flowmeter["range_l_min"] == pytest.approx([0.0, 10.0])
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


def test_disturbance_benchmark_adds_seeded_bypasses_to_matching_tracking_case():
    model = ThreeTankModel()
    modes = Counter()
    single_branches = Counter()
    simultaneous_pairs = Counter()
    simultaneous_triples = 0
    for seed in range(20):
        episode = BENCHMARKS["disturbance-rejection"].make_episode(model, seed)
        paired = BENCHMARKS["tracking"].make_episode(model, seed)
        assert episode.initial_state == pytest.approx(paired.initial_state)
        assert episode.initial_action == pytest.approx(paired.initial_action)
        assert episode.reference == pytest.approx(paired.reference)
        assert not episode.reference_schedule
        assert episode.horizon == 600
        assert all(
            set(changes) <= {"bv12_open", "bv23_open", "bv34_open"}
            for changes in episode.disturbance_schedule.values()
        )

        branch_events = {}
        for name in ("bv12_open", "bv23_open", "bv34_open"):
            events = [
                (step, changes[name])
                for step, changes in episode.disturbance_schedule.items()
                if name in changes
            ]
            if events:
                assert len(events) == 2
                assert events[0][1] == 1.0
                assert events[1][0] == events[0][0] + 360
                assert events[1][1] == 0.0
                branch_events[name] = events
        opens = sorted((events[0][0], name) for name, events in branch_events.items())
        if len(opens) == 1:
            modes["single"] += 1
            start, name = opens[0]
            assert 30 <= start <= 90
            single_branches[name] += 1
        elif len(opens) == 2 and opens[0][0] == opens[1][0]:
            modes["simultaneous-pair"] += 1
            assert 30 <= opens[0][0] <= 90
            simultaneous_pairs[tuple(sorted(name for _, name in opens))] += 1
        elif len(opens) == 3 and len({step for step, _ in opens}) == 1:
            modes["simultaneous-triple"] += 1
            simultaneous_triples += 1
            assert 30 <= opens[0][0] <= 90
        else:
            raise AssertionError(f"unexpected bypass event pattern: {opens}")

        open_step, close_step = sorted(episode.disturbance_schedule)
        assert close_step == open_step + 360

        flow = model.process_info(
            episode.initial_state,
            episode.initial_action,
            episode.disturbances,
        )["P101_flow_m3s"]
        assert 3.0 <= flow * 60000.0 <= 8.0
        persistent = dict(episode.disturbances)
        persistent.update(episode.disturbance_schedule[open_step])
        target = model.nominal_steady_state(
            levels=episode.reference,
            flow=flow,
            env=persistent,
        )
        assert target["feasible"]
        derivative = model.dynamics(target["state"], target["action"], persistent)
        assert max(abs(float(value)) for value in derivative) < 1e-10
        restored = dict(persistent)
        restored.update(episode.disturbance_schedule[close_step])
        restored_target = model.nominal_steady_state(
            levels=episode.reference,
            flow=flow,
            env=restored,
        )
        assert restored_target["feasible"]

    assert modes == {
        "single": 6,
        "simultaneous-pair": 9,
        "simultaneous-triple": 5,
    }
    assert set(single_branches) == {"bv12_open", "bv23_open", "bv34_open"}
    assert set(simultaneous_pairs) == {
        ("bv12_open", "bv23_open"),
        ("bv12_open", "bv34_open"),
        ("bv23_open", "bv34_open"),
    }
    assert simultaneous_triples == 5


def test_bypass_and_operating_flow_ranges_are_physically_separated():
    model = ThreeTankModel()
    levels = (0.125, 0.4)
    bypass_l_min = [
        model.parameter("cv_bypass")[0]
        * np.sqrt(level + model.parameter("gravity_drop")[0])
        * 60000.0
        for level in levels
    ]
    valve_l_min_at_benchmark_limit = (
        model.parameter("cv_valves")[0]
        * 0.85
        * np.sqrt(levels[0] + model.parameter("gravity_drop")[0])
        * 60000.0
    )
    assert bypass_l_min == pytest.approx([1.56460858, 2.00798406])
    assert valve_l_min_at_benchmark_limit == pytest.approx(16.62396613)


def test_disturbance_benchmark_uses_whole_episode_tracking_metrics_only():
    env = aiogym.make_env("three_tank", benchmark="disturbance-rejection")
    try:
        result = aiogym.evaluate(env=env, policies={"policy": "pid"}, seeds=(0,))["evaluations"]["policy"]
    finally:
        env.close()

    metrics = result["episodes"][0]["metrics"]
    expected = {"tracking_iae", "final_error", "settling_time", "return"}
    assert expected <= set(metrics)
    assert "disturbance_iae" not in metrics
    assert "recovery_time" not in metrics
    assert not any(name.startswith(("bypass_", "post_bypass_")) for name in metrics)


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
    assert observation_with_other_previous_action[:3] == pytest.approx(observation[:3])
    assert observation_with_other_previous_action[3:7] == pytest.approx([0.0] * 4)
    assert observation_with_other_previous_action[7:] == pytest.approx(observation[7:])
    assert observation[3:7] != pytest.approx([0.0] * 4)
    assert measurement["flow_measurement_m3s"] == pytest.approx(
        np.asarray(observation[3:7]) * np.asarray(model.parameter("flow_observation_scale"))
    )
    assert len(observation) == 10
    assert [row["kind"] for row in model.observation_schema()] == [
        *("measurement",) * 7,
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


def test_binary_bypass_adds_a_bounded_parallel_transfer_flow():
    model = ThreeTankModel()
    action = model.default_action()
    nominal = model.process_info(model.initial_state(), action)
    disturbed = model.process_info(
        model.initial_state(),
        action,
        {**model.default_disturbances(), "bv12_open": 1.0},
    )
    added_l_min = disturbed["BV12_flow_m3s"] * 60000.0
    assert added_l_min == pytest.approx(1.738965, rel=1e-5)
    assert disturbed["BV12_flow_m3s"] * 60000.0 == pytest.approx(
        added_l_min
    )
    assert disturbed["BV23_flow_m3s"] == 0.0
    assert disturbed["BV34_flow_m3s"] == 0.0
    assert disturbed["V12_flow_m3s"] == pytest.approx(nominal["V12_flow_m3s"])
    assert disturbed["FT12_flow_m3s"] == pytest.approx(nominal["FT12_flow_m3s"])
    closed_observation = model.observation(
        model.initial_state(),
        model.default_setpoint_vector(),
        action,
        model.default_disturbances(),
    )
    open_observation = model.observation(
        model.initial_state(),
        model.default_setpoint_vector(),
        action,
        {**model.default_disturbances(), "bv12_open": 1.0},
    )
    assert open_observation[3] == pytest.approx(closed_observation[3])
    assert open_observation[4] == pytest.approx(closed_observation[4])
    assert open_observation[5] == pytest.approx(closed_observation[5])
    assert open_observation[6] == pytest.approx(closed_observation[6])

    outlet_bypass = model.process_info(
        model.initial_state(),
        action,
        {**model.default_disturbances(), "bv34_open": 1.0},
    )
    assert outlet_bypass["FT34_flow_m3s"] == pytest.approx(
        nominal["FT34_flow_m3s"]
    )
    assert outlet_bypass["V34_flow_m3s"] == pytest.approx(nominal["V34_flow_m3s"])
    assert model.dynamics(
        model.initial_state(),
        action,
        {**model.default_disturbances(), "bv34_open": 1.0},
    )[2] < model.dynamics(model.initial_state(), action)[2]
    with pytest.raises(ValueError, match="must be binary"):
        model.dynamics(
            model.initial_state(),
            action,
            {**model.default_disturbances(), "bv12_open": 0.5},
        )


def test_physical_io_declares_plan_flowmeters_bypasses_and_v34():
    io = ThreeTankModel().physical_io_schema()
    names = {row["name"] for row in io["measurements"]}
    assert {
        "LT101",
        "LT201",
        "LT301",
    } == names - {"FT101", "FT12", "FT23", "FT34"}
    assert {"FT101", "FT12", "FT23", "FT34"} <= names
    assert [row["name"] for row in io["actuators"]] == [
        "P101",
        "V12",
        "V23",
        "V34",
    ]
    assert [row["name"] for row in io["disturbance_actuators"]] == [
        "BV12",
        "BV23",
        "BV34",
    ]
    assert io["boundary"]["effective_capacity_m3"] == 0.18


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
