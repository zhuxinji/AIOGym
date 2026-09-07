from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import aiogym
from aiogym.core.model import integrate_process_state
from aiogym.scenarios.cascade.model import (
    BOM_CONFIGURATION,
    HEATER_POWER_W,
    TRACKING_SETTLING_TOLERANCES,
    CascadeModel,
)
from aiogym.scenarios.cascade.metrics import regulation_episode_metrics
from aiogym.scenarios.three_tank.model import (
    BOM_CONFIGURATION as THREE_TANK_BOM_CONFIGURATION,
    ThreeTankModel,
)


def test_cascade_extends_three_tank_hydraulics_with_three_heaters():
    model = CascadeModel()
    three_tank = ThreeTankModel()
    assert [row["name"] for row in model.state_schema()] == [
        "h1",
        "T1",
        "h2",
        "T2",
        "h3",
        "T3",
        "reservoir_volume",
        "reservoir_temperature",
    ]
    assert [row["name"] for row in model.output_schema()] == [
        "tank_1_level",
        "tank_2_level",
        "tank_3_level",
        "tank_1_temperature",
        "tank_2_temperature",
        "tank_3_temperature",
    ]
    assert [row["name"] for row in model.action_schema()] == [
        "pump_P101",
        "valve_V12",
        "valve_V23",
        "valve_V34",
        "heater_H1",
        "heater_H2",
        "heater_H3",
    ]
    assert [row["id"] for row in BOM_CONFIGURATION["installed_heaters"]] == [
        "H1",
        "H2",
        "H3",
    ]
    assert {
        name: model.resolved_parameters[name]
        for name in three_tank.resolved_parameters
    } == dict(three_tank.resolved_parameters)
    assert {
        name: BOM_CONFIGURATION[name]
        for name in THREE_TANK_BOM_CONFIGURATION
        if name != "source"
    } == {
        name: value
        for name, value in THREE_TANK_BOM_CONFIGURATION.items()
        if name != "source"
    }
    assert model.parameter("heater_power_max") == [HEATER_POWER_W] * 3
    assert model.parameter("reservoir_capacity") == pytest.approx(0.180)
    assert model.parameter("reservoir_initial_volume") == pytest.approx(0.090)
    assert model.parameter("reservoir_heat_loss_coefficient") == pytest.approx(40.0)
    assert model.heater == (1, 0, 0)
    assert [
        row["power_w"] for row in BOM_CONFIGURATION["installed_heaters"]
    ] == [HEATER_POWER_W] * 3


def test_cascade_settling_uses_physical_level_and_temperature_tolerances():
    reference = np.asarray([0.30, 0.25, 0.325, 24.0, 27.0, 30.0])
    output = reference + np.asarray([0.004, -0.004, 0.004, 0.15, -0.15, 0.15])
    transition = SimpleNamespace(
        info={
            "constraint_costs": {},
            "energy_kw": 0.0,
            "minimum_safety_margin": 1.0,
            "transition_reference": reference,
            "y": output,
        },
        step_index=0,
        terminated=False,
        truncated=True,
    )
    episode = SimpleNamespace(
        episode_return=0.0,
        reset_info={"episode_spec": {"disturbance_schedule": {}}},
        transitions=(transition,),
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(control_dt=1.0, model=SimpleNamespace(time_unit="s"))
    )

    metrics = regulation_episode_metrics(env, episode)

    assert TRACKING_SETTLING_TOLERANCES == pytest.approx(
        (0.005, 0.005, 0.005, 0.2, 0.2, 0.2)
    )
    assert metrics["settling_time"] == 0.0
    assert metrics["settling_rate"] == 1.0
    assert metrics["final_error"] == pytest.approx(0.04)

    transition.info["y"] = reference + np.asarray(
        [0.006, -0.004, 0.004, 0.15, -0.15, 0.15]
    )
    outside_metrics = regulation_episode_metrics(env, episode)
    assert outside_metrics["settling_time"] == 1.0
    assert outside_metrics["settling_rate"] == 0.0


def test_default_operating_point_is_a_hydraulic_and_thermal_equilibrium():
    model = CascadeModel()
    equilibrium = model.nominal_steady_state()
    derivative = model.dynamics(
        equilibrium["state"],
        equilibrium["action"],
        model.default_disturbances(),
    )
    assert equilibrium["feasible"]
    assert max(abs(float(value)) for value in derivative) < 1e-10
    assert model.outputs(equilibrium["state"]) == pytest.approx(
        equilibrium["y_sp"]
    )
    assert np.all(0.0 < np.asarray(equilibrium["action"][:5]))
    assert np.all(np.asarray(equilibrium["action"][:5]) < 1.0)
    assert equilibrium["action"][5:] == pytest.approx([0.0, 0.0])


def test_reservoir_starts_half_full_at_ambient_and_stays_hidden_from_policy():
    model = CascadeModel()
    state = model.initial_state()
    assert state[6] == pytest.approx(0.090)
    assert state[7] == pytest.approx(model.parameter("ambient_temperature"))
    reference = model.default_setpoint_vector()
    action = model.default_action()
    observation = model.observation(
        state,
        reference,
        action,
        model.default_disturbances(),
    )
    changed_hidden_state = [*state[:6], 0.120, state[7] + 5.0]
    changed_observation = model.observation(
        changed_hidden_state,
        reference,
        action,
        model.default_disturbances(),
    )
    assert len(state) == 8
    assert len(observation) == 16
    assert changed_observation == pytest.approx(observation)


def test_closed_reservoir_conserves_liquid_and_warms_from_return_flow():
    model = CascadeModel()
    state = np.asarray(model.initial_state(), dtype=float)
    action = np.asarray(model.default_action(), dtype=float)
    action[0] += 0.05
    area = np.asarray(model.parameter("area"), dtype=float)

    def total_liquid(values):
        return float(values[6] + np.sum(area * values[[0, 2, 4]]))

    initial_total = total_liquid(state)
    result = integrate_process_state(
        model,
        state,
        action,
        model.default_disturbances(),
        duration=120.0,
    )
    assert total_liquid(result) == pytest.approx(initial_total, abs=1e-10)
    assert 0.0 < result[6] < model.parameter("reservoir_capacity")
    assert result[7] > state[7]


def test_reservoir_inventory_limits_are_safety_constraints():
    model = CascadeModel()
    empty = model.initial_state()
    empty[6] = 0.0
    full = model.initial_state()
    full[6] = model.parameter("reservoir_capacity") + 1e-4
    assert "reservoir_empty" in model.hard_termination_reasons(empty)
    assert "reservoir_overflow_limit" in model.hard_termination_reasons(full)


def test_bypass_flow_affects_cascade_balances_but_not_branch_flowmeters():
    model = CascadeModel()
    state = model.initial_state()
    action = model.default_action()
    disturbances = model.default_disturbances()
    nominal_info = model.process_info(state, action, disturbances)
    nominal_observation = model.observation(
        state,
        model.default_setpoint_vector(),
        action,
        disturbances,
    )
    nominal_derivative = np.asarray(model.dynamics(state, action, disturbances))

    for branch_index, branch in enumerate(("12", "23", "34")):
        disturbed = {**disturbances, f"bv{branch}_open": 1.0}
        disturbed_info = model.process_info(state, action, disturbed)
        disturbed_observation = model.observation(
            state,
            model.default_setpoint_vector(),
            action,
            disturbed,
        )
        disturbed_derivative = np.asarray(model.dynamics(state, action, disturbed))

        assert disturbed_info[f"BV{branch}_flow_m3s"] > 0.0
        assert disturbed_info[f"FT{branch}_flow_m3s"] == pytest.approx(
            nominal_info[f"FT{branch}_flow_m3s"]
        )
        assert disturbed_observation[6:10] == pytest.approx(
            nominal_observation[6:10]
        )
        assert disturbed_derivative[2 * branch_index] < nominal_derivative[
            2 * branch_index
        ]


def test_each_heater_independently_heats_its_own_tank():
    model = CascadeModel(heater=[1, 1, 1])
    state = model.initial_state()
    base_action = np.asarray(model.default_action(), dtype=float)
    base = np.asarray(model.dynamics(state, base_action), dtype=float)
    temperature_indexes = (1, 3, 5)
    for heater_index, state_index in enumerate(temperature_indexes):
        action = base_action.copy()
        action[4 + heater_index] += 0.1
        changed = np.asarray(model.dynamics(state, action), dtype=float)
        difference = changed - base
        assert difference[state_index] > 0.0
        assert np.count_nonzero(np.abs(difference) > 1e-12) == 1


def test_low_level_and_temperature_interlocks_are_per_tank():
    model = CascadeModel(heater=[1, 1, 1])
    action = np.ones(7)
    state = model.initial_state()
    state[0] = model.parameter("heater_min_level") - 0.01
    state[3] = model.parameter("temperature_trip")
    info = model.process_info(state, action)
    assert info["heater_interlocked"] == [True, True, False]
    assert info["heater_electric_power_w"] == pytest.approx([0.0, 0.0, 2000.0])


def test_observation_round_trip_includes_shared_flow_measurements():
    model = CascadeModel()
    state = [0.10, 21.0, 0.20, 24.0, 0.30, 28.0, 0.09, 22.0]
    reference = model.outputs(state)
    disturbances = {
        **model.default_disturbances(),
        "ambient_temperature": 27.0,
    }
    observation = model.observation(
        state,
        reference,
        model.default_action(),
        disturbances,
    )
    other_observation = model.observation(
        state,
        reference,
        model.default_action(),
        model.default_disturbances(),
    )
    measurement = model.measurement_from_observation(observation, disturbances)
    assert len(observation) == 16
    assert observation == pytest.approx(other_observation)
    assert measurement["x"][:6] == pytest.approx(state[:6])
    assert len(measurement["x"]) == 8
    assert measurement["y"] == pytest.approx(reference)
    assert measurement["reservoir_volume_m3"] == pytest.approx(0.09)
    assert 20.0 < measurement["reservoir_temperature"] < state[5]
    assert measurement["ambient_temperature"] == pytest.approx(27.0)
    assert measurement["flow_measurement_m3s"] == pytest.approx(
        np.asarray(observation[6:10])
        * np.asarray(model.parameter("flow_observation_scale"))
    )
    assert [row["kind"] for row in model.observation_schema()] == [
        *("measurement",) * 10,
        *("reference",) * 6,
    ]


def test_runtime_thermal_disturbances_stay_outside_policy_observation():
    env = aiogym.make_env("cascade")
    try:
        env.set_disturbances({"ambient_temperature": 31.0})
        observation, info = env.reset(seed=3)
        assert observation.shape == (16,)
        assert "reservoir_temperature" not in info["disturbance"]
        assert info["reservoir_temperature"] == pytest.approx(
            env.unwrapped.episode.initial_state[7]
        )
        assert info["disturbance"]["ambient_temperature"] == pytest.approx(31.0)
        env.set_disturbances({"ambient_temperature": 24.0})
        observation, _, _, _, info = env.step(env.unwrapped.model.default_action())
        assert observation.shape == (16,)
        assert "reservoir_temperature" not in info["disturbance"]
        assert info["disturbance"]["ambient_temperature"] == pytest.approx(24.0)
    finally:
        env.close()


def test_cascade_parameter_overrides_preserve_the_interface_and_change_dynamics():
    base = CascadeModel(heater=[1, 1, 1])
    changed = CascadeModel(
        {"heater_power_max": [1500.0, 1800.0, 2200.0]},
        heater=[1, 1, 1],
    )
    state = base.initial_state()
    action = [*base.default_action()[:4], 0.5, 0.5, 0.5]
    base_derivative = np.asarray(base.dynamics(state, action), dtype=float)
    changed_derivative = np.asarray(changed.dynamics(state, action), dtype=float)
    assert changed.resolved_parameters["heater_power_max"] == (
        1500.0,
        1800.0,
        2200.0,
    )
    assert not np.allclose(base_derivative[[1, 3, 5]], changed_derivative[[1, 3, 5]])
    assert base.action_dim() == changed.action_dim() == 7


def test_cascade_tracking_cases_are_reproducible_feasible_equilibrium_moves():
    env = aiogym.make_env("cascade", benchmark="tracking")
    serialized = []
    sampled_flows_l_min = []
    temperature_move_lower = np.asarray([-4.5, -3.5, -3.0])
    temperature_move_upper = np.asarray([6.5, 5.0, 3.5])
    try:
        for seed in range(20):
            _, info = env.reset(seed=seed)
            episode = env.unwrapped.episode
            assert episode.horizon == 2100
            serialized.append(repr(info["episode_spec"]))
            start = np.asarray(env.model.outputs(episode.initial_state))
            target = np.asarray(episode.reference)
            flow_m3s = env.model.process_info(
                episode.initial_state,
                episode.initial_action,
                episode.disturbances,
            )["P101_flow_m3s"]
            initial_derivative = env.model.dynamics(
                episode.initial_state,
                episode.initial_action,
                episode.disturbances,
            )
            assert max(abs(float(value)) for value in initial_derivative) < 1e-10
            assert episode.initial_state[7] > env.model.parameter(
                "ambient_temperature"
            )
            sampled_flows_l_min.append(60000.0 * flow_m3s)
            assert np.all((0.125 <= start[:3]) & (start[:3] <= 0.40))
            assert np.all((0.125 <= target[:3]) & (target[:3] <= 0.40))
            assert np.all(np.abs(target[:3] - start[:3]) >= 0.05)
            temperature_move = target[3:] - start[3:]
            assert np.all(np.abs(temperature_move) >= 1.0)
            assert np.all(temperature_move_lower <= temperature_move)
            assert np.all(temperature_move <= temperature_move_upper)
            area = np.asarray(env.model.parameter("area"), dtype=float)
            target_reservoir_volume = episode.initial_state[6] + np.sum(
                area * (start[:3] - target[:3])
            )
            assert 0.0 < target_reservoir_volume < env.model.parameter(
                "reservoir_capacity"
            )
            for reference in (start, target):
                equilibrium = env.model.nominal_steady_state(
                    flow=flow_m3s,
                    levels=reference[:3],
                    temperatures=reference[3:],
                    env=episode.disturbances,
                )
                assert equilibrium["feasible"]
                state = equilibrium["state"]
                action = np.asarray(equilibrium["action"])
                available = np.asarray(env.model.heater, dtype=bool)
                assert np.all(0.02 <= action[:4])
                assert np.all(action[:4] <= 0.85)
                assert np.all(0.05 <= action[4:][available])
                assert np.all(action[4:][available] <= 0.80)
                assert action[4:][~available] == pytest.approx(0.0)
                derivative = env.unwrapped.model.dynamics(
                    state,
                    action,
                    episode.disturbances,
                )
                assert max(abs(float(value)) for value in derivative) < 1e-10
    finally:
        env.close()
    assert len(set(serialized)) == 20
    assert all(2.0 <= value <= 6.0 for value in sampled_flows_l_min)
    assert any(value > 4.0 for value in sampled_flows_l_min)


def test_cascade_boundary_cases_start_from_forward_prerun_boundary_states():
    env = aiogym.make_env("cascade", benchmark="boundary-safety")
    specs = []
    try:
        for seed in range(20):
            _, info = env.reset(seed=seed)
            episode = env.unwrapped.episode
            assert episode.horizon == 600
            specs.append(repr(info["episode_spec"]))
            state = np.asarray(episode.initial_state, dtype=float)
            action = np.asarray(episode.initial_action, dtype=float)
            levels = state[[0, 2, 4]]
            temperatures = state[[1, 3, 5]]
            derivative = env.model.dynamics(
                state,
                action,
                episode.disturbances,
            )
            assert np.any(levels >= 0.41)
            assert np.all((0.0 < levels) & (levels < 0.5))
            assert np.all((20.0 < temperatures) & (temperatures < 80.0))
            available = np.asarray(env.model.heater, dtype=bool)
            assert np.all(
                (0.80 <= action[4:][available])
                & (action[4:][available] <= 0.95)
            )
            assert action[4:][~available] == pytest.approx(0.0)
            assert max(abs(float(value)) for value in derivative) > 1e-4
            assert not any(
                float(value) > 0.0
                for value in env.model.constraint_costs(
                    state, episode.disturbances
                ).values()
            )
    finally:
        env.close()
    assert len(set(specs)) == 20


def test_randomized_boundary_starts_reuse_the_forward_prerun_family():
    env = aiogym.make_env(
        "cascade", randomize=True, boundary_probability=1.0
    )
    try:
        for seed in (0, 1):
            _, info = env.reset(seed=seed)
            assert info["episode_parameters"]["initial_family"] == "boundary-prerun"
            episode = env.unwrapped.episode
            assert episode.horizon == 600
            state = np.asarray(episode.initial_state, dtype=float)
            levels = state[[0, 2, 4]]
            assert np.any(levels >= 0.40)
            derivative = env.unwrapped.model.dynamics(
                state,
                episode.initial_action,
                episode.disturbances,
            )
            assert max(abs(float(value)) for value in derivative) > 1e-4
    finally:
        env.close()


@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_cascade_baseline_controller_emits_direct_physical_actions(controller_id):
    env = aiogym.make_env("cascade", benchmark="tracking")
    try:
        observation, info = env.reset(seed=3)
        policy = aiogym.make_controller(controller_id, env=env)
        action = policy.act(observation, {"reference": info["reference"]})
        assert action.shape == (7,)
        assert env.action_space.contains(action)
        _, reward, terminated, truncated, step_info = env.step(action)
        assert np.isfinite(reward)
        assert not terminated
        assert not truncated
        expected_applied = np.asarray(action, dtype=float)
        expected_applied[5:] = 0.0
        assert step_info["commanded_action"] == pytest.approx(action)
        assert step_info["applied_action"] == pytest.approx(expected_applied)
    finally:
        env.close()


def test_cascade_pid_uses_target_steady_action_as_reference_scoped_bias():
    env = aiogym.make_env("cascade", benchmark="tracking")
    try:
        policy = aiogym.make_controller(
            "pid",
            env=env,
            config={"matrix_terms": []},
        )
        observation, info = env.reset(seed=3)
        first_reference = np.asarray(info["reference"], dtype=float)
        first_expected = env.unwrapped.model.tracking_steady_state_action(
            first_reference
        )
        first_action = policy.act(
            observation,
            {"reference": first_reference},
        )
        assert first_expected is not None
        assert first_action == pytest.approx(first_expected)

        policy.integral.fill(0.25)
        policy.act(
            observation,
            {"reference": first_reference.copy()},
        )
        assert policy.integral == pytest.approx(0.25)

        observation, info = env.reset(seed=4)
        second_reference = np.asarray(info["reference"], dtype=float)
        second_expected = env.unwrapped.model.tracking_steady_state_action(
            second_reference
        )
        second_action = policy.act(
            observation,
            {"reference": second_reference},
        )
        assert second_expected is not None
        assert second_action == pytest.approx(second_expected)
        assert policy.integral == pytest.approx(0.0)
        assert not np.allclose(first_action, second_action)
    finally:
        env.close()


def test_cascade_tuned_controller_config_is_shared_by_all_benchmarks():
    metadata = []
    for benchmark in (
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    ):
        env = aiogym.make_env("cascade", benchmark=benchmark)
        try:
            metadata.append(
                (
                    aiogym.make_controller("pid", env=env).metadata(),
                    aiogym.make_controller("mpc", env=env).metadata(),
                )
            )
        finally:
            env.close()

    assert metadata[0] == metadata[1] == metadata[2]
    pid, mpc = metadata[0]
    assert pid["kp"][0] == [0.0] * 6
    assert pid["ki"][0] == [0.0] * 6
    assert max(abs(value) for row in pid["kp"][1:4] for value in row) == 48.0
    assert max(abs(value) for row in pid["ki"][1:4] for value in row) == 0.12
    assert pid["kp"][4] == [0.0, 0.0, 0.0, 0.75, 0.15, 0.10]
    assert pid["ki"][4] == [0.0, 0.0, 0.0, 0.0075, 0.0015, 0.0010]
    assert pid["kp"][5][4] == 1.0
    assert pid["ki"][5][4] == 0.01
    assert pid["kp"][6][5] == 1.0
    assert pid["ki"][6][5] == 0.01
    assert pid["feedforward"] == (
        "tracking_steady_state_action_on_reference_change"
    )
    assert mpc["Ts"] == 2.0
    assert mpc["horizon"] == 30
    assert mpc["move_supp"] == [50.0] * 4 + [2.0] * 3
    assert mpc["steady_input_weight"] == [1.0] * 7
    assert mpc["q_y"] == [3.0, 3.0, 3.0, 2.0, 2.5, 4.0]


def test_cascade_benchmarks_rank_settling_before_return():
    for benchmark in (
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    ):
        env = aiogym.make_env("cascade", benchmark=benchmark)
        try:
            assert env.unwrapped.benchmark.ranking_metrics == (
                ("unsafe_rate", "minimize"),
                ("settling_rate", "maximize"),
                ("return", "maximize"),
            )
        finally:
            env.close()


def test_cascade_evaluation_aggregates_settling_as_a_case_rate():
    env = aiogym.make_env("cascade", benchmark="tracking")
    try:
        result = aiogym.evaluate(
            env=env,
            policy="hold",
            seeds=(0, 1),
            max_steps=2,
        )
    finally:
        env.close()
    assert result["ranking_metrics"] == [
        {"name": "unsafe_rate", "direction": "minimize", "aggregate": "mean"},
        {"name": "settling_rate", "direction": "maximize", "aggregate": "mean"},
        {"name": "return", "direction": "maximize", "aggregate": "mean"},
    ]


def test_physical_io_declares_level_temperature_and_all_actuator_channels():
    io = CascadeModel().physical_io_schema()
    assert {row["name"] for row in io["measurements"]} == {
        "LT101",
        "LT201",
        "LT301",
        "TT101",
        "TT201",
        "TT301",
        "FT101",
        "FT12",
        "FT23",
        "FT34",
    }
    assert [row["name"] for row in io["actuators"]] == [
        "P101",
        "V12",
        "V23",
        "V34",
        "H1",
        "H2",
        "H3",
    ]
    assert [row.get("available") for row in io["actuators"][4:]] == [
        True,
        False,
        False,
    ]
    assert [row["name"] for row in io["disturbance_actuators"]] == [
        "BV12",
        "BV23",
        "BV34",
    ]


def test_public_heater_configuration_masks_unavailable_actions_and_power():
    env = aiogym.make_env("cascade", heater=[1, 0, 1], benchmark="tracking")
    try:
        _, info = env.reset(seed=3)
        assert env.unwrapped.model.heater == (1, 0, 1)
        assert env.unwrapped.runtime_config["parameters"]["heater"] == [1, 0, 1]
        _, _, terminated, truncated, step_info = env.step(np.ones(7))
        assert not terminated
        assert not truncated
        assert info["episode_spec"]["initial_action"][5] == pytest.approx(0.0)
        assert step_info["commanded_action"][4:] == pytest.approx([1.0, 1.0, 1.0])
        assert step_info["applied_action"][4:] == pytest.approx([1.0, 0.0, 1.0])
        assert step_info["heater_available"] == [True, False, True]
        assert step_info["heater_electric_power_w"] == pytest.approx(
            [2000.0, 0.0, 2000.0]
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    "heater",
    (
        [1, 0],
        [1, 0, 0, 1],
        [1, 0.0, 0],
        [1, 2, 0],
        "100",
    ),
)
def test_public_heater_configuration_rejects_invalid_values(heater):
    with pytest.raises((TypeError, ValueError), match="heater"):
        aiogym.make_env("cascade", heater=heater)


def test_public_heater_configuration_rejects_other_scenarios():
    with pytest.raises(ValueError, match="only by the cascade"):
        aiogym.make_env("three_tank", heater=[1, 0, 0])


@pytest.mark.parametrize(
    "heater",
    (
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ),
)
def test_training_and_benchmarks_respect_each_heater_configuration(heater):
    available = np.asarray(heater, dtype=bool)
    configurations = (
        {"randomize": True},
        {"benchmark": "tracking"},
        {"benchmark": "disturbance-rejection"},
        {"benchmark": "boundary-safety"},
    )
    for configuration in configurations:
        env = aiogym.make_env("cascade", heater=heater, **configuration)
        try:
            _, _ = env.reset(seed=4)
            episode = env.unwrapped.episode
            initial_action = np.asarray(episode.initial_action, dtype=float)
            assert initial_action[4:][~available] == pytest.approx(0.0)

            target_action = env.unwrapped.model.tracking_steady_state_action(
                episode.reference,
                episode.disturbances,
            )
            assert target_action is not None
            target_action = np.asarray(target_action, dtype=float)
            assert target_action[4:][~available] == pytest.approx(0.0)

            if configuration.get("benchmark") == "disturbance-rejection":
                event = next(iter(episode.disturbance_schedule.values()))
                efficiency_names = {
                    name
                    for name in event
                    if name.startswith("heater_")
                }
                assert efficiency_names == {
                    f"heater_H{index + 1}_efficiency_factor"
                    for index, enabled in enumerate(heater)
                    if enabled
                }
        finally:
            env.close()
