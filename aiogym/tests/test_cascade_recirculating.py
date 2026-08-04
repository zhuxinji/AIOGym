"""Foundational acceptance tests for the PDF-derived recirculating scenario."""
from __future__ import annotations

import math

import numpy as np
import pytest

import aiogym
from aiogym.models.integration import Integrator
from aiogym.models.parameter_profiles import load_parameter_profile
from aiogym.models.registry import make_model
from aiogym.models.validation import validate_model_readiness
from aiogym.tests._env import make_test_env


def test_scenario_is_registered_with_pdf_hardware_contract():
    assert "cascade-recirculating" in aiogym.list_scenarios()
    model = make_model("cascade-recirculating")

    assert model.action_names == (
        "pump_P101", "valve_V12", "valve_V23", "heater_H1"
    )
    assert model.action_dim() == 4
    assert model.state_names == ("h1", "T1", "h2", "T2", "h3", "T3")
    assert model.p["heater_power"] == pytest.approx(2000.0)
    default_y_sp = model.default_setpoint_vector()
    assert default_y_sp[3] >= default_y_sp[4] >= default_y_sp[5]


def test_parameter_profile_separates_design_value_from_provisional_values():
    profile = load_parameter_profile("cascade-recirculating")

    assert profile["status"] == "design-provisional"
    assert profile["parameters"]["heater_power"]["status"] == "design-specified"
    assert profile["parameters"]["area"]["status"] == "design-specified-provisional"
    assert profile["parameters"]["height_max"]["status"] == "design-specified-provisional"
    assert profile["parameters"]["pump_power_max"]["status"] == "design-specified-provisional"
    assert profile["parameters"]["pump_flow_max"]["status"] == "design-specified-provisional"
    assert profile["parameters"]["area"]["value"] == pytest.approx([0.075] * 3)
    assert profile["parameters"]["height_max"]["value"] == pytest.approx([0.4] * 3)
    assert np.multiply(profile["parameters"]["area"]["value"], 0.24) == pytest.approx(
        [0.018, 0.018, 0.018]
    )
    assert profile["parameters"]["pump_flow_max"]["value"] * 60000 == pytest.approx(25.0)
    assert profile["references"][0]["revision"] == "V2.0-baseline"
    assert "非最终施工图" in profile["references"][0]["limitations"]


def test_nominal_closed_loop_conserves_total_liquid_volume():
    model = make_model("cascade-recirculating")
    x = model.initial_state()
    dx = model.dynamics(x, model.default_action(), model.disturbance_defaults())

    volume_rate = sum(model.p["area"][i] * dx[2 * i] for i in range(3))
    assert volume_rate == pytest.approx(0.0, abs=1e-15)
    assert max(abs(dx[2 * i]) for i in range(3)) == pytest.approx(0.0, abs=1e-15)


def test_pump_recirculates_tank3_temperature_into_tank1():
    model = make_model("cascade-recirculating")
    action = [0.5, 0.0, 0.0, 0.0]
    cold_return = [0.24, 30.0, 0.24, 25.0, 0.24, 20.0]
    hot_return = [0.24, 30.0, 0.24, 25.0, 0.24, 40.0]

    cold_dx = model.dynamics(cold_return, action, model.disturbance_defaults())
    hot_dx = model.dynamics(hot_return, action, model.disturbance_defaults())
    assert hot_dx[1] > cold_dx[1]


def test_foundational_model_metadata_and_environment_step_are_finite():
    model = make_model("cascade-recirculating")
    readiness = validate_model_readiness(model)
    assert readiness["passed"], readiness

    metadata = model.metadata()
    assert metadata["scenario"] == "cascade-recirculating"
    assert metadata["action_vector"]["length"] == 4
    assert metadata["physical_metadata"]["parameter_status"] == "design-provisional"

    env = make_test_env(
        "cascade-recirculating",
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
    )
    observation, info = env.reset(seed=7)
    assert all(math.isfinite(float(value)) for value in observation)
    observation, reward, terminated, truncated, info = env.step(model.default_action())
    assert all(math.isfinite(float(value)) for value in observation)
    assert math.isfinite(float(reward))
    assert not terminated
    assert not truncated
    assert info["closed_loop_nominal"] is True
    assert info["H1_electric_power_w"] <= 2000.0


def test_v2_physical_io_contract_exposes_transmitters_switches_and_flow_meters():
    model = make_model("cascade-recirculating")
    physical_io = model.metadata()["physical_io"]

    analog_names = {row["name"] for row in physical_io["analog_measurements"]}
    digital_names = {row["name"] for row in physical_io["digital_inputs"]}
    assert analog_names == {
        "LT101", "LT201", "LT301", "TT101", "TT201", "TT301", "FT12", "FT23"
    }
    assert digital_names == {
        "LSL101", "LSL201", "LSL301", "LSH101", "LSH201", "LSH301"
    }
    assert physical_io["network"].startswith("Modbus RTU")


def test_maximum_declared_actuator_power_matches_foundational_budget():
    model = make_model("cascade-recirculating")
    assert model.energy_kw([1.0, 1.0, 1.0, 1.0]) == pytest.approx(2.37)


def test_declared_default_is_a_model_consistent_closed_loop_equilibrium():
    model = make_model("cascade-recirculating")
    equilibrium = model.nominal_steady_state()
    dx = model.dynamics(
        equilibrium["state"],
        equilibrium["action"],
        model.disturbance_defaults(),
    )

    assert equilibrium["feasible"]
    assert equilibrium["y_sp"] == pytest.approx(model.default_setpoint_vector())
    assert equilibrium["action"] == pytest.approx(model.default_action())
    assert equilibrium["circulation_flow_m3s"] == pytest.approx(5.0 / 60000.0)
    assert equilibrium["ideal_energy_kw"] == pytest.approx(1.2520223701170998)
    assert model.ideal_energy_kw(
        equilibrium["state"], equilibrium["y_sp"], model.disturbance_defaults(),
        [0.0, 0.0, 0.0, 0.0],
    ) == pytest.approx(equilibrium["ideal_energy_kw"])
    assert max(abs(value) for value in dx) < 1e-14


def test_random_internal_points_satisfy_independent_mass_and_energy_balances():
    model = make_model("cascade-recirculating")
    rng = np.random.default_rng(20260720)
    max_mass = 0.0
    max_energy = 0.0

    for _ in range(200):
        state = []
        for _tank in range(3):
            state.extend((rng.uniform(0.09, 0.33), rng.uniform(15.0, 75.0)))
        action = rng.uniform(0.0, 1.0, 4).tolist()
        disturbance = {
            "t_amb": rng.uniform(5.0, 35.0),
            "pump_flow_factor": rng.uniform(0.5, 1.3),
            "heater_efficiency": rng.uniform(0.5, 1.0),
            "heat_loss_factor": rng.uniform(0.4, 2.5),
        }
        residuals = model.balance_residuals(state, action, disturbance)
        max_mass = max(
            max_mass,
            abs(residuals["total_mass_balance_m3s"]),
            *(abs(value) for value in residuals["tank_mass_balance_m3s"]),
        )
        max_energy = max(
            max_energy,
            abs(residuals["total_energy_balance_w"]),
            *(abs(value) for value in residuals["tank_energy_balance_w"]),
        )

    assert max_mass < 1e-18
    assert max_energy < 1e-8


def test_model_readiness_records_mass_and_energy_as_checked():
    report = validate_model_readiness("cascade-recirculating")
    checks = {row["name"]: row for row in report["checks"]}

    assert checks["mass_balance"]["passed"]
    assert checks["energy_balance"]["passed"]
    assert report["not_checked"] == ["reference_parameter_fidelity"]


@pytest.mark.oracle
def test_numeric_and_casadi_dynamics_match_across_internal_domain():
    ca = pytest.importorskip("casadi")
    model = make_model("cascade-recirculating")
    rng = np.random.default_rng(20260720)

    for _ in range(40):
        state = []
        for _tank in range(3):
            state.extend((rng.uniform(0.09, 0.33), rng.uniform(15.0, 75.0)))
        action = rng.uniform(-0.25, 1.25, 4).tolist()
        disturbance = {
            "t_amb": rng.uniform(5.0, 35.0),
            "pump_flow_factor": rng.uniform(0.5, 1.3),
            "heater_efficiency": rng.uniform(0.5, 1.0),
            "heat_loss_factor": rng.uniform(0.4, 2.5),
        }
        numeric = np.asarray(
            model.dynamics(state, action, disturbance), dtype=np.float64
        )
        symbolic = np.asarray(
            model.dynamics(
                ca.DM(state),
                ca.DM(action),
                ca.DM(model.disturbance_vector(disturbance)),
                backend="casadi",
                ca=ca,
            ),
            dtype=np.float64,
        ).reshape(-1)
        assert symbolic == pytest.approx(numeric, rel=1e-12, abs=1e-12)


@pytest.mark.oracle
def test_numeric_and_casadi_use_the_same_action_clipping():
    ca = pytest.importorskip("casadi")
    model = make_model("cascade-recirculating")
    state = [0.25, 35.0, 0.30, 30.0, 0.32, 25.0]
    disturbance = model.disturbance_defaults()
    raw = [-2.0, 1.5, 0.2, 4.0]
    clipped = [0.0, 1.0, 0.2, 1.0]

    assert model.dynamics(state, raw, disturbance) == pytest.approx(
        model.dynamics(state, clipped, disturbance)
    )
    symbolic = np.asarray(
        model.dynamics(
            ca.DM(state),
            ca.DM(raw),
            ca.DM(model.disturbance_vector(disturbance)),
            backend="casadi",
            ca=ca,
        ),
        dtype=np.float64,
    ).reshape(-1)
    assert symbolic == pytest.approx(model.dynamics(state, clipped, disturbance))


def _integrate(model, state, action, disturbance, max_step, duration=120.0):
    integrator = Integrator(model, max_step=max_step)
    integrator.reset(state)
    return np.asarray(
        integrator.step(duration, action, disturbance), dtype=np.float64
    )


def test_rk4_step_refinement_converges_on_smooth_nominal_trajectory():
    model = make_model("cascade-recirculating")
    state = [0.24, 50.0, 0.24, 35.0, 0.24, 25.0]
    action = model.default_action()
    disturbance = model.disturbance_defaults()

    # Deliberately use steps much larger than the declared 0.02 s maximum so
    # the fourth-order convergence trend is measurable above roundoff.
    coarse = _integrate(model, state, action, disturbance, 2.0)
    medium = _integrate(model, state, action, disturbance, 1.0)
    fine = _integrate(model, state, action, disturbance, 0.5)
    coarse_error = float(np.max(np.abs(coarse - medium)))
    fine_error = float(np.max(np.abs(medium - fine)))

    assert fine_error < coarse_error / 8.0
    assert fine_error < 1e-9


def _process_info(model, state, action):
    outputs = model.outputs(state)
    return model.process_info(
        state,
        outputs["levels"],
        outputs["temps"],
        model.disturbance_defaults(),
        action,
    )


@pytest.mark.parametrize("tank_index", [0, 1])
def test_passive_overflow_returns_mass_and_enthalpy_to_tank3(tank_index):
    model = make_model("cascade-recirculating")
    state = [0.24, 35.0, 0.24, 30.0, 0.25, 20.0]
    state[2 * tank_index] = 0.37
    state[2 * tank_index + 1] = 60.0 if tank_index == 0 else 50.0
    action = [0.0, 0.0, 0.0, 0.0]

    dx = model.dynamics(state, action, model.disturbance_defaults())
    residuals = model.balance_residuals(state, action, model.disturbance_defaults())
    info = _process_info(model, state, action)

    overflow_key = f"tank_{tank_index + 1}_overflow_return_m3s"
    event = f"tank_{tank_index + 1}_passive_overflow"
    assert info[overflow_key] > 0.0
    assert event in info["passive_safety_events"]
    assert dx[2 * tank_index] < 0.0
    assert dx[4] > 0.0
    assert dx[5] > 0.0
    assert residuals["total_mass_balance_m3s"] == pytest.approx(0.0, abs=1e-15)
    assert max(abs(value) for value in residuals["tank_energy_balance_w"]) < 1e-8
    assert model.hard_termination_reasons(state, [], [], {}) == ()


def test_passive_overflow_is_a_protective_flow_not_a_hard_termination():
    model = make_model("cascade-recirculating")
    state = [0.37, 40.0, 0.24, 30.0, 0.25, 25.0]
    constraints = model.process_constraint_info(state, [], [], {})
    info = _process_info(model, state, [0.0, 0.0, 0.0, 0.0])

    assert constraints["passive_overflow_head"] == pytest.approx(0.01)
    assert constraints["level_overflow"] == pytest.approx(0.0)
    assert info["closed_loop_nominal"] is False
    assert info["total_overflow_return_m3s"] > 0.0
    assert model.hard_termination_reasons(state, [], [], {}) == ()


@pytest.mark.parametrize(
    ("state", "expected_event"),
    [
        ([0.07, 40.0, 0.24, 30.0, 0.24, 25.0], "L2_H1_dry_fire"),
        ([0.24, 80.0, 0.24, 30.0, 0.24, 25.0], "L4_H1_over_temperature"),
    ],
)
def test_h1_hardwired_interlocks_remove_actual_heater_power(state, expected_event):
    model = make_model("cascade-recirculating")
    action = [0.0, 0.0, 0.0, 1.0]
    info = _process_info(model, state, action)

    assert info["H1_enabled"] is False
    assert info["H1_electric_power_w"] == pytest.approx(0.0)
    assert expected_event in info["hardware_interlocks_active"]
    assert model.action_energy_kw(action, state, model.disturbance_defaults()) == 0.0


def test_tank1_or_tank2_high_level_stops_p101_without_disabling_h1():
    model = make_model("cascade-recirculating")
    state = [0.35, 40.0, 0.24, 30.0, 0.24, 25.0]
    action = [1.0, 0.0, 0.0, 1.0]
    info = _process_info(model, state, action)

    assert info["P101_enabled"] is False
    assert info["H1_enabled"] is True
    assert info["circulation_flow_m3s"] == pytest.approx(0.0)
    assert "L3_P101_high_level" in info["hardware_interlocks_active"]


def test_tank3_low_level_interlock_stops_p101_without_terminating():
    model = make_model("cascade-recirculating")
    state = [0.24, 30.0, 0.24, 28.0, 0.07, 26.0]
    action = [1.0, 0.0, 0.0, 0.0]
    dx = model.dynamics(state, action, model.disturbance_defaults())
    info = _process_info(model, state, action)

    assert info["P101_enabled"] is False
    assert info["circulation_flow_m3s"] == pytest.approx(0.0)
    assert "P101_tank_3_low_level" in info["hardware_interlocks_active"]
    assert dx[0] == pytest.approx(0.0)
    assert dx[4] == pytest.approx(0.0)
    assert model.action_energy_kw(action, state, model.disturbance_defaults()) == 0.0
    assert model.hard_termination_reasons(state, [], [], {}) == ()


def test_hard_boundaries_remain_distinct_from_passive_protection():
    model = make_model("cascade-recirculating")

    assert model.hard_termination_reasons(
        [0.41, 30.0, 0.24, 28.0, 0.24, 26.0], [], [], {}
    ) == ("tank_1_hard_overflow",)
    assert model.hard_termination_reasons(
        [0.24, 30.0, 0.41, 28.0, 0.24, 26.0], [], [], {}
    ) == ("tank_2_hard_overflow",)
    assert model.hard_termination_reasons(
        [0.24, 30.0, 0.24, 28.0, 0.41, 26.0], [], [], {}
    ) == ("tank_3_hard_overflow",)
    assert model.hard_termination_reasons(
        [0.24, 100.0, 0.24, 28.0, 0.24, 26.0], [], [], {}
    ) == ("temperature_hard_limit",)


@pytest.mark.oracle
def test_numeric_and_casadi_match_with_both_overflow_branches_active():
    ca = pytest.importorskip("casadi")
    model = make_model("cascade-recirculating")
    state = [0.37, 60.0, 0.375, 45.0, 0.25, 25.0]
    action = [0.08, 0.20, 0.30, 1.0]
    disturbance = {
        "t_amb": 18.0,
        "pump_flow_factor": 0.9,
        "heater_efficiency": 0.8,
        "heat_loss_factor": 1.4,
    }

    numeric = np.asarray(model.dynamics(state, action, disturbance))
    symbolic = np.asarray(
        model.dynamics(
            ca.DM(state),
            ca.DM(action),
            ca.DM(model.disturbance_vector(disturbance)),
            backend="casadi",
            ca=ca,
        ),
        dtype=np.float64,
    ).reshape(-1)
    assert symbolic == pytest.approx(numeric, rel=1e-12, abs=1e-12)


def test_environment_reports_passive_protection_separately_from_hard_stop():
    env = make_test_env(
        "cascade-recirculating",
        control_dt=0.01,
        episode_steps=3,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
    )
    env.reset(seed=3)
    env.integ.reset([0.37, 40.0, 0.24, 30.0, 0.25, 25.0])
    _, _, terminated, _, passive_info = env.step([0.0, 0.0, 0.0, 0.0])

    assert not terminated
    assert passive_info["safety_events"] == []
    assert "tank_1_passive_overflow" in passive_info["passive_safety_events"]
    assert "L3_P101_high_level" in passive_info["hardware_interlocks_active"]
    assert "termination_reason" not in passive_info

    env.reset(seed=3)
    env.integ.reset([0.43, 40.0, 0.24, 30.0, 0.25, 25.0])
    _, _, terminated, _, hard_info = env.step([0.0, 0.0, 0.0, 0.0])

    assert terminated
    assert hard_info["termination_reason"] == "tank_1_hard_overflow"
    assert "tank_1_hard_overflow" in hard_info["safety_events"]


def test_device_cases_are_explicit_and_goal_independent():
    expected = {
        "cascade-recirculating/commissioning",
        "cascade-recirculating/disturbance-rejection",
        "cascade-recirculating/hydraulic-commissioning",
        "cascade-recirculating/safety-recovery",
        "cascade-recirculating/temperature-step",
    }
    assert set(aiogym.list_cases("cascade-recirculating")) == expected
    for name in expected:
        profile = aiogym.load_case(name)
        assert not {
            "default_objective",
            "supported_objectives",
            "objectives",
        }.intersection(profile)

def test_commissioning_case_starts_from_a_feasible_nontrivial_offset():
    model = make_model("cascade-recirculating")
    case = aiogym.load_case("cascade-recirculating/commissioning")
    equilibrium = model.nominal_steady_state()
    env = make_test_env(
        "cascade-recirculating", case=case, reward_spec="regulation-v1"
    )
    env.reset(seed=4)

    assert env.integ.x == pytest.approx(case["initialization"]["state"])
    assert env.integ.x != pytest.approx(equilibrium["state"])
    assert env.y_sp == pytest.approx(equilibrium["y_sp"])
    _, _, terminated, _, info = env.step(equilibrium["action"])
    assert not terminated
    assert info["closed_loop_nominal"] is True
    assert info["tracking_error_cost"] > 1e-8


def test_phase1_hydraulic_commissioning_physically_disables_h1():
    case = aiogym.load_case("cascade-recirculating/hydraulic-commissioning")
    env = make_test_env(
        "cascade-recirculating",
        case=case,
        reward_spec="regulation-v1",
    )
    _, info = env.reset(seed=8)
    assert env.model.p["heater_power"] == pytest.approx(0.0)
    assert env.model.default_action()[3] == pytest.approx(0.0)

    _, _, terminated, _, info = env.step([1.0, 0.2, 0.2, 1.0])
    assert not terminated
    assert info["H1_electric_power_w"] == pytest.approx(0.0)


def test_temperature_step_is_visible_before_its_control_step():
    case = aiogym.load_case("cascade-recirculating/temperature-step")
    env = make_test_env(
        "cascade-recirculating", case=case, reward_spec="regulation-v1"
    )
    env.reset(seed=5)
    action = env.model.default_action()

    for _ in range(120):
        env.step(action)
    raised = case["setpoints"]["schedule"][0]["values"]
    assert env.y_sp == pytest.approx(raised)
    assert raised[3] >= raised[4] >= raised[5]


def test_disturbance_case_applies_and_exposes_scheduled_p101_loss():
    case = aiogym.load_case("cascade-recirculating/disturbance-rejection")
    env = make_test_env(
        "cascade-recirculating", case=case, reward_spec="regulation-v1"
    )
    env.reset(seed=6)
    action = env.model.default_action()

    info = None
    for _ in range(201):
        _, _, _, _, info = env.step(action)
    assert info is not None
    assert info["pump_flow_factor"] == pytest.approx(0.75)
    assert env._env()["pump_flow_factor"] == pytest.approx(0.75)


def test_safety_recovery_case_starts_with_recoverable_protection_layers():
    case = aiogym.load_case("cascade-recirculating/safety-recovery")
    env = make_test_env(
        "cascade-recirculating", case=case, reward_spec="regulation-v1"
    )
    env.reset(seed=7)
    _, _, terminated, _, info = env.step(env.model.default_action())

    assert not terminated
    assert "tank_1_passive_overflow" in info["passive_safety_events"]
    assert "P101_tank_3_low_level" in info["hardware_interlocks_active"]
    assert "L3_P101_high_level" in info["hardware_interlocks_active"]
    assert info["safety_events"] == []


@pytest.mark.parametrize(
    ("controller", "controller_config"),
    [
        ("pid", {}),
        ("mpc", {"P": 2, "Ts": 0.5}),
        pytest.param(
            "oracle",
            {
                "horizon": 1,
                "ipopt_max_iter": 60,
                "warm_start": False,
                "terminal_weight": 0.0,
                "r_move": 0.01,
            },
            marks=pytest.mark.oracle,
        ),
    ],
)
def test_four_action_controllers_complete_short_commissioning_run(
    controller, controller_config
):
    agent = aiogym.make_controller(
        controller,
        scenario="cascade-recirculating",
        config=controller_config,
    )
    env = make_test_env(
        "cascade-recirculating",
        case="commissioning",
        reward_spec="regulation-v1",
        episode_steps=2,
    )
    result = aiogym.evaluate_controller(agent, env, seed=0)
    controller_model = getattr(agent, "model", None)
    if controller_model is None:
        controller_model = agent.m
    assert controller_model.action_dim() == 4
    assert result["execution_status"] == "passed"
    assert result["controller_status"] == "ok"
    assert result["constraint_violation_count"] == 0
