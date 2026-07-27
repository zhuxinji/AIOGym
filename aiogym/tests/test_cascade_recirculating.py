"""Foundational acceptance tests for the PDF-derived recirculating scenario."""
from __future__ import annotations

import math

import numpy as np
import pytest

import aiogym


def test_scenario_is_registered_with_pdf_hardware_contract():
    assert "cascade-recirculating" in aiogym.list_scenarios()
    model = aiogym.make_model("cascade-recirculating")

    assert model.action_names == (
        "pump_P101", "valve_V12", "valve_V23", "heater_H1"
    )
    assert model.action_dim() == 4
    assert model.state_names == ("h1", "T1", "h2", "T2", "h3", "T3")
    assert model.p["heater_power"] == pytest.approx(2000.0)
    default_y_sp = model.default_setpoint_vector()
    assert default_y_sp[3] >= default_y_sp[4] >= default_y_sp[5]


def test_parameter_profile_separates_design_value_from_provisional_values():
    profile = aiogym.load_parameter_profile("cascade-recirculating")

    assert profile["status"] == "design-provisional"
    assert profile["parameters"]["heater_power"]["status"] == "design-specified"
    assert profile["parameters"]["area"]["status"] == "design-range-derived"
    assert profile["parameters"]["height_max"]["status"] == "design-range-derived"
    assert profile["parameters"]["pump_power_max"]["status"] == "design-range-informed"
    assert profile["parameters"]["pump_flow_max"]["status"] == "legacy-unverified"
    assert profile["parameters"]["area"]["value"] == pytest.approx([0.04, 0.04, 0.15])
    assert profile["parameters"]["height_max"]["value"] == pytest.approx([0.5, 0.5, 0.5])
    assert np.multiply(profile["parameters"]["area"]["value"], 0.4) == pytest.approx(
        [0.016, 0.016, 0.060]
    )
    assert profile["references"][0]["drawing"] == "L-001"
    assert "非最终施工图" in profile["references"][0]["limitations"]


def test_nominal_closed_loop_conserves_total_liquid_volume():
    model = aiogym.make_model("cascade-recirculating")
    x = model.initial_state()
    dx = model.dynamics(x, model.default_action(), model.disturbance_defaults())

    volume_rate = sum(model.p["area"][i] * dx[2 * i] for i in range(3))
    assert volume_rate == pytest.approx(0.0, abs=1e-15)
    assert max(abs(dx[2 * i]) for i in range(3)) == pytest.approx(0.0, abs=1e-15)


def test_pump_recirculates_tank3_temperature_into_tank1():
    model = aiogym.make_model("cascade-recirculating")
    action = [0.05, 0.0, 0.0, 0.0]
    cold_return = [0.40, 30.0, 0.40, 25.0, 0.40, 20.0]
    hot_return = [0.40, 30.0, 0.40, 25.0, 0.40, 40.0]

    cold_dx = model.dynamics(cold_return, action, model.disturbance_defaults())
    hot_dx = model.dynamics(hot_return, action, model.disturbance_defaults())
    assert hot_dx[1] > cold_dx[1]


def test_foundational_model_metadata_and_environment_step_are_finite():
    model = aiogym.make_model("cascade-recirculating")
    readiness = aiogym.validate_model_readiness(model)
    assert readiness["passed"], readiness

    metadata = model.metadata()
    assert metadata["scenario"] == "cascade-recirculating"
    assert metadata["action_vector"]["length"] == 4
    assert metadata["physical_metadata"]["parameter_status"] == "design-provisional"

    env = aiogym.AIOGymNativeEnv(
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


def test_maximum_declared_actuator_power_matches_foundational_budget():
    model = aiogym.make_model("cascade-recirculating")
    assert model.energy_kw([1.0, 1.0, 1.0, 1.0]) == pytest.approx(2.5)


def test_declared_default_is_a_model_consistent_closed_loop_equilibrium():
    model = aiogym.make_model("cascade-recirculating")
    equilibrium = model.nominal_steady_state()
    dx = model.dynamics(
        equilibrium["state"],
        equilibrium["action"],
        model.disturbance_defaults(),
    )

    assert equilibrium["feasible"]
    assert equilibrium["y_sp"] == pytest.approx(model.default_setpoint_vector())
    assert equilibrium["action"] == pytest.approx(model.default_action())
    assert equilibrium["circulation_flow_m3s"] == pytest.approx(8.0e-5)
    assert equilibrium["ideal_energy_kw"] == pytest.approx(1.2368598383681584)
    assert model.ideal_energy_kw(
        equilibrium["state"], equilibrium["y_sp"], model.disturbance_defaults(),
        [0.0, 0.0, 0.0, 0.0],
    ) == pytest.approx(equilibrium["ideal_energy_kw"])
    assert max(abs(value) for value in dx) < 1e-14


def test_random_internal_points_satisfy_independent_mass_and_energy_balances():
    model = aiogym.make_model("cascade-recirculating")
    rng = np.random.default_rng(20260720)
    max_mass = 0.0
    max_energy = 0.0

    for _ in range(200):
        state = []
        for _tank in range(3):
            state.extend((rng.uniform(0.06, 0.44), rng.uniform(15.0, 85.0)))
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
    report = aiogym.validate_model_readiness("cascade-recirculating")
    checks = {row["name"]: row for row in report["checks"]}

    assert checks["mass_balance"]["passed"]
    assert checks["energy_balance"]["passed"]
    assert report["not_checked"] == ["reference_parameter_fidelity"]


def test_numeric_and_casadi_dynamics_match_across_internal_domain():
    ca = pytest.importorskip("casadi")
    model = aiogym.make_model("cascade-recirculating")
    rng = np.random.default_rng(20260720)

    for _ in range(40):
        state = []
        for _tank in range(3):
            state.extend((rng.uniform(0.06, 0.44), rng.uniform(15.0, 85.0)))
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


def test_numeric_and_casadi_use_the_same_action_clipping():
    ca = pytest.importorskip("casadi")
    model = aiogym.make_model("cascade-recirculating")
    state = [0.35, 35.0, 0.40, 30.0, 0.42, 25.0]
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
    integrator = aiogym.Integrator(model, max_step=max_step)
    integrator.reset(state)
    return np.asarray(
        integrator.step(duration, action, disturbance), dtype=np.float64
    )


def test_rk4_step_refinement_converges_on_smooth_nominal_trajectory():
    model = aiogym.make_model("cascade-recirculating")
    state = [0.40, 50.0, 0.40, 35.0, 0.40, 25.0]
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
    model = aiogym.make_model("cascade-recirculating")
    state = [0.40, 35.0, 0.40, 30.0, 0.35, 20.0]
    state[2 * tank_index] = 0.49
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
    model = aiogym.make_model("cascade-recirculating")
    state = [0.49, 40.0, 0.40, 30.0, 0.35, 25.0]
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
        ([0.04, 40.0, 0.40, 30.0, 0.40, 25.0], "L2_H1_dry_fire"),
        ([0.40, 40.0, 0.46, 30.0, 0.40, 25.0], "L3_high_level"),
        ([0.40, 92.0, 0.40, 30.0, 0.40, 25.0], "L4_H1_over_temperature"),
    ],
)
def test_h1_hardwired_interlocks_remove_actual_heater_power(state, expected_event):
    model = aiogym.make_model("cascade-recirculating")
    action = [0.0, 0.0, 0.0, 1.0]
    info = _process_info(model, state, action)

    assert info["H1_enabled"] is False
    assert info["H1_electric_power_w"] == pytest.approx(0.0)
    assert expected_event in info["hardware_interlocks_active"]
    assert model.action_energy_kw(action, state, model.disturbance_defaults()) == 0.0


def test_tank3_low_level_interlock_stops_p101_without_terminating():
    model = aiogym.make_model("cascade-recirculating")
    state = [0.40, 30.0, 0.40, 28.0, 0.04, 26.0]
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
    model = aiogym.make_model("cascade-recirculating")

    assert model.hard_termination_reasons(
        [0.51, 30.0, 0.40, 28.0, 0.40, 26.0], [], [], {}
    ) == ("tank_1_hard_overflow",)
    assert model.hard_termination_reasons(
        [0.40, 30.0, 0.51, 28.0, 0.40, 26.0], [], [], {}
    ) == ("tank_2_hard_overflow",)
    assert model.hard_termination_reasons(
        [0.40, 30.0, 0.40, 28.0, 0.51, 26.0], [], [], {}
    ) == ("tank_3_hard_overflow",)
    assert model.hard_termination_reasons(
        [0.40, 100.0, 0.40, 28.0, 0.40, 26.0], [], [], {}
    ) == ("temperature_hard_limit",)


def test_numeric_and_casadi_match_with_both_overflow_branches_active():
    ca = pytest.importorskip("casadi")
    model = aiogym.make_model("cascade-recirculating")
    state = [0.49, 60.0, 0.495, 45.0, 0.30, 25.0]
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
    env = aiogym.AIOGymNativeEnv(
        "cascade-recirculating",
        control_dt=0.01,
        episode_steps=3,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
    )
    env.reset(seed=3)
    env.integ.reset([0.49, 40.0, 0.40, 30.0, 0.35, 25.0])
    _, _, terminated, _, passive_info = env.step([0.0, 0.0, 0.0, 0.0])

    assert not terminated
    assert passive_info["safety_events"] == []
    assert "tank_1_passive_overflow" in passive_info["passive_safety_events"]
    assert "L3_high_level" in passive_info["hardware_interlocks_active"]
    assert "termination_reason" not in passive_info

    env.reset(seed=3)
    env.integ.reset([0.55, 40.0, 0.40, 30.0, 0.35, 25.0])
    _, _, terminated, _, hard_info = env.step([0.0, 0.0, 0.0, 0.0])

    assert terminated
    assert hard_info["termination_reason"] == "tank_1_hard_overflow"
    assert "tank_1_hard_overflow" in hard_info["safety_events"]


def test_device_tasks_are_explicit_and_exclude_production_economics():
    expected = {
        "cascade-recirculating/commissioning",
        "cascade-recirculating/disturbance-rejection",
        "cascade-recirculating/safety-recovery",
        "cascade-recirculating/temperature-step",
    }
    assert set(aiogym.list_tasks("cascade-recirculating")) == expected
    for name in expected:
        profile = aiogym.load_task_profile(name)
        assert "economic" not in profile["supported_objectives"]
        assert profile["default_objective"] in profile["supported_objectives"]

    with pytest.raises(ValueError, match="does not support objective 'economic'"):
        aiogym.BenchmarkProtocol.economic("cascade-recirculating")
    with pytest.raises(ValueError, match="does not support objective 'economic'"):
        aiogym.BenchmarkProtocol.economic(
            "cascade-recirculating", task="commissioning"
        )


def test_commissioning_task_starts_at_the_declared_equilibrium():
    model = aiogym.make_model("cascade-recirculating")
    task = aiogym.load_task_profile("cascade-recirculating/commissioning")
    equilibrium = model.nominal_steady_state()
    env = aiogym.AIOGymNativeEnv(
        "cascade-recirculating", task=task, reward_mode="tracking"
    )
    env.reset(seed=4)

    assert env.integ.x == pytest.approx(equilibrium["state"])
    assert env.y_sp == pytest.approx(equilibrium["y_sp"])
    _, _, terminated, _, info = env.step(equilibrium["action"])
    assert not terminated
    assert info["closed_loop_nominal"] is True
    assert info["tracking_error_cost"] < 1e-18


def test_temperature_step_is_visible_before_its_control_step():
    task = aiogym.load_task_profile("cascade-recirculating/temperature-step")
    env = aiogym.AIOGymNativeEnv(
        "cascade-recirculating", task=task, reward_mode="tracking"
    )
    env.reset(seed=5)
    action = env.model.default_action()

    for _ in range(120):
        env.step(action)
    raised = task["setpoints"]["schedule"][0]["values"]
    assert env.y_sp == pytest.approx(raised)
    assert raised[3] >= raised[4] >= raised[5]


def test_disturbance_task_applies_and_exposes_scheduled_p101_loss():
    task = aiogym.load_task_profile("cascade-recirculating/disturbance-rejection")
    env = aiogym.AIOGymNativeEnv(
        "cascade-recirculating", task=task, reward_mode="kpi"
    )
    env.reset(seed=6)
    action = env.model.default_action()

    info = None
    for _ in range(201):
        _, _, _, _, info = env.step(action)
    assert info is not None
    assert info["pump_flow_factor"] == pytest.approx(0.75)
    assert env._env()["pump_flow_factor"] == pytest.approx(0.75)


def test_safety_recovery_task_starts_with_recoverable_protection_layers():
    task = aiogym.load_task_profile("cascade-recirculating/safety-recovery")
    env = aiogym.AIOGymNativeEnv(
        "cascade-recirculating", task=task, reward_mode="kpi"
    )
    env.reset(seed=7)
    _, _, terminated, _, info = env.step(env.model.default_action())

    assert not terminated
    assert "tank_1_passive_overflow" in info["passive_safety_events"]
    assert "P101_tank_3_low_level" in info["hardware_interlocks_active"]
    assert "L3_high_level" in info["hardware_interlocks_active"]
    assert info["safety_events"] == []


@pytest.mark.parametrize(
    ("controller", "controller_config"),
    [
        ("pid", {}),
        ("mpc", {"P": 2, "Ts": 0.5}),
        (
            "oracle",
            {
                "horizon": 1,
                "ipopt_max_iter": 60,
                "warm_start": False,
                "terminal_weight": 0.0,
                "r_move": 0.01,
            },
        ),
    ],
)
def test_four_action_controllers_complete_short_commissioning_run(
    controller, controller_config
):
    from aiogym.evaluation.execution import run_evaluation_case

    protocol = aiogym.BenchmarkProtocol.tracking(
        "cascade-recirculating", task="commissioning", episode_steps=2
    )
    case = run_evaluation_case(
        scenario="cascade-recirculating",
        controller=controller,
        protocol=protocol,
        seeds=[0],
        controller_config=controller_config,
        include_episodes=False,
    )

    controller_model = getattr(case["controller"], "model", None)
    if controller_model is None:
        controller_model = case["controller"].m
    assert controller_model.action_dim() == 4
    assert case["row"]["execution_status"] == "passed"
    assert case["row"]["controller_status"] == "ok"
    assert case["row"]["constraint_violation_count"] == 0
