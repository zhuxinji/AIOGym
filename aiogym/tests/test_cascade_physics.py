"""Physical acceptance and known-gap tests for the heated-tank cascade.

Phase 0 intentionally records unimplemented requirements as strict xfails.
When a later phase implements one of these requirements, remove the matching
xfail marker in the same change; an unexpected pass is treated as a failure so
the temporary marker cannot silently survive.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from aiogym.env import AIOGymEnv
from aiogym.models import RHO_CP, apply_model_params, load_parameter_profile, make_model


@pytest.fixture
def model():
    return make_model("cascade")


@pytest.fixture
def nominal_env(model):
    return model.runtime_env(model.disturbance_defaults())


def _nominal_target_equilibrium(model, env, *, flow=8.0e-4):
    """Return the analytic equilibrium used by the existing benchmark target."""

    target = model.default_setpoint_vector()
    levels = target[:3]
    temperatures = target[3:]
    pump = flow / model.p["pump_flow_max"]
    valves = [
        flow / (model.p["cv_out"] * math.sqrt(level))
        for level in levels
    ]
    heaters = []
    for i, temperature in enumerate(temperatures):
        inlet_temperature = env["t_cold"] if i == 0 else temperatures[i - 1]
        thermal_load = (
            RHO_CP * flow * (temperature - inlet_temperature)
            + model.p["ua_loss"] * (temperature - env["t_amb"])
        )
        heaters.append(thermal_load / model.p["heater_max"])
    state = [
        levels[0], temperatures[0],
        levels[1], temperatures[1],
        levels[2], temperatures[2],
    ]
    return state, [pump, *valves, *heaters]


def test_public_vector_order_and_default_target_are_frozen(model):
    assert model.state_names == ("h0", "T0", "h1", "T1", "h2", "T2")
    assert model.action_names == (
        "feed_pump",
        "outlet_valve_0",
        "outlet_valve_1",
        "outlet_valve_2",
        "heater_0",
        "heater_1",
        "heater_2",
    )
    assert model.output_names == (
        "tank_0_level",
        "tank_1_level",
        "tank_2_level",
        "tank_0_temperature",
        "tank_1_temperature",
        "tank_2_temperature",
    )
    assert model.default_setpoint_vector() == [0.45, 0.45, 0.45, 35.0, 50.0, 65.0]


def test_rho_cp_is_water_volumetric_heat_capacity():
    assert 1.0e6 < RHO_CP < 1.0e7
    assert RHO_CP == pytest.approx(4.186e6)


def test_zero_flow_zero_heat_at_ambient_is_an_equilibrium(model, nominal_env):
    state = [0.3, 20.0, 0.3, 20.0, 0.3, 20.0]
    derivative = model.dynamics(state, [0.0] * 7, nominal_env)
    assert derivative == pytest.approx([0.0] * 6, abs=1e-15)


def test_default_target_has_a_feasible_analytic_equilibrium(model, nominal_env):
    state, action = _nominal_target_equilibrium(model, nominal_env)
    assert all(0.0 <= value <= 1.0 for value in action)
    derivative = model.dynamics(state, action, nominal_env)
    assert max(abs(value) for value in derivative) < 1e-12


def test_mass_and_energy_balances_match_independent_equations(model, nominal_env):
    state = [0.37, 42.0, 0.44, 51.0, 0.29, 63.0]
    action = [0.43, 0.62, 0.38, 0.71, 0.51, 0.27, 0.66]
    derivative = model.dynamics(state, action, nominal_env)
    p = model.p
    levels = [state[0], state[2], state[4]]
    temperatures = [state[1], state[3], state[5]]
    pump_flow = action[0] * p["pump_flow_max"]
    outlet_flows = [
        p["cv_out"] * action[1 + i] * math.sqrt(levels[i])
        for i in range(3)
    ]

    accumulation = p["area"] * (derivative[0] + derivative[2] + derivative[4])
    assert accumulation == pytest.approx(pump_flow - outlet_flows[2], abs=1e-15)

    for i in range(3):
        inlet_flow = pump_flow if i == 0 else outlet_flows[i - 1]
        inlet_temperature = nominal_env["t_cold"] if i == 0 else temperatures[i - 1]
        heater_power = action[4 + i] * p["heater_max"]
        heat_loss = p["ua_loss"] * (temperatures[i] - nominal_env["t_amb"])
        energy_accumulation = RHO_CP * p["area"] * (
            levels[i] * derivative[2 * i + 1]
            + temperatures[i] * derivative[2 * i]
        )
        energy_boundary = (
            RHO_CP
            * (inlet_flow * inlet_temperature - outlet_flows[i] * temperatures[i])
            + heater_power
            - heat_loss
        )
        assert energy_accumulation == pytest.approx(energy_boundary, abs=1e-8)


def test_height_metadata_is_currently_consistent(model):
    assert model.height_max == [model.p["height_max"]] * 3
    for name in ("h0", "h1", "h2"):
        assert model.state_bounds[name] == (0.0, model.p["height_max"])
    assert model.safety_constraints[0]["bounds"] == (0.0, model.p["height_max"])


def test_safety_parameters_and_constraints_are_exposed(model):
    assert model.p["heater_min_level"] == pytest.approx(0.05)
    assert model.p["temperature_trip"] == pytest.approx(92.0)
    assert model.p["temperature_hard_limit"] == pytest.approx(120.0)
    constraints = {row["name"]: row for row in model.constraint_schema()}
    assert constraints["heater_min_level"]["bounds"] == (0.05, None)
    assert constraints["temperature_trip"]["bounds"] == (None, 92.0)
    assert constraints["temperature_hard_limit"]["bounds"] == (None, 120.0)


def test_direct_dynamics_defensively_bounds_actions(model, nominal_env):
    state = [0.3, 40.0, 0.3, 50.0, 0.3, 60.0]
    raw = [-0.2, 1.3, -0.4, 1.2, -0.1, 1.4, 2.0]
    bounded = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    assert model.dynamics(state, raw, nominal_env) == pytest.approx(
        model.dynamics(state, bounded, nominal_env)
    )


def test_direct_dynamics_uses_environment_defaults(model, nominal_env):
    state = model.initial_state()
    action = model.default_action()
    assert model.dynamics(state, action, {}) == pytest.approx(
        model.dynamics(state, action, nominal_env)
    )


@pytest.mark.parametrize(
    "bad_env",
    [
        {"t_cold": math.nan},
        {"t_amb": math.inf},
        {"t_cold": -1.0},
        {"t_amb": 46.0},
        {"extra_outflow": -1.0e-4},
        {"extra_outflow": 0.0021},
        {"pump_flow_factor": 0.39},
        {"heater_efficiency": 1.01},
        {"heat_loss_factor": 3.01},
    ],
)
def test_invalid_environment_values_are_rejected(model, nominal_env, bad_env):
    with pytest.raises(ValueError):
        model.dynamics(
            model.initial_state(),
            model.default_action(),
            {**nominal_env, **bad_env},
        )


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_action_values_are_rejected(model, nominal_env, bad_value):
    action = model.default_action()
    action[0] = bad_value
    with pytest.raises(ValueError, match="cascade action values must be finite"):
        model.dynamics(model.initial_state(), action, nominal_env)
    with pytest.raises(ValueError, match="cascade action values must be finite"):
        model.energy_kw(action)


def test_casadi_dynamics_defensively_bounds_actions(model, nominal_env):
    ca = pytest.importorskip("casadi")
    state = ca.DM([0.3, 40.0, 0.3, 50.0, 0.3, 60.0])
    raw = ca.DM([-0.2, 1.3, -0.4, 1.2, -0.1, 1.4, 2.0])
    bounded = ca.DM([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    disturbance = ca.DM(model.disturbance_vector(nominal_env))
    raw_derivative = np.asarray(
        model.dynamics(state, raw, disturbance, backend="casadi", ca=ca),
        dtype=float,
    ).reshape(-1)
    bounded_derivative = np.asarray(
        model.dynamics(state, bounded, disturbance, backend="casadi", ca=ca),
        dtype=float,
    ).reshape(-1)
    assert raw_derivative == pytest.approx(bounded_derivative)


def test_action_energy_defensively_bounds_actions(model):
    raw = [-0.2, 1.3, -0.4, 1.2, -0.1, 1.4, 2.0]
    bounded = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    assert model.energy_kw(raw) == pytest.approx(model.energy_kw(bounded))


def test_casadi_action_energy_defensively_bounds_actions(model):
    ca = pytest.importorskip("casadi")
    raw = ca.DM([-0.2, 1.3, -0.4, 1.2, -0.1, 1.4, 2.0])
    bounded = ca.DM([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    raw_energy = float(model.energy_kw(raw, backend="casadi", ca=ca))
    bounded_energy = float(model.energy_kw(bounded, backend="casadi", ca=ca))
    assert raw_energy == pytest.approx(bounded_energy)


def test_empty_tank_valve_has_exactly_zero_outflow(model, nominal_env):
    state = [0.0, 20.0, 0.3, 20.0, 0.3, 20.0]
    action = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    derivative = model.dynamics(state, action, nominal_env)
    assert derivative[0] == 0.0


def test_empty_third_tank_disables_extra_outflow(model, nominal_env):
    state = [0.3, 20.0, 0.3, 20.0, 0.0, 20.0]
    action = [0.0] * 7
    derivative = model.dynamics(
        state,
        action,
        {**nominal_env, "extra_outflow": 0.002},
    )
    assert derivative[4] == 0.0


@pytest.mark.parametrize("tank_index", [0, 1, 2])
def test_low_level_interlock_removes_heater_power(model, nominal_env, tank_index):
    state = [0.3, 40.0, 0.3, 50.0, 0.3, 60.0]
    state[2 * tank_index] = 0.001
    heater_off = [0.0] * 7
    heater_on = [0.0] * 7
    heater_on[4 + tank_index] = 1.0
    derivative_off = model.dynamics(state, heater_off, nominal_env)
    derivative_on = model.dynamics(state, heater_on, nominal_env)
    assert derivative_on[2 * tank_index + 1] == pytest.approx(
        derivative_off[2 * tank_index + 1]
    )


@pytest.mark.parametrize("tank_index", [0, 1, 2])
def test_temperature_trip_removes_heater_power(model, nominal_env, tank_index):
    state = [0.3, 40.0, 0.3, 50.0, 0.3, 60.0]
    state[2 * tank_index + 1] = 92.0
    heater_off = [0.0] * 7
    heater_on = [0.0] * 7
    heater_on[4 + tank_index] = 1.0
    derivative_off = model.dynamics(state, heater_off, nominal_env)
    derivative_on = model.dynamics(state, heater_on, nominal_env)
    assert derivative_on[2 * tank_index + 1] == pytest.approx(
        derivative_off[2 * tank_index + 1]
    )


def test_numeric_and_casadi_safety_gates_match(model, nominal_env):
    ca = pytest.importorskip("casadi")
    state = [0.01, 40.0, 0.3, 92.0, 0.3, 60.0]
    action = [1.0] * 7
    numeric = np.asarray(model.dynamics(state, action, nominal_env), dtype=float)
    symbolic = np.asarray(
        model.dynamics(
            ca.DM(state),
            ca.DM(action),
            ca.DM(model.disturbance_vector(nominal_env)),
            backend="casadi",
            ca=ca,
        ),
        dtype=float,
    ).reshape(-1)
    assert numeric == pytest.approx(symbolic)


def test_process_info_exposes_flows_powers_and_interlocks(model, nominal_env):
    state = [0.01, 40.0, 0.3, 92.0, 0.3, 60.0]
    action = [0.5] * 7
    context = {**nominal_env, "extra_outflow": 2.0e-4}
    outputs = model.outputs(state)
    info = model.process_info(
        state,
        outputs["levels"],
        outputs["temps"],
        context,
        action,
    )
    assert info["feed_flow_m3s"] == pytest.approx(8.0e-4)
    assert info["interstage_flow_01_m3s"] >= 0.0
    assert info["interstage_flow_12_m3s"] >= 0.0
    assert info["product_flow_m3s"] > info["extra_outflow_m3s"]
    assert info["extra_outflow_m3s"] == pytest.approx(2.0e-4)
    assert info["heater_interlocked"] == [True, True, False]
    assert info["low_level_interlock_active"] == [True, False, False]
    assert info["temperature_trip_active"] == [False, True, False]
    assert info["heater_electric_power_w"][:2] == [0.0, 0.0]
    assert info["heater_to_liquid_power_w"][:2] == [0.0, 0.0]
    assert info["heater_electric_power_w"][2] > 0.0


def test_actual_energy_rate_respects_heater_interlocks(model, nominal_env):
    action = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    normal_state = [0.3, 40.0, 0.3, 50.0, 0.3, 60.0]
    low_level_state = [0.01, 40.0, 0.3, 50.0, 0.3, 60.0]
    tripped_state = [0.3, 92.0, 0.3, 50.0, 0.3, 60.0]
    assert model.energy_kw(action) == pytest.approx(90.0)
    assert model.action_energy_kw(action, normal_state, nominal_env) == pytest.approx(90.0)
    assert model.action_energy_kw(action, low_level_state, nominal_env) == pytest.approx(0.0)
    assert model.action_energy_kw(action, tripped_state, nominal_env) == pytest.approx(0.0)


def test_overflow_is_an_unconditional_structured_termination():
    env = AIOGymEnv(
        "cascade",
        control_dt=0.5,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
        terminate_on_runaway=False,
    )
    env.reset(seed=0)
    env.integ.reset([0.799, 20.0, 0.3, 20.0, 0.3, 20.0])
    _, _, terminated, _, info = env.step(
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    assert terminated is True
    assert info["termination_reason"] == "overflow"


def test_negative_level_is_an_unconditional_structured_termination():
    env = AIOGymEnv(
        "cascade",
        control_dt=0.5,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
        terminate_on_runaway=False,
    )
    env.reset(seed=0)
    env.extra_outflow = 0.002
    env.integ.reset([0.3, 20.0, 0.3, 20.0, 0.001, 20.0])
    _, _, terminated, _, info = env.step(np.zeros(7, dtype=np.float32))
    assert terminated is True
    assert info["termination_reason"] == "negative_level"


def test_temperature_hard_limit_is_an_unconditional_structured_termination():
    env = AIOGymEnv(
        "cascade",
        control_dt=0.5,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
        terminate_on_runaway=False,
    )
    env.reset(seed=0)
    env.integ.reset([0.3, 121.0, 0.3, 20.0, 0.3, 20.0])
    _, _, terminated, _, info = env.step(np.zeros(7, dtype=np.float32))
    assert terminated is True
    assert info["termination_reason"] == "temperature_hard_limit"
    assert "temperature_hard_limit" in info["safety_events"]


def test_numeric_and_casadi_controlled_outputs_match_for_same_state(model):
    ca = pytest.importorskip("casadi")
    state = [-0.01, 35.0, 0.2, 50.0, 0.3, 65.0]
    numeric = np.asarray(model.controlled_output(state), dtype=float)
    symbolic = np.asarray(
        model.controlled_output(ca.DM(state), backend="casadi", ca=ca),
        dtype=float,
    ).reshape(-1)
    assert numeric == pytest.approx(symbolic)
    assert numeric[0] == pytest.approx(-0.01)
    assert model.display_outputs(state)["levels"][0] == pytest.approx(0.0)


def test_physical_output_bounds_are_separate_from_setpoint_bounds(model):
    output_rows = {row["name"]: row for row in model.controlled_output_schema()}
    setpoint_rows = {row["name"]: row for row in model.setpoint_schema()}
    assert output_rows["tank_0_temperature"]["bounds"] == (0.0, 120.0)
    assert output_rows["tank_1_temperature"]["bounds"] == (0.0, 120.0)
    assert output_rows["tank_2_temperature"]["bounds"] == (0.0, 120.0)
    assert setpoint_rows["tank_0_temperature"]["bounds"] == (25.0, 80.0)
    assert setpoint_rows["tank_1_temperature"]["bounds"] == (30.0, 82.0)
    assert setpoint_rows["tank_2_temperature"]["bounds"] == (35.0, 85.0)


def test_parameter_overrides_update_all_physical_bound_metadata():
    model = apply_model_params(
        make_model("cascade"),
        {
            "height_max": 0.9,
            "heater_min_level": 0.08,
            "temperature_trip": 90.0,
            "temperature_hard_limit": 110.0,
        },
    )
    assert model.height_max == [0.9, 0.9, 0.9]
    assert model.state_bounds["h0"] == (0.0, 0.9)
    assert model.state_bounds["T0"] == (0.0, 110.0)
    output_rows = {row["name"]: row for row in model.controlled_output_schema()}
    setpoint_rows = {row["name"]: row for row in model.setpoint_schema()}
    constraints = {row["name"]: row for row in model.constraint_schema()}
    assert output_rows["tank_2_level"]["bounds"] == (0.0, 0.9)
    assert output_rows["tank_2_temperature"]["bounds"] == (0.0, 110.0)
    assert setpoint_rows["tank_2_level"]["bounds"] == (0.0, 0.9)
    assert setpoint_rows["tank_2_temperature"]["bounds"] == (35.0, 85.0)
    assert constraints["level_bounds"]["bounds"] == (0.0, 0.9)
    assert constraints["heater_min_level"]["bounds"] == (0.08, None)
    assert constraints["temperature_trip"]["bounds"] == (None, 90.0)
    assert constraints["temperature_hard_limit"]["bounds"] == (None, 110.0)


def test_heater_efficiency_is_a_bounded_fraction(model, nominal_env):
    disturbance = {
        row["name"]: row for row in model.disturbance_schema()
    }["heater_efficiency"]
    assert disturbance["bounds"] == (0.4, 1.0)
    assert model.runtime_env({**nominal_env, "heater_efficiency": 1.0})[
        "heater_efficiency"
    ] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="heater_efficiency"):
        model.runtime_env({**nominal_env, "heater_efficiency": 1.01})
    rng = np.random.default_rng(3)
    samples = [
        model.sample_disturbance("heater_efficiency_shift", 1.0, rng)
        for _ in range(100)
    ]
    assert all(0.4 <= value <= 1.0 for value in samples)


def test_every_cascade_parameter_has_honest_provenance_metadata(model):
    profile = load_parameter_profile("cascade")
    schema = model.parameter_schema()
    assert set(profile["parameters"]) == set(model.p)
    assert set(schema) == set(model.p)
    for row in schema.values():
        assert row["status"] in {"legacy-unverified", "assumed-benchmark"}
        assert row["source"]
        assert row["source_locator"]


def test_steady_state_requirements_report_feasibility(model, nominal_env):
    requirements = model.steady_state_requirements(
        model.default_setpoint_vector(),
        nominal_env,
        product_flow_sp=4.0e-4,
    )
    assert requirements["feasible"] is True
    assert requirements["ideal_energy_kw"] > 0.0
    assert 0.0 <= float(requirements["pump_command"]) <= 1.0
    assert len(requirements["valve_commands"]) == 3
    assert len(requirements["heater_commands"]) == 3
    assert all(0.0 <= value <= 1.0 for value in requirements["action"])
    assert requirements["ideal_energy_kw"] == pytest.approx(
        (
            requirements["pump_power_w"]
            + sum(requirements["heater_electric_power_w"])
        )
        / 1000.0
    )
    target = model.default_setpoint_vector()
    state = [target[0], target[3], target[1], target[4], target[2], target[5]]
    assert model.dynamics(state, requirements["action"], nominal_env) == pytest.approx(
        [0.0] * 6, abs=1e-12
    )

    impossible = model.steady_state_requirements(
        target,
        nominal_env,
        product_flow_sp=2.0e-3,
    )
    assert impossible["feasible"] is False
    assert "feed_pump" in impossible["infeasible_reasons"]


def test_zero_throughput_target_needs_no_separate_operation_mode(model, nominal_env):
    assert model.operation == {
        "product_flow_sp": 0.0,
        "min_product_flow": 0.0,
    }
    state = [0.45, 35.0, 0.45, 50.0, 0.45, 65.0]
    flowing_action = [0.25, 0.25, 0.25, 0.25, 0.4, 0.3, 0.3]

    assert model.production(state, flowing_action, nominal_env) == pytest.approx(0.0)
    assert model.product_flow_shortfall(0.0) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="unknown cascade operation fields: mode"):
        model.configure_operation({
            "mode": "continuous",
            "product_flow_sp": 4.0e-4,
        })


def test_configured_throughput_sets_cascade_default_and_mpc_actions(model):
    model.configure_operation({
        "product_flow_sp": 4.0e-4,
        "min_product_flow": 4.0e-4,
    })
    expected = model.steady_state_requirements(
        model.default_setpoint_vector()
    )["action"]

    assert model.default_action() == pytest.approx(expected)
    assert model.mpc_init() == pytest.approx(expected)


def test_lower_heater_efficiency_increases_ideal_electric_power(model, nominal_env):
    efficient = model.steady_state_requirements(
        model.default_setpoint_vector(),
        {**nominal_env, "heater_efficiency": 1.0},
        product_flow_sp=4.0e-4,
    )
    inefficient = model.steady_state_requirements(
        model.default_setpoint_vector(),
        {**nominal_env, "heater_efficiency": 0.5},
        product_flow_sp=4.0e-4,
    )
    assert inefficient["ideal_energy_kw"] > efficient["ideal_energy_kw"]
    assert inefficient["pump_power_w"] == pytest.approx(efficient["pump_power_w"])
    assert sum(inefficient["heater_electric_power_w"]) == pytest.approx(
        2.0 * sum(efficient["heater_electric_power_w"])
    )


def test_ideal_energy_uses_task_throughput_not_current_pump_action(model, nominal_env):
    model.configure_operation(
        {"product_flow_sp": 4.0e-4, "min_product_flow": 4.0e-4}
    )
    target = model.default_setpoint_vector()
    requirements = model.steady_state_requirements(target, nominal_env)
    state = model.initial_state()
    assert model.ideal_energy_kw(state, target, nominal_env, [0.0] * 7) == pytest.approx(
        requirements["ideal_energy_kw"]
    )
    assert model.ideal_energy_kw(state, target, nominal_env, [1.0] * 7) == pytest.approx(
        requirements["ideal_energy_kw"]
    )


def test_numeric_and_casadi_continuous_economics_match(model, nominal_env):
    ca = pytest.importorskip("casadi")
    model.configure_operation(
        {"product_flow_sp": 4.0e-4, "min_product_flow": 4.0e-4}
    )
    target = model.default_setpoint_vector()
    state = [target[0], target[3], target[1], target[4], target[2], target[5]]
    action = model.steady_state_requirements(target, nominal_env)["action"]
    disturbance = ca.DM(model.disturbance_vector(nominal_env))
    numeric_value = model.economic_value(state, action, nominal_env)
    symbolic_value = float(
        model.economic_value(
            ca.DM(state), ca.DM(action), model.dynamics_disturbance_map(disturbance),
            backend="casadi", ca=ca,
        )
    )
    assert numeric_value == pytest.approx(4.0e-4)
    assert symbolic_value == pytest.approx(numeric_value)
    assert model.product_flow_shortfall(numeric_value) == pytest.approx(0.0)
    symbolic_shortfall = float(
        model.product_flow_shortfall(0.0, backend="casadi", ca=ca)
    )
    assert symbolic_shortfall == pytest.approx(1.0)
