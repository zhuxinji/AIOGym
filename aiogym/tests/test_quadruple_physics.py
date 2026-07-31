"""Physics and canonical Case checks for the Johansson quadruple tank."""
from __future__ import annotations

import numpy as np
import pytest

import aiogym
from aiogym.tests._env import make_test_env
from aiogym.controllers.contracts import build_context
from aiogym.models.integration import Integrator
from aiogym.models.registry import apply_model_params, make_model


def test_reference_parameters_and_phase_configuration():
    model = make_model("quadruple")
    assert model.p == {
        "tank_area": [28.0, 32.0, 28.0, 32.0],
        "outlet_area": [0.071, 0.057, 0.071, 0.057],
        "pump_gain": [3.33, 3.35],
        "gamma": [0.70, 0.60],
        "gravity": 981.0,
        "max_voltage": 10.0,
        "max_level": 20.0,
        "nominal_voltage": [3.0, 3.0],
    }
    assert model.phase_configuration == "minimum-phase"
    shifted = apply_model_params(model, {"gamma": [0.43, 0.34]})
    assert shifted.phase_configuration == "nonminimum-phase"


def test_nominal_state_is_exact_nonlinear_equilibrium():
    model = make_model("quadruple")
    dx = model.dynamics(
        model.initial_state(),
        model.default_action(),
        model.disturbance_defaults(),
    )
    assert np.max(np.abs(dx)) < 1e-12


@pytest.mark.parametrize(
    "state,action",
    [
        ([12.0, 13.0, 2.0, 1.5], [0.3, 0.3]),
        ([5.0, 17.0, 8.0, 3.0], [0.1, 0.8]),
        ([19.0, 4.0, 0.2, 12.0], [1.0, 0.0]),
    ],
)
def test_total_volume_balance_closes(state, action):
    assert abs(make_model("quadruple").mass_balance_residual(state, action)) < 1e-12


def test_pump_routing_matches_physical_diagonals():
    model = make_model("quadruple")
    state = model.initial_state()
    disturbance = model.disturbance_defaults()
    base = np.asarray(model.dynamics(state, [0.3, 0.3], disturbance))
    pump_1 = np.asarray(model.dynamics(state, [0.4, 0.3], disturbance)) - base
    pump_2 = np.asarray(model.dynamics(state, [0.3, 0.4], disturbance)) - base
    assert pump_1[0] > 0 and pump_1[3] > 0
    assert pump_1[1] == pytest.approx(0.0)
    assert pump_1[2] == pytest.approx(0.0)
    assert pump_2[1] > 0 and pump_2[2] > 0
    assert pump_2[0] == pytest.approx(0.0)
    assert pump_2[3] == pytest.approx(0.0)


def test_rk4_solution_is_step_converged():
    model = make_model("quadruple")
    disturbance = model.disturbance_defaults()
    coarse = Integrator(model, max_step=0.1)
    fine = Integrator(model, max_step=0.05)
    for _ in range(60):
        coarse.step(1.0, [0.36, 0.24], disturbance)
        fine.step(1.0, [0.36, 0.24], disturbance)
    assert np.max(np.abs(np.asarray(coarse.x) - np.asarray(fine.x))) < 1e-8


def test_minimum_phase_case_applies_schedule():
    env = make_test_env(
        "quadruple",
        case="minimum-phase",
        reward_spec="regulation-v1",
    )
    try:
        env.reset(seed=0)
        assert env.control_dt == 1.0
        assert env.episode_steps == 600
        initial = list(env.y_sp)
        for _ in range(120):
            env.step([0.3, 0.3])
        assert env.y_sp == pytest.approx([initial[0] + 1.0, initial[1] - 1.0])
        assert build_context(env).setpoint["y_sp"] == pytest.approx(env.y_sp)
    finally:
        env.close()


def test_randomized_case_schedule_is_seeded():
    env = make_test_env(
        "quadruple",
        case="nonminimum-phase",
        reward_spec="regulation-v1",
        randomize_setpoints=True,
    )
    try:
        env.reset(seed=9000)
        first = dict(env._episode_setpoint_events)
        env.reset(seed=9001)
        second = dict(env._episode_setpoint_events)
        env.reset(seed=9000)
        repeated = dict(env._episode_setpoint_events)
    finally:
        env.close()
    assert first == repeated
    assert first != second


def test_nonminimum_case_uses_shifted_plant_and_cross_pid():
    env = make_test_env(
        "quadruple",
        case="nonminimum-phase",
        reward_spec="regulation-v1",
        episode_steps=2,
    )
    controller = aiogym.make_controller(
        "pid",
        scenario="quadruple",
        config={
            "profile": "quadruple-nonminimum-phase-benchmark",
            "case": "nonminimum-phase",
            "policy_scope": "specialist",
            "goal": "regulation",
            "reward_spec": "regulation-v1",
        },
    )
    try:
        assert env.model.phase_configuration == "nonminimum-phase"
        assert {(u_index, y_index) for u_index, y_index, _ in controller.loops} == {
            (0, 1),
            (1, 0),
        }
        result = aiogym.evaluate_controller(controller, env, seed=0)
    finally:
        env.close()
    assert result["execution_status"] == "passed"


def test_zero_boundary_case_places_transmission_zero_at_origin():
    env = make_test_env(
        "quadruple",
        case="zero-boundary-stress",
        reward_spec="regulation-v1",
        episode_steps=2,
    )
    try:
        assert min(abs(value) for value in env.model.transmission_zeros()) < 1e-10
    finally:
        env.close()


def test_disturbance_case_is_reproducible():
    def rollout(seed):
        env = make_test_env(
            "quadruple",
            case="disturbance-rejection",
            reward_spec="regulation-v1",
            episode_steps=3,
        )
        try:
            env.reset(seed=seed)
            return [env.step([0.3, 0.3])[4] for _ in range(3)]
        finally:
            env.close()

    assert rollout(17) == rollout(17)
