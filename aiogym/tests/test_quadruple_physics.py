"""Physical and task acceptance tests for the Johansson quadruple tank."""
from __future__ import annotations

import numpy as np
import pytest

import aiogym
from aiogym.controllers import build_context, make_controller
from aiogym.evaluation import evaluate_controller
from aiogym.models import Integrator, apply_model_params, make_model


def test_reference_parameters_and_minimum_phase_configuration():
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
    nonminimum = apply_model_params(model, {"gamma": [0.43, 0.34]})
    assert nonminimum.phase_configuration == "nonminimum-phase"


def test_nominal_state_is_exact_nonlinear_equilibrium():
    model = make_model("quadruple")
    dx = model.dynamics(
        model.initial_state(), model.default_action(), model.disturbance_defaults()
    )
    assert np.max(np.abs(dx)) < 1e-12
    assert model.initial_state() == pytest.approx([
        12.2629675195507,
        12.783158403008972,
        1.6339411322567796,
        1.409044702533737,
    ])


@pytest.mark.parametrize(
    "state,action",
    [
        ([12.0, 13.0, 2.0, 1.5], [0.3, 0.3]),
        ([5.0, 17.0, 8.0, 3.0], [0.1, 0.8]),
        ([19.0, 4.0, 0.2, 12.0], [1.0, 0.0]),
    ],
)
def test_total_volume_balance_closes(state, action):
    model = make_model("quadruple")
    assert abs(model.mass_balance_residual(state, action)) < 1e-12


def test_pump_routing_matches_physical_diagonals():
    model = make_model("quadruple")
    x = model.initial_state()
    env = model.disturbance_defaults()
    base = np.asarray(model.dynamics(x, [0.3, 0.3], env))
    pump_1 = np.asarray(model.dynamics(x, [0.4, 0.3], env)) - base
    pump_2 = np.asarray(model.dynamics(x, [0.3, 0.4], env)) - base
    assert pump_1[0] > 0 and pump_1[3] > 0
    assert pump_1[1] == pytest.approx(0.0) and pump_1[2] == pytest.approx(0.0)
    assert pump_2[1] > 0 and pump_2[2] > 0
    assert pump_2[0] == pytest.approx(0.0) and pump_2[3] == pytest.approx(0.0)


def test_rk4_solution_is_step_converged_for_reference_transient():
    model = make_model("quadruple")
    env = model.disturbance_defaults()
    coarse = Integrator(model, max_step=0.1)
    fine = Integrator(model, max_step=0.05)
    for _ in range(60):
        coarse.step(1.0, [0.36, 0.24], env)
        fine.step(1.0, [0.36, 0.24], env)
    assert np.max(np.abs(np.asarray(coarse.x) - np.asarray(fine.x))) < 1e-8


def test_reference_task_applies_equilibrium_and_setpoint_schedule():
    env = aiogym.make_env(
        "quadruple", objective="tracking", task="minimum-phase"
    )
    env.reset(seed=0)
    assert env.control_dt == 1.0 and env.episode_steps == 600
    assert env.integ.x == pytest.approx(env.model.initial_state())
    initial = list(env.y_sp)
    for _ in range(119):
        env.step([0.3, 0.3])
    assert env.y_sp == initial
    env.step([0.3, 0.3])
    scheduled = [initial[0] + 1.0, initial[1] - 1.0]
    assert env.y_sp == pytest.approx(scheduled)
    assert build_context(env).setpoint["y_sp"] == pytest.approx(scheduled)


def test_randomized_reference_schedule_is_seeded_and_reachable():
    env = aiogym.make_env(
        "quadruple",
        objective="tracking",
        task="nonminimum-phase",
        randomize_setpoints=True,
    )
    nominal_initial = list(env._ysp0)
    nominal_schedule = dict(env._task_setpoint_events)

    env.reset(seed=9000)
    first = dict(env._episode_setpoint_events)
    assert env.y_sp == pytest.approx(nominal_initial)

    env.reset(seed=9001)
    second = dict(env._episode_setpoint_events)
    env.reset(seed=9000)
    repeated = dict(env._episode_setpoint_events)

    assert first == repeated
    assert first != second
    assert first != nominal_schedule
    assert env._task_setpoint_events == nominal_schedule
    for _ in range(120):
        env.step(env.model.default_action())
    assert env.y_sp == pytest.approx(first[120])
    for target in first.values():
        assert env.model.is_setpoint_reachable(target)
        action = env.model.tracking_steady_state_action(target)
        equilibrium = env.model.equilibrium_state(
            env.model.physical_action_vector(action)
        )
        assert env.model.controlled_output(equilibrium) == pytest.approx(target)
        assert all(0.0 <= value <= 20.0 for value in equilibrium)


def test_zero_boundary_randomized_schedule_stays_on_reachable_manifold():
    env = aiogym.make_env(
        "quadruple",
        objective="tracking",
        task="zero-boundary-stress",
        randomize_setpoints=True,
    )
    env.reset(seed=9000)
    target = env._episode_setpoint_events[120]
    env.reset(seed=9001)
    other_target = env._episode_setpoint_events[120]

    assert target != pytest.approx(other_target)
    # The singular zero-boundary map has no unique inverse, so the sampler
    # generates a safe actuator equilibrium and maps it forward to this target.
    assert env.model.tracking_steady_state_action(target) is None
    gamma1, gamma2 = env.model.p["gamma"]
    k1, k2 = env.model.p["pump_gain"]
    a1, a2 = env.model.p["outlet_area"][:2]
    required = [
        a1 * np.sqrt(2.0 * env.model.p["gravity"] * target[0]),
        a2 * np.sqrt(2.0 * env.model.p["gravity"] * target[1]),
    ]
    matrix = np.asarray([
        [gamma1 * k1, (1.0 - gamma2) * k2],
        [(1.0 - gamma1) * k1, gamma2 * k2],
    ])
    assert np.linalg.matrix_rank(matrix) == 1
    assert np.linalg.matrix_rank(np.column_stack([matrix, required])) == 1


def test_evaluation_records_each_seeded_setpoint_schedule():
    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple",
        task="nonminimum-phase",
        randomize_setpoints=True,
        episode_steps=1,
    )
    controller = make_controller(
        "pid",
        model=protocol.make_env().model,
        scenario="quadruple",
        config={"profile": "quadruple-nonminimum-phase-benchmark"},
    )
    result = evaluate_controller(
        controller,
        protocol.make_env(),
        seed_list=[9000, 9001],
        protocol=protocol,
    )
    schedules = result["disturbance"]["episode_schedules"]

    assert len(schedules) == 2
    assert schedules[0]["setpoints"] != schedules[1]["setpoints"]
    assert "120" in schedules[0]["setpoints"]


def test_reference_pi_bias_and_closed_loop_smoke():
    env = aiogym.make_env(
        "quadruple", objective="tracking", task="minimum-phase"
    )
    controller = make_controller("pid", scenario="quadruple")
    env.reset(seed=0)
    nominal_action = controller.compute(
        env.model.measurement(env.integ.x, env._env()), {"y_sp": env.y_sp}, 1.0
    )
    assert env.model.action_vector(nominal_action) == pytest.approx([0.3, 0.3])
    result = evaluate_controller(controller, env, episodes=1, seed_list=[0])
    assert result["tracking_error_cost"] < 40.0
    assert result["tracking_move_cost"] < 20.0
    assert result["tracking_cost"] == pytest.approx(
        result["tracking_error_cost"]
        + result["tracking_move_cost"]
    )
    assert result["constraint_violation_count"] == 0
    assert result["normalized_score"] > 99.0


def test_nonminimum_phase_task_has_rhp_zero_and_matched_controller_profile():
    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple", task="nonminimum-phase", episode_steps=2
    )
    env = protocol.make_env()
    env.reset(seed=0)
    zeros = env.model.transmission_zeros()
    assert env.model.phase_configuration == "nonminimum-phase"
    assert max(zeros) > 0.0
    assert env.integ.x == pytest.approx([
        12.441864220183483,
        13.166812925360238,
        4.730260670665678,
        4.986334403669725,
    ])
    controller = make_controller(
        "pid",
        model=env.model,
        scenario="quadruple",
        config={"profile": "quadruple-nonminimum-phase-benchmark"},
    )
    nominal_action = controller.compute(
        env.model.measurement(env.integ.x, env._env()), {"y_sp": env.y_sp}, 1.0
    )
    assert env.model.action_vector(nominal_action) == pytest.approx([0.315, 0.315])


def test_nonminimum_steady_input_inverse_matches_requested_lower_levels():
    model = apply_model_params(make_model("quadruple"), {
        "gamma": [0.43, 0.34],
        "nominal_voltage": [3.15, 3.15],
    })
    initial = model.initial_state()
    target = [initial[0] + 1.0, initial[1]]
    action = model.tracking_steady_state_action(target)
    equilibrium = model.equilibrium_state([
        action[0] * model.p["max_voltage"],
        action[1] * model.p["max_voltage"],
    ])
    assert action == pytest.approx([0.2962154714, 0.3463036995], abs=1e-9)
    assert model.controlled_output(equilibrium) == pytest.approx(target)
    # This cross-coupled allocation is the key nonminimum-phase distinction:
    # raise lower tank 1 by reducing pump 1 and increasing pump 2.
    assert action[0] < model.default_action()[0]
    assert action[1] > model.default_action()[1]


def test_zero_boundary_has_no_unique_steady_input_target():
    model = apply_model_params(make_model("quadruple"), {"gamma": [0.55, 0.45]})
    initial = model.initial_state()

    assert model.tracking_steady_state_action([initial[0] + 0.5, initial[1]]) is None


def test_nonminimum_benchmark_pid_uses_cross_pairing():
    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple", task="nonminimum-phase", episode_steps=2
    )
    controller = make_controller(
        "pid",
        model=protocol.make_env().model,
        scenario="quadruple",
        config={"profile": "quadruple-nonminimum-phase-benchmark"},
    )
    loops = controller.metadata()["loops"]
    assert [(loop["u_index"], loop["y_index"]) for loop in loops] == [(0, 1), (1, 0)]


def test_nonminimum_mpc_profile_regularizes_toward_steady_input():
    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple", task="nonminimum-phase", episode_steps=2
    )
    controller = make_controller(
        "mpc",
        model=protocol.make_env().model,
        scenario="quadruple",
        config={"profile": "quadruple-nonminimum-phase"},
    )
    metadata = controller.metadata()
    assert metadata["horizon"] == 48
    assert metadata["cv_scale"] == pytest.approx([1.0, 1.0])
    assert metadata["initialization"] == "tracking_steady_state_action"
    assert metadata["feedforward_reseed"] == "setpoint_change"
    assert metadata["steady_input_weight"] == pytest.approx(20000.0)
    assert "du_max" not in metadata


def test_mpc_steady_feedforward_reseeds_on_setpoint_change(monkeypatch):
    model = aiogym.make_model("quadruple")
    controller = make_controller(
        "mpc",
        model=model,
        scenario="quadruple",
        config={"profile": "quadruple-minimum-phase", "P": 1},
    )
    target = [model.default_setpoint_vector()[0] + 1.0, model.default_setpoint_vector()[1]]
    changed_target = [target[0], target[1] + 1.0]
    seed = model.tracking_steady_state_action(target)
    changed_seed = model.tracking_steady_state_action(changed_target)
    calls = []

    def steady_action(y_sp):
        calls.append(list(y_sp))
        return seed if list(y_sp) == target else changed_seed

    monkeypatch.setattr(model, "tracking_steady_state_action", steady_action)
    monkeypatch.setattr(np.linalg, "solve", lambda H, g: np.zeros(model.action_dim()))
    measurement = {"x": model.initial_state(), **model.disturbance_defaults()}

    first = controller.compute(measurement, {"y_sp": target}, 1.0)
    second = controller.compute(measurement, {"y_sp": target}, 1.0)
    third = controller.compute(measurement, {"y_sp": changed_target}, 1.0)

    assert first == pytest.approx(seed)
    assert second == pytest.approx(seed)
    assert third == pytest.approx(changed_seed)
    assert calls == [target, changed_target]


@pytest.mark.parametrize(
    "profile,horizon,solve_every,r_move,terminal_weight,steady_input_weight",
    [
        ("quadruple-minimum-phase", 12, 1, 1.0, 0.0, 0.0),
        ("quadruple-nonminimum-phase", 8, 8, 1.0, 0.0, 1.0),
        ("quadruple-zero-boundary", 4, 1, 1.0, 0.0, 0.0),
        ("quadruple-disturbance-rejection", 3, 2, 1.0, 0.0, 0.0),
    ],
)
def test_quadruple_oracle_profiles_expose_task_specific_tuning(
    profile, horizon, solve_every, r_move, terminal_weight, steady_input_weight
):
    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple", task="minimum-phase", episode_steps=2
    )
    controller = make_controller(
        "oracle",
        model=protocol.make_env().model,
        scenario="quadruple",
        config={"profile": profile},
    )
    metadata = controller.metadata()
    assert metadata["horizon"] == horizon
    assert metadata["solve_every"] == solve_every
    assert metadata["q_y"] == pytest.approx([1.0, 1.0])
    assert metadata["r_move"] == pytest.approx(r_move)
    assert metadata["terminal_weight"] == pytest.approx(terminal_weight)
    assert metadata["steady_input_weight"] == pytest.approx(steady_input_weight)
    assert metadata["initialization"] == "tracking_steady_state_action"
    assert metadata["integration_substeps"] == 10
    assert metadata["integration_max_step"] == pytest.approx(0.1)
    assert "du_max" not in metadata


def test_oracle_setpoint_preview_is_opt_in_and_uses_future_task_events():
    env = aiogym.make_env(
        "quadruple", objective="tracking", task="minimum-phase"
    )
    env.reset(seed=0)
    controller = make_controller(
        "oracle",
        model=env.model,
        scenario="quadruple",
        config={
            "profile": "quadruple-minimum-phase",
            "horizon": 4,
            "preview_setpoints": True,
        },
    )
    initial = list(env.y_sp)
    scheduled = list(env._task_setpoint_events[120])

    env._k = 117
    preview = controller._preview_targets(build_context(env))

    assert controller.metadata()["preview_setpoints"] is True
    assert np.asarray(preview[:3]) == pytest.approx(np.asarray([initial, initial, initial]))
    assert preview[3] == pytest.approx(scheduled)
    current, trajectory = controller.orc._setpoint_trajectory(initial, preview)
    assert current == pytest.approx(initial)
    assert trajectory.T == pytest.approx(np.asarray(preview))


def test_standard_oracle_does_not_preview_setpoint_schedule():
    controller = make_controller(
        "oracle",
        model=aiogym.make_model("quadruple"),
        scenario="quadruple",
        config={"profile": "quadruple-minimum-phase", "horizon": 1},
    )

    assert controller.metadata()["preview_setpoints"] is False


def test_zero_boundary_oracle_falls_back_to_nominal_initial_guess():
    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple", task="zero-boundary-stress", episode_steps=2
    )
    controller = make_controller(
        "oracle",
        model=protocol.make_env().model,
        scenario="quadruple",
        config={"profile": "quadruple-zero-boundary", "horizon": 1},
    )
    target = protocol.make_env().model.default_setpoint_vector()

    assert controller.orc._steady_action_target(target) == pytest.approx(controller.orc.u_init)
    assert "u_target" in controller.orc.par
    assert controller.orc.steady_input_weight == pytest.approx(0.0)
    assert "steady_weight" not in controller.orc.par


def test_oracle_replans_immediately_for_setpoint_and_disturbance_changes(monkeypatch):
    model = aiogym.make_model("quadruple")
    controller = make_controller(
        "oracle",
        model=model,
        scenario="quadruple",
        config={"profile": "quadruple-minimum-phase", "horizon": 1, "solve_every": 5},
    )
    calls = []

    def fake_solve(x, t_cold, t_amb, disturbances=None, y_sp=None, advance_steps=1):
        calls.append({"x": list(x), "disturbances": dict(disturbances or {}), "y_sp": list(y_sp)})
        controller.orc.last_error = None
        controller.orc.last_plan = None
        return model.default_action()

    monkeypatch.setattr(controller.orc, "solve", fake_solve)
    measurement = {"x": model.initial_state(), **model.disturbance_defaults()}
    setpoint = {"y_sp": model.default_setpoint_vector()}

    controller.compute(measurement, setpoint, 1.0)
    controller.compute({**measurement, "x": [value + 0.1 for value in measurement["x"]]}, setpoint, 1.0)
    assert len(calls) == 1

    changed_setpoint = {"y_sp": [setpoint["y_sp"][0] + 1.0, setpoint["y_sp"][1]]}
    controller.compute(measurement, changed_setpoint, 1.0)
    assert len(calls) == 2

    changed_disturbance = {**measurement, "pump_flow_factor": 0.8}
    controller.compute(changed_disturbance, changed_setpoint, 1.0)
    assert len(calls) == 3


def test_oracle_replays_the_optimized_plan_between_periodic_solves(monkeypatch):
    model = aiogym.make_model("quadruple")
    controller = make_controller(
        "oracle",
        model=model,
        scenario="quadruple",
        config={"profile": "quadruple-minimum-phase", "horizon": 3, "solve_every": 3},
    )
    plan = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)
    calls = []

    def fake_solve(x, t_cold, t_amb, disturbances=None, y_sp=None, advance_steps=1):
        calls.append(advance_steps)
        controller.orc.last_error = None
        controller.orc.last_plan = plan.copy()
        return list(plan[:, 0])

    monkeypatch.setattr(controller.orc, "solve", fake_solve)
    measurement = {"x": model.initial_state(), **model.disturbance_defaults()}
    setpoint = {"y_sp": model.default_setpoint_vector()}

    actions = [controller.compute(measurement, setpoint, 1.0) for _ in range(4)]

    assert np.asarray(actions[:3]) == pytest.approx(plan.T)
    assert actions[3] == pytest.approx(plan[:, 0])
    assert calls == [1, 3]


@pytest.mark.parametrize(
    "task,horizon,phase,initial_action,setpoint",
    [
        (
            "minimum-phase",
            360,
            "minimum-phase",
            [0.3, 0.3],
            [14.2629675195507, 12.783158403008972],
        ),
        (
            "nonminimum-phase",
            3600,
            "nonminimum-phase",
            [0.315, 0.315],
            [14.441864220183483, 13.166812925360238],
        ),
    ],
)
def test_custom_reference_steps_are_active_in_initial_context(
    task, horizon, phase, initial_action, setpoint
):
    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple",
        task=task,
        episode_steps=horizon,
        setpoint_schedule=[{"at_step": 0, "values": setpoint}],
    )
    env = protocol.make_env()
    env.reset(seed=0)
    equilibrium = env.model.initial_state()
    assert env.episode_steps == horizon
    assert env.model.phase_configuration == phase
    assert env.y_sp == pytest.approx([equilibrium[0] + 2.0, equilibrium[1]])
    assert env.model.default_action() == pytest.approx(initial_action)


def test_zero_boundary_task_places_transmission_zero_at_origin():
    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple", task="zero-boundary-stress", episode_steps=2
    )
    env = protocol.make_env()
    env.reset(seed=0)
    assert env.model.phase_configuration == "zero-at-origin"
    assert min(abs(value) for value in env.model.transmission_zeros()) < 1e-12
    dx = env.model.dynamics(env.integ.x, env.model.default_action(), env._env())
    assert max(abs(value) for value in dx) < 1e-12


def test_task_disturbance_schedule_is_executable_and_reproducible():
    env = aiogym.make_env(
        "quadruple",
        objective="robustness",
        task="disturbance-rejection",
        episode_steps=422,
    )
    env.reset(seed=0)
    for _ in range(121):
        env.step([0.3, 0.3])
    assert env._env()["pump_flow_factor"] == pytest.approx(0.8)
    for _ in range(180):
        env.step([0.3, 0.3])
    assert env._env()["pump_flow_factor"] == pytest.approx(1.0)
    for _ in range(121):
        env.step([0.3, 0.3])
    assert env._env()["outlet_area_factor"] == pytest.approx(0.8)
