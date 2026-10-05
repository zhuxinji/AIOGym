from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import aiogym.scenarios  # noqa: F401
from aiogym.controllers.base import make_controller
from aiogym.controllers.mpc import SuccessiveLinearizationMPC
from aiogym.core.env import make_env
from aiogym.core.rollout import rollout
from aiogym.rl.sb3 import SB3CheckpointPolicy


class _LinearMPCModel:
    scenario = "linear-mpc-test"
    time_unit = "s"
    dt_micro = 0.1

    def __init__(self, gains, bounds):
        self.gains = np.asarray(gains, dtype=float)
        self.bounds = np.asarray(bounds, dtype=float)

    def action_dim(self):
        return self.gains.shape[1]

    def action_schema(self):
        return [{"low": low, "high": high} for low, high in self.bounds]

    def state_schema(self):
        return [{"low": -np.inf, "high": np.inf} for _ in range(self.gains.shape[0])]

    def clamp_state(self, state):
        return state

    def initial_state(self):
        return np.zeros(self.gains.shape[0])

    def default_action(self):
        return np.clip(np.zeros(self.action_dim()), *self.bounds.T)

    def action_vector(self, action):
        return list(action)

    def outputs(self, state):
        return state

    def output_scales(self):
        return np.ones(self.gains.shape[0])

    def dynamics(self, state, action, disturbances):
        return self.gains @ action

    def tracking_steady_state_action(self, target, env):
        return None


def test_mpc_reoptimizes_coupled_actions_at_bounds():

    model = _LinearMPCModel([[1, 0], [1, 1]], [[0, 1], [0, 1]])
    controller = SuccessiveLinearizationMPC(model, control_dt=1, prediction_horizon=1, move_supp=0)
    action = controller.compute({"x": [0, 0]}, {"y_sp": [-1, 0]})
    assert action == pytest.approx([0, 0], abs=1e-9)
    assert np.sum((model.gains @ action - [-1, 0]) ** 2) == pytest.approx(1)
    assert np.sum((model.gains @ [0, 1] - [-1, 0]) ** 2) == 2  # Old clipped solution.

    controller.reset()
    action = controller.compute({"x": [0, 0]}, {"y_sp": [0.2, 0.6]})
    assert action == pytest.approx([0.2, 0.4], abs=1e-9)


def test_mpc_handles_singular_objective_and_model_specific_fixed_bounds():
    model = _LinearMPCModel([[1, 1, 2], [0, 0, 0]], [[2, 3], [-1, 1], [0.25, 0.25]])
    model.tracking_steady_state_action = lambda target, env: [100, -100, 1]
    controller = SuccessiveLinearizationMPC(model, control_dt=1, prediction_horizon=1, move_supp=0)
    action = controller.compute({"x": [0, 0]}, {"y_sp": [10, 0]})
    assert action == pytest.approx([3, 1, 0.25], abs=1e-9)
    assert controller.metadata()["action_bounds"] == model.bounds.tolist()


def test_mpc_reports_bounded_solver_failure(monkeypatch):
    from types import SimpleNamespace
    import aiogym.controllers.mpc as mpc

    model = _LinearMPCModel([[1, 0], [1, 1]], [[0, 1], [0, 1]])
    controller = SuccessiveLinearizationMPC(model, control_dt=1, prediction_horizon=1, move_supp=0)
    monkeypatch.setattr(mpc, "lsq_linear", lambda *args, **kwargs: SimpleNamespace(
        success=False, message="iteration limit reached",
    ))
    with pytest.raises(RuntimeError, match="MPC action sequence solve failed: iteration limit"):
        controller.compute({"x": [0, 0]}, {"y_sp": [-1, 0]})


def test_mpc_optimizes_distinct_moves_with_the_previous_action_penalty():
    model = _LinearMPCModel([[1]], [[0, 1]])
    model.tracking_steady_state_action = lambda target, env: [1.0]
    controller = SuccessiveLinearizationMPC(
        model, control_dt=1, prediction_horizon=2, solve_every=2, move_supp=0.1,
    )
    # Exact minimum of (u0-1)^2 + (u0+u1-1)^2 + .1*u0^2 + .1*(u1-u0)^2.
    first = controller.compute({"x": [0]}, {"y_sp": [1]})
    assert first == pytest.approx([130 / 161], abs=1e-8)
    second = controller.compute({"x": first}, {"y_sp": [1]})
    assert second == pytest.approx([40 / 161], abs=1e-8)


def test_mpc_replans_after_two_moves_and_reset_discards_the_cached_sequence(monkeypatch):
    model = _LinearMPCModel([[1]], [[0, 1]])
    controller = SuccessiveLinearizationMPC(
        model, control_dt=1, prediction_horizon=10, solve_every=2, move_supp=1,
    )
    solves = []
    solve = controller._solve

    def record_solve(measurement, setpoint):
        solves.append(list(measurement["x"]))
        solve(measurement, setpoint)

    monkeypatch.setattr(controller, "_solve", record_solve)
    first = controller.compute({"x": [0]}, {"y_sp": [1]})
    original_plan = controller._plan.copy()
    second = controller.compute({"x": first}, {"y_sp": [1]})
    assert second == pytest.approx(original_plan[1])
    assert len(solves) == 1
    third = controller.compute({"x": [-1]}, {"y_sp": [1]})
    assert solves == [[0], [-1]]
    assert third != pytest.approx(original_plan[2])
    controller.reset()
    assert controller.compute({"x": [0]}, {"y_sp": [1]}) == pytest.approx(first)


def test_mpc_linearizes_before_rk4_with_affine_drift_and_substeps():
    from aiogym.core.model import integrate_process_state

    model = _LinearMPCModel([[1]], [[-5, 5]])
    calls = []

    def dynamics(state, action, disturbances):
        calls.append(1)
        return [-state[0] ** 2 + action[0] + disturbances["bias"]]

    model.dynamics = dynamics
    controller = SuccessiveLinearizationMPC(
        model, control_dt=0.35, prediction_horizon=1, move_supp=0,
    )
    action = controller.compute({"x": [0.8], "bias": 0.3}, {"y_sp": [0.6]})
    # At x0=.8 the tangent ODE is dx/dt = -1.6*x + u + .94.
    # Integrate it independently with the shared stage-by-stage RK4 helper.
    linear = _LinearMPCModel([[1]], [[-5, 5]])
    linear.dynamics = lambda state, action, disturbances: [-1.6*state[0] + action[0] + 0.94]
    free = integrate_process_state(linear, [0.8], [0], {}, duration=0.35)
    unit = integrate_process_state(linear, [0.8], [1], {}, duration=0.35)
    assert action == pytest.approx((0.6-free)/(unit-free), abs=1e-9)
    assert len(calls) <= 10  # Derivative sampling must not repeat nonlinear integration.
    assert controller.metadata()["linearization"] == "continuous_dynamics"
    assert controller.metadata()["discretization"] == "rk4"
    assert controller.metadata()["min_integration_substeps"] == 4
    assert controller.metadata()["max_integration_dt"] == pytest.approx(0.0875)


def test_mpc_rk4_refines_stiff_cascade_boundary_linearization():
    env = make_env("cascade", benchmark="boundary-safety")
    try:
        policy = make_controller("mpc", env=env, config={"prediction_horizon": 30})
        result = rollout(env, policy, seed=1000, max_steps=2)
        assert len(result.transitions) == 2
        assert all(env.action_space.contains(row.action) for row in result.transitions)
        assert np.isfinite(result.episode_return)
    finally:
        env.close()


def test_mpc_rejects_an_execution_interval_longer_than_its_plan():
    with pytest.raises(ValueError, match="solve_every must not exceed prediction_horizon"):
        SuccessiveLinearizationMPC(
            _LinearMPCModel([[1]], [[0, 1]]), control_dt=1,
            prediction_horizon=2, solve_every=3,
        )


@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_stable_scenario_pid_and_mpc_emit_environment_actions(controller_id):
    env = make_env("quadruple", benchmark="tracking")
    try:
        policy = make_controller(controller_id, env=env)
        result = rollout(env, policy, seed=0, max_steps=2)
        assert len(result.transitions) == 2
        assert all(env.action_space.contains(row.action) for row in result.transitions)
        assert np.isfinite(result.episode_return)
    finally:
        env.close()


@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_fixed_three_tank_pid_and_mpc_contracts(controller_id):
    env = make_env("three_tank", reward="regulation")
    try:
        result = rollout(
            env,
            make_controller(controller_id, env=env),
            seed=7,
            max_steps=2,
        )
        assert len(result.transitions) == 2
        assert all(env.action_space.contains(row.action) for row in result.transitions)
        assert np.isfinite(result.episode_return)
        if controller_id == "mpc":
            assert result.policy_metadata["initialization"] == "default_action"
    finally:
        env.close()


@pytest.mark.parametrize("controller_id", ("pid", "mpc"))
def test_tank3_tracking_pid_and_mpc_use_the_physical_interface(controller_id):
    env = make_env("three_tank", benchmark="tracking")
    try:
        result = rollout(
            env,
            make_controller(controller_id, env=env),
            seed=2,
            max_steps=2,
        )
        assert len(result.transitions) == 2
        assert not result.transitions[-1].terminated
        assert result.transitions[0].action.shape == (4,)
        assert result.transitions[0].info["applied_action"].shape == (4,)
    finally:
        env.close()


def test_three_tank_controller_config_is_shared_by_all_benchmarks():
    metadata = []
    for benchmark in (
        "tracking",
        "disturbance-rejection",
        "boundary-safety",
    ):
        env = make_env("three_tank", benchmark=benchmark)
        try:
            metadata.append(
                (
                    make_controller("pid", env=env).metadata(),
                    make_controller("mpc", env=env).metadata(),
                )
            )
        finally:
            env.close()
    assert metadata[0] == metadata[1] == metadata[2]
    pid, mpc = metadata[0]
    assert max(abs(value) for row in pid["kp"] for value in row) == 32.0
    assert pid["kp"][-1] == pytest.approx([0.0, 0.0, -24.0])
    assert pid["ki"][-1] == pytest.approx([0.0, 0.0, -0.06])
    assert max(abs(value) for row in pid["ki"][:4] for value in row) == pytest.approx(
        0.08
    )
    assert mpc["q_y"] == [1.0] * 3
    assert "feedforward" not in pid
    assert mpc["steady_input_role"] == "linearization_reference_and_penalty"
    assert mpc["prediction_horizon"] == 20
    assert mpc["control_horizon"] == 20
    assert mpc["action_bound_handling"] == "box_constrained_lsq"
    assert mpc["action_bounds"] == [[0.0, 1.0]] * 4
    assert mpc["prediction_state_constraints"] is False
    assert mpc["move_supp"] == [5.0] * 4
    assert mpc["steady_input_weight"] == [0.5] * 4


def test_name_bound_pid_rejects_unknown_actuator_before_rollout():
    env = make_env("three_tank", reward="regulation")
    try:
        with pytest.raises(ValueError, match="unknown PID actuator"):
            make_controller(
                "pid",
                env=env,
                config={
                    "matrix_terms": [
                        {
                            "actuator": "not-installed",
                            "output": "tank_1_level",
                            "kp": 1.0,
                            "ki": 0.0,
                            "kd": 0.0,
                        }
                    ]
                },
            )
    finally:
        env.close()


def test_mpc_does_not_receive_hidden_disturbance_values():
    env = make_env("three_tank", reward="regulation")
    try:
        nominal = make_controller(
            "mpc",
            env=env,
            config={
                "prediction_horizon": 5,
                "move_supp": 1.0,
                "steady_input_weight": 1.0,
            },
        )
        repeated = make_controller(
            "mpc",
            env=env,
            config={
                "prediction_horizon": 5,
                "move_supp": 1.0,
                "steady_input_weight": 1.0,
            },
        )
        observation, info = env.reset(seed=0)
        context = {
            "step_index": 0,
            "physical_time": 0.0,
            "reference": info["reference"],
        }
        first = nominal.act(observation, context)
        env.set_disturbances({"pump_flow_factor": 0.85})
        second = repeated.act(observation, context)
        assert second == pytest.approx(first)
    finally:
        env.close()


def test_mpc_accepts_per_actuator_regularization_weights():
    env = make_env("three_tank", reward="regulation")
    try:
        policy = make_controller(
            "mpc",
            env=env,
            config={
                "move_supp": [1.0, 1.0, 1.0, 0.01],
                "steady_input_weight": [0.1, 0.1, 0.1, 0.0],
            },
        )
        result = rollout(env, policy, seed=0, max_steps=2)
        assert len(result.transitions) == 2
        assert result.policy_metadata["move_supp"][-1] == pytest.approx(0.01)
    finally:
        env.close()


def test_hold_and_random_are_seeded_environment_action_policies():
    env = make_env("three_tank", reward="regulation")
    try:
        held = rollout(env, make_controller("hold", env=env), seed=2, max_steps=2)
        first = rollout(env, make_controller("random", env=env), seed=3, max_steps=2)
        second = rollout(env, make_controller("random", env=env), seed=3, max_steps=2)
        assert np.array_equal(held.transitions[0].action, held.transitions[1].action)
        assert all(
            np.array_equal(left.action, right.action)
            for left, right in zip(first.transitions, second.transitions)
        )
    finally:
        env.close()


def test_pid_is_deterministic_for_a_repeated_benchmark_case_seed():
    env = make_env("quadruple", benchmark="tracking")
    try:
        first = rollout(
            env,
            make_controller("pid", env=env),
            seed=3,
            max_steps=1,
        )
        second = rollout(
            env,
            make_controller("pid", env=env),
            seed=3,
            max_steps=1,
        )
        assert np.array_equal(
            first.transitions[0].action,
            second.transitions[0].action,
        )
    finally:
        env.close()


def test_sb3_adapter_uses_predict_without_importing_sb3_at_module_import():
    code = (
        "import sys; import aiogym.rl.sb3; "
        "assert 'stable_baselines3' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

    class FakeModel:
        def predict(self, observation, deterministic=True):
            assert deterministic
            return np.asarray([0.25, 0.75], dtype=np.float32), None

    policy = SB3CheckpointPolicy(FakeModel(), algorithm="sac", checkpoint="fake.zip")
    action = policy.act(np.zeros(3), {})
    assert action.tolist() == pytest.approx([0.25, 0.75])
    assert policy.metadata()["algorithm"] == "sac"
