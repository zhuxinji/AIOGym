"""Acceptance tests for the formal open-cascade control tasks."""
from types import SimpleNamespace

import pytest

import aiogym


TASK_NAMES = {
    "commissioning",
    "temperature-step",
    "disturbance-rejection",
    "safety-recovery",
}
NOMINAL_STATE = [0.45, 35.0, 0.45, 50.0, 0.45, 65.0]
NOMINAL_SETPOINT = [0.45, 0.45, 0.45, 35.0, 50.0, 65.0]


def test_open_cascade_tasks_are_discoverable_and_use_continuous_operation():
    expected = {f"cascade/{name}" for name in TASK_NAMES}
    assert expected < set(aiogym.list_tasks("cascade"))

    for task_id in expected:
        profile = aiogym.load_task_profile(task_id)
        assert profile["scenario"] == "cascade"
        assert profile["default_objective"] in profile["supported_objectives"]
        assert "economic" not in profile["supported_objectives"]
        assert profile["operation"] == {
            "product_flow_sp": 0.0004,
            "min_product_flow": 0.0004,
        }
        assert "model_params" not in profile


def test_commissioning_task_starts_at_a_model_consistent_equilibrium():
    env = aiogym.AIOGymNativeEnv(
        "cascade", task="commissioning", reward_mode="tracking"
    )
    env.reset(seed=4)
    requirements = env.model.steady_state_requirements(env.y_sp)

    assert env.integ.x == pytest.approx(NOMINAL_STATE)
    assert env.y_sp == pytest.approx(NOMINAL_SETPOINT)
    assert requirements["feasible"] is True
    assert env.model.default_action() == pytest.approx(requirements["action"])
    assert env.model.mpc_init() == pytest.approx(requirements["action"])
    assert env.model.tracking_steady_state_action(env.y_sp) == pytest.approx(
        requirements["action"]
    )
    assert max(
        abs(value)
        for value in env.model.dynamics(
            env.integ.x, requirements["action"], env.model.disturbance_defaults()
        )
    ) < 1e-12

    _, _, terminated, _, info = env.step(requirements["action"])
    assert not terminated
    assert info["tracking_error_cost"] < 1e-18
    assert info["product_flow_sp_m3s"] == pytest.approx(0.0004)


def test_temperature_step_is_visible_and_both_targets_are_feasible():
    task = aiogym.load_task_profile("cascade/temperature-step")
    env = aiogym.AIOGymNativeEnv(
        "cascade", task=task, reward_mode="tracking"
    )
    env.reset(seed=5)
    action = env.model.tracking_steady_state_action(env.y_sp)

    for _ in range(120):
        env.step(action)

    raised = task["setpoints"]["schedule"][0]["values"]
    assert env.y_sp == pytest.approx(raised)
    assert env.model.steady_state_requirements(NOMINAL_SETPOINT)["feasible"] is True
    assert env.model.steady_state_requirements(raised)["feasible"] is True


def test_disturbance_task_applies_and_exposes_feed_pump_loss():
    env = aiogym.AIOGymNativeEnv(
        "cascade", task="disturbance-rejection", reward_mode="kpi"
    )
    env.reset(seed=6)
    action = env.model.tracking_steady_state_action(env.y_sp)

    info = None
    for _ in range(201):
        _, _, _, _, info = env.step(action)

    assert info is not None
    assert info["pump_flow_factor"] == pytest.approx(0.75)
    assert env._env()["pump_flow_factor"] == pytest.approx(0.75)


def test_safety_recovery_starts_inside_hard_limits_and_clears_interlocks():
    env = aiogym.AIOGymNativeEnv(
        "cascade", task="safety-recovery", reward_mode="kpi"
    )
    env.reset(seed=7)
    action = env.model.tracking_steady_state_action(env.y_sp)

    _, _, terminated, _, info = env.step(action)
    assert not terminated
    assert info["low_level_interlock_active"] == [True, False, False]
    assert info["temperature_trip_active"] == [False, False, True]
    assert info["safety_events"] == []

    for _ in range(99):
        _, _, terminated, _, info = env.step(action)
        assert not terminated

    assert info["low_level_interlock_active"] == [False, False, False]
    assert info["temperature_trip_active"] == [False, False, False]
    assert info["safety_events"] == []


@pytest.mark.parametrize("controller", ["pid", "mpc", "oracle"])
def test_seven_action_controllers_complete_short_commissioning_run(controller):
    from aiogym.evaluation.execution import run_evaluation_case

    protocol = aiogym.BenchmarkProtocol.tracking(
        "cascade", task="commissioning", episode_steps=2
    )
    config = (
        {"profile": "cascade-control-benchmark"}
        if controller in {"pid", "oracle"}
        else {}
    )
    case = run_evaluation_case(
        scenario="cascade",
        controller=controller,
        protocol=protocol,
        seeds=[0],
        controller_config=config,
        include_episodes=False,
    )

    controller_model = getattr(case["controller"], "model", None)
    if controller_model is None:
        controller_model = case["controller"].m
    assert controller_model.action_dim() == 7
    assert case["row"]["execution_status"] == "passed"
    assert case["row"]["controller_status"] == "ok"
    if controller == "oracle":
        assert case["row"]["tracking_error_cost"] < 1e-10
        assert case["row"]["controller_fallback_count"] == 0
        metadata = case["controller"].metadata()
        assert metadata["objective"] == "tracking"
        assert metadata["warm_start"] is True
        assert metadata["ipopt_max_iter"] == 1000
        assert metadata["ipopt_tol"] == pytest.approx(1e-4)
        assert metadata["horizon"] == 16
        assert metadata["solve_every"] == 4
        assert metadata["terminal_weight"] == pytest.approx(20.0)
        assert metadata["q_y"] == pytest.approx([0.7] * 6)
        assert metadata["r_move"] == pytest.approx(0.3)
        assert metadata["enforce_temperature_cap"] is False


@pytest.mark.parametrize(
    ("task", "objective"),
    [
        ("disturbance-rejection", "robustness"),
        ("safety-recovery", "safety"),
        ("continuous-benchmark", "economic"),
    ],
)
def test_oracle_reads_resolved_objective_from_task(task, objective):
    from aiogym.evaluation import resolve_protocol
    from aiogym.evaluation.execution import run_evaluation_case

    protocol = resolve_protocol(
        "cascade",
        data={"task": task, "episode_steps": 1},
    )
    case = run_evaluation_case(
        scenario="cascade",
        controller="oracle",
        protocol=protocol,
        seeds=[0],
        controller_config={"profile": "cascade-control-benchmark"},
        include_episodes=False,
    )

    assert protocol.objective == objective
    assert case["row"]["execution_status"] == "passed"
    assert case["row"]["controller_fallback_count"] == 0
    metadata = case["controller"].metadata()
    assert metadata["objective"] == objective
    assert metadata["horizon"] == 4
    assert metadata["terminal_weight"] == pytest.approx(0.0)


def test_cascade_control_suite_uses_task_defaults_and_pid_profile():
    from aiogym.cli.benchmark import build_cases

    args = SimpleNamespace(
        suite="cascade-control",
        scenarios=None,
        objectives=None,
        controllers=None,
        seed_list=None,
        seed=7,
        episodes=1,
        episode_steps=None,
        control_dt=None,
        sb3_path=None,
        sb3_algo="sac",
        onnx_path=None,
    )
    _, cases = build_cases(args)

    assert len(cases) == 8
    by_task = {}
    for case in cases:
        by_task.setdefault(case["task"], set()).add(case["controller"])
        assert case["scenario"] == "cascade"
        assert case["objective_source"] == "task-default"
        assert case["protocol"].model_params == {}
        if case["controller"] == "pid":
            assert case["controller_config"]["profile"] == "cascade-control-benchmark"
    assert by_task == {name: {"pid", "mpc"} for name in TASK_NAMES}
    assert {
        case["task"]: case["objective"]
        for case in cases
        if case["controller"] == "pid"
    } == {
        "commissioning": "tracking",
        "temperature-step": "tracking",
        "disturbance-rejection": "robustness",
        "safety-recovery": "safety",
    }


def test_cascade_tracking_steady_state_action_rejects_infeasible_target():
    model = aiogym.make_model("cascade")
    model.configure_operation({
        "product_flow_sp": 0.0004,
        "min_product_flow": 0.0004,
    })

    assert model.tracking_steady_state_action(
        [0.45, 0.45, 0.45, 35.0, 50.0, 65.0]
    ) is not None
    assert model.tracking_steady_state_action(
        [0.45, 0.45, 0.45, 35.0, 20.0, 65.0]
    ) is None
