"""Tests for physical metadata, task profiles, and generic model readiness."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import aiogym


def test_all_builtins_have_non_mutating_parameter_profiles():
    assert set(aiogym.list_parameter_profiles()) == set(aiogym.list_scenarios())
    for scenario in aiogym.list_scenarios():
        model = aiogym.make_model(scenario)
        before = dict(model.p)
        profile = aiogym.load_parameter_profile(scenario)
        metadata = model.metadata()
        assert profile["scenario"] == scenario
        expected_status = {
            "quadruple": "reference-parameterized",
            "cascade-recirculating": "design-provisional",
        }.get(scenario, "legacy-unverified")
        assert profile["status"] == expected_status
        assert model.p == before
        assert metadata["physical_metadata"]["parameter_status"] == expected_status
        assert metadata["solver"]["method"] == "rk4"
        assert metadata["solver"]["max_step"] == pytest.approx(model.dt_micro)
        if scenario == "cascade":
            assert {
                row["status"] for row in metadata["parameters"].values()
            } == {"legacy-unverified", "assumed-benchmark"}
        elif scenario == "cascade-recirculating":
            statuses = {row["status"] for row in metadata["parameters"].values()}
            assert "design-specified" in statuses
            assert "assumed-benchmark" in statuses
            assert "legacy-unverified" in statuses
        elif scenario != "quadruple":
            assert all(
                row["status"] == "legacy-unverified"
                for row in metadata["parameters"].values()
            )


def test_all_builtins_pass_generic_readiness_checks():
    for scenario in aiogym.list_scenarios():
        report = aiogym.validate_model_readiness(scenario)
        assert report["passed"], report
        if scenario == "quadruple":
            expected_not_checked = []
        elif scenario == "cascade-recirculating":
            expected_not_checked = ["reference_parameter_fidelity"]
        else:
            expected_not_checked = [
                "mass_balance", "energy_balance", "reference_parameter_fidelity"
            ]
        assert report["not_checked"] == expected_not_checked
        if scenario == "quadruple":
            assert report["not_applicable"] == ["energy_balance"]


def test_models_use_single_dynamics_and_flat_action_contract():
    for scenario in aiogym.list_scenarios():
        model = aiogym.make_model(scenario)
        assert not hasattr(model, "derivatives")
        assert not hasattr(model, "action_vector_to_dict")
        assert not hasattr(model, "uses_legacy_actions")
        assert isinstance(model.default_action(), list)
        assert len(model.default_action()) == model.action_dim()
        with pytest.raises(ValueError, match="numeric vector"):
            model.action_vector({"pumps": [0.5]})


def test_task_profiles_are_opt_in_and_explicit_timing_wins():
    assert not any(name.endswith("/legacy-default") for name in aiogym.list_tasks())
    assert "quadruple/minimum-phase" in aiogym.list_tasks()
    assert "quadruple/nonminimum-phase" in aiogym.list_tasks()
    assert "cascade/continuous-benchmark" in aiogym.list_tasks()
    assert {
        "quadruple/zero-boundary-stress",
        "quadruple/disturbance-rejection",
    }.issubset(set(aiogym.list_tasks()))

    direct_default = aiogym.AIOGymNativeEnv("cstr")
    task_env = aiogym.AIOGymNativeEnv("quadruple", task="minimum-phase")
    overridden = aiogym.AIOGymNativeEnv(
        "quadruple", task="minimum-phase", control_dt=0.25, episode_steps=12
    )
    assert (direct_default.control_dt, direct_default.episode_steps) == (0.5, 600)
    assert (task_env.control_dt, task_env.episode_steps) == (1.0, 600)
    assert task_env.task_profile["name"] == "minimum-phase"
    assert task_env.tracking_q_y == pytest.approx([0.7, 0.7])
    assert task_env.tracking_r_move == pytest.approx(0.3)
    assert task_env.disturbance_obs is False
    assert task_env.previous_action_obs is True
    assert task_env.normalize_observations is True
    assert task_env.tracking_error_obs is True
    assert (overridden.control_dt, overridden.episode_steps) == (0.25, 12)

    protocol_default = aiogym.BenchmarkProtocol.tracking("cstr")
    protocol_task = aiogym.BenchmarkProtocol.tracking("quadruple", task="minimum-phase")
    factory_task = aiogym.make_env("quadruple", task="minimum-phase")
    assert (protocol_default.control_dt, protocol_default.episode_steps) == (0.5, 400)
    assert (protocol_task.control_dt, protocol_task.episode_steps) == (1.0, 600)
    assert protocol_task.metadata()["task"]["name"] == "minimum-phase"
    assert (factory_task.control_dt, factory_task.episode_steps) == (1.0, 600)


def test_task_setpoint_overrides_are_executable():
    schedule = [{"at_step": 0, "values": [14.0, 13.0]}]
    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple",
        task="minimum-phase",
        episode_steps=360,
        setpoint_schedule=schedule,
    )
    env = protocol.make_env()
    env.reset(seed=0)

    assert env.episode_steps == 360
    assert env.y_sp == pytest.approx([14.0, 13.0])


def test_cascade_operation_profiles_are_validated_and_shared_with_controller_model():
    from aiogym.evaluation.execution import run_evaluation_case

    continuous = aiogym.load_task_profile("cascade/continuous-benchmark")
    assert aiogym.task_operation(continuous) == {
        "product_flow_sp": 4.0e-4,
        "min_product_flow": 4.0e-4,
    }
    invalid = dict(continuous)
    invalid["operation"] = {"min_product_flow": 0.0}
    with pytest.raises(ValueError, match="operation requires product_flow_sp"):
        aiogym.validate_task_profile(invalid)

    protocol = aiogym.BenchmarkProtocol.economic(
        "cascade", task="continuous-benchmark", episode_steps=1
    )
    environment = protocol.make_env()
    case = run_evaluation_case(
        scenario="cascade",
        controller="pid",
        protocol=protocol,
        seeds=[0],
        include_episodes=False,
    )
    assert environment.model.operation == continuous["operation"]
    assert case["controller"].model.operation == environment.model.operation


def test_task_owns_conditions_while_objective_owns_scoring():
    task = aiogym.load_task_profile("quadruple/nonminimum-phase")
    protocol = aiogym.BenchmarkProtocol.kpi(
        "quadruple", task=task, episode_steps=7
    )
    metadata = protocol.metadata()
    assert protocol.objective == "kpi"
    assert protocol.env_reward_mode == "kpi"
    assert protocol.auto_events is False
    assert not hasattr(protocol, "dynamic")
    assert protocol.randomize is False
    assert protocol.noise is False
    assert protocol.episode_steps == 7
    assert protocol.model_params["gamma"] == [0.43, 0.34]
    assert metadata["task_identity"]["name"] == "nonminimum-phase"
    assert len(metadata["task_identity"]["profile_hash"]) == 64
    with pytest.raises(ValueError, match="does not support objective"):
        aiogym.BenchmarkProtocol.economic("quadruple", task=task)


def test_removed_dynamic_option_is_rejected():
    assert aiogym.AIOGymNativeEnv("cstr").auto_events is True
    assert aiogym.BenchmarkProtocol.tracking("cstr").auto_events is False

    with pytest.raises(TypeError, match="dynamic"):
        aiogym.AIOGymNativeEnv("cstr", dynamic=True)


def test_auto_events_is_the_only_canonical_protocol_and_case_field():
    protocol = aiogym.BenchmarkProtocol.tracking(
        "cstr", auto_events=True, episode_steps=1
    )
    protocol_metadata = protocol.metadata()
    case_metadata = aiogym.BenchmarkCase.from_protocol(
        protocol, controller="pid", seeds=[0]
    ).metadata()

    assert protocol.env_kwargs()["auto_events"] is True
    assert protocol_metadata["auto_events"] is True
    assert case_metadata["environment"]["auto_events"] is True
    assert "dynamic" not in protocol.env_kwargs()
    assert "dynamic" not in protocol_metadata
    assert "dynamic" not in case_metadata["environment"]

    with pytest.raises(TypeError, match="dynamic"):
        aiogym.BenchmarkProtocol.tracking("cstr", dynamic=True)


def test_removed_task_and_suite_dynamic_fields_are_rejected(tmp_path):
    task = aiogym.load_task_profile("quadruple/minimum-phase")
    legacy_task = dict(task)
    legacy_task["environment"] = dict(task["environment"])
    legacy_task["environment"]["dynamic"] = legacy_task["environment"].pop(
        "auto_events"
    )
    with pytest.raises(ValueError, match="unknown task environment fields: dynamic"):
        aiogym.load_task_profile(legacy_task)

    suite_path = tmp_path / "legacy-suite.json"
    suite_path.write_text(json.dumps({
        "scenarios": ["cstr"],
        "objectives": ["tracking"],
        "controllers": ["pid"],
        "dynamic": True,
    }))
    from aiogym.cli.benchmark import load_suite

    with pytest.raises(ValueError, match="unsupported suite option.*dynamic"):
        load_suite(str(suite_path))


def test_benchmark_runner_aligns_oracle_with_primary_tracking_error_weights():
    from aiogym.evaluation.execution import run_evaluation_case

    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple",
        task="minimum-phase",
        episode_steps=1,
        tracking_q_y=[2.0, 3.0],
        tracking_r_move=0.7,
    )
    case = run_evaluation_case(
        scenario="quadruple",
        controller="oracle",
        protocol=protocol,
        seeds=[0],
        controller_config={
            "profile": "quadruple-minimum-phase",
            "horizon": 1,
            "r_move": 0.7,
            "terminal_weight": 0.6,
        },
        include_episodes=False,
    )
    metadata = case["controller"].metadata()

    assert metadata["q_y"] == pytest.approx([2.0, 3.0])
    assert metadata["r_move"] == pytest.approx(0.7)
    assert metadata["terminal_weight"] == pytest.approx(0.6)
    assert metadata["steady_input_weight"] == pytest.approx(0.0)


def test_benchmark_runner_aligns_mpc_move_weight_with_tracking_protocol():
    from aiogym.evaluation.execution import run_evaluation_case

    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple",
        task="minimum-phase",
        episode_steps=1,
        tracking_q_y=[0.7, 0.7],
        tracking_r_move=0.0,
    )
    case = run_evaluation_case(
        scenario="quadruple",
        controller="mpc",
        protocol=protocol,
        seeds=[0],
        controller_config={"q_y": [9.0, 9.0], "move_supp": 0.8},
        include_episodes=False,
    )

    metadata = case["controller"].metadata()
    assert metadata["q_y"] == pytest.approx([0.7, 0.7])
    assert metadata["move_supp"] == pytest.approx(0.0)


def test_scheduled_setpoint_is_visible_before_its_control_step():
    from aiogym.controllers import build_context

    model = aiogym.make_model("quadruple")
    initial = model.default_setpoint_vector()
    scheduled = [initial[0] + 1.0, initial[1]]
    task = {
        "schema_version": "aiogym.task_profile.v1",
        "name": "setpoint-timing-test",
        "scenario": "quadruple",
        "status": "test-only",
        "environment": {
            "control_dt": 1.0,
            "episode_steps": 3,
            "auto_events": False,
            "randomize": False,
            "randomize_setpoints": False,
            "randomize_plant": False,
            "plant_drift": False,
            "noise": False,
        },
        "setpoints": {
            "initial": initial,
            "schedule": [{"at_step": 1, "values": scheduled}],
        },
    }
    env = aiogym.AIOGymNativeEnv("quadruple", task=task, reward_mode="tracking")
    env.reset(seed=0)

    assert build_context(env).setpoint["y_sp"] == pytest.approx(initial)
    _, _, _, _, first_info = env.step(model.default_action())

    # Step 0 was controlled and scored with the initial reference. The returned
    # observation/context for step 1 already exposes the scheduled reference.
    assert first_info["y_sp"] == pytest.approx(initial)
    assert build_context(env, first_info).setpoint["y_sp"] == pytest.approx(scheduled)

    _, _, _, _, second_info = env.step(model.default_action())
    assert second_info["y_sp"] == pytest.approx(scheduled)


def test_tracking_action_weights_default_to_seventy_thirty():
    env = aiogym.AIOGymNativeEnv("quadruple", reward_mode="tracking")
    protocol = aiogym.BenchmarkProtocol.tracking("quadruple")

    assert env.tracking_q_y == pytest.approx([0.7, 0.7])
    assert env.tracking_r_move == pytest.approx(0.3)
    assert protocol.tracking_q_y == pytest.approx(0.7)
    assert protocol.tracking_r_move == pytest.approx(0.3)


def test_default_tracking_weights_are_forwarded_to_oracle():
    from aiogym.evaluation.execution import run_evaluation_case

    protocol = aiogym.BenchmarkProtocol.tracking(
        "cascade",
        task="commissioning",
        episode_steps=1,
    )
    case = run_evaluation_case(
        scenario="cascade",
        controller="oracle",
        protocol=protocol,
        seeds=[0],
        controller_config={"profile": "cascade-control-benchmark"},
        include_episodes=False,
    )
    metadata = case["controller"].metadata()

    assert metadata["q_y"] == pytest.approx([0.7] * 6)
    assert metadata["r_move"] == pytest.approx(0.3)


def test_profile_scenario_mismatches_are_rejected():
    with pytest.raises(ValueError, match="expected task"):
        aiogym.load_task_profile("quadruple/minimum-phase", scenario="hvac")
    with pytest.raises(ValueError, match="expected parameter profile"):
        aiogym.load_parameter_profile("cstr", scenario="hvac")


def test_explicit_suite_cases_keep_task_identity_and_controller_profiles():
    from aiogym.cli.benchmark import build_cases

    args = SimpleNamespace(
        suite="quadruple-phase-comparison",
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
    assert len(cases) == 6
    assert {case["task"] for case in cases} == {
        "minimum-phase", "nonminimum-phase",
    }
    minimum_pid = next(
        case for case in cases
        if case["task"] == "minimum-phase" and case["controller"] == "pid"
    )
    nonminimum_pid = next(
        case for case in cases
        if case["task"] == "nonminimum-phase" and case["controller"] == "pid"
    )
    assert minimum_pid["protocol"].episode_steps == 600
    assert nonminimum_pid["protocol"].episode_steps == 1800
    assert minimum_pid["controller_config"]["profile"] == "quadruple-minimum-phase-benchmark"
    assert nonminimum_pid["controller_config"]["profile"] == "quadruple-nonminimum-phase-benchmark"


def test_single_and_suite_cases_share_task_controller_defaults():
    from aiogym.cli.benchmark import build_cases
    from aiogym.evaluation import BenchmarkCase, resolve_protocol

    for suite_name in ("quadruple", "cascade-control"):
        args = SimpleNamespace(
            suite=suite_name,
            scenarios=None,
            objectives=None,
            controllers=None,
            seed_list=None,
            seed=7,
            episodes=1,
            episode_steps=None,
            control_dt=None,
            randomize_setpoints=None,
            sb3_path=None,
            sb3_algo="sac",
            onnx_path=None,
        )
        _, suite_cases = build_cases(args)
        for suite_case in suite_cases:
            protocol = resolve_protocol(
                suite_case["scenario"],
                None,
                {
                    "task": suite_case["task"],
                    "action_mode": suite_case["action_mode"],
                },
            )
            single_case = BenchmarkCase.from_protocol(
                protocol,
                controller=suite_case["controller"],
                seeds=[7],
            )
            assert (
                dict(single_case.controller_config)
                == suite_case["controller_config"]
                == dict(suite_case["case_spec"].controller_config)
            )


def test_suite_cli_can_override_task_setpoint_randomization():
    from aiogym.cli.benchmark import build_cases

    args = SimpleNamespace(
        suite="quadruple-phase-comparison",
        scenarios=None,
        objectives=None,
        controllers=None,
        seed_list=None,
        seed=9000,
        episodes=3,
        episode_steps=None,
        control_dt=None,
        randomize_setpoints=True,
        sb3_path=None,
        sb3_algo="sac",
        onnx_path=None,
    )
    _, cases = build_cases(args)

    assert len(cases) == 6
    assert all(case["protocol"].randomize_setpoints for case in cases)
    assert all(case["seeds"] == [9000, 9001, 9002] for case in cases)


def test_scenario_target_expands_registered_tasks_without_a_suite_file():
    from aiogym.cli.benchmark import build_cases

    args = SimpleNamespace(
        suite="cascade",
        scenarios=None,
        objectives=None,
        controllers="pid",
        seed_list=None,
        seed=9000,
        episodes=1,
        episode_steps=None,
        control_dt=None,
        randomize_setpoints=None,
        sb3_path=None,
        sb3_algo="sac",
        onnx_path=None,
    )
    suite, cases = build_cases(args)

    assert suite["name"] == "cascade"
    assert {case["task"] for case in cases} == {
        "commissioning",
        "continuous-benchmark",
        "disturbance-rejection",
        "safety-recovery",
        "temperature-step",
    }
    assert len(cases) == 5


def test_taskless_scenario_target_has_a_default_tracking_case():
    from aiogym.cli.benchmark import build_cases

    args = SimpleNamespace(
        suite="cstr",
        scenarios=None,
        objectives=None,
        controllers="pid",
        seed_list=None,
        seed=9000,
        episodes=1,
        episode_steps=None,
        control_dt=None,
        randomize_setpoints=None,
        sb3_path=None,
        sb3_algo="sac",
        onnx_path=None,
    )
    suite, cases = build_cases(args)

    assert suite["name"] == "cstr"
    assert len(cases) == 1
    assert cases[0]["task"] == "default"
    assert cases[0]["objective"] == "tracking"
    assert cases[0]["protocol"].episode_steps == 80
    assert cases[0]["protocol"].control_dt == 0.5


def test_quadruple_suite_runs_all_three_controllers_on_every_formal_task():
    from aiogym.cli.benchmark import build_cases

    args = SimpleNamespace(
        suite="quadruple",
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
    assert len(cases) == 12
    by_task = {}
    for case in cases:
        by_task.setdefault(case["task"], set()).add(case["controller"])
    assert by_task == {
        "minimum-phase": {"pid", "mpc", "oracle"},
        "nonminimum-phase": {"pid", "mpc", "oracle"},
        "zero-boundary-stress": {"pid", "mpc", "oracle"},
        "disturbance-rejection": {"pid", "mpc", "oracle"},
    }
    assert {case["objective"] for case in cases} == {"tracking", "robustness"}
    assert "legacy-default" not in by_task
    pid_profiles = {
        case["task"]: case["controller_config"]["profile"]
        for case in cases
        if case["controller"] == "pid"
    }
    assert pid_profiles == {
        "minimum-phase": "quadruple-minimum-phase-benchmark",
        "nonminimum-phase": "quadruple-nonminimum-phase-benchmark",
        "zero-boundary-stress": "quadruple-zero-boundary",
        "disturbance-rejection": "quadruple-disturbance-rejection",
    }
    mpc_profiles = {
        case["task"]: case["controller_config"]["profile"]
        for case in cases
        if case["controller"] == "mpc"
    }
    assert mpc_profiles == {
        "minimum-phase": "quadruple-minimum-phase",
        "nonminimum-phase": "quadruple-nonminimum-phase",
        "zero-boundary-stress": "quadruple-zero-boundary",
        "disturbance-rejection": "quadruple-disturbance-rejection",
    }
    oracle_profiles = {
        case["task"]: case["controller_config"]["profile"]
        for case in cases
        if case["controller"] == "oracle"
    }
    assert oracle_profiles == {
        "minimum-phase": "quadruple-minimum-phase",
        "nonminimum-phase": "quadruple-nonminimum-phase",
        "zero-boundary-stress": "quadruple-zero-boundary",
        "disturbance-rejection": "quadruple-disturbance-rejection",
    }


def test_cascade_recirculating_suite_uses_formal_tasks_and_task_objectives():
    from aiogym.cli.benchmark import build_cases

    args = SimpleNamespace(
        suite="cascade-recirculating",
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

    assert "cascade-recirculating" in aiogym.list_suites()
    assert len(cases) == 8
    by_task = {}
    for case in cases:
        by_task.setdefault(case["task"], set()).add(case["controller"])
        assert case["scenario"] == "cascade-recirculating"
        assert case["objective_source"] == "task-default"
        assert case["protocol"].model_params == {}
    assert by_task == {
        "commissioning": {"pid", "mpc"},
        "temperature-step": {"pid", "mpc"},
        "disturbance-rejection": {"pid", "mpc"},
        "safety-recovery": {"pid", "mpc"},
    }
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
    assert "legacy-default" not in by_task


def test_single_benchmark_passes_oracle_specific_profile():
    from aiogym.cli.benchmark import controller_specs

    args = SimpleNamespace(
        controllers="oracle",
        controller_profile=None,
        oracle_profile="quadruple-minimum-phase",
        episodes=3,
        seed=9000,
        seed_list=None,
        sb3_path=None,
        sb3_algo="sac",
        sb3_action_mode="setpoint",
        onnx_path=None,
        onnx_action_mode="setpoint",
    )
    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple", task="minimum-phase"
    )

    specs = controller_specs(args, protocol)

    assert specs[0]["config"] == {
        "profile": "quadruple-minimum-phase",
    }
    assert specs[0]["seed_list"] == [9000, 9001, 9002]


@pytest.mark.parametrize("objective", ["kpi", "robustness", "safety"])
def test_single_benchmark_does_not_set_oracle_objective(objective):
    from aiogym.cli.benchmark import controller_specs

    args = SimpleNamespace(
        controllers="oracle",
        controller_profile="cascade-control-benchmark",
        oracle_profile=None,
        episodes=1,
        seed=7,
        seed_list=None,
        sb3_path=None,
        sb3_algo="sac",
        sb3_action_mode="setpoint",
        onnx_path=None,
        onnx_action_mode="setpoint",
    )
    protocol = getattr(aiogym.BenchmarkProtocol, objective)(
        "cascade",
        episode_steps=1,
    )

    specs = controller_specs(args, protocol)

    assert specs[0]["config"] == {
        "profile": "cascade-control-benchmark",
    }


def test_tracking_plots_show_task_identity_setpoints_and_controls(tmp_path):
    from aiogym.evaluation.artifact import plot_results, write_benchmark_artifacts

    task = "minimum-phase"
    rollout = {
        "controller_name": "PID",
        "objective": "tracking",
        "protocol": {
            "scenario": "quadruple",
            "task": {"name": task},
            "objective": "tracking",
        },
        "rollout": [
            {
                "time": 0.0,
                "next_state": [12.0, 13.0, 2.0, 1.5],
                "action": [0.3, 0.3],
                "info": {"y": [12.0, 13.0], "y_sp": [13.0, 12.0]},
            },
            {
                "time": 1.0,
                "next_state": [12.1, 12.9, 2.1, 1.6],
                "action": [0.4, 0.2],
                "info": {"y": [12.1, 12.9], "y_sp": [13.0, 12.0]},
            },
        ],
    }
    rows = [
        {
            "scenario": "quadruple",
            "task": task,
            "objective": "tracking",
            "controller": "PID",
            "execution_status": "passed",
            "metric": "tracking_error_cost",
            "tracking_cost": 1.0,
            "tracking_error_cost": 1.0,
            "runtime_seconds_per_step": 0.001,
            "runtime_total_seconds": 0.25,
        },
        {
            "scenario": "quadruple",
            "task": "zero-boundary-stress",
            "objective": "tracking",
            "controller": "PID",
            "execution_status": "passed",
            "metric": "tracking_error_cost",
            "tracking_cost": 2.0,
            "tracking_error_cost": 2.0,
            "runtime_seconds_per_step": 0.001,
            "runtime_total_seconds": 0.5,
        },
    ]
    payload = {
        "benchmark": "benchmark_suite",
        "suite": "plot-smoke",
        "suite_config": {"scenarios": ["quadruple"]},
        "rows": rows,
        "results": [],
        "report": {},
        "rollouts": [rollout],
    }
    artifacts = write_benchmark_artifacts(tmp_path, payload)
    payload["artifacts"] = artifacts
    payload["artifacts"]["rollout_figure"] = str(tmp_path / "figures" / "quadruple_rollout.svg")
    payload["artifacts"]["constraint_timeline_figure"] = str(tmp_path / "figures" / "constraint_timeline.svg")
    (tmp_path / "figures").mkdir(exist_ok=True)
    (tmp_path / "figures" / "quadruple_rollout.svg").write_text("stale")
    (tmp_path / "figures" / "constraint_timeline.svg").write_text("stale")
    (tmp_path / "benchmark.json").write_text(json.dumps(payload))
    figures = plot_results(tmp_path)

    comparison = (tmp_path / "figures" / "tracking_comparison.svg").read_text()
    assert "Task" in comparison
    assert "minimum-phase" in comparison
    assert "zero-boundary-stress" in comparison
    leaderboard = (tmp_path / "figures" / "leaderboard.svg").read_text()
    assert "minimum-phase / tracking" in leaderboard
    assert "tracking / quadruple / minimum-phase" not in leaderboard
    controls = figures["tracking_control_by_scenario"]
    assert set(controls) == {"quadruple / minimum-phase"}
    control_svg = (tmp_path / "figures" / "tracking_control_quadruple-minimum-phase.svg").read_text()
    assert "quadruple / minimum-phase tracking control" in control_svg
    assert "Setpoint" in control_svg
    assert 'stroke-dasharray="6 5"' in control_svg
    assert "h3 (cm)" not in control_svg and "h4 (cm)" not in control_svg
    assert "u1 (V)" in control_svg and "u2 (V)" in control_svg
    assert "pump_1_voltage" not in control_svg
    assert "PID" in control_svg
    assert "Time (s)" in control_svg
    assert ">4.08</text>" in control_svg and ">1.92</text>" in control_svg
    assert 'font-size="18" text-anchor="middle"' in control_svg
    assert 'font-size="20" text-anchor="start"' in control_svg
    assert not (tmp_path / "figures" / "quadruple_rollout.svg").exists()
    assert not (tmp_path / "figures" / "constraint_timeline.svg").exists()
    plotted_payload = json.loads((tmp_path / "benchmark.json").read_text())
    assert "rollout_figure" not in plotted_payload["artifacts"]
    assert "constraint_timeline_figure" not in plotted_payload["artifacts"]
    assert plotted_payload["rollouts"][0]["scenario"] == "quadruple"
    assert plotted_payload["rollouts"][0]["task"] == task


def test_summary_bars_preserve_small_nonzero_value_labels(tmp_path):
    from aiogym.evaluation.artifact.svg import plot_summary

    path = tmp_path / "small-values.svg"
    plot_summary([
        {
                "controller": "tiny-but-nonzero",
            "objective": "tracking",
            "tracking_cost": 0.004321,
            "tracking_error_cost": 0.0001234,
            "tracking_move_cost": 0.00001234,
        }
    ], path, "quadruple")
    svg = path.read_text()
    assert ">0.004321</text>" in svg
    assert ">0.0001234</text>" in svg
    assert ">1.234e-05</text>" in svg
    assert ">0.00</text>" not in svg


def test_quadruple_tracking_plot_uses_controlled_levels_and_pump_signals(tmp_path):
    from aiogym.evaluation.artifact.svg import plot_tracking_control

    rollout = {
        "controller_name": "PID",
        "rollout": [
            {
                "time": 0.0,
                "next_state": [12.0, 13.0, 2.0, 1.5],
                "action": [0.3, 0.3],
                "info": {"y": [12.0, 13.0], "y_sp": [14.0, 13.0]},
            },
            {
                "time": 1.0,
                "next_state": [12.1, 13.0, 2.1, 1.6],
                "action": [0.4, 0.2],
                "info": {"y": [12.1, 13.0], "y_sp": [14.0, 13.0]},
            },
        ],
    }
    path = tmp_path / "quadruple-tracking.svg"
    plot_tracking_control(
        [rollout], path, "quadruple", task="minimum-phase"
    )
    svg = path.read_text()
    assert "h1 (cm)" in svg and "h2 (cm)" in svg
    assert "h3 (cm)" not in svg and "h4 (cm)" not in svg
    assert "u1 (V)" in svg and "u2 (V)" in svg
    assert ">4.08</text>" in svg and ">1.92</text>" in svg


def test_named_task_suite_inherits_task_timing_instead_of_runner_defaults():
    from aiogym.cli.benchmark import build_cases

    args = SimpleNamespace(
        suite="quadruple",
        scenarios=None,
        objectives=None,
        controllers="pid",
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
    assert len(cases) == 4
    disturbance = next(
        case for case in cases if case["task"] == "disturbance-rejection"
    )
    assert disturbance["protocol"].episode_steps == 900
    assert disturbance["protocol"].control_dt == 1.0


def test_economic_suites_exclude_models_without_meaningful_economics():
    from aiogym.cli.benchmark import build_cases, load_suite

    valid_economic = {"cascade", "cstr", "hvac", "heater"}
    args = SimpleNamespace(
        suite="standard-baselines",
        scenarios=None,
        objectives=None,
        controllers="pid",
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
    tracking_scenarios = {
        case["scenario"] for case in cases if case["objective"] == "tracking"
    }
    economic_scenarios = {
        case["scenario"] for case in cases if case["objective"] == "economic"
    }
    assert tracking_scenarios == set(aiogym.list_scenarios())
    assert economic_scenarios == valid_economic
    assert {case["protocol"].episode_steps for case in cases} == {80}
    assert {case["protocol"].control_dt for case in cases} == {0.5}
    assert {
        case["task"]
        for case in cases
        if case["scenario"] == "cascade" and case["objective"] == "economic"
    } == {"continuous-benchmark"}

    assert set(load_suite("economic-actuator")["scenarios"]) == valid_economic
    cascade_cases = [
        case for case in load_suite("economic-actuator")["cases"]
        if case.get("scenarios") == ["cascade"]
    ]
    assert len(cascade_cases) == 1
    assert cascade_cases[0]["task"] == "continuous-benchmark"


def test_leaderboard_ranks_restart_for_each_task():
    from aiogym.evaluation.artifact.tables import _leaderboard

    rows = []
    for task, pid_cost, mpc_cost in (
        ("minimum-phase", 1.0, 2.0),
        ("nonminimum-phase", 5.0, 4.0),
    ):
        for controller, value in (("PID", pid_cost), ("MPC", mpc_cost)):
            rows.append({
                "scenario": "quadruple",
                "task": task,
                "objective": "tracking",
                "controller": controller,
                "execution_status": "passed",
                "metric": "tracking_cost",
                "tracking_cost": value,
            })
    board = _leaderboard(rows)
    winners = {(row["task"], row["controller"]) for row in board if row["rank"] == 1}
    assert winners == {
        ("minimum-phase", "PID"),
        ("nonminimum-phase", "MPC"),
    }


def test_tracking_artifact_count_distinguishes_tasks_within_one_scenario():
    from aiogym.evaluation.artifact.report import _tracking_benchmark_case_count

    rows = [
        {
            "scenario": "quadruple",
            "task": task,
            "objective": "tracking",
            "execution_status": "passed",
        }
        for task in ("minimum-phase", "nonminimum-phase")
    ]
    assert _tracking_benchmark_case_count(rows) == 2


def test_objective_resolution_has_one_documented_precedence_order():
    from aiogym.evaluation import resolve_objective, resolve_protocol

    task = aiogym.load_task_profile("quadruple/minimum-phase")
    assert resolve_objective(
        explicit="safety",
        case_config="kpi",
        suite_config="tracking",
        task_profile=task,
    ).metadata()["source"] == "explicit"
    assert resolve_objective(
        case_config="kpi",
        suite_config="safety",
        task_profile=task,
    ).name == "kpi"
    assert resolve_objective(
        suite_config="safety",
        task_profile=task,
    ).name == "safety"
    assert resolve_objective(task_profile=task).name == "tracking"
    with pytest.raises(ValueError, match="no objective was resolved"):
        resolve_objective()

    protocol = resolve_protocol(
        "quadruple",
        data={"task": "minimum-phase"},
    )
    assert protocol.objective == "tracking"
    assert protocol.objective_source == "task-default"

    task_env = aiogym.make_env(
        "quadruple", task="minimum-phase", episode_steps=1
    )
    assert task_env.reward_mode == "tracking"
    explicit_env = aiogym.make_env(
        "cstr",
        objective="tracking",
        config={"objective": "economic"},
        episode_steps=1,
    )
    assert explicit_env.reward_mode == "tracking"


def test_objective_selection_does_not_change_implicit_environment_conditions():
    fields = (
        "action_mode", "auto_events", "randomize", "randomize_setpoints",
        "randomize_plant", "plant_drift", "integral_obs",
        "disturbance_obs", "previous_action_obs", "normalize_observations",
        "tracking_error_obs",
        "terminate_on_runaway", "noise",
    )
    protocols = [
        aiogym.BenchmarkProtocol.tracking("cstr"),
        aiogym.BenchmarkProtocol.economic("cstr"),
        aiogym.BenchmarkProtocol.kpi("cstr"),
        aiogym.BenchmarkProtocol.robustness("cstr"),
        aiogym.BenchmarkProtocol.safety("cstr"),
    ]
    conditions = [{field: getattr(protocol, field) for field in fields} for protocol in protocols]
    assert conditions == [conditions[0]] * len(conditions)
    assert {protocol.env_reward_mode for protocol in protocols} == {
        "tracking", "economic", "kpi"
    }


@pytest.mark.parametrize(
    ("objective", "reward_mode"),
    [
        ("economic", "economic"),
        ("tracking", "tracking"),
        ("kpi", "kpi"),
        ("robustness", "kpi"),
        ("safety", "kpi"),
    ],
)
def test_public_objectives_resolve_one_internal_reward_mode(objective, reward_mode):
    from aiogym.evaluation import resolve_protocol

    assert aiogym.reward_mode_for_objective(objective) == reward_mode

    protocol = resolve_protocol(
        "cstr", objective, {"episode_steps": 1}
    )
    assert protocol.objective == objective
    assert protocol.env_reward_mode == reward_mode
    assert protocol.metadata()["resolved_reward_mode"] == reward_mode
    assert protocol.metadata()["objective_spec"]["resolved_reward_mode"] == reward_mode


def test_high_level_reward_mode_option_is_rejected():
    safety_env = aiogym.make_env("cstr", objective="safety", episode_steps=1)
    assert safety_env.reward_mode == "kpi"

    with pytest.raises(ValueError, match="unsupported environment option.*reward_mode"):
        aiogym.make_env(
            "cstr",
            objective="tracking",
            reward_mode="economic",
            episode_steps=1,
        )


def test_training_cli_objective_resolution_and_metadata():
    from aiogym.rl.train_rlpd import (
        configure_training_objective as configure_rlpd,
        configure_training_task as configure_rlpd_task,
    )
    from aiogym.rl.train_sb3 import (
        configure_training_auto_events,
        configure_training_objective as configure_sb3,
        configure_training_task,
        training_metadata,
    )

    sb3 = SimpleNamespace(objective="safety", reward_mode=None, eval_objective=None)
    configure_sb3(sb3)
    assert sb3.objective == "safety"
    assert sb3.resolved_reward_mode == "kpi"
    assert sb3.eval_objective == "safety"

    task_objective = SimpleNamespace(
        objective=None,
        reward_mode=None,
        eval_objective=None,
        scenario="quadruple",
        task="minimum-phase",
    )
    configure_sb3(task_objective)
    assert task_objective.objective == "tracking"
    assert task_objective.resolved_reward_mode == "tracking"
    assert task_objective.eval_objective == "tracking"

    with pytest.raises(ValueError, match="reward_mode is not supported"):
        configure_rlpd(SimpleNamespace(objective=None, reward_mode="economic"))
    with pytest.raises(ValueError, match="dynamic is not supported"):
        configure_training_auto_events(SimpleNamespace(auto_events=None, dynamic=True))

    task_args = SimpleNamespace(
        scenario="quadruple",
        task="minimum-phase",
        objective="tracking",
        action_mode="actuator",
        control_dt=None,
        train_episode_steps=None,
        eval_episode_steps=None,
        tracking_q_y=None,
        tracking_r_move=None,
        disturbance_obs=None,
        previous_action_obs=None,
        normalize_observations=None,
        tracking_error_obs=None,
    )
    configure_training_task(task_args)
    assert task_args.control_dt == pytest.approx(1.0)
    assert task_args.train_episode_steps == 600
    assert task_args.eval_episode_steps == 600
    assert task_args.tracking_q_y == pytest.approx(0.7)
    assert task_args.tracking_r_move == pytest.approx(0.3)
    assert task_args.disturbance_obs is False
    assert task_args.previous_action_obs is True
    assert task_args.normalize_observations is True
    assert task_args.tracking_error_obs is True

    rlpd_task_args = SimpleNamespace(
        scenario="quadruple",
        task="minimum-phase",
        objective="tracking",
        action_mode="setpoint",
        control_dt=None,
        episode_steps=None,
        tracking_q_y=None,
        tracking_r_move=None,
        disturbance_obs=None,
        previous_action_obs=None,
        normalize_observations=None,
        tracking_error_obs=None,
    )
    configure_rlpd_task(
        rlpd_task_args,
        episode_steps_attr="episode_steps",
        eval_episode_steps_attr=None,
    )
    assert rlpd_task_args.control_dt == pytest.approx(task_args.control_dt)
    assert rlpd_task_args.episode_steps == task_args.train_episode_steps
    assert rlpd_task_args.tracking_q_y == pytest.approx(task_args.tracking_q_y)
    assert rlpd_task_args.tracking_r_move == pytest.approx(task_args.tracking_r_move)
    assert rlpd_task_args.previous_action_obs is task_args.previous_action_obs
    assert (
        rlpd_task_args.normalize_observations
        is task_args.normalize_observations
    )

    metadata_args = SimpleNamespace(
        algo="sac",
        scenario="cstr",
        task=None,
        action_mode="actuator",
        objective="robustness",
        resolved_reward_mode="kpi",
        steps=10,
        seed=3,
        n_envs=1,
        vec_env="dummy",
        subproc_start_method="fork",
        device="cpu",
        torch_threads=1,
        train_episode_steps=10,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
        randomize_plant=False,
        plant_drift=False,
        integral_obs=False,
        disturbance_obs=False,
        previous_action_obs=True,
        normalize_observations=True,
        tracking_error_obs=True,
        terminate_on_runaway=False,
        noise=False,
        noise_pct=0.01,
        control_dt=0.5,
        tracking_q_y=0.7,
        tracking_r_move=0.3,
        gamma=0.999,
        learning_rate=1e-4,
        batch_size=256,
        buffer_size=500000,
        learning_starts=10000,
        train_freq=1,
        gradient_steps=4,
        learning_curve_every=10000,
        learning_curve_episodes=1,
        save_rollout=True,
    )
    metadata = training_metadata(metadata_args, "checkpoint.zip")
    assert metadata["objective"] == "robustness"
    assert metadata["resolved_reward_mode"] == "kpi"
    assert metadata["tracking_q_y"] == pytest.approx(0.7)
    assert metadata["tracking_r_move"] == pytest.approx(0.3)
    assert metadata["disturbance_obs"] is False
    assert metadata["previous_action_obs"] is True
    assert metadata["normalize_observations"] is True
    assert metadata["tracking_error_obs"] is True
    assert metadata["gamma"] == pytest.approx(0.999)
    assert metadata["gradient_steps"] == 4
    assert metadata["learning_curve_every"] == 10000
    assert metadata["save_rollout"] is True
    assert "reward_mode" not in metadata
    assert metadata["env_kwargs"]["auto_events"] is False
    assert metadata["env_kwargs"]["disturbance_obs"] is False
    assert metadata["env_kwargs"]["previous_action_obs"] is True
    assert metadata["env_kwargs"]["normalize_observations"] is True
    assert metadata["env_kwargs"]["tracking_error_obs"] is True
    assert "dynamic" not in metadata["env_kwargs"]


def test_sb3_training_env_receives_task_and_tracking_weights():
    from aiogym.rl.train_sb3 import make_training_env

    args = SimpleNamespace(
        scenario="quadruple",
        task="minimum-phase",
        resolved_reward_mode="tracking",
        action_mode="actuator",
        control_dt=1.0,
        train_episode_steps=600,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
        randomize_plant=False,
        plant_drift=False,
        integral_obs=False,
        disturbance_obs=False,
        previous_action_obs=True,
        normalize_observations=True,
        tracking_error_obs=True,
        terminate_on_runaway=False,
        noise=False,
        noise_pct=0.01,
        tracking_q_y=0.7,
        tracking_r_move=0.3,
        seed=1000,
    )
    env = make_training_env(args)()
    try:
        assert env.task_profile["name"] == "minimum-phase"
        assert env.control_dt == pytest.approx(1.0)
        assert env.episode_steps == 600
        assert env.tracking_q_y == pytest.approx([0.7, 0.7])
        assert env.tracking_r_move == pytest.approx(0.3)
        assert env.disturbance_obs is False
        assert env.previous_action_obs is True
        assert env.normalize_observations is True
        assert env.tracking_error_obs is True
        observation, _ = env.reset(seed=1000)
        assert observation.shape == (8,)
        assert observation.tolist() == pytest.approx([
            12.2629675195507 / 20.0,
            12.783158403008972 / 20.0,
            1.6339411322567796 / 20.0,
            1.409044702533737 / 20.0,
            0.0,
            0.0,
            0.3,
            0.3,
        ])
        next_observation, *_ = env.step([0.4, 0.2])
        assert next_observation[4] == pytest.approx(
            12.2629675195507 / 20.0 - next_observation[0],
            abs=1e-6,
        )
        assert next_observation[5] == pytest.approx(
            12.783158403008972 / 20.0 - next_observation[1],
            abs=1e-6,
        )
        assert next_observation[-2:].tolist() == pytest.approx([0.4, 0.2])
    finally:
        env.close()


def test_protocol_preserves_markov_normalized_observation_contract():
    protocol = aiogym.BenchmarkProtocol.tracking(
        "quadruple",
        task="minimum-phase",
        disturbance_obs=False,
        previous_action_obs=True,
        normalize_observations=True,
        tracking_error_obs=True,
    )

    assert protocol.metadata()["disturbance_obs"] is False
    assert protocol.metadata()["previous_action_obs"] is True
    assert protocol.metadata()["normalize_observations"] is True
    assert protocol.metadata()["tracking_error_obs"] is True
    environment = protocol.environment_spec()
    assert environment.disturbance_obs is False
    assert environment.previous_action_obs is True
    assert environment.normalize_observations is True
    assert environment.tracking_error_obs is True
    env = protocol.make_env()
    try:
        observation, _ = env.reset(seed=1000)
        assert observation.shape == (8,)
    finally:
        env.close()

    with pytest.raises(TypeError, match="previous_action_obs must be a boolean"):
        aiogym.BenchmarkProtocol.tracking(
            "quadruple",
            previous_action_obs="yes",
        )


def test_learning_curve_best_checkpoint_comparison_respects_metric_direction():
    from aiogym.rl.train_sb3 import _learning_curve_point_is_better

    assert _learning_curve_point_is_better(
        {"metric_value": 0.08, "metric_direction": "minimize"},
        None,
    )
    assert _learning_curve_point_is_better(
        {"metric_value": 0.07, "metric_direction": "minimize"},
        0.08,
    )
    assert not _learning_curve_point_is_better(
        {"metric_value": 0.09, "metric_direction": "minimize"},
        0.08,
    )
    assert _learning_curve_point_is_better(
        {"metric_value": 12.0, "metric_direction": "maximize"},
        10.0,
    )


def test_benchmark_case_supports_injected_environment_factory():
    from aiogym.evaluation import BenchmarkCase
    from aiogym.evaluation.execution import run_evaluation_case

    protocol = aiogym.BenchmarkProtocol.tracking(
        "cstr", episode_steps=1
    )
    case = BenchmarkCase.from_protocol(
        protocol,
        controller="pid",
        seeds=[3],
        case_id="test:cstr:tracking:pid",
    )
    captured = []

    def env_factory(environment, objective):
        captured.append((environment, objective))
        return environment.make_env(objective)

    artifact = run_evaluation_case(
        case=case,
        env_factory=env_factory,
        save_rollout=True,
        rollout_steps=1,
    )
    assert captured[0][0].scenario == "cstr"
    assert captured[0][1].name == "tracking"
    assert artifact["case"]["case_id"] == "test:cstr:tracking:pid"
    assert artifact["result"]["execution_status"] == "passed"
    assert artifact["result"]["objective_status"] == "not-defined"
    assert artifact["rollout"]["scenario"] == "cstr"
    assert artifact["rollout"]["task"] == "default"
    assert artifact["rollout"]["objective"] == "tracking"


def test_task_acceptance_is_separate_from_execution_status():
    from aiogym.evaluation import evaluate_task_acceptance

    task = {
        "schema_version": "aiogym.task_profile.v1",
        "name": "acceptance-test",
        "scenario": "cstr",
        "status": "test-only",
        "environment": {},
        "acceptance": {
            "metrics": {
                "tracking_error_cost": {"max": 1.0},
                "normalized_score": {"min": 90.0},
            }
        },
    }
    assert evaluate_task_acceptance(
        task,
        {"tracking_error_cost": 0.5, "normalized_score": 95.0},
    )["status"] == "met"
    assert evaluate_task_acceptance(
        task,
        {"tracking_error_cost": 2.0, "normalized_score": 95.0},
    )["status"] == "not-met"
