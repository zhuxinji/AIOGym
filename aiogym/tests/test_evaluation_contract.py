"""Controller evaluation, metrics, artifacts, and oracle tests."""
import pytest

from aiogym.tests.interface_support import *  # noqa: F403

def test_controller_evaluation_protocol():
    """PID/MPC/oracle-style controllers and learned policies share one evaluator."""
    fixed = BenchmarkProtocol.tracking("cstr", action_mode="actuator", episode_steps=8,
                                       auto_events=False, randomize=False,
                                       randomize_setpoints=False, randomize_plant=False,
                                       plant_drift=False)
    names = set(list_controllers())
    registry_ok = {"pid", "mpc", "oracle", "policy", "sb3", "onnx"}.issubset(names)
    expected_keys = {"name", "action_mode", "control_structure", "parameters", "scenarios"}
    controller_scenarios = {
        name: set(load_controller_config(name).get("scenarios", {}))
        for name in ("pid", "mpc", "oracle", "sb3", "onnx")
    }
    expected_config_scenarios = {
        "pid": controller_scenarios["pid"],
        "mpc": controller_scenarios["mpc"],
        "oracle": controller_scenarios["oracle"],
        "sb3": controller_scenarios["sb3"],
        "onnx": controller_scenarios["onnx"],
    }
    config_shape_ok = True
    for ctl in ("pid", "mpc", "oracle", "sb3", "onnx"):
        cfg = load_controller_config(ctl, "cstr")
        raw_scenarios = load_controller_config(ctl).get("scenarios", {})
        config_shape_ok = config_shape_ok and expected_keys.issubset(cfg)
        config_shape_ok = config_shape_ok and set(raw_scenarios) == expected_config_scenarios[ctl]
    mpc_defaults = load_controller_config("mpc", "cstr")
    pid_defaults = load_controller_config("pid", "cstr")
    extraction_pid_defaults = load_controller_config("pid", "extraction")
    pid_controller = make_controller("pid", scenario="cstr")
    mpc_controller = make_controller("mpc", scenario="cstr", config={"P": 4})
    oracle_controller = make_controller(
        "oracle",
        scenario="cstr",
        config={"objective": "economic", "horizon": 4},
    )
    extraction_pid_controller = make_controller("pid", scenario="extraction")
    registry_ok = registry_ok and config_shape_ok
    registry_ok = registry_ok and pid_defaults["parameters"]["loops"][0]["pid"] == [0.08, 0.02, 0.0]
    registry_ok = registry_ok and extraction_pid_defaults["parameters"]["loops"][0]["pid"] == [0.0, 0.25, 0.0]
    registry_ok = registry_ok and extraction_pid_defaults["parameters"]["loops"][0]["y_index"] == 4
    registry_ok = registry_ok and pid_controller.metadata()["api"] == "aiogym.controller.v1"
    registry_ok = registry_ok and pid_controller.__class__.__name__ == "PIDAgent"
    registry_ok = registry_ok and pid_controller.metadata()["loops"][0]["pid"] == [0.08, 0.02, 0.0]
    registry_ok = registry_ok and extraction_pid_controller.metadata()["loops"][0]["y_index"] == 4
    registry_ok = registry_ok and mpc_defaults["parameters"]["P"] == 40
    registry_ok = registry_ok and mpc_controller.metadata()["control_structure"] == "fixed_sp_mpc"
    registry_ok = registry_ok and mpc_controller.__class__.__name__ == "MPCAgent"
    registry_ok = registry_ok and mpc_controller.metadata()["horizon"] == 4
    registry_ok = registry_ok and oracle_controller.metadata()["control_structure"] == "nmpc_oracle"
    registry_ok = registry_ok and oracle_controller.__class__.__name__ == "OracleAgent"
    registry_ok = registry_ok and oracle_controller.metadata()["horizon"] == 4
    check("controller registry builds built-ins", registry_ok)

    pid = evaluate_controller(pid_controller, fixed.make_env(), episodes=2,
                              seed=123, protocol=fixed, include_episodes=True)
    fixed_ok = (
        pid["controller_name"] == "PID"
        and pid["episodes"] == 2
        and pid["seed_list"] == [123, 124]
        and pid["schema_version"] == "aiogym.evaluation.v4"
        and pid["protocol"]["objective"] == "tracking"
        and pid["controller"]["api"] == "aiogym.controller.v1"
        and pid["controller"]["control_structure"] == "fixed_sp_pid"
        and "tracking_iae" in pid
        and pid["tracking_raw_by_output"]
        and pid["tracking_raw_by_output_std"]
        and pid["episode_metrics"][0]["tracking_raw_by_output"]
        and "energy_kwh" in pid
        and "constraint_violation_count" in pid
        and "metric_definitions" in pid
        and "result_schema" in pid
        and pid["reproducibility"]["seed_list"] == [123, 124]
        and pid["episode_metrics"][0]["seed"] == 123
        and "return_std" in pid
    )
    check("actuator controller evaluation includes reproducibility metadata", fixed_ok)

    extraction_protocol = BenchmarkProtocol.tracking("extraction", action_mode="actuator", episode_steps=5,
                                                     auto_events=False, randomize=False,
                                                     randomize_setpoints=False, randomize_plant=False,
                                                     plant_drift=False)
    extraction = evaluate_controller(
        extraction_pid_controller,
        extraction_protocol.make_env(),
        seed_list=[22],
        protocol=extraction_protocol,
    )
    extraction_ok = (
        extraction["controller_name"] == "PID"
        and extraction["controller"]["scenario"] == "extraction"
        and extraction["episodes"] == 1
        and extraction["constraint_violation_count"] == 0.0
        and np.isfinite(extraction["normalized_score"])
    )
    check("extraction PID baseline evaluates through controller registry", extraction_ok)

    sup = BenchmarkProtocol.economic("cstr", action_mode="setpoint", episode_steps=8,
                                     auto_events=False, randomize=False,
                                     randomize_setpoints=False, randomize_plant=False,
                                     plant_drift=False)
    env = sup.make_env()
    default_action = env.default_sp_action()

    class ConstantPolicy:
        name = "constant-setpoint-policy"

        def act(self, obs, deterministic=True):
            return default_action

    policy_controller = as_controller(ConstantPolicy(), action_mode="setpoint",
                                      control_structure="constant_setpoint_policy")
    pol = evaluate_controller(policy_controller, env, episodes=2, seed=321, protocol=sup)
    policy_ok = (
        pol["controller_name"] == "constant-setpoint-policy"
        and pol["metric"] == "profit"
        and pol["protocol"]["action_mode"] == "setpoint"
        and pol["controller"]["api"] == "aiogym.controller.v1"
        and pol["controller"]["control_structure"] == "constant_setpoint_policy"
        and np.isfinite(pol["profit"])
    )
    check("policy evaluation uses the same rollout path", policy_ok)

    class DegradedPolicy:
        name = "diagnostic-setpoint-policy"

        def act(self, obs, deterministic=True):
            return default_action

        def diagnostics(self):
            return {
                "solve_count": 1,
                "solver_success_count": 0,
                "solver_failure_count": 1,
                "fallback_count": 1,
                "degraded": True,
                "last_solver_error": "synthetic failure",
            }

    diagnostic_controller = as_controller(DegradedPolicy(), action_mode="setpoint",
                                          control_structure="diagnostic_setpoint_policy")
    diag = evaluate_controller(diagnostic_controller, sup.make_env(), episodes=1, seed=322,
                               protocol=sup, include_episodes=True)
    diag_ok = (
        diag["controller_status"] == "degraded"
        and diag["controller_solver_failure_count"] == 1.0
        and diag["controller_fallback_count"] == 1.0
        and diag["controller_diagnostics"]["solver_failure_count"] == 1
        and diag["controller_diagnostics"]["last_solver_error"] == "synthetic failure"
        and diag["episode_metrics"][0]["controller_degraded_count"] == 1.0
    )
    check("controller diagnostics mark degraded evaluations", diag_ok)


def test_generic_controller_rollout():
    """The common rollout recorder should work across built-in process models."""
    rollout_ok = True
    for scn in load_controller_config("pid").get("scenarios", {}):
        protocol = BenchmarkProtocol.tracking(
            scn,
            action_mode="actuator",
            episode_steps=4,
            auto_events=False,
            randomize=False,
            randomize_setpoints=False,
            randomize_plant=False,
            plant_drift=False,
        )
        env = protocol.make_env()
        artifact = rollout_controller(make_controller("pid", scenario=scn), env, seed=77, protocol=protocol)
        first = artifact["rollout"][0]
        rollout_ok = rollout_ok and artifact["protocol"]["scenario"] == protocol.scenario
        rollout_ok = rollout_ok and artifact["steps"] == 4
        rollout_ok = rollout_ok and artifact["rollout_schema"]["disturbance"]
        rollout_ok = rollout_ok and len(first["action"]) == env.action_space.shape[0]
        rollout_ok = rollout_ok and {"obs", "state", "setpoint", "measurement", "disturbance", "profit", "constraint", "info"}.issubset(first)
    check("generic rollout recorder works across scenarios", rollout_ok)


def test_benchmark_config_and_report_schema():
    """Roadmap phase-3 evaluation protocols stay separated and reproducible."""
    protocols = {
        "tracking": BenchmarkProtocol.tracking("cstr", episode_steps=3, auto_events=False,
                                               randomize=False, randomize_setpoints=False,
                                               randomize_plant=False, plant_drift=False),
        "economic": BenchmarkProtocol.economic("cstr", episode_steps=3, auto_events=False,
                                               randomize=False, randomize_setpoints=False,
                                               randomize_plant=False, plant_drift=False),
        "robustness": BenchmarkProtocol.robustness("cstr", episode_steps=3),
        "safety": BenchmarkProtocol.safety("cstr", episode_steps=3),
    }
    tracking_metrics = protocols["tracking"].metadata()["metrics"]
    tracking_meta = protocols["tracking"].metadata()
    tracking_env_kwargs = protocols["tracking"].env_kwargs()
    protocol_ok = (
        tracking_metrics[0] == "tracking_cost"
        and all(key.startswith("tracking_") for key in tracking_metrics)
        and "energy_kwh" not in tracking_metrics
        and "constraint_violation_count" not in tracking_metrics
        and tracking_meta["primary_metric"] == "tracking_cost"
        and tracking_meta["primary_metric_direction"] == "minimize"
        and tracking_meta["env_reward_mode"] == "tracking"
        and "reward_mode" not in tracking_meta
        and tracking_env_kwargs["reward_mode"] == "tracking"
        and "env_reward_mode" not in tracking_env_kwargs
        and protocols["economic"].metadata()["metrics"][0] == "profit"
        and protocols["economic"].metadata()["primary_metric"] == "profit"
        and protocols["robustness"].metadata()["noise"] is False
        and protocols["safety"].metadata()["metrics"][0] == "constraint_violation_count"
    )
    config = BenchmarkCase.from_protocol(protocols["tracking"], controller="pid", seeds=[11, 12])
    config_meta = config.metadata()
    config_ok = (
        config_meta["objective"] == "tracking"
        and config_meta["scenario"] == "cstr"
        and config_meta["controller"] == "pid"
        and config_meta["seed_list"] == [11, 12]
        and config_meta["episode_steps"] == 3
        and config_meta["primary_metric"] == "tracking_cost"
        and config_meta["primary_metric_direction"] == "minimize"
        and config_meta["protocol"]["env_reward_mode"] == "tracking"
        and "reward_mode" not in config_meta["protocol"]
        and "tracking_cost" in config_meta["metric_definitions"]
    )

    result = evaluate_controller(
        make_controller("pid", scenario="cstr"),
        protocols["tracking"].make_env(),
        seed_list=[11, 12],
        protocol=protocols["tracking"],
        include_episodes=True,
    )
    report = build_evaluation_report([result])
    report_ok = (
        {"tracking", "economic", "safety", "robustness"}.issubset(report)
        and result["metric"] == "tracking_cost"
        and result["metric_direction"] == "minimize"
        and report["tracking"][0]["controller"] == "PID"
        and "tracking_cost" in report["tracking"][0]
        and "tracking_mse" in report["tracking"][0]
        and "tracking_iae" in report["tracking"][0]
        and "profit" in report["economic"][0]
        and "constraint_violation_count" in report["safety"][0]
        and "return_std" in report["robustness"][0]
    )
    check("BenchmarkCase + separated evaluation report schema", protocol_ok and config_ok and report_ok)


def test_kpi_tracking_setpoint_alignment():
    """KPI and tracking metrics should use the same active setpoint semantics."""
    from aiogym.evaluation.metrics.tracking import (
        tracking_step_metrics as _tracking_step_metrics,
    )
    from aiogym.evaluation.metrics.kpi import W_TRACKING

    env = AIOGymNativeEnv("heater", reward_mode="tracking", action_mode="actuator",
                          auto_events=False, randomize=False, randomize_setpoints=False,
                          episode_steps=1)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.full(env.action_space.shape[0], 0.5, np.float32))
    raw_level_err = abs(info["y"][0] - env.y_sp[0])
    raw_temp_err = abs(info["y"][1] - env.y_sp[1])
    normalized_level_err = raw_level_err / (5.0 - 1.8)
    normalized_temp_err = raw_temp_err / (372.0 - 364.0)
    expected_error_cost = 0.7 * (
        normalized_level_err ** 2 + normalized_temp_err ** 2
    )
    expected_cost = expected_error_cost + info["tracking_move_cost"]
    mean_output_err = 0.5 * (normalized_temp_err + normalized_level_err)
    report = env.scorer.report()
    tracking = _tracking_step_metrics(info, {"y_sp": env.y_sp}, 0.0, env.control_dt, env)
    raw_by_output = tracking["tracking_raw_by_output"]
    scaled_ok = (
        abs(report["comp_tracking"] - W_TRACKING * mean_output_err) < 1e-9
        and abs(
            tracking["tracking_iae"]
            - (normalized_temp_err + normalized_level_err) * env.control_dt
        ) < 1e-9
        and abs(info["tracking_error_cost"] - expected_error_cost) < 1e-9
        and abs(tracking["tracking_cost"] - expected_cost) < 1e-9
        and abs(info["tracking_cost"] - expected_cost) < 1e-9
        and abs(reward + info["tracking_cost"]) < 1e-9
        and abs(info["tracking_return"] + info["tracking_cost"]) < 1e-9
        and abs(
            tracking["tracking_mse"]
            - 0.5
            * (normalized_level_err ** 2 + normalized_temp_err ** 2)
            * env.control_dt
        ) < 1e-9
        and abs(info["track"] - (normalized_temp_err + normalized_level_err)) < 1e-9
        and raw_by_output["flue_o2"]["unit"] == "%"
        and raw_by_output["outlet_temperature"]["unit"] == "degC"
        and abs(
            raw_by_output["flue_o2"]["iae"]
            - raw_level_err * env.control_dt
        ) < 1e-9
        and abs(
            raw_by_output["outlet_temperature"]["iae"]
            - raw_temp_err * env.control_dt
        ) < 1e-9
    )

    class HighSetpointPolicy:
        name = "high-setpoint-policy"

        def act(self, obs, deterministic=True):
            return np.array([1.0, 1.0], dtype=np.float32)

    protocol = BenchmarkProtocol.economic("heater", action_mode="setpoint", episode_steps=1,
                                          auto_events=False, randomize=False,
                                          randomize_setpoints=False, randomize_plant=False,
                                          plant_drift=False)
    result = evaluate_controller(
        as_controller(HighSetpointPolicy(), action_mode="setpoint"),
        protocol.make_env(),
        seed_list=[7],
        protocol=protocol,
        include_episodes=True,
    )
    replay = protocol.make_env()
    replay.reset(seed=7)
    _, _, _, _, replay_info = replay.step(np.array([1.0, 1.0], dtype=np.float32))
    post = _tracking_step_metrics(
        replay_info,
        {"y_sp": replay.y_sp},
        0.0,
        replay.control_dt,
        replay,
    )
    pre = _tracking_step_metrics(
        replay_info,
        {"y_sp": [3.0, 370.0]},
        0.0,
        replay.control_dt,
        replay,
    )
    setpoint_ok = (
        abs(result["episode_metrics"][0]["tracking_iae"] - post["tracking_iae"]) < 1e-9
        and abs(post["tracking_iae"] - pre["tracking_iae"]) > 1e-6
    )
    check("KPI and tracking metrics use active scaled setpoints", scaled_ok and setpoint_ok)


def test_pure_tracking_reward_mode():
    """Tracking reward should be pure SP error; safety/energy stay report-only."""
    from aiogym.evaluation import metric_for_reward_mode, primary_metric_for_objective

    env = AIOGymNativeEnv("hvac", reward_mode="tracking", action_mode="actuator",
                          auto_events=False, randomize=False, randomize_setpoints=False,
                          episode_steps=1)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(env.action_space.shape[0], np.float32))
    pure_reward_ok = abs(reward + info["tracking_cost"]) < 1e-9
    diagnostics_ok = (
        "constraint" in info
        and "energy_kw" in info
        and "heat_kw" not in info
        and "pump_kw" not in info
        and info["constraint"] >= 0.0
    )
    metric_ok = metric_for_reward_mode("tracking") == "return" and primary_metric_for_objective("tracking") == "tracking_cost"
    check("Pure SP tracking reward excludes energy and constraints", pure_reward_ok and diagnostics_ok and metric_ok)


def test_setpoint_randomization_uses_model_bounds():
    """Setpoint moves should honor model-specific SP bounds such as heater O2."""
    env = AIOGymNativeEnv("heater", auto_events=False, randomize=False,
                          randomize_setpoints=True, episode_steps=1)
    reset_bounds_ok = True
    move_bounds_ok = True
    for seed in range(8):
        env.reset(seed=seed)
        reset_bounds_ok = reset_bounds_ok and 1.8 <= env.y_sp[0] <= 5.0
        reset_bounds_ok = reset_bounds_ok and 364.0 <= env.y_sp[1] <= 372.0
        env._apply_disturbance("setpoint_move")
        move_bounds_ok = move_bounds_ok and 1.8 <= env.y_sp[0] <= 5.0
        move_bounds_ok = move_bounds_ok and 364.0 <= env.y_sp[1] <= 372.0

    legacy_water_range_ok = True
    tank = AIOGymNativeEnv("cascade", auto_events=False, randomize=False,
                           randomize_setpoints=True, episode_steps=1)
    tank.reset(seed=3)
    for j in range(len(tank.model.economic_config.get("level_band", []))):
        legacy_water_range_ok = legacy_water_range_ok and 0.15 <= tank.y_sp[j] <= 0.70
    check("setpoint randomization uses model-specific bounds", reset_bounds_ok and move_bounds_ok and legacy_water_range_ok)


def test_seeded_setpoint_randomization_covers_all_builtin_sp_models():
    for scenario in aiogym.list_scenarios():
        env = AIOGymNativeEnv(
            scenario,
            reward_mode="tracking",
            action_mode="actuator",
            auto_events=False,
            randomize=False,
            randomize_setpoints=True,
            episode_steps=1,
        )
        env.reset(seed=9000)
        first = list(env.y_sp)
        env.reset(seed=9001)
        second = list(env.y_sp)
        env.reset(seed=9000)
        repeated = list(env.y_sp)

        assert first == pytest.approx(repeated)
        assert first != pytest.approx(second)
        for value, row in zip(first, env.model.setpoint_schema()):
            bounds = row.get("bounds")
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                continue
            lo, hi = bounds
            assert lo is None or value >= float(lo)
            assert hi is None or value <= float(hi)


def test_benchmark_suite_configs():
    """Named benchmark suites are data configs, not hidden script constants."""
    from aiogym.evaluation import plot_results
    from aiogym.cli.benchmark import load_suite
    from aiogym.evaluation.suite import (
        SUMMARY_COLUMNS,
        artifact_dir_for,
        build_summary_table,
        effective_suite_config,
    )
    from aiogym.evaluation.artifact import write_benchmark_artifacts

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cfg_dir = os.path.join(root, "aiogym", "evaluation", "suites")
    required = {
        "cascade-control.json",
        "cascade-recirculating.json",
        "crystallization-tracking.json",
        "economic-actuator.json",
        "quadruple.json",
        "quadruple-phase-comparison.json",
        "robustness-actuator.json",
        "standard-baselines.json",
        "tracking-actuator.json",
    }
    present = {name for name in os.listdir(cfg_dir) if name.endswith(".json")}
    configs_ok = required == present
    for name in required:
        cfg = load_suite(os.path.splitext(name)[0])
        configs_ok = configs_ok and {"description", "scenarios", "controllers", "action_mode"}.issubset(cfg)
        configs_ok = configs_ok and cfg["action_mode"] in {"actuator", "setpoint"}
        configs_ok = configs_ok and bool(cfg["controllers"])
        configs_ok = configs_ok and bool(cfg.get("objectives") or cfg.get("cases"))
    standard = load_suite("standard-baselines")
    configs_ok = configs_ok and set(standard["scenarios"]) == set(aiogym.list_scenarios())
    configs_ok = configs_ok and standard["objectives"] == ["tracking", "economic"]
    configs_ok = configs_ok and standard["controllers"] == ["pid", "mpc", "oracle"]
    crystal = load_suite("crystallization-tracking")
    configs_ok = configs_ok and crystal["scenarios"] == ["crystallization"]
    configs_ok = configs_ok and crystal["controllers"] == ["pid", "mpc", "oracle"]
    configs_ok = configs_ok and crystal["episode_steps"] == 30
    configs_ok = configs_ok and crystal["control_dt"] == 1.0
    summary = build_summary_table([{
        "suite_case": "tracking:cstr:pid",
        "scenario": "cstr",
        "objective": "tracking",
        "action_mode": "actuator",
        "controller": "PID",
        "control_structure": "fixed_sp_pid",
        "execution_status": "passed",
            "metric": "tracking_error_cost",
        "normalized_score": 90.0,
        "normalized_score_std": 1.5,
        "profit": 0.0,
        "return": -1.0,
        "return_std": 0.2,
        "track": 1.0,
            "tracking_cost": 0.5,
            "tracking_error_cost": 0.5,
            "tracking_error_cost_std": 0.1,
        "tracking_cost_std": 0.1,
        "tracking_return": -0.5,
        "tracking_mse": 0.5,
        "tracking_mse_std": 0.1,
        "tracking_iae": 1.0,
        "tracking_iae_std": 0.3,
        "energy_kwh": 0.1,
        "constraint": 0.0,
        "constraint_violation_count": 0.0,
        "constraint_violation_severity": 0.0,
        "safety_margin_min": 0.0,
        "runtime_seconds_per_step": 0.001,
        "controller_fallback_count": 0,
        "controller_solver_failure_count": 0,
        "episodes": 2,
        "seed_list": [11, 12],
    }])
    summary_ok = (
        set(SUMMARY_COLUMNS).issubset(summary[0])
        and summary[0]["metric_mean"] == 0.5
        and summary[0]["metric_std"] == 0.1
        and summary[0]["seed_list"] == [11, 12]
        and artifact_dir_for("example") == "runs/example_suite"
        and artifact_dir_for(
            "example",
            overwrite=False,
            run_id="20260708T120000000000Z",
        )
        == "runs/example_suite_20260708T120000000000Z"
        and artifact_dir_for("example", artifact_dir="custom/out") == "custom/out"
    )
    effective = effective_suite_config(standard, [
        {"scenario": "cstr", "objective": "tracking", "controller": "pid"},
        {"scenario": "hvac", "objective": "tracking", "controller": "pid"},
    ], episode_steps=4, control_dt=0.5)
    summary_ok = summary_ok and effective["scenarios"] == ["cstr", "hvac"]
    summary_ok = summary_ok and effective["objectives"] == ["tracking"]
    summary_ok = summary_ok and effective["controllers"] == ["pid"]

    with tempfile.TemporaryDirectory() as tmpdir:
        payload = {
            "benchmark": "benchmark_suite",
            "suite": "registry-smoke",
            "suite_config": {
                "name": "registry-smoke",
                "scenarios": ["cstr", "hvac"],
                "objectives": ["tracking", "economic"],
                "controllers": ["pid"],
                "action_mode": "actuator",
            },
            "configs": [{"scenario": "cstr"}, {"scenario": "hvac"}],
            "rows": [
                {
                    "suite_case": "tracking:cstr:pid",
                    "scenario": "cstr",
                    "objective": "tracking",
                    "action_mode": "actuator",
                    "controller": "PID",
                    "execution_status": "passed",
                        "metric": "tracking_error_cost",
                        "tracking_cost": 0.5,
                        "tracking_error_cost": 0.5,
                    "tracking_return": -0.5,
                    "tracking_mse": 0.5,
                    "tracking_iae": 1.0,
                    "normalized_score": 90.0,
                    "profit": 0.0,
                    "return": -1.0,
                    "track": 1.0,
                    "constraint": 0.0,
                    "runtime_total_seconds": 0.25,
                    "episodes": 1,
                    "seed_list": [11],
                },
                {
                    "suite_case": "tracking:hvac:pid",
                    "scenario": "hvac",
                    "objective": "tracking",
                    "action_mode": "actuator",
                    "controller": "PID",
                    "execution_status": "passed",
                        "metric": "tracking_error_cost",
                        "tracking_cost": 1.5,
                        "tracking_error_cost": 1.5,
                    "tracking_return": -1.5,
                    "tracking_mse": 1.5,
                    "tracking_iae": 2.0,
                    "normalized_score": 80.0,
                    "profit": 0.0,
                    "return": -2.0,
                    "track": 2.0,
                    "constraint": 0.0,
                    "runtime_total_seconds": 0.5,
                    "episodes": 1,
                    "seed_list": [12],
                },
                {
                    "suite_case": "economic:cstr:pid",
                    "scenario": "cstr",
                    "objective": "economic",
                    "action_mode": "actuator",
                    "controller": "PID",
                    "execution_status": "passed",
                    "metric": "profit",
                    "tracking_cost": 1.5,
                    "tracking_return": -1.5,
                    "tracking_mse": 1.5,
                    "tracking_iae": 2.0,
                    "normalized_score": 75.0,
                    "profit": 12.0,
                    "production": 4.0,
                    "return": 12.0,
                    "track": 2.0,
                    "constraint": 0.0,
                    "episodes": 1,
                    "seed_list": [13],
                },
            ],
            "results": [],
            "report": {"tracking": [{"metric": "tracking_cost", "controller": "PID"}]},
            "training": {"algo": "smoke", "total_timesteps": 2},
            "learning_curve": [
                {"step": 0, "phase": "start", "metric": "tracking_cost", "metric_value": 3.0, "tracking_cost": 3.0, "tracking_mse": 3.0, "tracking_iae": 3.0},
                {"step": 2, "phase": "final", "metric": "tracking_cost", "metric_value": 1.0, "tracking_cost": 1.0, "tracking_mse": 1.0, "tracking_iae": 1.0},
            ],
            "errors": [],
        }
        artifacts = write_benchmark_artifacts(tmpdir, payload)
        payload["artifacts"] = artifacts
        with open(os.path.join(tmpdir, "benchmark.json"), "w") as f:
            json.dump(payload, f, indent=2)
        figures = plot_results(tmpdir)
        report_path = os.path.join(tmpdir, "report.md")
        report_text = aiogym.render_benchmark_report(tmpdir, out_path=report_path)
        suite_artifacts_ok = True
        for key in ("benchmark", "input_config", "benchmark_config", "model_metadata_manifest",
                    "rows", "summary_csv", "tracking_comparison", "leaderboard", "all_summary_csv", "all_leaderboard", "results", "report",
                    "training", "learning_curve", "learning_curve_csv"):
            suite_artifacts_ok = suite_artifacts_ok and os.path.exists(artifacts[key])
        for key in ("learning_curve",):
            suite_artifacts_ok = suite_artifacts_ok and os.path.exists(figures[key])
        suite_artifacts_ok = suite_artifacts_ok and set(artifacts["summary_csvs"]) == {"tracking", "economic"}
        suite_artifacts_ok = suite_artifacts_ok and set(artifacts["leaderboards"]) == {"tracking", "economic"}
        for path in artifacts["summary_csvs"].values():
            suite_artifacts_ok = suite_artifacts_ok and os.path.exists(path)
        for path in artifacts["leaderboards"].values():
            suite_artifacts_ok = suite_artifacts_ok and os.path.exists(path)
        with open(artifacts["summary_csvs"]["tracking"]) as f:
            tracking_header = f.readline().strip().split(",")
        with open(artifacts["summary_csvs"]["economic"]) as f:
            economic_header = f.readline().strip().split(",")
        with open(artifacts["all_summary_csv"]) as f:
            all_header = f.readline().strip().split(",")
        with open(artifacts["tracking_comparison"]) as f:
            tracking_comparison_header = f.readline().strip().split(",")
            tracking_comparison_rows = f.read()
        suite_artifacts_ok = suite_artifacts_ok and "tracking_cost" in tracking_header and "tracking_mse" in tracking_header and "tracking_iae" in tracking_header and "profit" not in tracking_header
        suite_artifacts_ok = suite_artifacts_ok and "profit" in economic_header and "tracking_mse" not in economic_header
        suite_artifacts_ok = suite_artifacts_ok and {"profit", "tracking_cost", "tracking_mse", "tracking_iae"}.issubset(all_header)
        suite_artifacts_ok = suite_artifacts_ok and {"scenario", "best_controller", "best_tracking_cost", "best_runtime_total_seconds"}.issubset(set(tracking_comparison_header))
        suite_artifacts_ok = suite_artifacts_ok and "PID_tracking_cost" in tracking_comparison_header and "PID_runtime_total_seconds" in tracking_comparison_header
        suite_artifacts_ok = suite_artifacts_ok and "cstr" in tracking_comparison_rows and "hvac" in tracking_comparison_rows
        suite_artifacts_ok = suite_artifacts_ok and set(figures["summary_by_scenario"]) == {"tracking", "economic"}
        suite_artifacts_ok = suite_artifacts_ok and set(figures["summary_by_scenario"]["tracking"]) == {"cstr", "hvac"}
        suite_artifacts_ok = suite_artifacts_ok and os.path.exists(figures["leaderboard_by_scenario"])
        for paths_by_scenario in figures["summary_by_scenario"].values():
            for path in paths_by_scenario.values():
                suite_artifacts_ok = suite_artifacts_ok and os.path.exists(path)
        with open(figures["summary_by_scenario"]["tracking"]["cstr"]) as f:
            tracking_summary_svg = f.read()
        with open(figures["summary_by_scenario"]["economic"]["cstr"]) as f:
            economic_summary_svg = f.read()
        with open(figures["leaderboard_by_scenario"]) as f:
            leaderboard_svg = f.read()
        with open(figures["tracking_comparison"]) as f:
            tracking_comparison_svg = f.read()
        with open(figures["learning_curve"]) as f:
            learning_curve_svg = f.read()
        suite_artifacts_ok = suite_artifacts_ok and all(
            label in tracking_summary_svg
            for label in ("Tracking Cost", "Tracking Error Cost", "Move Cost")
        )
        suite_artifacts_ok = suite_artifacts_ok and "Normalized Tracking MSE" not in tracking_summary_svg
        suite_artifacts_ok = suite_artifacts_ok and "Normalized Tracking IAE" not in tracking_summary_svg and "Profit" not in tracking_summary_svg
        suite_artifacts_ok = suite_artifacts_ok and "Profit" in economic_summary_svg and "Tracking Cost" not in economic_summary_svg
        suite_artifacts_ok = suite_artifacts_ok and "cstr / tracking" in leaderboard_svg and "cstr / economic" in leaderboard_svg
        suite_artifacts_ok = suite_artifacts_ok and "tracking comparison" in tracking_comparison_svg
        suite_artifacts_ok = suite_artifacts_ok and "Normalized tracking cost" in tracking_comparison_svg
        suite_artifacts_ok = suite_artifacts_ok and "Total runtime" in tracking_comparison_svg
        suite_artifacts_ok = suite_artifacts_ok and "Oracle gap" in tracking_comparison_svg
        suite_artifacts_ok = suite_artifacts_ok and "tracking_cost" in learning_curve_svg
        suite_artifacts_ok = suite_artifacts_ok and "reward" not in learning_curve_svg
        suite_artifacts_ok = suite_artifacts_ok and "metric_value" not in learning_curve_svg
        suite_artifacts_ok = suite_artifacts_ok and ">0</text>" in learning_curve_svg
        suite_artifacts_ok = suite_artifacts_ok and os.path.exists(report_path)
        suite_artifacts_ok = suite_artifacts_ok and "AIO-Gym Benchmark Report" in report_text
        suite_artifacts_ok = suite_artifacts_ok and "Tracking Comparison" in report_text
        suite_artifacts_ok = suite_artifacts_ok and "| cstr |" in report_text and "| hvac |" in report_text
        suite_artifacts_ok = suite_artifacts_ok and "summary/leaderboard.json" in report_text
        suite_artifacts_ok = suite_artifacts_ok and "training/learning_curve.json" in report_text
        check_result = aiogym.check_benchmark_artifacts(tmpdir)
        suite_artifacts_ok = suite_artifacts_ok and check_result["ok"]
        with open(artifacts["model_metadata_manifest"]) as f:
            manifest = json.load(f)
        suite_artifacts_ok = suite_artifacts_ok and set(manifest["scenarios"]) == {"cstr", "hvac"}
        stale_card = os.path.join(tmpdir, "metadata", "models", "cascade.json")
        with open(stale_card, "w") as f:
            json.dump({"scenario": "cascade"}, f)
        stale_result = aiogym.check_benchmark_artifacts(tmpdir)
        suite_artifacts_ok = suite_artifacts_ok and not stale_result["ok"]
        suite_artifacts_ok = suite_artifacts_ok and any(row["name"] == "stale_model_metadata" for row in stale_result["failed"])
        single_payload = {
            "benchmark": "public_benchmark",
            "scenario": "cstr",
            "config": {},
            "benchmark_config": {},
            "rows": payload["rows"][:1],
            "results": [],
            "report": {},
        }
        single_artifacts = write_benchmark_artifacts(tmpdir, single_payload)
        stale_hvac_card = os.path.join(tmpdir, "metadata", "models", "hvac.json")
        suite_artifacts_ok = suite_artifacts_ok and os.path.exists(single_artifacts["model_metadata"])
        suite_artifacts_ok = suite_artifacts_ok and not os.path.exists(stale_hvac_card)

    check("benchmark suite configs declare controller permissions", configs_ok and summary_ok and suite_artifacts_ok)


def test_oracle():
    """NMPC oracle solves and beats PID on CSTR economic (it's the upper bound)."""
    try:
        from aiogym.controllers.oracle import OracleAgent
    except RuntimeError as ex:
        print(f"  (skip oracle: {ex})"); return
    from aiogym.evaluation import evaluate_controller
    from aiogym.controllers.pid import PIDAgent
    from aiogym.models import make_model
    mk = lambda: AIOGymNativeEnv("cstr", reward_mode="economic", episode_steps=120, auto_events=True, randomize_plant=True)
    orc = evaluate_controller(
        OracleAgent("cstr", horizon=12, objective="economic"),
        mk(),
        episodes=2,
    )["profit"]
    pid = evaluate_controller(PIDAgent(make_model("cstr")), mk(), episodes=2)["profit"]
    check(f"NMPC oracle {orc:.0f} > PID {pid:.0f}", orc > pid)
