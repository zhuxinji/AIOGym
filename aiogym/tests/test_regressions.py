"""Regression tests for public construction and benchmark edge cases."""
from __future__ import annotations

import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

import aiogym
from aiogym import make_env
from aiogym._internal.config import parse_seed_list
from aiogym.cli.benchmark import main as benchmark_main
from aiogym.cli.benchmark import run_case
from aiogym.evaluation.suite import controller_config_for
from aiogym.controllers import (
    ControllerContext,
    PolicyController,
    load_controller_config,
    make_controller,
    register_controller,
    unregister_controller,
    validate_action,
)
from aiogym.controllers.pid import PIDAgent
from aiogym.controllers.onnx import ONNXPolicyController
from aiogym.env import AIOGymNativeEnv
from aiogym.evaluation import (
    BenchmarkCase,
    BenchmarkProtocol,
    check_benchmark_artifacts,
    evaluate_controller,
    plot_results,
    resolve_protocol,
    run_benchmark,
    write_benchmark_artifacts,
)
from aiogym.evaluation.metrics.robustness import robustness_extrema
from aiogym.evaluation.protocols import METRIC_DIRECTIONS
from aiogym.models import (
    apply_model_params,
    define_model,
    make_model,
    register_model,
)
from aiogym.models.tasks import load_task_profile
from aiogym.rl.train_rlpd import output_base_for
from aiogym.rl.train_sb3 import run_name_for

SCENARIO_IDS = aiogym.list_scenarios()
list_controllers = aiogym.list_controllers


def test_skipped_rows_are_not_plotted():
    with tempfile.TemporaryDirectory() as tmpdir:
        benchmark = {
            "scenario": "cstr",
            "rows": [{
                "scenario": "cstr",
                "objective": "tracking",
                "controller": "missing",
                "execution_status": "skipped",
                "metric": "tracking_cost",
            }],
        }
        Path(tmpdir, "benchmark.json").write_text(json.dumps(benchmark))
        figures = plot_results(tmpdir)
        assert figures == {}
        assert not list(Path(tmpdir, "figures").glob("*.svg"))


def test_robustness_extrema_respect_metric_direction():
    extrema = robustness_extrema(
        [
            {"normalized_score": 80.0, "tracking_iae": 4.0, "energy_kwh": 2.0},
            {"normalized_score": 90.0, "tracking_iae": 2.0, "energy_kwh": 3.0},
        ],
        ("normalized_score", "tracking_iae", "energy_kwh"),
        METRIC_DIRECTIONS,
    )
    assert extrema["normalized_score_best"] == 90.0
    assert extrema["normalized_score_worst"] == 80.0
    assert extrema["tracking_iae_best"] == 2.0
    assert extrema["tracking_iae_worst"] == 4.0
    assert extrema["energy_kwh_best"] == 2.0
    assert extrema["energy_kwh_worst"] == 3.0


def test_make_env_preserves_non_protocol_overrides():
    env = make_env(
        scenario="cstr",
        objective="tracking",
        custom_stage_reward=lambda *_: 123.0,
        episode_steps=1,
        auto_events=False,
        randomize=False,
        randomize_setpoints=False,
    )
    env.reset(seed=0)
    _, reward, _, _, _ = env.step(np.array([0.5, 0.5], dtype=np.float32))
    assert callable(env.custom_stage_reward)
    assert reward == 123.0


def test_direct_model_instances_are_isolated_per_environment():
    shared = make_model("cstr")
    randomized = AIOGymNativeEnv(shared, randomize_plant=True)
    nominal = AIOGymNativeEnv(shared)

    randomized.reset(seed=1)
    assert randomized.model is not nominal.model
    assert randomized.model is not shared
    assert nominal.model is not shared
    assert randomized.model.p["Uc"] != nominal.model.p["Uc"]
    assert nominal.model.p["Uc"] == shared.p["Uc"] == 0.05


def test_model_parameter_overrides_follow_schema_bounds():
    invalid = (
        ("cstr", {"Dmax": float("nan")}, "finite"),
        ("cstr", {"Dmax": -1.0}, ">= 0.0"),
        ("cstr", {"Dmax": "fast"}, "numeric"),
        ("quadruple", {"outlet_area": [0.071]}, "must contain 4 values"),
        ("quadruple", {"outlet_area": [0.071, 0.057, 0.071, 100.0]}, "<= 10.0"),
    )
    for scenario, params, expected in invalid:
        try:
            apply_model_params(make_model(scenario), params)
        except (TypeError, ValueError) as ex:
            assert expected in str(ex)
        else:
            raise AssertionError(f"invalid {scenario} model parameters should be rejected")

    model = make_model("cstr")
    original = dict(model.p)
    try:
        apply_model_params(model, {"Dmax": 0.03, "Uc": float("inf")})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid parameter batches should fail transactionally")
    assert model.p == original


def test_registered_name_must_match_model_scenario():
    model = define_model({
        "scenario": "actual_name",
        "states": {"x0": {"initial": 0.0, "bounds": (0.0, 1.0)}},
        "actions": {"u0": {"bounds": (0.0, 1.0)}},
        "params": {"gain": 1.0},
        "dynamics": {"x0": "gain * u0"},
        "outputs": {"y0": {"expr": "x0", "bounds": (0.0, 1.0)}},
    })
    try:
        register_model("alias_name", model)
    except ValueError as ex:
        assert "must match model.scenario" in str(ex)
    else:
        raise AssertionError("mismatched registry name should be rejected")


def test_controller_registration_requires_explicit_replacement():
    try:
        register_controller("pid", lambda **_: None)
    except ValueError as ex:
        assert "already registered" in str(ex)
    else:
        raise AssertionError("controller registration should not silently overwrite built-ins")

    name = "regression-temporary-controller"
    register_controller(name, lambda **_: None)
    assert name in list_controllers()
    unregister_controller(name)
    assert name not in list_controllers()

    register_controller("pid", lambda **_: None, replace=True)
    unregister_controller("pid")
    assert make_controller("pid", scenario="cstr").metadata()["name"] == "PID"


def test_policy_adapters_do_not_mask_internal_type_errors():
    class BrokenPolicy:
        def act(self, obs, deterministic=False):
            raise TypeError("policy implementation failed")

    context = ControllerContext({}, {}, {}, "actuator", 0.5)
    try:
        PolicyController(BrokenPolicy()).act(np.zeros(1), context)
    except TypeError as ex:
        assert str(ex) == "policy implementation failed"
    else:
        raise AssertionError("adapter should preserve policy implementation errors")

    class ContextPolicy:
        def act(self, obs, context):
            return [context.control_dt]

    action = PolicyController(ContextPolicy()).act(np.zeros(1), context)
    assert np.allclose(action, [0.5])

    env = AIOGymNativeEnv("cstr", episode_steps=1)
    env.model.process_info = lambda x, levels, temps, process_env: {}
    env.reset(seed=0)
    env.step([0.5, 0.5])

    def broken_process_info(x, levels, temps, process_env):
        raise TypeError("process_info implementation failed")

    env.model.process_info = broken_process_info
    env.reset(seed=0)
    try:
        env.step([0.5, 0.5])
    except TypeError as ex:
        assert str(ex) == "process_info implementation failed"
    else:
        raise AssertionError("environment should preserve process_info implementation errors")

    class BrokenDiagnosticsPolicy:
        def act(self, obs, context):
            return [0.5, 0.5]

        def diagnostics(self):
            raise TypeError("diagnostics implementation failed")

    env = AIOGymNativeEnv("cstr", episode_steps=1)
    try:
        evaluate_controller(BrokenDiagnosticsPolicy(), env, episodes=1)
    except TypeError as ex:
        assert str(ex) == "diagnostics implementation failed"
    else:
        raise AssertionError("evaluation should preserve diagnostics implementation errors")


def test_sb3_in_memory_policy_does_not_receive_loader_algorithm():
    class InMemoryPolicy:
        def predict(self, obs, deterministic=True):
            return np.asarray([0.25, 0.75], dtype=np.float32), None

    controller = make_controller(
        "sb3",
        scenario="quadruple",
        policy=InMemoryPolicy(),
        config={
            "algo": "sac",
            "action_mode": "actuator",
            "name": "SB3-SAC",
        },
    )

    assert controller.name == "SB3-SAC"
    assert controller.action_mode == "actuator"


def test_onnx_policy_controller_validates_and_runs_export_contract():
    class Node:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape

    class Session:
        def get_inputs(self):
            return [Node("obs", ["batch", 5])]

        def get_outputs(self):
            return [Node("action", ["batch", 2])]

        def run(self, output_names, feeds):
            assert output_names == ["action"]
            assert feeds["obs"].shape == (1, 5)
            return [np.array([[0.25, 0.75]], dtype=np.float32)]

    controller = ONNXPolicyController(
        Session(),
        path="policy.onnx",
        action_mode="setpoint",
        expected_action_dim=2,
        scenario="cstr",
    )
    context = ControllerContext({}, {}, {}, "setpoint", 0.5)
    assert np.allclose(controller.act(np.zeros(5, dtype=np.float32), context), [0.25, 0.75])
    assert controller.metadata()["input_shape"] == ["batch", 5]

    try:
        controller.act(np.zeros(4, dtype=np.float32), context)
    except ValueError as ex:
        assert "expects 5 observations" in str(ex)
    else:
        raise AssertionError("ONNX policies should reject mismatched observation dimensions")

    try:
        ONNXPolicyController(Session(), path="policy.onnx", expected_action_dim=3)
    except ValueError as ex:
        assert "outputs 2 actions" in str(ex)
    else:
        raise AssertionError("ONNX policies should reject mismatched action dimensions")

def test_invalid_suite_controller_is_failed_not_skipped():
    protocol = BenchmarkProtocol.tracking("cstr", episode_steps=1)
    case = BenchmarkCase.from_protocol(
        protocol,
        controller="typo",
        seeds=[0],
        case_id="tracking:cstr:default:typo",
    )
    result = run_case({
        "name": "tracking:cstr:default:typo",
        "scenario": "cstr",
        "task": "default",
        "objective": "tracking",
        "controller": "typo",
        "action_mode": "actuator",
        "controller_config": {},
        "protocol": protocol,
        "case_spec": case,
        "seeds": [0],
    }, include_tracebacks=False)
    assert result["status"] == "failed"
    assert result["row"]["execution_status"] == "failed"


def test_hvac_oracle_builds_with_casadi_outputs():
    controller = make_controller(
        "oracle",
        scenario="hvac",
        config={"horizon": 1, "ipopt_max_iter": 20},
    )
    protocol = BenchmarkProtocol.tracking("hvac", episode_steps=1)
    result = evaluate_controller(controller, protocol.make_env(), episodes=1, protocol=protocol)
    assert result["episodes"] == 1
    assert result["controller"]["scenario"] == "hvac"


def test_oracle_scenario_overrides_are_exposed_in_metadata():
    hvac = make_controller(
        "oracle",
        scenario="hvac",
        config={"profile": "tracking", "objective": "tracking"},
    ).metadata()
    extraction = make_controller(
        "oracle",
        scenario="extraction",
        config={"profile": "tracking", "objective": "tracking"},
    ).metadata()
    heater = make_controller(
        "oracle",
        scenario="heater",
        config={"profile": "tracking", "objective": "tracking"},
    ).metadata()
    heater_economic = make_controller("oracle", scenario="heater").metadata()
    assert hvac["warm_start"] is True
    assert extraction["control_dt"] == 0.05 and extraction["ipopt_max_iter"] == 120
    for tracking_oracle in (hvac, extraction, heater):
        assert tracking_oracle["r_move"] == 1.0
        assert tracking_oracle["terminal_weight"] == 0.0
        assert "du_max" not in tracking_oracle
        assert tracking_oracle["steady_input_weight"] == pytest.approx(0.0)
    assert heater["transcription"] == "single_shooting"
    assert heater["enforce_state_bounds"] is False
    assert heater["enforce_temperature_cap"] is False
    assert heater_economic["transcription"] == "multiple_shooting"
    assert heater_economic["enforce_state_bounds"] is True


def test_suite_controller_defaults_do_not_depend_on_objective():
    args = SimpleNamespace(sb3_path=None, sb3_algo="sac", onnx_path=None)
    assert controller_config_for(args, "oracle", "actuator", "tracking") == {}
    assert controller_config_for(args, "oracle", "actuator", "economic") == {}


def test_artifact_check_accepts_failed_rows_and_model_metadata_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        payload = {
            "benchmark": "benchmark_suite",
            "suite": "regression",
            "suite_config": {
                "scenarios": ["cstr", "hvac"],
                "objectives": ["tracking"],
                "controllers": ["pid", "oracle"],
                "action_mode": "actuator",
            },
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
                    "episodes": 1,
                    "seed_list": [0],
                },
                {
                    "suite_case": "tracking:hvac:oracle",
                    "scenario": "hvac",
                    "objective": "tracking",
                    "action_mode": "actuator",
                    "controller": "oracle",
                    "execution_status": "failed",
                    "metric": "tracking_error_cost",
                    "message": "boom",
                    "episodes": 0,
                    "seed_list": [0],
                },
            ],
            "results": [],
            "report": {},
        }
        artifacts = write_benchmark_artifacts(tmpdir, payload)
        payload["artifacts"] = artifacts
        Path(tmpdir, "benchmark.json").write_text(json.dumps(payload, indent=2))
        with open(artifacts["model_metadata_manifest"]) as f:
            manifest = json.load(f)
        assert manifest["models"]["cstr"] == "metadata/models/cstr.json"
        plot_results(tmpdir)
        check_result = check_benchmark_artifacts(tmpdir)
        assert check_result["ok"], check_result["failed"]


def test_single_benchmark_rejects_removed_legacy_output_option():
    with pytest.raises(SystemExit):
        benchmark_main(["cstr", "--out", "report.json"])


def test_single_benchmark_writes_checkable_standard_artifacts(tmp_path):
    artifact_dir = tmp_path / "single-artifacts"

    benchmark_main([
        "quadruple", "minimum-phase",
        "--objective", "tracking",
        "--controllers", "pid",
        "--episodes", "1",
        "--episode-steps", "1",
        "--tracking-q-y", "0.7",
        "--tracking-r-move", "0",
        "--no-disturbance-obs",
        "--previous-action-obs",
        "--normalize-observations",
        "--tracking-error-obs",
        "--artifact-dir", str(artifact_dir),
    ])

    benchmark = json.loads((artifact_dir / "benchmark.json").read_text())
    assert benchmark["benchmark"] == "public_benchmark"
    assert benchmark["scenario"] == "quadruple"
    assert benchmark["controllers"] == ["pid"]
    assert benchmark["config"]["output_dir"] == str(artifact_dir)
    assert benchmark["config"]["tracking_q_y"] == 0.7
    assert benchmark["config"]["tracking_r_move"] == 0.0
    assert benchmark["config"]["disturbance_obs"] is False
    assert benchmark["config"]["previous_action_obs"] is True
    assert benchmark["config"]["normalize_observations"] is True
    assert benchmark["config"]["tracking_error_obs"] is True
    assert benchmark["protocol"]["disturbance_obs"] is False
    assert benchmark["protocol"]["previous_action_obs"] is True
    assert benchmark["protocol"]["normalize_observations"] is True
    assert benchmark["protocol"]["tracking_error_obs"] is True
    assert benchmark["protocol"]["objective_spec"]["reward_options"]["tracking_q_y"] == 0.7
    assert benchmark["protocol"]["objective_spec"]["reward_options"]["tracking_r_move"] == 0.0
    assert benchmark["config"]["save_rollouts"] is True
    assert "rollouts" not in benchmark
    assert (artifact_dir / "config" / "config.json").is_file()
    assert (artifact_dir / "metadata" / "model_metadata.json").is_file()
    assert (artifact_dir / "summary" / "summary.csv").is_file()
    assert (artifact_dir / "results" / "results.json").is_file()
    rollout_path = artifact_dir / "rollouts" / "rollouts.json"
    assert rollout_path.is_file()
    assert len(json.loads(rollout_path.read_text())) == 1
    assert (artifact_dir / "figures" / "summary.svg").is_file()
    assert any(
        (artifact_dir / "figures").glob("tracking_control_quadruple*.svg")
    )
    assert (artifact_dir / "report.md").is_file()
    check_result = check_benchmark_artifacts(artifact_dir)
    assert check_result["ok"], check_result["failed"]


def test_single_benchmark_default_dir_replaces_managed_artifacts(
    monkeypatch,
    tmp_path,
):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("AIOGYM_RUNS_DIR", str(runs_root))
    artifact_dir = runs_root / "quadruple" / "minimum-phase"
    stale_rollout = artifact_dir / "rollouts" / "rollouts.json"
    stale_rollout.parent.mkdir(parents=True)
    stale_rollout.write_text("stale")
    review_notes = artifact_dir / "review-notes.md"
    review_notes.write_text("keep")

    benchmark_main([
        "quadruple", "minimum-phase",
        "--objective", "tracking",
        "--controllers", "pid",
        "--episodes", "1",
        "--episode-steps", "1",
    ])

    assert (artifact_dir / "benchmark.json").is_file()
    assert len(json.loads(stale_rollout.read_text())) == 1
    assert review_notes.read_text() == "keep"


def test_pid_json_is_the_single_default_config_source():
    for scenario in SCENARIO_IDS:
        expected = load_controller_config("pid", scenario)["parameters"]
        direct = PIDAgent(make_model(scenario)).metadata()
        registered = make_controller("pid", scenario=scenario).metadata()
        assert direct["loops"] == expected["loops"]
        assert direct["holds"] == expected.get("holds", [])
        assert direct["demand_u_index"] == expected.get("demand_u_index")
        assert registered["loops"] == expected["loops"]


def test_predictive_controller_configs_fail_at_construction():
    invalid_mpc = (
        ({"P": 0}, "P"),
        ({"Ts": 0}, "Ts"),
        ({"move_supp": -1}, "move_supp"),
        ({"cv_scale": [1.0, 2.0, 3.0]}, "cv_scale"),
        ({"cv_scale": [0.0]}, "cv_scale"),
        ({"q_y": [1.0, 2.0, 3.0]}, "q_y"),
        ({"q_y": -1.0}, "q_y"),
    )
    for config, expected in invalid_mpc:
        try:
            make_controller("mpc", scenario="cstr", config=config)
        except ValueError as ex:
            assert expected in str(ex)
        else:
            raise AssertionError(f"invalid MPC config {config} should fail during construction")

    invalid_oracle = (
        ({"horizon": 0}, "horizon"),
        ({"control_dt": 0}, "control_dt"),
        ({"objective": "trackingg"}, "objective"),
        ({"ipopt_tol": float("inf")}, "ipopt_tol"),
        ({"terminal_weight": -1.0}, "terminal_weight"),
        ({"q_y": [1.0, 2.0]}, "q_y"),
        ({"transcription": "collocation"}, "transcription"),
        ({"solve_every": 0}, "solve_every"),
    )
    for config, expected in invalid_oracle:
        try:
            make_controller("oracle", scenario="cstr", config=config)
        except ValueError as ex:
            assert expected in str(ex)
        else:
            raise AssertionError(f"invalid Oracle config {config} should fail during construction")


def test_invalid_environment_configuration_fails_early():
    invalid_options = (
        ({"control_dt": 0}, "control_dt"),
        ({"episode_steps": 0}, "episode_steps"),
        ({"reward_mode": "trackingg"}, "reward_mode"),
        ({"action_mode": "direct"}, "action_mode"),
        ({"noise_pct": -0.1}, "noise_pct"),
        ({"noise_pct": float("nan")}, "noise_pct"),
        ({"control_dt": float("inf")}, "control_dt"),
        ({"crystal_ln_range": (2.0, 1.0)}, "crystal_ln_range"),
        ({"crystal_ln_range": (1.0, float("nan"))}, "crystal_ln_range"),
        ({"crystal_cv_range": (0.5,)}, "crystal_cv_range"),
    )
    for options, expected in invalid_options:
        try:
            AIOGymNativeEnv("cstr", **options)
        except ValueError as ex:
            assert expected in str(ex)
        else:
            raise AssertionError(f"invalid {expected} should be rejected")


def test_removed_compatibility_surfaces_stay_removed():
    import aiogym.env as env_module
    from aiogym.rl import TransitionDataset

    assert importlib.util.find_spec("aiogym.objectives") is None
    assert importlib.util.find_spec("aiogym.evaluation.core") is None
    assert not hasattr(env_module, "make_env")
    assert not hasattr(TransitionDataset(), "rl_tuples")

    for build in (
        lambda: AIOGymNativeEnv("cstr", reward_mode="track"),
        lambda: AIOGymNativeEnv(custom_model=make_model("cstr")),
        lambda: BenchmarkProtocol.tracking("cstr", reward_mode="track"),
        lambda: make_env(config={"model": "cstr", "objective": "tracking"}),
        lambda: run_benchmark({"scenario": "cstr", "objective": "tracking", "controller": "pid"}),
        lambda: load_task_profile("quadruple/minimum-phase-tracking"),
        lambda: make_controller("nmpc", scenario="cstr"),
        lambda: make_controller(
            "oracle",
            scenario="cstr",
            config={"mode": "economic"},
        ),
    ):
        try:
            build()
        except (FileNotFoundError, KeyError, TypeError, ValueError):
            pass
        else:
            raise AssertionError("removed compatibility input should be rejected")


def test_invalid_actions_fail_before_reaching_dynamics():
    env = AIOGymNativeEnv("cstr", episode_steps=1)
    env.reset(seed=0)
    invalid_actions = (
        ([0.5], "2 values"),
        ([0.5, float("nan")], "finite"),
        ([float("inf"), 0.5], "finite"),
    )
    for action, expected in invalid_actions:
        try:
            env.step(action)
        except ValueError as ex:
            assert expected in str(ex)
        else:
            raise AssertionError("invalid actions should be rejected by the environment")

    try:
        validate_action([0.5, float("nan")], env, "broken-policy")
    except ValueError as ex:
        assert "broken-policy produced a non-finite action" in str(ex)
    else:
        raise AssertionError("controller actions must be finite")


def test_explicit_seed_list_overrides_episode_count():
    assert parse_seed_list("7, 11", seed=0, episodes=0) == [7, 11]
    try:
        parse_seed_list(None, seed=0, episodes=0)
    except ValueError as ex:
        assert "episodes must be positive" in str(ex)
    else:
        raise AssertionError("generated seed lists require a positive episode count")


def test_training_outputs_are_unique_unless_explicitly_named():
    sb3_args = SimpleNamespace(
        name=None,
        algo="sac",
        scenario="cstr",
        action_mode="actuator",
        reward_mode="kpi",
        seed=3,
    )
    first = run_name_for(sb3_args, "run-a")
    second = run_name_for(sb3_args, "run-b")
    assert first != second and first.endswith("run-a") and second.endswith("run-b")
    sb3_args.name = "stable-sb3"
    assert run_name_for(sb3_args, "ignored") == "stable-sb3"

    rlpd_args = SimpleNamespace(out=None, scenario="cascade")
    first = output_base_for(rlpd_args, "run-a")
    second = output_base_for(rlpd_args, "run-b")
    assert first != second and first.endswith("cascade_run-a")
    rlpd_args.out = "custom/rlpd"
    assert output_base_for(rlpd_args, "ignored") == "custom/rlpd"


def test_public_benchmark_supports_strict_and_batch_error_modes():
    with tempfile.TemporaryDirectory() as tmpdir:
        base_config = {
            "scenario": "cstr",
            "objective": "tracking",
            "controllers": ["typo"],
            "seeds": [0],
            "episode_steps": 1,
        }
        batch = run_benchmark({**base_config, "output_dir": Path(tmpdir, "batch")})
        assert batch["rows"][0]["execution_status"] == "failed"
        assert batch["errors"][0]["controller"] == "typo"

        try:
            run_benchmark({
                **base_config,
                "output_dir": Path(tmpdir, "strict"),
                "strict": True,
            })
        except RuntimeError as ex:
            assert "controller 'typo' benchmark failed" in str(ex)
        else:
            raise AssertionError("strict benchmark mode should propagate controller failures")


def test_public_benchmark_rejects_empty_worklists():
    for options, expected in (({"controllers": []}, "controller"), ({"seeds": []}, "seed")):
        try:
            run_benchmark({"scenario": "cstr", "objective": "tracking", **options})
        except ValueError as ex:
            assert expected in str(ex)
        else:
            raise AssertionError(f"empty benchmark {expected} list should be rejected")


def test_direct_evaluation_rejects_empty_episode_worklists():
    env = AIOGymNativeEnv("cstr", episode_steps=1)
    controller = make_controller("pid", scenario="cstr")
    for options, expected in (
        ({"episodes": 0}, "episodes must be positive"),
        ({"seed_list": []}, "seed_list must contain"),
    ):
        try:
            evaluate_controller(controller, env, **options)
        except ValueError as ex:
            assert expected in str(ex)
        else:
            raise AssertionError("empty direct evaluations should be rejected")

    explicit = evaluate_controller(controller, env, episodes=0, seed_list=[7])
    assert explicit["seed_list"] == [7]

    protocol = BenchmarkProtocol.tracking("cstr", episode_steps=1)
    try:
        BenchmarkCase.from_protocol(protocol, controller="pid", seeds=[])
    except ValueError as ex:
        assert "seeds must not be empty" in str(ex)
    else:
        raise AssertionError("BenchmarkCase should reject explicit empty seeds")


def test_benchmark_protocol_rejects_invalid_metadata():
    invalid = (
        ({"scenario": ""}, "scenario"),
        ({"objective": "typo"}, "objective"),
        ({"env_reward_mode": "reward"}, "env_reward_mode"),
        ({"action_mode": "direct"}, "action_mode"),
        ({"control_dt": 0}, "control_dt"),
        ({"episode_steps": 1.5}, "episode_steps"),
        ({"noise_pct": float("nan")}, "noise_pct"),
        ({"model_params": []}, "model_params"),
    )
    for options, expected in invalid:
        try:
            BenchmarkProtocol(**options)
        except (TypeError, ValueError) as ex:
            assert expected in str(ex)
        else:
            raise AssertionError(f"invalid protocol field {expected} should be rejected")

    try:
        resolve_protocol("cstr", "typo", {})
    except ValueError as ex:
        assert "objective must be one of" in str(ex)
    else:
        raise AssertionError("unknown protocol objectives should produce a clear ValueError")
