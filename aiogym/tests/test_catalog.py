"""Public resource discovery and canonical-ID error messages."""
from __future__ import annotations

from pathlib import Path
import importlib.util
import json

import pytest

import aiogym
from aiogym.cli import benchmark
from aiogym.cli.benchmark import load_suite
from aiogym.evaluation import finalize_benchmark_artifacts, resolve_protocol


def test_public_catalog_lists_current_canonical_ids():
    assert "cascade-recirculating" in aiogym.list_scenarios()
    assert "cascade_recirculating" not in aiogym.list_scenarios()
    assert len(aiogym.list_scenarios()) == 8

    tasks = aiogym.list_tasks()
    assert not hasattr(aiogym, "list_task_profiles")
    assert len(tasks) == 13
    assert aiogym.list_tasks("cascade") == (
        "cascade/commissioning",
        "cascade/continuous-benchmark",
        "cascade/disturbance-rejection",
        "cascade/safety-recovery",
        "cascade/temperature-step",
    )
    assert aiogym.list_tasks("cstr") == ()
    with pytest.raises(ValueError, match="not canonical"):
        aiogym.list_tasks("cascade_recirculating")

    suites = aiogym.list_suites()
    assert suites == (
        "cascade-control",
        "cascade-recirculating",
        "crystallization-tracking",
        "economic-actuator",
        "quadruple",
        "quadruple-phase-comparison",
        "robustness-actuator",
        "standard-baselines",
        "tracking-actuator",
    )
    assert "quadruple-paper-reference" not in suites

    assert not hasattr(aiogym, "registered_controllers")
    assert aiogym.list_controllers() == (
        "mpc", "onnx", "oracle", "pid", "policy", "sb3",
    )


def test_task_runtime_is_owned_by_models_without_evaluation_facade():
    import aiogym.models.tasks as model_tasks
    from aiogym.evaluation import evaluate_task_acceptance

    assert aiogym.load_task_profile is model_tasks.load_task_profile
    assert importlib.util.find_spec("aiogym.evaluation.task_profiles") is None
    assert model_tasks.load_task_profile.__module__ == "aiogym.models.tasks.registry"
    assert model_tasks.validate_task_profile.__module__ == "aiogym.models.tasks.schema"
    assert evaluate_task_acceptance.__module__ == "aiogym.evaluation.results"

    builtin_dir = Path(model_tasks.__file__).with_name("builtin")
    assert builtin_dir.is_dir()
    assert len(tuple(builtin_dir.glob("*/*.json"))) == 13


def test_public_catalog_tracks_runtime_registrations():
    from aiogym.evaluation.suite import scenario_groups

    model = aiogym.make_model("cstr")
    model.scenario = "catalog_test_model"
    aiogym.register_model("catalog_test_model", model)
    aiogym.register_controller("catalog_test_controller", lambda **_: None)
    try:
        assert "catalog_test_model" in aiogym.list_scenarios()
        assert "catalog_test_controller" in aiogym.list_controllers()
        assert "catalog_test_model" in scenario_groups()["ALL_SCENARIOS"]
        assert "catalog_test_model" in scenario_groups()["ECONOMIC_SCENARIOS"]
    finally:
        aiogym.unregister_controller("catalog_test_controller")
        aiogym.unregister_model("catalog_test_model")

    assert "catalog_test_model" not in aiogym.list_scenarios()
    assert "catalog_test_controller" not in aiogym.list_controllers()


def test_noncanonical_scenario_task_and_suite_ids_are_rejected():
    with pytest.raises(ValueError, match="not canonical"):
        aiogym.make_model("cascade_recirculating")
    with pytest.raises(ValueError, match="not canonical"):
        aiogym.load_task_profile("cascade_recirculating/commissioning")
    with pytest.raises(FileNotFoundError, match="unknown suite ID"):
        load_suite("standard_baselines")
    with pytest.raises(ValueError, match="not canonical"):
        resolve_protocol("cascade_recirculating", "tracking", {"task": "commissioning"})


def test_builtin_suites_resolve_to_self_contained_declarations():
    standard = load_suite("standard-baselines")
    quadruple = load_suite("quadruple")

    assert standard["controllers"] == ["pid", "mpc", "oracle"]
    assert len(standard["cases"]) == 3
    assert quadruple["cases"][0]["task"] == "minimum-phase"
    assert "controller_configs" not in quadruple["cases"][0]
    assert aiogym.load_task_profile(
        "quadruple/minimum-phase"
    )["controllers"]["pid"] == {
        "profile": "quadruple-minimum-phase-benchmark"
    }

    def assert_resolved(value):
        if isinstance(value, dict):
            assert not ({"extends", "preset", "case_ref"} & set(value))
            for item in value.values():
                assert_resolved(item)
        elif isinstance(value, list):
            for item in value:
                assert_resolved(item)

    assert_resolved(standard)
    assert_resolved(quadruple)


def test_external_suite_extends_relative_file_and_deep_overrides(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "scenarios": ["cstr"],
        "objectives": ["tracking"],
        "controllers": ["pid", "mpc"],
        "action_mode": "actuator",
        "environment": {"preset": "environment/deterministic"},
        "controller_configs": {
            "pid": {"profile": "base", "parameters": {"kp": 1.0}}
        },
    }))
    child = tmp_path / "child.json"
    child.write_text(json.dumps({
        "extends": "base.json",
        "controllers": ["pid"],
        "environment": {"randomize": True},
        "controller_configs": {
            "pid": {"parameters": {"ki": 2.0}}
        },
    }))

    suite = load_suite(str(child))

    assert suite["controllers"] == ["pid"]
    assert suite["environment"]["auto_events"] is False
    assert suite["environment"]["randomize"] is True
    assert suite["controller_configs"]["pid"] == {
        "profile": "base",
        "parameters": {"kp": 1.0, "ki": 2.0},
    }

    case_override = tmp_path / "case-override.json"
    case_override.write_text(json.dumps({
        "scenarios": ["cstr"],
        "objectives": ["tracking"],
        "controllers": ["pid"],
        "action_mode": "actuator",
        "cases": [{
            "case_ref": "economic/randomized",
            "scenarios": ["cstr"],
            "objective": "tracking",
            "environment": {"preset": "environment/deterministic"},
        }],
    }))
    resolved_case = load_suite(str(case_override))["cases"][0]
    assert resolved_case["objective"] == "tracking"
    assert resolved_case["environment"]["auto_events"] is False
    assert resolved_case["environment"]["randomize"] is False


def test_suite_reuse_rejects_cycles_and_unknown_references(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"extends": "second.json"}))
    second.write_text(json.dumps({"extends": "first.json"}))
    with pytest.raises(ValueError, match="cyclic suite extends"):
        load_suite(str(first))

    unknown_preset = tmp_path / "unknown-preset.json"
    unknown_preset.write_text(json.dumps({"preset": "missing/preset"}))
    with pytest.raises(ValueError, match="unknown suite preset ID"):
        load_suite(str(unknown_preset))

    unknown_case = tmp_path / "unknown-case.json"
    unknown_case.write_text(json.dumps({
        "cases": [{"case_ref": "missing/case"}]
    }))
    with pytest.raises(ValueError, match="unknown suite case ID"):
        load_suite(str(unknown_case))


def test_suite_preset_and_case_reference_cycles_are_rejected(tmp_path, monkeypatch):
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()
    (preset_dir / "cycles.json").write_text(json.dumps({
        "presets": {
            "preset/a": {"preset": "preset/b"},
            "preset/b": {"preset": "preset/a"},
        },
        "cases": {
            "case/a": {"case_ref": "case/b"},
            "case/b": {"case_ref": "case/a"},
        },
    }))
    monkeypatch.setattr(benchmark, "PRESET_DIR", preset_dir)

    preset_suite = tmp_path / "preset-cycle.json"
    preset_suite.write_text(json.dumps({"preset": "preset/a"}))
    with pytest.raises(ValueError, match="cyclic suite preset"):
        load_suite(str(preset_suite))

    case_suite = tmp_path / "case-cycle.json"
    case_suite.write_text(json.dumps({"cases": [{"case_ref": "case/a"}]}))
    with pytest.raises(ValueError, match="cyclic suite case"):
        load_suite(str(case_suite))


def test_custom_ids_are_not_normalized_by_character_replacement():
    snake_model = aiogym.make_model("cstr")
    snake_model.scenario = "catalog_custom_model"
    kebab_model = aiogym.make_model("cstr")
    kebab_model.scenario = "catalog-custom-model"
    aiogym.register_model("catalog_custom_model", snake_model)
    aiogym.register_model("catalog-custom-model", kebab_model)
    try:
        assert aiogym.make_model("catalog_custom_model").scenario == "catalog_custom_model"
        assert aiogym.make_model("catalog-custom-model").scenario == "catalog-custom-model"
        assert {"catalog_custom_model", "catalog-custom-model"}.issubset(
            aiogym.list_scenarios()
        )
    finally:
        aiogym.unregister_model("catalog_custom_model")
        aiogym.unregister_model("catalog-custom-model")

    reserved = aiogym.make_model("cstr")
    reserved.scenario = "cascade-recirculating"
    with pytest.raises(ValueError, match="canonical alias"):
        aiogym.register_model("cascade-recirculating", reserved)


def test_standard_artifacts_serialize_canonical_scenario_ids(tmp_path):
    payload = {
        "benchmark": "public_benchmark",
        "scenario": "cascade_recirculating",
        "config": {"scenario": "cascade_recirculating"},
        "benchmark_config": {"scenario": "cascade_recirculating"},
        "rows": [],
        "results": [],
        "report": {},
    }

    finalize_benchmark_artifacts(tmp_path, payload)

    benchmark = json.loads((tmp_path / "benchmark.json").read_text())
    model_metadata = json.loads(
        (tmp_path / "metadata" / "model_metadata.json").read_text()
    )
    assert benchmark["scenario"] == "cascade-recirculating"
    assert benchmark["config"]["scenario"] == "cascade-recirculating"
    assert benchmark["benchmark_config"]["scenario"] == "cascade-recirculating"
    assert model_metadata["scenario"] == "cascade-recirculating"


def test_unknown_ids_show_their_canonical_catalogs():
    with pytest.raises(
        ValueError,
        match="available scenario IDs:.*cascade-recirculating",
    ):
        aiogym.make_model("missing-scenario")

    with pytest.raises(FileNotFoundError, match="available task IDs for scenario 'quadruple'"):
        aiogym.load_task_profile("quadruple/missing-task")

    with pytest.raises(FileNotFoundError, match="available suite IDs:.*standard-baselines"):
        load_suite("missing-suite")

    with pytest.raises(KeyError, match="available controller IDs:.*oracle"):
        aiogym.make_controller("missing-controller", scenario="cstr")


def test_explicit_missing_task_and_suite_files_keep_file_errors(tmp_path):
    missing_task = tmp_path / "missing-task.json"
    missing_suite = tmp_path / "missing-suite.json"

    with pytest.raises(FileNotFoundError, match="task profile not found"):
        aiogym.load_task_profile(missing_task)
    with pytest.raises(FileNotFoundError) as error:
        load_suite(str(missing_suite))
    assert str(missing_suite) in str(error.value)
    assert not Path(missing_suite).exists()
