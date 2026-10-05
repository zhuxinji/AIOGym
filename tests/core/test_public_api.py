from __future__ import annotations

import inspect
import json
from pathlib import Path

import aiogym
import aiogym.core
import aiogym.rl
import aiogym.workflows


def test_public_api_is_scenario_reward_oriented_and_small():
    for name in ("collect", "evaluate", "train", "load_policy", "plot_training_curve"):
        assert getattr(aiogym, name) is getattr(aiogym.workflows, name)
    assert not hasattr(aiogym, "compare_policies")
    evaluation_parameters = inspect.signature(aiogym.evaluate).parameters
    assert "policies" in evaluation_parameters and "policy" not in evaluation_parameters
    assert evaluation_parameters["output"].default is None
    parameters = inspect.signature(aiogym.train).parameters
    assert parameters["steps"].default == 500_000
    assert parameters["evaluate_every"].default == 5_000
    assert parameters["record_every"].default == 500
    assert set(aiogym.__all__) == {
        "__version__",
        "DatasetReader",
        "FunctionPolicy",
        "Policy",
        "Scenario",
        "collect",
        "environment_metadata",
        "evaluate",
        "list_algorithms",
        "list_actions",
        "list_info",
        "list_observations",
        "list_outputs",
        "list_safety_rules",
        "list_scenarios",
        "list_states",
        "list_benchmarks",
        "list_parameters",
        "list_rewards",
        "load_policy",
        "make_controller",
        "make_env",
        "plot_training_curve",
        "train",
    }
    assert "three_tank" in aiogym.list_scenarios()
    assert "cascade" in aiogym.list_scenarios()
    assert "regulation" in aiogym.list_rewards("three_tank")


def test_public_parameter_listing_includes_defaults_and_native_units():
    quadruple = {row["name"]: row for row in aiogym.list_parameters("quadruple")}
    assert quadruple["tank_area"]["default"] == [28.0, 32.0, 28.0, 32.0]
    assert quadruple["tank_area"]["value"] == quadruple["tank_area"]["default"]
    assert quadruple["tank_area"]["unit"] == "cm^2"
    assert quadruple["tank_area"]["allowed_values"] == "Four finite positive values"
    assert quadruple["pump_gain"]["unit"] == "cm^3/(s*V)"

    three_tank = {row["name"]: row for row in aiogym.list_parameters("three_tank")}
    assert three_tank["pump_flow_max"]["unit"] == "m^3/s"
    assert "heater_power" not in three_tank
    cascade = {row["name"]: row for row in aiogym.list_parameters("cascade")}
    assert cascade["heater_power_max"]["unit"] == "W"
    assert cascade["heat_loss_coefficient"]["unit"] == "W/K"


def test_every_built_in_scenario_has_a_documentation_page():
    scenario_docs = Path(__file__).parents[2] / "docs" / "scenarios"
    for scenario in aiogym.list_scenarios():
        assert (scenario_docs / f"{scenario}.md").is_file()


def test_information_covers_builtin_definitions_and_declared_benchmark_horizons():
    for scenario in aiogym.list_scenarios():
        env = aiogym.make_env(scenario)
        try:
            report = env.describe()
            json.dumps(report, allow_nan=False)
            assert report["episode"]["status"] == "template"
            for section in ("parameters", "states", "actions", "observations", "outputs"):
                assert report[section]
                assert all(row["description"] != "Not provided" for row in report[section].values())
            assert report["safety_rules"]
            assert env.parameters == {name: row["default"] for name, row in report["parameters"].items()}
            assert env.states == aiogym.list_states(scenario)
            assert env.actions == aiogym.list_actions(scenario)
            assert env.observations == aiogym.list_observations(scenario)
            assert env.outputs == aiogym.list_outputs(scenario)
            assert env.safety_rules == aiogym.list_safety_rules(scenario)
            assert env.rewards == aiogym.list_rewards(scenario)
            assert env.benchmarks == aiogym.list_benchmarks(scenario)
            for name, benchmark in env.scenario.benchmarks.items():
                assert env.benchmarks[name]["horizon"] == benchmark.make_episode(env.model, 0).horizon
                assert env.benchmarks[name]["description"] != "Not provided"
        finally:
            env.close()


def test_instance_information_uses_effective_parameters_and_independent_snapshots():
    env = aiogym.make_env("cstr", parameters={"temperature_trip": 95.0})
    try:
        assert env.parameters["temperature_trip"] == 95.0
        assert env.describe()["parameters"]["temperature_trip"]["default"] == 92.0
        assert env.states["reactor_temperature"]["high"] == 200.0
        assert "> 95 degC" in env.safety_rules["reactor_temperature"]["condition"]
        assert env.observations["reactor_temperature"]["unit"] == "normalized"
        assert env.observations["reactor_temperature"]["physical_unit"] == "degC"
        report = env.describe()
        report["states"]["reactor_temperature"]["high"] = 999.0
        report["configuration"]["parameters"]["temperature_trip"] = 1.0
        assert env.states["reactor_temperature"]["high"] == 200.0
        assert env.parameters["temperature_trip"] == 95.0
    finally:
        env.close()
    env = aiogym.make_env("cascade", parameters={"heater": [1, 0, 1]})
    try:
        snapshot = env.parameters
        snapshot["heater"][0] = 0
        assert env.parameters["heater"] == [1, 0, 1]
        assert env.actions["heater_H2"]["enabled"] is False
        assert env.actions["heater_H3"]["enabled"] is True
    finally:
        env.close()


def test_information_queries_preserve_sampling_and_do_not_generate_benchmark_cases(monkeypatch):
    from aiogym.core.io import jsonable
    from aiogym.core.specs import Benchmark

    env = aiogym.make_env("cstr", randomize=True, noise=True, delay=True)
    control = aiogym.make_env("cstr", randomize=True, noise=True, delay=True)
    try:
        env.reset(seed=7)
        control.reset(seed=7)
        interface = env.policy_interface()
        assert env.describe()["episode"]["status"] == "resolved"
        assert env.describe()["episode"]["selection"]["reset_seed"] == 7
        assert env.policy_interface() == interface
        assert jsonable(env.reset()) == jsonable(control.reset())
    finally:
        env.close()
        control.close()

    env = aiogym.make_env("cstr", benchmark="tracking")
    env.reset(seed=3)
    def unexpected_case(*args, **kwargs):
        raise AssertionError("information query generated a benchmark case")
    monkeypatch.setattr(Benchmark, "make_episode", unexpected_case)
    try:
        assert env.describe()["episode"]["selection"] == {"case_seed": 3}
        assert env.benchmarks["tracking"]["horizon"] == env.episode_steps
        assert aiogym.list_benchmarks("cstr") == env.benchmarks
    finally:
        env.close()


def test_hybrid_information_describes_the_outer_policy_interface():
    from aiogym.scenarios.cascade.hybrid import three_tank_pid_temperature_control

    env = three_tank_pid_temperature_control(aiogym.make_env("cascade"))
    try:
        assert list(env.actions) == ["heater_H1", "heater_H2", "heater_H3"]
        assert [row["index"] for row in env.actions.values()] == [0, 1, 2]
        assert len(env.observations) == env.observation_space.shape[0]
        assert all(row["description"] != "Not provided" for row in env.observations.values())
        assert env.observations["tank_1_temperature_error"]["normalization"] == "(reference - measurement) / 5 degC"
        assert len(env.describe()["physical_actions"]) == 7
        assert env.describe()["environment"]["reward_adapter"] == "temperature_tracking"
    finally:
        env.close()


def test_subpackages_do_not_export_internal_definition_types():
    assert aiogym.core.__all__ == ()
    assert set(aiogym.workflows.__all__) == {
        "collect",
        "DatasetReader",
        "evaluate",
        "load_policy",
        "plot_training_curve",
        "train",
    }
    assert set(aiogym.rl.__all__) == {"list_algorithms"}


def test_public_environment_and_controller_flow():
    env = aiogym.make_env("three_tank", reward="regulation")
    try:
        policy = aiogym.make_controller("hold", env=env)
        assert isinstance(policy, aiogym.Policy)
        result = aiogym.evaluate(env=env, policies={"policy": policy}, seeds=[0], max_steps=2)["evaluations"]["policy"]
    finally:
        env.close()
    assert result["environment"]["reward"] == "regulation"
    assert result["episodes"][0]["length"] == 2
    assert "tracking_ise" in result["aggregate"]


def test_scenario_only_environment_uses_defaults():
    env = aiogym.make_env("three_tank")
    try:
        assert env.benchmark is None
        assert env.episode_family == "default"
        assert env.reward.id == "regulation"
    finally:
        env.close()
