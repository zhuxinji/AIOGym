from __future__ import annotations

from pathlib import Path

import aiogym
import aiogym.core
import aiogym.workflows
import pytest


def test_public_api_is_scenario_reward_oriented_and_small():
    assert aiogym.__version__ == "0.7.0"
    assert set(aiogym.__all__) == {
        "__version__",
        "DatasetReader",
        "collect",
        "compare_policies",
        "evaluate",
        "list_scenarios",
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
    assert "regulation" in aiogym.list_rewards("three_tank")


def test_public_parameter_listing_includes_defaults_and_native_units():
    quadruple = {row["name"]: row for row in aiogym.list_parameters("quadruple")}
    assert quadruple["tank_area"] == {
        "name": "tank_area",
        "default": (28.0, 32.0, 28.0, 32.0),
        "unit": "cm^2",
    }
    assert quadruple["pump_gain"]["unit"] == "cm^3/(s*V)"

    three_tank = {
        row["name"]: row for row in aiogym.list_parameters("three_tank")
    }
    assert three_tank["heater_power"]["default"] == 2000.0
    assert three_tank["heater_power"]["unit"] == "W"
    assert three_tank["pump_flow_max"]["unit"] == "m^3/s"


def test_every_built_in_scenario_has_a_documentation_page():
    scenario_docs = Path(__file__).parents[2] / "docs" / "scenarios"
    for scenario in aiogym.list_scenarios():
        assert (scenario_docs / f"{scenario}.md").is_file()


def test_subpackages_do_not_export_internal_definition_types():
    assert aiogym.core.__all__ == ()
    assert set(aiogym.workflows.__all__) == {
        "collect",
        "compare_policies",
        "DatasetReader",
        "evaluate",
        "load_policy",
        "plot_training_curve",
        "train",
    }
    assert not hasattr(aiogym.workflows, "TrainConfig")


def test_public_environment_and_controller_flow():
    env = aiogym.make_env("three_tank", reward="regulation")
    try:
        policy = aiogym.make_controller("hold", env=env)
        result = aiogym.evaluate(env=env, policy=policy, seeds=[0], max_steps=2)
    finally:
        env.close()
    assert result["environment"]["reward"] == "regulation"
    assert result["episodes"][0]["length"] == 2
    assert "tracking_iae" in result["aggregate"]


def test_scenario_only_environment_uses_defaults():
    env = aiogym.make_env("three_tank")
    try:
        assert env.benchmark is None
        assert env.episode_family == "default"
        assert env.reward.id == "regulation"
    finally:
        env.close()


def test_composite_scenario_reward_selector_is_not_registered():
    with pytest.raises(KeyError, match="unknown scenario"):
        aiogym.make_env("three_tank/economic")


@pytest.mark.parametrize(
    "legacy",
    [
        {"condition": "reference-step"},
        {"preset": "reference-step"},
        {"case": "reference-step"},
        {"reward_spec": "regulation"},
    ],
)
def test_removed_make_env_keywords_are_rejected(legacy):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        aiogym.make_env(
            "quadruple",
            **legacy,
        )
