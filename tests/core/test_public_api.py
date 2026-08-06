from __future__ import annotations

import warnings

import aiogym


def test_public_api_is_task_oriented_and_small():
    assert aiogym.__version__ == "0.3.0"
    assert set(aiogym.__all__) == {
        "__version__",
        "collect",
        "evaluate",
        "list_scenarios",
        "list_plants",
        "list_conditions",
        "list_tasks",
        "load_plant",
        "make_controller",
        "make_env",
        "study",
        "train",
    }
    assert "three_tank" in aiogym.list_scenarios()
    assert "three_tank/regulation" in aiogym.list_tasks("three_tank")


def test_public_environment_and_controller_flow():
    env = aiogym.make_env("three_tank/regulation", condition="commissioning")
    try:
        policy = aiogym.make_controller("hold", env=env)
        result = aiogym.evaluate(policy, seeds=[0], max_steps=2)
    finally:
        env.close()
    assert result["task_id"] == "three_tank/regulation"
    assert result["episodes"][0]["steps"] == 2
    assert "tracking_iae" in result["aggregate"]


def test_old_make_env_shape_is_narrowly_deprecated():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        env = aiogym.make_env(
            "quadruple",
            case="minimum-phase",
            reward_spec="regulation",
        )
    try:
        assert env.task.id == "quadruple/regulation"
        assert env.condition.id == "minimum-phase"
    finally:
        env.close()
    assert captured
    assert captured[0].category is DeprecationWarning
