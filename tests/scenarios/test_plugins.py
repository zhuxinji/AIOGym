from __future__ import annotations

import warnings

import pytest

import aiogym.scenarios  # noqa: F401
from aiogym.core import get_task, list_scenarios, list_tasks, make_env
from aiogym.scenarios.quadruple.model import QuadrupleModel
from aiogym.scenarios.quadruple.physics import QuadruplePhysicsModel


def test_open_cascade_tasks_share_plant_with_distinct_objectives():
    regulation = make_env(
        "three_tank/regulation",
        plant="open-cascade-v1",
        condition="continuous-benchmark",
    )
    economic = make_env(
        "three_tank/economic",
        plant="open-cascade-v1",
        condition="continuous-benchmark",
    )
    try:
        assert regulation.plant.plant_hash == economic.plant.plant_hash
        assert regulation.task.task_hash != economic.task.task_hash
        assert regulation.action_space.shape == economic.action_space.shape == (7,)
    finally:
        regulation.close()
        economic.close()


def test_only_quadruple_and_three_tank_are_registered():
    assert set(list_scenarios()) == {"quadruple", "three_tank"}
    assert set(list_tasks(scenario="three_tank")) == {
        "three_tank/regulation",
        "three_tank/economic",
    }
    assert get_task("three_tank/economic").primary_metric == "economic_objective"


def test_quadruple_model_is_a_thin_adapter_over_its_physics_model():
    assert QuadrupleModel.numerical_type is QuadruplePhysicsModel


@pytest.mark.parametrize(
    ("legacy", "plant_id", "condition_id"),
    (
        ("cascade/regulation", "open-cascade-v1", "continuous-benchmark"),
        ("cascade/economic", "open-cascade-v1", "continuous-benchmark"),
        (
            "cascade_recirculating/regulation",
            "recirculating-h1-v1",
            "commissioning",
        ),
    ),
)
def test_legacy_task_aliases_are_central_and_warn(legacy, plant_id, condition_id):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        env = make_env(legacy)
    try:
        assert env.task.scenario == "three_tank"
        assert env.plant.id == plant_id
        assert env.condition.id == condition_id
    finally:
        env.close()
    assert [row.category for row in captured] == [DeprecationWarning]


def test_legacy_alias_rejects_explicit_conflicts():
    with pytest.raises(ValueError, match="legacy alias requires plant"):
        make_env("cascade/regulation", plant="recirculating-h1-v1")
    with pytest.raises(ValueError, match="legacy alias requires condition"):
        make_env("cascade/regulation", condition="commissioning")
