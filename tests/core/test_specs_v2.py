from __future__ import annotations

from dataclasses import replace

from aiogym.core import EnvironmentIdentity, OperatingCondition, PlantConfig, TaskSpec


def _condition(**changes):
    values = {
        "id": "nominal",
        "initial_state": (0.2, 20.0),
        "reference": (0.3,),
        "control_dt": 1.0,
        "horizon": 10,
        "disturbances": {"gain": 1.0},
        "reference_schedule": {5: (0.4,)},
        "disturbance_schedule": {7: {"gain": 0.8}},
    }
    values.update(changes)
    return OperatingCondition(**values)


def _plant(**changes):
    values = {
        "id": "toy-v1",
        "scenario": "toy",
        "plant": {"gain": 2.0},
        "conditions": {"nominal": _condition()},
        "default_condition": "nominal",
        "study": {"requirements": {"limit": 3.0}},
        "references": ("source-a",),
    }
    values.update(changes)
    return PlantConfig(**values)


def _reward(state, action, next_state, context):
    del state, action, context
    return -float(next_state[0])


def _task(**changes):
    values = {
        "id": "toy/regulation",
        "scenario": "toy",
        "objective": "regulation",
        "reward": _reward,
        "metrics": ("return", "iae"),
        "primary_metric": "iae",
        "metric_direction": "minimize",
        "revision": 1,
        "reward_id": "tracking-v1",
        "metric_suite_id": "regulation-v1",
        "reward_term_names": ("tracking",),
        "required_capabilities": ("tracking",),
    }
    values.update(changes)
    return TaskSpec(**values)


def test_plant_hashes_separate_physics_condition_study_and_declaration():
    base = _plant()
    physical = _plant(plant={"gain": 2.1})
    condition = _plant(conditions={"nominal": _condition(reference=(0.4,))})
    study = _plant(study={"requirements": {"limit": 4.0}})
    note = _plant(description="new note", references=("source-b",))

    assert physical.plant_hash != base.plant_hash
    assert physical.config_hash != base.config_hash
    assert condition.plant_hash == base.plant_hash
    assert condition.condition_hashes != base.condition_hashes
    assert condition.config_hash != base.config_hash
    assert study.plant_hash == base.plant_hash
    assert study.condition_hashes == base.condition_hashes
    assert study.study_hash != base.study_hash
    assert note.plant_hash == base.plant_hash
    assert note.config_hash != base.config_hash


def test_task_hash_uses_explicit_reward_and_metric_identity():
    base = _task()
    assert replace(base, reward_id="tracking-v2").task_hash != base.task_hash
    assert replace(base, revision=2).task_hash != base.task_hash
    assert replace(base, reward=lambda *_: 0.0).task_hash == base.task_hash


def test_environment_identity_hashes_all_resolved_contracts():
    condition = _condition()
    plant = _plant()
    task = _task()
    base = EnvironmentIdentity(
        task_id=task.id,
        task_hash=task.task_hash,
        plant_id=plant.id,
        plant_hash=plant.plant_hash,
        condition_id=condition.id,
        condition_hash=condition.condition_hash,
        interface_hash="interface-a",
    )
    changed = replace(base, interface_hash="interface-b")
    assert base.env_hash != changed.env_hash
    assert base.as_dict()["integrator_id"] == "rk4-v1"
