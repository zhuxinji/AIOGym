from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import aiogym
from aiogym.workflows import DatasetReader, load_checkpoint


pytest.importorskip("stable_baselines3")

SMALL_POLICY = {"policy_kwargs": {"net_arch": [8, 8]}}
PLANT = Path("aiogym/scenarios/three_tank/default-design-v1.json")


def _sac_smoke(tmp_path, *, task, condition, plant=None):
    result = aiogym.train(
        task=task,
        plant=plant,
        condition=condition,
        algorithm="sac",
        steps=2,
        seed=4,
        eval_seeds=(14,),
        eval_max_steps=2,
        algorithm_kwargs=SMALL_POLICY,
        output=tmp_path / "train",
    )
    loaded = load_checkpoint(result["checkpoint"]["path"], algorithm="sac", device="cpu")
    replay = aiogym.evaluate(
        loaded,
        task=task,
        plant=plant,
        condition=condition,
        seeds=(15,),
        max_steps=2,
    )
    assert replay["plant_hash"] == result["plant_hash"]
    assert replay["task_hash"] == result["task_hash"]
    return result


def test_three_tank_plant_to_study_data_train_and_evaluate(tmp_path):
    plant = aiogym.load_plant(PLANT)
    study = aiogym.study(plant, robustness_samples=0, output=tmp_path / "study")
    env = aiogym.make_env(
        "three_tank/regulation", plant=plant, condition="commissioning"
    )
    try:
        pid = aiogym.make_controller("pid", env=env)
        baseline = aiogym.evaluate(pid, seeds=(7,), max_steps=2)
        mpc = aiogym.make_controller("mpc", env=env)
        mpc_result = aiogym.evaluate(mpc, seeds=(7,), max_steps=2)
        assert np.isfinite(mpc_result["aggregate"]["return"]["mean"])
        collected = aiogym.collect(
            task="three_tank/regulation",
            plant=plant,
            condition="commissioning",
            policy=pid,
            episodes=1,
            max_steps=2,
            output=tmp_path / "dataset",
        )
    finally:
        env.close()
    dataset = DatasetReader(tmp_path / "dataset", verify_checksums=True)
    assert dataset.load_episode(0).transition_count == 2
    trained = _sac_smoke(
        tmp_path,
        task="three_tank/regulation",
        condition="commissioning",
        plant=plant,
    )
    hashes = {
        study["plant_hash"],
        baseline["plant_hash"],
        collected["plant_hash"],
        trained["plant_hash"],
    }
    assert hashes == {plant.plant_hash}
    assert {
        study["task_hash"],
        baseline["task_hash"],
        collected["task_hash"],
        trained["task_hash"],
    } == {baseline["task_hash"]}


def test_quadruple_controller_data_and_sac_path(tmp_path):
    env = aiogym.make_env("quadruple/regulation", condition="minimum-phase")
    try:
        pid = aiogym.make_controller("pid", env=env)
        mpc = aiogym.make_controller("mpc", env=env)
        for policy in (pid, mpc):
            result = aiogym.evaluate(policy, seeds=(0,), max_steps=2)
            assert result["episodes"][0]["steps"] == 2
        collected = aiogym.collect(
            task="quadruple/regulation",
            condition="minimum-phase",
            policy=pid,
            episodes=1,
            max_steps=2,
            output=tmp_path / "dataset",
        )
    finally:
        env.close()
    trained = _sac_smoke(
        tmp_path, task="quadruple/regulation", condition="minimum-phase"
    )
    assert collected["plant_hash"] == trained["plant_hash"]


def test_open_cascade_tasks_share_plant_but_separate_objectives(tmp_path):
    regulation = aiogym.evaluate(
        "pid",
        task="three_tank/regulation",
        plant="open-cascade-v1",
        condition="continuous-benchmark",
        seeds=(0,),
        max_steps=2,
    )
    economic = aiogym.evaluate(
        "hold",
        task="three_tank/economic",
        plant="open-cascade-v1",
        condition="continuous-benchmark",
        seeds=(0,),
        max_steps=2,
    )
    assert regulation["plant_hash"] == economic["plant_hash"]
    assert regulation["task_hash"] != economic["task_hash"]
    assert "tracking_iae" in regulation["aggregate"]
    assert "economic_objective" in economic["aggregate"]
    trained = _sac_smoke(
        tmp_path,
        task="three_tank/regulation",
        plant="open-cascade-v1",
        condition="continuous-benchmark",
    )
    assert trained["plant_hash"] == regulation["plant_hash"]
