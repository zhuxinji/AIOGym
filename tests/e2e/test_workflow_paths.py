from __future__ import annotations

import numpy as np
import pytest

import aiogym


pytest.importorskip("stable_baselines3")

pytestmark = [pytest.mark.rl, pytest.mark.e2e]

SMALL_POLICY = {"policy_kwargs": {"net_arch": [8, 8]}}


@pytest.mark.parametrize(
    ("scenario", "reward"),
    [
        ("quadruple", "regulation"),
        ("three_tank", "regulation"),
    ],
)
def test_data_train_load_evaluate_and_compare_pipeline(
    tmp_path,
    scenario,
    reward,
):
    env = aiogym.make_env(scenario, reward=reward)
    try:
        pid = aiogym.make_controller("pid", env=env)
        mpc = aiogym.make_controller("mpc", env=env)
        collected = aiogym.collect(
            env=env,
            policy=pid,
            episodes=1,
            seed=3,
            max_steps=2,
            output=tmp_path / scenario / "dataset",
        )
        trained = aiogym.train(
            env=env,
            algorithm="sac",
            steps=2,
            seed=4,
            algorithm_kwargs=SMALL_POLICY,
            output=tmp_path / scenario / "train",
        )
        learned = aiogym.load_policy(
            trained["checkpoint"],
            algorithm="sac",
            env=env,
        )
        single = aiogym.evaluate(
            env=env,
            policy=learned,
            seeds=(7, 8),
            max_steps=2,
        )
        comparison = aiogym.compare_policies(
            env=env,
            policies={"pid": pid, "mpc": mpc, "sac": learned},
            seeds=(7, 8),
            max_steps=2,
            output=tmp_path / scenario / "comparison",
        )
    finally:
        env.close()

    dataset = aiogym.DatasetReader(collected["path"])
    assert len(dataset) == 1
    assert dataset.transition_count == 2
    assert single["seeds"] == [7, 8]
    assert comparison["seeds"] == [7, 8]
    assert comparison["evaluations"]["sac"] == single
    assert set(comparison["ordering"]) == {"pid", "mpc", "sac"}
    assert np.isfinite(
        comparison["evaluations"]["pid"]["aggregate"]["episode_return"][
            "mean"
        ]
    )


def test_three_tank_rewards_have_separate_descriptive_objectives():
    regulation_env = aiogym.make_env("three_tank", reward="regulation")
    economic_env = aiogym.make_env("three_tank", reward="economic")
    try:
        regulation = aiogym.evaluate(
            env=regulation_env,
            policy="pid",
            seeds=(0,),
            max_steps=2,
        )
        economic = aiogym.evaluate(
            env=economic_env,
            policy="hold",
            seeds=(0,),
            max_steps=2,
        )
    finally:
        regulation_env.close()
        economic_env.close()
    assert "tracking_iae" in regulation["aggregate"]
    assert "economic_objective" in economic["aggregate"]
