from __future__ import annotations

import numpy as np
import pytest

import aiogym


pytest.importorskip("stable_baselines3")

pytestmark = [pytest.mark.rl, pytest.mark.e2e]


def test_cstr_collect_train_load_evaluate_compare_pipeline(tmp_path):
    env = aiogym.make_env("cstr", reward="regulation")
    try:
        pid = aiogym.make_controller("pid", env=env)
        mpc = aiogym.make_controller("mpc", env=env)
        collected = aiogym.collect(
            env=env,
            policy=pid,
            episodes=1,
            seed=3,
            max_steps=2,
            output=tmp_path / "dataset",
        )
        trained = aiogym.train(
            env=env,
            algorithm="sac",
            steps=2,
            seed=4,
            algorithm_kwargs={"policy_kwargs": {"net_arch": [8, 8]}},
            dataset=collected["path"],
            behavior_cloning_epochs=1,
            behavior_cloning_batch_size=2,
            output=tmp_path / "train",
        )
        learned = aiogym.load_policy(trained["checkpoint"], env=env)
        evaluation = aiogym.evaluate(
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
            output=tmp_path / "comparison",
        )
    finally:
        env.close()

    dataset = aiogym.DatasetReader(collected["path"])
    assert len(dataset) == 1
    assert dataset.transition_count == 2
    assert comparison["evaluations"]["sac"]["aggregate"] == evaluation["aggregate"]
    assert (tmp_path / "comparison" / "trajectories.npz").is_file()
    assert set(comparison["ordering"]) == {"pid", "mpc", "sac"}
    assert np.isfinite(
        comparison["evaluations"]["pid"]["aggregate"]["return"]["mean"]
    )
